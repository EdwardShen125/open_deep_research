"""P0 真集成验证: Embedder 接口(真 BGE-M3 + hash fallback)。

设计依据: notes/evidence-pipeline-runbook-v1.md 阶段 P0 — "成本花在验证上"。

接口:`embed_texts(texts, *, model="bge-m3") -> np.ndarray` (N, 1024)。

真模型优先级:
  1. BGE-M3 (sentence-transformers,BAAI/bge-m3,1024 dim) — 首选
  2. fallback: 基于 SHA-256 的 deterministic pseudo-vectors(只为让 pipeline
     跑通 + HNSW 索引能建;**不**做真语义相似度检索)。

调用方(EuDAO.upsert_many 前的 batch hook):
    from open_deep_research.evidence.embedder import embed_texts
    vecs = embed_texts([eu.claim for eu in eus])
    for eu, v in zip(eus, vecs):
        eu.embedding = v.tolist()
    EuDAO(...).upsert_many(eus)
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBED_DIM = 1024


def _hash_pseudo_vector(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic 1024-dim pseudo-vector from text SHA-256.

    性质:
      - 同一 text → 同一 vector(可复现)
      - 不同 text → 完全不同 vector(雪崩)
      - L2-normalized(满足 cosine 距离定义,向量能用 HNSW 索引)
      - 完全不语义,但能验证 pipeline 通 + HNSW 真能跑
    """
    seed = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    # 用 seed 反复 digest 直到够 dim 字节
    buf = b""
    i = 0
    while len(buf) < dim:
        buf += hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        i += 1
    # int8(-128..127)→ float32,避免 byte 0..255 → float32 后的 sum overflow
    arr = np.frombuffer(buf[: dim], dtype=np.int8).astype(np.float32)
    arr = arr - float(arr.astype(np.float64).mean())
    norm = float(np.linalg.norm(arr.astype(np.float64)))
    if norm < 1e-12:
        # 极罕见:全 0 字节 → 退化为均匀分布
        arr = np.ones(dim, dtype=np.float32)
        norm = float(np.sqrt(float(dim)))
    arr = (arr.astype(np.float64) / norm).astype(np.float32)
    # 终极保护:NaN/Inf 替换
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        n3 = float(np.linalg.norm(arr.astype(np.float64)))
        if n3 > 1e-12:
            arr = (arr.astype(np.float64) / n3).astype(np.float32)
        else:
            arr = np.ones(dim, dtype=np.float32) / np.sqrt(float(dim))
    return arr.astype(np.float32)


class _BGEModelSingleton:
    """Lazy-load a sentence-transformers model on first call.

    Default model: sentence-transformers/all-MiniLM-L6-v2 (80MB, 384 dim).
    Why MiniLM over BGE-M3:
      - MiniLM downloads in ~10s on good connection vs BGE-M3 2.2GB
      - 384-dim embeddings are sufficient for cosine-based dedup / cluster
      - BGE-M3 shines for multi-lingual + long-doc (>512 token), neither of
        which our sentence-level EU uses (we extract sentences ≤200 chars)
    Model file 在 ~/.cache/huggingface/hub/;如果下载失败 / 超时,
    调用方自动 fallback 到 hash_pseudo_vector。

    关键: load 必须在子线程里跑 + 自身 timeout,防止网络挂起阻塞 caller。
    """

    _model: Optional[object] = None
    _model_name: Optional[str] = None
    _model_dim: Optional[int] = None
    _load_attempted: bool = False
    _load_failed: bool = False
    _last_error: Optional[str] = None
    _load_timeout_seconds: int = 180

    @classmethod
    def get(cls, model_name: Optional[str] = None):
        # model_name 切换会重置 singleton(支持运行时切换)
        if cls._model is not None and (model_name is None or model_name == cls._model_name):
            return cls._model
        if cls._load_attempted and model_name in (None, cls._model_name):
            return None  # 已失败过,直接返回 None 触发 fallback
        if model_name is not None and model_name != cls._model_name:
            # 显式切到别的模型 → 重置状态
            cls._model = None
            cls._load_attempted = False
            cls._load_failed = False
            cls._last_error = None
        cls._load_attempted = True
        target_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        try:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            from sentence_transformers import SentenceTransformer  # type: ignore

            # 在子线程里 load + 加 timeout,防止网络挂起阻塞 caller
            import threading

            result: dict[str, Any] = {}

            def _load() -> None:
                try:
                    result["model"] = SentenceTransformer(
                        target_name,
                        device="cpu",
                        trust_remote_code=False,
                    )
                except Exception as e:
                    result["error"] = e

            t = threading.Thread(target=_load, daemon=True)
            t.start()
            t.join(timeout=cls._load_timeout_seconds)
            if t.is_alive():
                # 子线程超时 → 主线程继续,fallback
                cls._load_failed = True
                cls._last_error = f"load timeout after {cls._load_timeout_seconds}s (likely network)"
                logger.warning("sentence-transformers %s", cls._last_error)
                return None
            if "error" in result:
                raise result["error"]
            cls._model = result["model"]
            cls._model_name = target_name
            cls._model_dim = cls._model.get_sentence_embedding_dimension()
            logger.info(
                "embedder loaded model=%s dim=%d",
                cls._model_name, cls._model_dim,
            )
            return cls._model
        except Exception as e:
            cls._load_failed = True
            cls._last_error = repr(e)[:200]
            logger.warning(
                "sentence-transformers load failed; falling back to hash pseudo-vectors: %s",
                cls._last_error,
            )
            return None

    @classmethod
    def status(cls) -> dict:
        return {
            "loaded": cls._model is not None,
            "model_name": cls._model_name,
            "model_dim": cls._model_dim,
            "load_attempted": cls._load_attempted,
            "load_failed": cls._load_failed,
            "last_error": cls._last_error,
        }


def embed_texts(
    texts: Iterable[str],
    *,
    model: str = "minilm",
    batch_size: int = 16,
) -> np.ndarray:
    """Embed a batch of texts → np.ndarray (N, D).

    Args:
        texts: any iterable of strings
        model:
          - "minilm" (default, ~80MB, dim=384) — sentence-transformers/all-MiniLM-L6-v2
          - "bge-m3" (~2.2GB, dim=1024) — BAAI/bge-m3 (heavyweight)
          - "hash" (强制 fallback) — SHA-256 pseudo-vec, useless for dedup
        batch_size: encode batch_size

    Returns:
        np.ndarray shape (N, D) where D = whatever the model produces,
        dtype float32, L2-normalized.

    Note: 返回维度由加载的模型决定(matching the model) — 不再硬编码 EMBED_DIM。
    Old callers that assume EMBED_DIM=1024 may need updating.
    """
    texts_list = [t if t else "" for t in texts]
    n = len(texts_list)

    if n == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    if model == "hash":
        return np.stack([_hash_pseudo_vector(t) for t in texts_list])

    # 真模型路径
    hf_name = {
        "minilm": "sentence-transformers/all-MiniLM-L6-v2",
        "bge-m3": "BAAI/bge-m3",
    }.get(model)
    if hf_name is None:
        logger.warning("embedder: unknown model=%r; using hash fallback", model)
        return np.stack([_hash_pseudo_vector(t) for t in texts_list])

    m = _BGEModelSingleton.get(hf_name)
    if m is None:
        logger.debug("embedder: using hash fallback for %d texts", n)
        return np.stack([_hash_pseudo_vector(t) for t in texts_list])

    try:
        vecs = m.encode(
            texts_list,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # 余弦距离用
            convert_to_numpy=True,
        )
        arr = np.asarray(vecs, dtype=np.float32)
        # Pad to EMBED_DIM (1024) for pgvector HNSW index compatibility.
        # HNSW index enforces a fixed dimension across all rows; older runs
        # have 1024-dim hash fallback vectors. MiniLM is 384; pad with
        # zeros to keep a single index shape across models.
        if arr.shape[1] < EMBED_DIM:
            pad = np.zeros((arr.shape[0], EMBED_DIM - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        return arr
    except Exception as e:
        logger.warning("embedder encode failed; fallback: %s", e)
        return np.stack([_hash_pseudo_vector(t) for t in texts_list])


def embedder_status() -> dict:
    """Return embedder backend status (for diagnostics / ReportResult metadata)."""
    s = _BGEModelSingleton.status()
    s["default_dim"] = EMBED_DIM  # legacy compat
    s["dim"] = s.get("model_dim") or EMBED_DIM
    return s


__all__ = [
    "EMBED_DIM",
    "embed_texts",
    "embedder_status",
    "_hash_pseudo_vector",
]