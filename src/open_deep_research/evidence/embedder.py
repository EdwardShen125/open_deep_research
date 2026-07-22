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
from typing import Iterable, Optional

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
    """Lazy-load BGE-M3 model on first call.

    Model file 在 ~/.cache/huggingface/hub/;如果下载失败(_load raises),
    调用方自动 fallback 到 hash_pseudo_vector。
    """

    _model: Optional[object] = None
    _load_attempted: bool = False
    _load_failed: bool = False
    _last_error: Optional[str] = None

    @classmethod
    def get(cls):
        if cls._model is not None:
            return cls._model
        if cls._load_attempted:
            return None  # 已失败过,直接返回 None 触发 fallback
        cls._load_attempted = True
        try:
            # 强制 CPU,避免无 GPU 环境崩
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            from sentence_transformers import SentenceTransformer  # type: ignore

            cls._model = SentenceTransformer(
                "BAAI/bge-m3",
                device="cpu",
                trust_remote_code=False,
            )
            logger.info("BGE-M3 loaded: dim=%d", cls._model.get_sentence_embedding_dimension())
            return cls._model
        except Exception as e:
            cls._load_failed = True
            cls._last_error = repr(e)[:200]
            logger.warning(
                "BGE-M3 load failed; falling back to hash pseudo-vectors: %s",
                cls._last_error,
            )
            return None

    @classmethod
    def status(cls) -> dict:
        return {
            "loaded": cls._model is not None,
            "load_attempted": cls._load_attempted,
            "load_failed": cls._load_failed,
            "last_error": cls._last_error,
        }


def embed_texts(
    texts: Iterable[str],
    *,
    model: str = "bge-m3",
    batch_size: int = 16,
) -> np.ndarray:
    """Embed a batch of texts → np.ndarray (N, EMBED_DIM).

    Args:
        texts: any iterable of strings
        model: "bge-m3" (真模型)/ "hash" (强制 fallback)
        batch_size: BGE-M3 encode batch_size

    Returns:
        np.ndarray shape (N, EMBED_DIM), dtype float32, L2-normalized
    """
    texts_list = [t if t else "" for t in texts]
    n = len(texts_list)

    if n == 0:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    if model == "hash":
        return np.stack([_hash_pseudo_vector(t) for t in texts_list])

    # 真 BGE-M3 路径
    m = _BGEModelSingleton.get()
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
        if arr.shape != (n, EMBED_DIM):
            logger.warning(
                "BGE-M3 returned shape %s, expected (%d, %d); fallback",
                arr.shape, n, EMBED_DIM,
            )
            return np.stack([_hash_pseudo_vector(t) for t in texts_list])
        return arr
    except Exception as e:
        logger.warning("BGE-M3 encode failed; fallback: %s", e)
        return np.stack([_hash_pseudo_vector(t) for t in texts_list])


def embedder_status() -> dict:
    """Return embedder backend status (for diagnostics / ReportResult metadata)."""
    s = _BGEModelSingleton.status()
    s["dim"] = EMBED_DIM
    return s


__all__ = [
    "EMBED_DIM",
    "embed_texts",
    "embedder_status",
    "_hash_pseudo_vector",
]