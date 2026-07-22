"""Phase P0 — 真集成验证的 embedder 测试。

覆盖:
  1. Hash pseudo-vector:deterministic / 1024-dim / L2-normalized / NaN-free
  2. embed_texts(model="hash"):接口形态 + 返回 dtype + batch 处理
  3. embedder_status():返回 dict
  4. 跟 EuDAO upsert_many 走真 PG(集成测试,需要 pgvector + 真连接)
  5. BGE-M3 真模型路径(可选 — 如果 sentence-transformers + model cache 可用)

不依赖任何外部 LLM / 网络。
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from open_deep_research.evidence.embedder import (
    EMBED_DIM,
    _hash_pseudo_vector,
    embed_texts,
    embedder_status,
)


# =============================================================================
# 1. Hash pseudo-vector
# =============================================================================

class TestHashPseudoVector:
    def test_dim(self):
        v = _hash_pseudo_vector("hello world")
        assert v.shape == (EMBED_DIM,), f"expected ({EMBED_DIM},), got {v.shape}"
        assert v.dtype == np.float32

    def test_deterministic(self):
        v1 = _hash_pseudo_vector("The AI market reached $94 billion in 2024")
        v2 = _hash_pseudo_vector("The AI market reached $94 billion in 2024")
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self):
        v1 = _hash_pseudo_vector("AI market $94B 2024")
        v2 = _hash_pseudo_vector("Coffee market $94B 2024")
        # cosine sim 应该 ≈ 0(不语义,只哈希)
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert cos < 0.5, f"hash vectors should be roughly orthogonal, got cos={cos}"

    def test_l2_normalized(self):
        v = _hash_pseudo_vector("test")
        norm = float(np.linalg.norm(v.astype(np.float64)))
        assert abs(norm - 1.0) < 1e-4, f"expected L2 norm ≈ 1.0, got {norm}"

    def test_no_nan_no_inf(self):
        """NaN/Inf 防护 — pgvector 不接受 NaN,这条断言很关键。"""
        for text in [
            "",
            " " * 100,
            "🐍 unicode emoji test 中文 测试 αβγ",
            "a" * 1000,
            "\x00\x00\x00",
        ]:
            v = _hash_pseudo_vector(text)
            assert np.all(np.isfinite(v)), f"NaN/Inf in hash vector for text={text!r}"
            assert not np.any(np.isnan(v))

    def test_custom_dim(self):
        v = _hash_pseudo_vector("test", dim=512)
        assert v.shape == (512,)

    def test_empty_string_safe(self):
        v = _hash_pseudo_vector("")
        assert v.shape == (EMBED_DIM,)
        assert np.all(np.isfinite(v))


# =============================================================================
# 2. embed_texts() interface
# =============================================================================

class TestEmbedTextsHash:
    """用 model='hash' 路径(强制 fallback,不需要 BGE-M3)。"""

    def test_basic_batch(self):
        texts = [
            "AI market reached $94 billion in 2024",
            "65% of organizations adopted AI",
            "EU AI Act entered force August 2024",
        ]
        vecs = embed_texts(texts, model="hash")
        assert vecs.shape == (3, EMBED_DIM)
        assert vecs.dtype == np.float32
        # L2-normalized
        norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_empty_input(self):
        vecs = embed_texts([], model="hash")
        assert vecs.shape == (0, EMBED_DIM)

    def test_iterator_input(self):
        """embed_texts 接受 Iterable,不是只能 list。"""
        gen = (t for t in ["a", "b", "c"])
        vecs = embed_texts(gen, model="hash")
        assert vecs.shape == (3, EMBED_DIM)

    def test_none_safe(self):
        """None / 空字符串不会崩(被转为 '')。"""
        texts: list[object] = ["hello", None, "", "world"]
        vecs = embed_texts(texts, model="hash")  # type: ignore[arg-type]
        assert vecs.shape == (4, EMBED_DIM)
        assert np.all(np.isfinite(vecs))


class TestEmbedTextsDefault:
    """embed_texts(model 默认 = 'bge-m3') — 没真模型时自动 fallback 到 hash。"""

    def test_default_falls_back_safely(self):
        """没有 BGE-M3 安装时,默认 model='bge-m3' 仍应工作(降级到 hash)。"""
        vecs = embed_texts(["test 1", "test 2"], model="bge-m3")
        # 不管真模型是否可用,都应返回 (2, EMBED_DIM) 数组
        assert vecs.shape == (2, EMBED_DIM)
        assert np.all(np.isfinite(vecs))


# =============================================================================
# 3. embedder_status()
# =============================================================================

class TestEmbedderStatus:
    def test_returns_dict(self):
        s = embedder_status()
        assert isinstance(s, dict)
        assert "dim" in s
        assert s["dim"] == EMBED_DIM
        assert "loaded" in s
        assert "load_attempted" in s


# =============================================================================
# 4. 集成测试(需要真 PG + pgvector)— 环境变量控开关
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Set INTEGRATION_TESTS=1 to run; requires live PG with pgvector",
)
class TestEmbedderPgIntegration:
    """end-to-end:embedder + EuDAO 真 PG 落库 + HNSW 真检索。"""

    PG_HOST = os.environ.get("POSTGRES_HOST", "172.17.0.2")
    PG_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
    PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "odr_v2_pg_pass_change_me")

    def _eu(self, run_id, claim: str):
        from uuid import uuid4
        from open_deep_research.evidence.schema import EvidenceUnitV2
        return EvidenceUnitV2(
            run_id=run_id,
            claim=claim[:300],
            claim_type="attribute",
            entities=[],
            source_url="https://example.com/test",
            source_domain="example.com",
            source_tier="secondary",
            source_span=claim[: min(220, len(claim))],
            span_start=0,
            span_end=min(220, len(claim)),
            extractor_model="test_embedder_pg",
        )

    def test_embedder_then_pg_roundtrip(self):
        """embedder 算向量 → upsert 进 PG(带 embedding 列)→ search_by_embedding 真检索回来。"""
        from open_deep_research.evidence.eu_dao import EuDAO

        os.environ["POSTGRES_HOST"] = self.PG_HOST
        os.environ["POSTGRES_PORT"] = str(self.PG_PORT)
        os.environ["POSTGRES_PASSWORD"] = self.PG_PASSWORD

        from uuid import uuid4
        rid = uuid4()

        eus = [self._eu(rid, c) for c in [
            "AI market reached $94 billion in 2024",
            "EU AI Act entered force in August 2024",
            "Coffee consumption grew 3% globally",
            "Renewable energy adoption hit 72%",
        ]]
        vecs = embed_texts([eu.claim for eu in eus], model="hash")
        for eu, v in zip(eus, vecs):
            eu.embedding = v.tolist()

        with EuDAO() as dao:
            ids = dao.upsert_many(eus)
            assert len(ids) == 4

            # search_by_embedding 真检索
            results = dao.search_by_embedding(rid, vecs[0].tolist(), limit=4)
            assert len(results) == 4
            # 第一个结果应该是 query 自身(sim=1.0)
            top_eu, top_sim = results[0]
            assert top_sim > 0.99, f"top result should be self (sim=1.0), got {top_sim}"


# =============================================================================
# 5. BGE-M3 真模型路径(可选)— 仅当 sentence-transformers + 模型 cache 可用
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("RUN_BGE_TESTS"),
    reason="Set RUN_BGE_TESTS=1 to attempt real BGE-M3 load (requires ~2.3GB model download)",
)
class TestEmbedderBGEReal:
    def test_bge_m3_real_model_load(self):
        """真 BGE-M3 路径(下载 ~2.3GB,首次冷启动慢)。

        测试目的:
          - 确认真模型能 load 进 CPU
          - 确认维度仍是 1024
          - 确认输出 L2-normalized
        """
        # 触发模型 load
        from open_deep_research.evidence.embedder import _BGEModelSingleton

        m = _BGEModelSingleton.get()
        if m is None:
            pytest.skip("BGE-M3 model load failed; see embedder_status() for reason")
        assert m is not None
        status = _BGEModelSingleton.status()
        assert status["loaded"] is True
        assert status["load_failed"] is False

        vecs = embed_texts(["test sentence"], model="bge-m3")
        assert vecs.shape == (1, EMBED_DIM)
        norms = np.linalg.norm(vecs.astype(np.float64), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-3)


__all__ = [
    "TestHashPseudoVector",
    "TestEmbedTextsHash",
    "TestEmbedTextsDefault",
    "TestEmbedderStatus",
    "TestEmbedderPgIntegration",
    "TestEmbedderBGEReal",
]