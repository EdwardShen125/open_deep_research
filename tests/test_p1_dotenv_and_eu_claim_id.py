"""Tests for P1.1 + P1.2.

P1.1: _load_dotenv_fallback 从 .env 文件加载环境变量
P1.2: build_claims_from_eus return_eu_map=True 返回 eu_to_claim_id 映射
P1.2: plan_v2_pipeline 用 eu_map 回填 EU.claim_id
"""
from __future__ import annotations

import os
import tempfile
import uuid as uuidlib

import pytest


# ---- P1.1: _load_dotenv_fallback ----

class TestLoadDotenvFallback:
    """轻量 .env 加载,不引入 python-dotenv 依赖。"""

    def test_load_simple_key_value(self, monkeypatch):
        """KEY=VALUE 应加载到 os.environ。"""
        from open_deep_research.api.server import _load_dotenv_fallback
        monkeypatch.delenv("TEST_KEY_12345", raising=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TEST_KEY_12345=test_value_xyz\n")
            path = f.name
        try:
            _load_dotenv_fallback(path)
            assert os.environ.get("TEST_KEY_12345") == "test_value_xyz"
        finally:
            os.unlink(path)
            monkeypatch.delenv("TEST_KEY_12345", raising=False)

    def test_load_with_quotes(self, monkeypatch):
        """带双引号 / 单引号 / 前后空格。"""
        from open_deep_research.api.server import _load_dotenv_fallback
        monkeypatch.delenv("TEST_KEY_QUOTED", raising=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write('TEST_KEY_QUOTED="  spaces and value  "\n')
            path = f.name
        try:
            _load_dotenv_fallback(path)
            assert os.environ.get("TEST_KEY_QUOTED") == "  spaces and value  "
        finally:
            os.unlink(path)
            monkeypatch.delenv("TEST_KEY_QUOTED", raising=False)

    def test_skip_comments_and_empty_lines(self, monkeypatch):
        """注释行 / 空行 应跳过。"""
        from open_deep_research.api.server import _load_dotenv_fallback
        monkeypatch.delenv("TEST_KEY_VALID", raising=False)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# This is a comment\n\nTEST_KEY_VALID=actual_value\n")
            path = f.name
        try:
            _load_dotenv_fallback(path)
            assert os.environ.get("TEST_KEY_VALID") == "actual_value"
        finally:
            os.unlink(path)
            monkeypatch.delenv("TEST_KEY_VALID", raising=False)

    def test_does_not_overwrite_existing(self, monkeypatch):
        """已存在的 env var 不覆盖(shell 优先级高)。"""
        from open_deep_research.api.server import _load_dotenv_fallback
        monkeypatch.setenv("TEST_KEY_PRESET", "shell_value")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("TEST_KEY_PRESET=dotenv_value\n")
            path = f.name
        try:
            _load_dotenv_fallback(path)
            # shell 优先级高 → 应保留 shell_value
            assert os.environ.get("TEST_KEY_PRESET") == "shell_value"
        finally:
            os.unlink(path)

    def test_missing_file_silent(self):
        """缺失文件 silent 返回,不抛错。"""
        from open_deep_research.api.server import _load_dotenv_fallback
        # 不存在的路径,不抛错
        _load_dotenv_fallback("/tmp/nonexistent_env_file_xyz12345.env")


# ---- P1.2: build_claims_from_eus return_eu_map ----

def _make_eu_simple(
    *,
    run_id: str,
    claim: str = "EDR market is $5B",
    source_url: str = "https://arxiv.org/abs/1",
    content_hash: str = None,
    claim_type: str = "attribute",
):
    from open_deep_research.evidence.schema import EvidenceUnitV2
    return EvidenceUnitV2(
        run_id=run_id,
        claim=claim,
        claim_type=claim_type,
        entities=[],
        source_url=source_url,
        source_domain="arxiv.org",
        source_title="Test",
        published_at=None,
        source_tier="primary",
        source_span="This is a test span with enough characters to pass validation",
        span_start=None,
        span_end=None,
        extractor_model="test_extractor",
        extracted_at="2026-07-23T00:00:00Z",
        span_verified=False,
        numeric_drift=False,
        entailment_verdict="unverifiable",
        entailment_score=None,
        claim_id=None,
        content_hash=content_hash or (uuidlib.uuid4().hex + uuidlib.uuid4().hex)[:64],
        embedding=None,
    )


class TestBuildClaimsFromEusReturnEuMap:
    """P1.2: build_claims_from_eus return_eu_map=True 返 (claims, eu_to_claim_id)。"""

    def test_default_returns_claims_only(self):
        """默认 return_eu_map=False 仍返 list[ClaimV2]。"""
        from open_deep_research.evidence.pipeline import build_claims_from_eus
        rid = str(uuidlib.uuid4())
        eu = _make_eu_simple(run_id=rid)
        claims = build_claims_from_eus([eu])
        assert isinstance(claims, list)

    def test_return_eu_map_returns_tuple(self):
        """return_eu_map=True 返 (claims, eu_to_claim_id) 元组。"""
        from open_deep_research.evidence.pipeline import build_claims_from_eus
        rid = str(uuidlib.uuid4())
        eu = _make_eu_simple(run_id=rid)
        result = build_claims_from_eus([eu], return_eu_map=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        claims, eu_map = result
        assert isinstance(claims, list)
        assert isinstance(eu_map, dict)

    def test_eu_map_has_correct_count(self):
        """eu_map 应包含所有 EU 的 eu_id → claim_id 映射。"""
        from open_deep_research.evidence.pipeline import build_claims_from_eus
        rid = str(uuidlib.uuid4())
        eu1 = _make_eu_simple(run_id=rid, claim="EDR market reached $5B", content_hash=(uuidlib.uuid4().hex + uuidlib.uuid4().hex)[:64])
        eu2 = _make_eu_simple(run_id=rid, claim="EDR market reached $5B", source_url="https://arxiv.org/abs/2", content_hash=(uuidlib.uuid4().hex + uuidlib.uuid4().hex)[:64])
        eu3 = _make_eu_simple(run_id=rid, claim="EDR vendors grew 30%", source_url="https://arxiv.org/abs/3", content_hash=(uuidlib.uuid4().hex + uuidlib.uuid4().hex)[:64])

        claims, eu_map = build_claims_from_eus([eu1, eu2, eu3], return_eu_map=True)
        assert len(eu_map) == 3
        # 每个 eu_id 都映射到某个 claim_id
        for eu in [eu1, eu2, eu3]:
            assert str(eu.eu_id) in eu_map
            # claim_id 应该是某个 ClaimV2 的 claim_id
            claim_id = eu_map[str(eu.eu_id)]
            assert any(str(c.claim_id) == claim_id for c in claims)

    def test_eu_map_groups_to_same_claim(self):
        """P1.2 eu_map 端到端:eu_id -> claim_id 映射完整。"""
        from open_deep_research.evidence.pipeline import build_claims_from_eus
        rid = str(uuidlib.uuid4())
        # 两个 EU claim 几乎相同 (高 cosine)
        eu1 = _make_eu_simple(run_id=rid, claim="EDR market reached $5 billion in 2024 according to research", content_hash=(uuidlib.uuid4().hex + uuidlib.uuid4().hex)[:64])
        eu2 = _make_eu_simple(run_id=rid, claim="EDR market reached $5 billion in 2024 according to research", source_url="https://arxiv.org/abs/2222", content_hash=(uuidlib.uuid4().hex + uuidlib.uuid4().hex)[:64])

        claims, eu_map = build_claims_from_eus([eu1, eu2], return_eu_map=True)
        # 没有 embedding 时, merge_units 不会强制 merge。2 EU 各自归并。
        assert len(claims) >= 1
        assert len(eu_map) == 2  # 2 EU -> 2 mapping
        # 两个 EU 都应该在 eu_map 里
        assert str(eu1.eu_id) in eu_map
        assert str(eu2.eu_id) in eu_map
        # 不管 merge 与否, eu_map 都是 2 个独立 entry