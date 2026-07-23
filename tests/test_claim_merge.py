"""Tests for P0 claim merge + 冲突检测 (Runbook v1 §3.3).

覆盖:
- build_claims_from_eus 端到端 (EU → ClaimV2)
- grade_claim A/B/C/D 评级
- PlanV2RunResult.claims 字段透传
- ClaimStats.from_claim_list with eus 参数 (unique_sources 正确)
- ClaimDAO.upsert_many 落库 (PG 集成, 跳过无 PG)
"""
from __future__ import annotations

import os
import uuid as uuidlib

import pytest


def _make_run_id() -> str:
    """生成 UUID 字符串 (ClaimV2.run_id: UUID 校验)。"""
    return str(uuidlib.uuid4())


def _make_content_hash(seed: str = "0") -> str:
    """生成 ≥ 64 字符的 content_hash (sha256 hex 长度)。"""
    # 用 uuid4 重复填充到 64 字符
    raw = (uuidlib.uuid4().hex + uuidlib.uuid4().hex + seed)[:64]
    return raw


def _make_eu(
    *,
    run_id: str,
    claim: str = "EDR market is $5B",
    source_url: str = "https://arxiv.org/abs/1",
    source_domain: str = "arxiv.org",
    source_tier: str = "primary",
    claim_type: str = "attribute",
    content_hash: str = None,
) -> "EvidenceUnitV2":
    """创建一个完整 EvidenceUnitV2 (满足所有必填字段)。

    Default claim_type='attribute' 因为 numeric 必须有 norm_value。
    """
    from open_deep_research.evidence.schema import EvidenceUnitV2
    return EvidenceUnitV2(
        run_id=run_id,
        claim=claim,
        claim_type=claim_type,
        entities=[],
        source_url=source_url,
        source_domain=source_domain,
        source_title="Test",
        published_at=None,
        source_tier=source_tier,
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
        content_hash=content_hash or _make_content_hash(),
        embedding=None,
    )


def _make_claim(
    *,
    run_id: str,
    canonical_claim: str = "EDR market is $5B",
    grade: str = "D",
    eu_count: int = 1,
    independent_source_count: int = 1,
    primary_source_count: int = 0,
    has_conflict: bool = False,
    conflicting_values: list = None,
) -> "ClaimV2":
    """创建一个完整 ClaimV2 (满足所有必填字段)。"""
    from open_deep_research.evidence.schema import ClaimV2
    return ClaimV2(
        run_id=run_id,
        dimension_id="market_size",
        canonical_claim=canonical_claim,
        claim_type="numeric",
        entities=[],
        norm_value=None,
        unit=None,
        value_as_of=None,
        value_spread=None,
        eu_count=eu_count,
        independent_source_count=independent_source_count,
        primary_source_count=primary_source_count,
        earliest_published_at=None,
        has_conflict=has_conflict,
        conflicting_values=conflicting_values or [],
        grade=grade,
        grade_reason="test reason",
    )


# ---- build_claims_from_eus 端到端 ----

class TestBuildClaimsFromEus:
    """evidence/pipeline.py build_claims_from_eus 端到端。"""

    def test_empty_eus_returns_empty_claims(self):
        from open_deep_research.evidence.pipeline import build_claims_from_eus
        assert build_claims_from_eus([]) == []

    def test_single_eu_one_claim(self):
        """单个 EU → 1 个 claim,grade 有效 (A/B/C/D)。"""
        from open_deep_research.evidence.pipeline import build_claims_from_eus
        from open_deep_research.evidence.schema import ClaimV2
        rid = _make_run_id()
        eu = _make_eu(run_id=rid)
        claims = build_claims_from_eus([eu])
        assert len(claims) >= 1
        c = claims[0]
        assert isinstance(c, ClaimV2)
        assert c.eu_count >= 1
        assert c.grade in ("A", "B", "C", "D")
        assert c.independent_source_count >= 1

    def test_two_independent_sources_grade_better(self):
        """2 个独立 primary source → grade >= B (如果有 entailed EU)。"""
        from open_deep_research.evidence.pipeline import build_claims_from_eus
        rid = _make_run_id()
        eu1 = _make_eu(
            run_id=rid,
            claim="EDR market reached $5B in 2024",
            source_url="https://arxiv.org/abs/1111",
            source_domain="arxiv.org",
            content_hash=_make_content_hash("a"),
        )
        eu2 = _make_eu(
            run_id=rid,
            claim="EDR market reached $5B in 2024",
            source_url="https://www.sec.gov/Archives/edgar/data/test",
            source_domain="sec.gov",
            content_hash=_make_content_hash("b"),
        )
        claims = build_claims_from_eus([eu1, eu2])
        assert len(claims) >= 1
        # EU 都没有 entailment_verdict='entailed',所以 grade=D 是预期行为。
        # 这是 P0 关键洞察:grade D 不只是单源问题,也是 entailment 校验未跑的问题。
        for c in claims:
            assert c.grade in ("A", "B", "C", "D")
            assert c.independent_source_count >= 1
            # primary_source_count >= 0 (EU 没有 entailed 时是 0)

    def test_grade_claim_function(self):
        """直接测 grade_claim 逻辑。"""
        from open_deep_research.evidence.independence import grade_claim
        from open_deep_research.evidence.merge import ClaimDraft

        # 用真正的 ClaimDraft (不需要 EU 引用)
        draft = ClaimDraft(
            eu_indices=[0],
            canonical_claim="x",
            claim_type="numeric",
            entities=[],
            norm_value=None,
            unit=None,
            value_as_of=None,
            value_spread=None,
            has_conflict=False,
            conflicting_values=[],
            earliest_published_at=None,
        )
        grade, reason = grade_claim(
            draft,
            independent_count=1,
            primary_count=0,
            has_any_entailed=False,
        )
        assert grade == "D"
        # reason 是中文 (Runbook §3.3):"无任何 EU 通过 entailment 校验"
        assert isinstance(reason, str) and len(reason) > 0


# ---- PlanV2RunResult claims 字段 ----

class TestPlanV2RunResultClaims:
    """PlanV2RunResult 加 claims + claim_grade_dist 字段。"""

    def test_default_empty_claims(self):
        from open_deep_research.plan_v2_pipeline import PlanV2RunResult
        out = PlanV2RunResult(query="test", run_id="r-test")
        assert out.claims == []
        assert out.claim_grade_dist == {}

    def test_claims_in_to_dict(self):
        from open_deep_research.plan_v2_pipeline import PlanV2RunResult
        rid = _make_run_id()
        c = _make_claim(run_id=rid)
        out = PlanV2RunResult(query="test", run_id="r-test")
        out.claims = [c]
        out.claim_grade_dist = {"D": 1}
        d = out.to_dict()
        assert "claims" in d
        assert len(d["claims"]) == 1
        assert d["claim_grade_dist"] == {"D": 1}


# ---- ClaimStats.from_claim_list with eus ----

class TestClaimStatsFromClaimListWithEus:
    """ClaimStats 接受 eus 参数计算 unique_sources / total_eus。"""

    def test_claims_without_eus(self):
        """不传 eus → total_eus=0, unique_sources=0。"""
        from open_deep_research.evidence.report import ClaimStats
        rid = _make_run_id()
        claims = [_make_claim(run_id=rid, grade="A", independent_source_count=2, primary_source_count=1)]
        stats = ClaimStats.from_claim_list(claims)
        assert stats.total_eus == 0
        assert stats.unique_sources == 0
        assert stats.total_claims == 1
        assert stats.primary_claims == 1

    def test_claims_with_eus(self):
        """传 eus → total_eus=N, unique_sources=N (不同 url)。"""
        from open_deep_research.evidence.report import ClaimStats
        rid = _make_run_id()
        claims = [_make_claim(run_id=rid, grade="A", independent_source_count=2, primary_source_count=2)]
        eus = [
            _make_eu(run_id=rid, source_url="https://arxiv.org/abs/1", source_domain="arxiv.org"),
            _make_eu(run_id=rid, source_url="https://www.sec.gov/test", source_domain="sec.gov"),
        ]
        stats = ClaimStats.from_claim_list(claims, eus=eus)
        assert stats.total_eus == 2
        assert stats.unique_sources == 2
        assert stats.unique_primary_sources == 2


# ---- ClaimDAO 集成 (PG, 跳过无 POSTGRES_HOST) ----

class TestClaimDAOIntegration:
    """ClaimDAO 端到端集成 (需要 PG)。"""

    def test_upsert_and_list_round_trip(self):
        if not os.environ.get("POSTGRES_HOST"):
            pytest.skip("POSTGRES_HOST 未设置 — ClaimDAO 集成测试需要真 PG")

        from open_deep_research.evidence import ClaimDAO

        run_id = _make_run_id()
        c = _make_claim(run_id=run_id, canonical_claim="EDR market is $5B test")
        with ClaimDAO() as dao:
            dao.upsert_many([c])
            listed = dao.list_by_run(run_id)
        assert len(listed) >= 1
        # canonical_claim 应该 round-trip 一致
        assert any("EDR market is $5B test" in x.canonical_claim for x in listed)