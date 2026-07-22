"""Phase 3 (= Runbook v1 阶段 1.1) schema 单元测试。

不需要 PG,纯 Pydantic 校验。
"""
from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from open_deep_research.evidence.schema import (
    ClaimV2,
    ClaimType,
    EvidenceUnitV2,
    Grade,
    SourceTier,
    Verdict,
)


# =============================================================================
# EvidenceUnitV2 校验
# =============================================================================

def _make_eu(**overrides) -> EvidenceUnitV2:
    """默认合法 EU;overrides 用于触发非法 case。"""
    defaults = dict(
        run_id=uuid.uuid4(),
        claim="Kompyte was acquired by Crayon in 2021.",
        claim_type="attribute",
        source_url="https://example.com/news/kompyte",
        source_domain="example.com",
        source_tier="secondary",
        source_span="Kompyte was acquired by Crayon in 2021.",
        extractor_model="deterministic_v1",
    )
    defaults.update(overrides)
    return EvidenceUnitV2(**defaults)


def test_eu_basic_creation():
    eu = _make_eu()
    assert isinstance(eu.eu_id, uuid.UUID)
    assert eu.usable is False  # 未过三道闸


def test_eu_usable_requires_three_signals():
    eu = _make_eu(span_verified=True, numeric_drift=False, entailment_verdict="entailed")
    assert eu.usable is True
    eu2 = _make_eu(span_verified=True, numeric_drift=True, entailment_verdict="entailed")
    assert eu2.usable is False  # numeric_drift 阻断
    eu3 = _make_eu(span_verified=True, entailment_verdict="contradicted")
    assert eu3.usable is False  # contradicted 不进 usable


def test_eu_partial_entailment_is_usable():
    eu = _make_eu(span_verified=True, entailment_verdict="partial")
    assert eu.usable is True


def test_eu_numeric_requires_value():
    with pytest.raises(Exception):  # ValidationError
        _make_eu(claim_type="numeric")
    # 提供 norm_value 应通过
    eu = _make_eu(claim_type="numeric", norm_value=Decimal("100.5"), unit="亿")
    assert eu.norm_value == Decimal("100.5")


def test_eu_span_min_length():
    with pytest.raises(Exception):
        _make_eu(source_span="short")  # < 10 字符


def test_eu_span_bounds_consistent():
    with pytest.raises(Exception):
        _make_eu(span_start=100, span_end=50)
    eu = _make_eu(span_start=10, span_end=50)
    assert eu.span_start == 10


def test_eu_pg_row_roundtrip():
    eu = _make_eu(
        norm_value=Decimal("99.5"),
        unit="USD",
        span_verified=True,
        entailment_verdict="entailed",
    )
    row = eu.to_pg_row()
    # 主键以 str 形式落地(UUID → str 兼容 PG)
    assert isinstance(row["eu_id"], str)
    assert isinstance(row["run_id"], str)
    # 复原
    eu2 = EvidenceUnitV2.from_pg_row(row)
    assert eu2.eu_id == eu.eu_id
    assert eu2.claim == eu.claim
    assert eu2.norm_value == Decimal("99.5")
    assert eu2.entailment_verdict == "entailed"


def test_eu_pg_row_strips_empty_strings():
    """PG 把 NULL 列读为 '' 时不应污染 model。"""
    eu = _make_eu()
    row = eu.to_pg_row()
    row["dimension_id"] = ""  # 模拟 PG 回读
    row["unit"] = ""
    row["entailment_verdict"] = ""
    eu2 = EvidenceUnitV2.from_pg_row(row)
    assert eu2.dimension_id is None
    assert eu2.unit is None
    assert eu2.entailment_verdict is None


# =============================================================================
# ClaimV2 校验
# =============================================================================

def _make_claim(**overrides) -> ClaimV2:
    defaults = dict(
        run_id=uuid.uuid4(),
        dimension_id="pricing-2024",
        canonical_claim="Kompyte pricing is $300/yr",
        claim_type="numeric",
        norm_value=Decimal("300"),
        unit="USD/year",
        eu_count=2,
        independent_source_count=2,
        primary_source_count=1,
        grade="A",
        grade_reason="2 independent sources consistent",
    )
    defaults.update(overrides)
    return ClaimV2(**defaults)


def test_claim_basic():
    c = _make_claim()
    assert c.grade == "A"
    assert c.grade_reason
    assert c.eu_count == 2


def test_claim_d_grade_allows_zero_or_more_eu():
    """D 级允许 eu_count=0(gap marker),也允许 eu_count>0(归并组均无 entailed)。"""
    # eu_count=0 → 合法
    c = _make_claim(grade="D", eu_count=0)
    assert c.grade == "D"
    # eu_count>0 → 也合法
    c2 = _make_claim(grade="D", eu_count=2)
    assert c2.grade == "D"


def test_claim_nond_requires_at_least_one_eu():
    """A/B/C 级必须有 ≥1 个 EU。"""
    for g in ("A", "B", "C"):
        with pytest.raises(Exception):
            _make_claim(grade=g, eu_count=0)


def test_claim_conflict_serialization():
    c = _make_claim(
        has_conflict=True,
        grade="C",
        conflicting_values=[
            {"source": "src-a", "value": "300 USD/yr"},
            {"source": "src-b", "value": "290 USD/yr"},
        ],
    )
    row = c.to_pg_row()
    assert row["has_conflict"] is True
    assert len(row["conflicting_values"]) == 2
    c2 = ClaimV2.from_pg_row(row)
    assert c2.has_conflict is True
    assert len(c2.conflicting_values) == 2


def test_claim_value_spread_optional():
    c = _make_claim(value_spread=0.05)
    assert c.value_spread == 0.05
    c2 = _make_claim()  # None
    assert c2.value_spread is None


def test_claim_pg_row_roundtrip():
    c = _make_claim()
    row = c.to_pg_row()
    c2 = ClaimV2.from_pg_row(row)
    assert c2.claim_id == c.claim_id
    assert c2.grade == "A"


# =============================================================================
# Literal type aliases
# =============================================================================

def test_literal_types_match_migration():
    """Schema Literal 必须与 migrations/002_claim_and_evidence_unit_v2.sql 对齐。"""
    import open_deep_research.evidence.schema as s
    assert set(s.ClaimType.__args__) == {"numeric", "event", "attribute", "relation", "opinion"}
    assert set(s.SourceTier.__args__) == {"primary", "secondary", "tertiary", "ugc"}
    assert set(s.Verdict.__args__) == {"entailed", "partial", "contradicted", "unverifiable"}
    assert set(s.Grade.__args__) == {"A", "B", "C", "D"}


# =============================================================================
# dataclass → V2 桥接
# =============================================================================

def test_legacy_dataclass_to_v2_bridge():
    """旧 dataclass EvidenceUnit.to_v2() 应产出合法 V2 EU。"""
    from open_deep_research.evidence_units import (
        EvidenceUnit as LegacyEU,
        NumberBinding,
        EntityRef,
    )
    legacy = LegacyEU(
        claim="Kompyte was acquired by Crayon in 2021.",
        quote="Kompyte was acquired by Crayon in 2021.",
        source_url="https://example.com/news/kompyte",
        source_title="Kompyte Acquisition News",
        numbers=[NumberBinding(text="2021", value_min=2021.0)],
        entities=[EntityRef(name="Kompyte", entity_type="company")],
        confidence=0.7,
    )
    rid = uuid.uuid4()
    v2 = legacy.to_v2(run_id=str(rid))
    assert v2.run_id == rid
    assert v2.claim == legacy.claim
    assert v2.claim_type == "numeric"  # 有 numbers → numeric
    assert "Kompyte" in v2.entities
    assert v2.norm_value == Decimal("2021.0")
    assert v2.source_domain == "example.com"
    assert v2.source_tier == "tertiary"  # 阶段 3 白名单升级前的 default
    assert v2.content_hash == legacy.content_hash  # 跨 run dedup 锚
    assert v2.usable is False  # 未过三道闸


def test_legacy_bridge_preserves_content_hash_across_invocation():
    """to_v2 应保持 content_hash 稳定 — 让跨 run dedup 锚保留。"""
    from open_deep_research.evidence_units import EvidenceUnit as LegacyEU
    legacy = LegacyEU(
        claim="this is a long enough claim",
        quote="this is a long enough quote",
        source_url="https://x.com/y",
    )
    h1 = legacy.content_hash
    v2 = legacy.to_v2(run_id=str(uuid.uuid4()))
    assert v2.content_hash == h1