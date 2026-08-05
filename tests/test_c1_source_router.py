"""C1 · 源意图路由验收测试

按 SPEC §5 C1(修订后):
- market_size 槽 → target_source_types 含 research_house/official,**前 2 条 query 含 site:**
- regulation 槽 → target_source_types=['official'],**前 1 条 query 含 site:.gov**
- 未知 claim_type → query_list 全无 site:
- query_list 长度 = N + len(fallback_templates),顺序固定
"""
from __future__ import annotations

from pathlib import Path

import pytest

from open_deep_research.evidence.caliber_registry import load_caliber_registry
from open_deep_research.evidence.framework import Slot, load_framework
from open_deep_research.evidence.source_registry import load_registry
from open_deep_research.evidence.source_router import (
    SourceIntent,
    claim_type_to_target_sources,
    route_sources,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def registry():
    return load_registry("us_livecommerce", base_dir="data/registry/sources")


@pytest.fixture
def calibers():
    return load_caliber_registry("us_livecommerce", base_dir="data/registry/calibers")


# -----------------------------------------------------------------------------
# 验收 1: market_size 槽 — site-scoped 强制前 2 条
# -----------------------------------------------------------------------------

class TestMarketSizeSlot:
    def test_target_source_types_includes_research_and_official(self, registry, calibers) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="2025 US live commerce market size?",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        intent = route_sources(slot, registry=registry, calibers=calibers)
        assert "research_house" in intent.target_source_types
        assert "official" in intent.target_source_types

    def test_first_n_queries_are_site_scoped(self, registry, calibers) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="2025 US live commerce market size?",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        intent = route_sources(slot, registry=registry, calibers=calibers)
        n = len(intent.target_source_types)
        # 前 N 条必须含 site:
        for q in intent.query_list[:n]:
            assert "site:" in q, f"前 {n} 条 query 之一不含 site:: {q}"
        # N 之后是 fallback(可不含 site:)
        # 但 query_list 总长度应 = N + len(fallback)
        fallback_count = len([t for t in registry.get_templates("market_size") if "site:" not in t])
        assert len(intent.query_list) == n + fallback_count


# -----------------------------------------------------------------------------
# 验收 2: regulation 槽 — site:.gov
# -----------------------------------------------------------------------------

class TestRegulationSlot:
    def test_target_is_official_only(self, registry, calibers) -> None:
        slot = Slot(
            slot_id="ftc_disclosure_rule",
            question="FTC disclosure requirements for live commerce?",
            expected_claim_type="regulation",
            required_tier_min="B",
        )
        intent = route_sources(slot, registry=registry, calibers=calibers)
        assert intent.target_source_types == ["official"]
        # 前 1 条 query 含 site:.gov(F1 注册表有 ftc.gov)
        assert intent.query_list
        assert "site:" in intent.query_list[0]
        assert ".gov" in intent.query_list[0]


# -----------------------------------------------------------------------------
# 验收 3: 未知 claim_type → 全无 site-scoped
# -----------------------------------------------------------------------------

class TestUnknownClaimType:
    def test_unknown_claim_type_no_site_scoped(self, registry) -> None:
        # Slot 的 expected_claim_type 是 Literal,无法构造未知值
        # 改用 validate 时长 type 检验:用 attribute 类型但 mock claim_type_to_target_sources 返 []
        from unittest.mock import patch
        slot = Slot(
            slot_id="x_obscure",
            question="something obscure",
            expected_claim_type="attribute",
        )
        with patch(
            "open_deep_research.evidence.source_router.claim_type_to_target_sources",
            return_value=[],
        ):
            intent = route_sources(slot, registry=registry)
        assert intent.target_source_types == []
        for q in intent.query_list:
            assert "site:" not in q


# -----------------------------------------------------------------------------
# 验收 4: 顺序固定 — 前 N 条 site-scoped,后跟 fallback
# -----------------------------------------------------------------------------

class TestOrdering:
    def test_ordering_market_size(self, registry, calibers) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="2025 US live commerce market size?",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        intent = route_sources(slot, registry=registry, calibers=calibers)
        n = len(intent.target_source_types)
        # 前 N 条含 site:
        assert all("site:" in q for q in intent.query_list[:n])
        # 第 N+1 条起不应含 site:
        for q in intent.query_list[n:]:
            assert "site:" not in q

    def test_ordering_regulation(self, registry) -> None:
        slot = Slot(
            slot_id="reg_test",
            question="regulation question",
            expected_claim_type="regulation",
            required_tier_min="B",
        )
        intent = route_sources(slot, registry=registry)
        n = len(intent.target_source_types)  # =1
        assert all("site:" in q for q in intent.query_list[:n])
        for q in intent.query_list[n:]:
            assert "site:" not in q


# -----------------------------------------------------------------------------
# 验收 5: claim_type_to_target_sources 直接 API
# -----------------------------------------------------------------------------

class TestClaimTypeMapping:
    @pytest.mark.parametrize("claim_type,expected", [
        ("regulation", ["official"]),
        ("financial", ["official"]),
        ("m_and_a", ["official", "research_house"]),
        ("quantitative", ["research_house", "official"]),  # 默认 quantitative_market
        ("trends", ["research_house", "industry_media", "mainstream_media"]),
        ("unknown_type", []),
        ("", []),
    ])
    def test_mapping(self, claim_type: str, expected: list[str]) -> None:
        result = claim_type_to_target_sources(claim_type)
        assert result == expected


# -----------------------------------------------------------------------------
# 验收 6: SourceIntent 自身
# -----------------------------------------------------------------------------

class TestSourceIntent:
    def test_site_scoped_count(self) -> None:
        intent = SourceIntent(
            target_source_types=["research_house", "official"],
            query_list=[
                "x site:a.com",
                "x site:b.com",
                "x",  # fallback
            ],
        )
        assert intent.site_scoped_count() == 2

    def test_first_n_site_scoped(self) -> None:
        intent = SourceIntent(
            target_source_types=["research_house", "official"],
            query_list=[
                "x site:a.com",
                "x site:b.com",
                "y site:c.com",  # 不应被前 2 取到
            ],
        )
        assert len(intent.first_n_site_scoped(2)) == 2
