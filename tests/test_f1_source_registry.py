"""F1 · 源注册表验收测试

按 SPEC §5 F1:
- loader 读入 YAML
- get_tier("emarketer.com") == "A"
- 未知域名返回 None(交给 A2 兜底判定)
- schema 校验拒绝缺 tier 的条目
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from open_deep_research.evidence.source_registry import (
    SourceEntry,
    SourceRegistry,
    load_registry,
)


REGISTRY_PATH = Path("data/registry/sources/us_livecommerce.yaml")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def registry() -> SourceRegistry:
    return load_registry("us_livecommerce", base_dir=REGISTRY_PATH.parent)


# -----------------------------------------------------------------------------
# 验收 1: loader 读入 YAML,返回 SourceRegistry 实例,vertical 字段正确
# -----------------------------------------------------------------------------

class TestLoader:
    def test_loads_yaml(self, registry: SourceRegistry) -> None:
        assert isinstance(registry, SourceRegistry)
        assert registry.vertical == "us_livecommerce"

    def test_sources_loaded(self, registry: SourceRegistry) -> None:
        assert len(registry.sources) >= 15  # SPEC F1 DoD:A/B ≥15,这里 total 也必 ≥15
        assert all(isinstance(s, SourceEntry) for s in registry.sources)

    def test_query_templates_loaded(self, registry: SourceRegistry) -> None:
        templates = registry.get_templates("market_size")
        assert isinstance(templates, list)
        assert len(templates) >= 1
        assert any("site:" in t for t in templates)  # 至少 1 条 site-scoped

    def test_ab_count_meets_dod(self, registry: SourceRegistry) -> None:
        # SPEC F1 DoD:种子 vertical 的 A/B 级源 ≥ 15 条
        assert registry.ab_count() >= 15, (
            f"got {registry.ab_count()} A/B sources; SPEC requires >= 15"
        )


# -----------------------------------------------------------------------------
# 验收 2: get_tier("emarketer.com") == "A"
# -----------------------------------------------------------------------------

class TestGetTier:
    @pytest.mark.parametrize("domain,expected", [
        ("emarketer.com", "A"),
        ("coresight.com", "A"),
        ("mckinsey.com", "A"),
        ("sec.gov", "A"),
        ("ftc.gov", "A"),
        ("reuters.com", "B"),
        ("ft.com", "B"),
        ("caixin.com", "B"),
        ("fastmoss.com", "C"),
        ("reddit.com", "D"),
    ])
    def test_known_domain(self, registry: SourceRegistry, domain: str, expected: str) -> None:
        assert registry.get_tier(domain) == expected

    def test_subdomain_matches_etld1(self, registry: SourceRegistry) -> None:
        # 简化的 eTLD+1:子域应命中注册表项
        assert registry.get_tier("blog.emarketer.com") == "A"
        assert registry.get_tier("www.sec.gov") == "A"

    def test_unknown_returns_none(self, registry: SourceRegistry) -> None:
        # plan F1:未知域名返回 None(交给 A2 信号兜底)
        assert registry.get_tier("never-seen-before.example") is None
        assert registry.get_tier("") is None

    def test_full_url_normalized(self, registry: SourceRegistry) -> None:
        # 传入完整 URL 应被规范化为 hostname
        assert registry.get_tier("https://www.emarketer.com/articles/foo") == "A"


# -----------------------------------------------------------------------------
# 验收 3: schema 校验拒绝缺 tier 的条目
# -----------------------------------------------------------------------------

class TestSchemaValidation:
    def test_missing_tier_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "sources": [{"domain": "x.com", "source_type": "research_house"}],
        }
        with pytest.raises(ValidationError):
            SourceRegistry(**raw)

    def test_missing_source_type_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "sources": [{"domain": "x.com", "tier": "A"}],
        }
        with pytest.raises(ValidationError):
            SourceRegistry(**raw)

    def test_invalid_tier_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "sources": [{"domain": "x.com", "tier": "Z", "source_type": "research_house"}],
        }
        with pytest.raises(ValidationError):
            SourceRegistry(**raw)

    def test_invalid_domain_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "sources": [{
                "domain": "not-a-domain",
                "tier": "A",
                "source_type": "research_house",
            }],
        }
        with pytest.raises(ValidationError):
            SourceRegistry(**raw)

    def test_duplicate_domain_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "sources": [
                {"domain": "x.com", "tier": "A", "source_type": "research_house"},
                {"domain": "x.com", "tier": "B", "source_type": "mainstream_media"},
            ],
        }
        with pytest.raises(ValidationError):
            SourceRegistry(**raw)

    def test_source_entry_normalizes_domain_case(self) -> None:
        e = SourceEntry(
            domain="EMARKETER.com",
            tier="A",
            source_type="research_house",
        )
        assert e.domain == "emarketer.com"


# -----------------------------------------------------------------------------
# 验收 4: loader 错误处理
# -----------------------------------------------------------------------------

class TestLoaderErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_registry("nonexistent_vertical", base_dir=tmp_path)

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("vertical: test\nsources: not-a-list\n", encoding="utf-8")
        with pytest.raises((ValidationError, yaml.YAMLError)):
            load_registry("bad", base_dir=tmp_path)


# -----------------------------------------------------------------------------
# Bonus: helpers + API(支撑后续卡)
# -----------------------------------------------------------------------------

class TestHelpers:
    def test_sources_by_type(self, registry: SourceRegistry) -> None:
        rh = registry.sources_by_type("research_house", min_tier="A")
        assert len(rh) >= 5  # emarketer/coresight/mckinsey/bain/forrester/gartner/iab
        assert all(s.source_type == "research_house" for s in rh)
        assert all(s.tier in ("A", "B") for s in rh)

    def test_sources_by_metric(self, registry: SourceRegistry) -> None:
        ms = registry.sources_by_metric("market_size", min_tier="A")
        assert len(ms) >= 3
        assert all("market_size" in s.covers for s in ms)

    def test_get_source_type(self, registry: SourceRegistry) -> None:
        assert registry.get_source_type("emarketer.com") == "research_house"
        assert registry.get_source_type("sec.gov") == "official"
        assert registry.get_source_type("reuters.com") == "mainstream_media"
        assert registry.get_source_type("nonexistent.com") is None
