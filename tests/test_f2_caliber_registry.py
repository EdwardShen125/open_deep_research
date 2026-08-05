"""F2 · 口径注册表验收测试

按 SPEC §5 F2:
- 为种子 vertical 定义 ≥ 2 个 caliber(直播零售 / 广义视频购物)
- loader 通过
- 两个 caliber 的 metric_type 相同但 id 不同、excludes 不同
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from open_deep_research.evidence.caliber_registry import (
    Caliber,
    CaliberRegistry,
    load_caliber_registry,
)


REGISTRY_PATH = Path("data/registry/calibers/us_livecommerce.yaml")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def registry() -> CaliberRegistry:
    return load_caliber_registry("us_livecommerce", base_dir=REGISTRY_PATH.parent)


# -----------------------------------------------------------------------------
# 验收 1: loader 读入 YAML
# -----------------------------------------------------------------------------

class TestLoader:
    def test_loads_yaml(self, registry: CaliberRegistry) -> None:
        assert isinstance(registry, CaliberRegistry)
        assert registry.vertical == "us_livecommerce"

    def test_seed_has_at_least_two_calibers(self, registry: CaliberRegistry) -> None:
        # plan F2 DoD:≥ 2 个 caliber
        assert len(registry.calibers) >= 2
        assert all(isinstance(c, Caliber) for c in registry.calibers)


# -----------------------------------------------------------------------------
# 验收 2: 两 caliber 同 metric_type 不同 id / 不同 excludes
# -----------------------------------------------------------------------------

class TestCaliberDifferentiation:
    def test_market_size_has_multiple_calibers(self, registry: CaliberRegistry) -> None:
        ms_calibers = registry.by_metric("market_size")
        assert len(ms_calibers) >= 2, "should have ≥2 calibers for market_size"

    def test_different_ids_same_metric(self, registry: CaliberRegistry) -> None:
        ms_calibers = registry.by_metric("market_size")
        ids = [c.id for c in ms_calibers]
        assert len(ids) == len(set(ids)), f"duplicate ids in market_size calibers: {ids}"

    def test_different_excludes(self, registry: CaliberRegistry) -> None:
        # 直播零售 excludes CTV;广义视频购物 includes CTV
        retail = registry.get("us_livestream_retail_emarketer")
        broad = registry.get("us_video_shopping_broad_coresight")
        assert retail is not None and broad is not None
        # 字符串包含校验(因为 excludes 元素是短语,而非纯关键词)
        retail_excludes_text = " ".join(retail.excludes)
        broad_excludes_text = " ".join(broad.excludes)
        broad_includes_text = " ".join(broad.includes)
        assert "CTV" in retail_excludes_text, (
            f"eMarketer 直播零售口径应排除 CTV, got excludes={retail.excludes}"
        )
        assert "CTV" not in broad_excludes_text, (
            f"Coresight 广义视频购物口径不应在 excludes 排除 CTV, got excludes={broad.excludes}"
        )
        assert "CTV" in broad_includes_text or "shoppable" in broad_includes_text, (
            f"Coresight 广义视频购物口径应在 includes 包含 CTV/shoppable, got includes={broad.includes}"
        )

    def test_different_entities(self, registry: CaliberRegistry) -> None:
        # 即使 metric_type 相同,entity 也可能不同
        retail = registry.get("us_livestream_retail_emarketer")
        broad = registry.get("us_video_shopping_broad_coresight")
        assert retail.entity == "us_live_commerce"
        assert broad.entity == "us_video_shopping"
        assert retail.entity != broad.entity

    def test_different_definitions(self, registry: CaliberRegistry) -> None:
        retail = registry.get("us_livestream_retail_emarketer")
        broad = registry.get("us_video_shopping_broad_coresight")
        assert retail.definition != broad.definition
        assert len(retail.definition) >= 10
        assert len(broad.definition) >= 10


# -----------------------------------------------------------------------------
# 验收 3: API 方法
# -----------------------------------------------------------------------------

class TestRegistryAPI:
    def test_get_existing(self, registry: CaliberRegistry) -> None:
        c = registry.get("us_livestream_retail_emarketer")
        assert c is not None
        assert c.metric_type == "market_size"

    def test_get_missing_returns_none(self, registry: CaliberRegistry) -> None:
        assert registry.get("nonexistent_caliber") is None

    def test_by_metric_filters_correctly(self, registry: CaliberRegistry) -> None:
        ms = registry.by_metric("market_size")
        assert all(c.metric_type == "market_size" for c in ms)

    def test_by_entity_filters_correctly(self, registry: CaliberRegistry) -> None:
        lc = registry.by_entity("us_live_commerce")
        assert all(c.entity == "us_live_commerce" for c in lc)
        assert len(lc) >= 2  # retail + mckinsey

    def test_applicable_in_filters_by_year(self, registry: CaliberRegistry) -> None:
        # McKinsey caliber 只适用 2024/2025
        applicable_2024 = registry.applicable_in(2024)
        mckinsey_ids = [c.id for c in applicable_2024 if c.id == "us_live_commerce_mckinsey"]
        assert len(mckinsey_ids) == 1, "mckinsey should apply in 2024"

        applicable_2030 = registry.applicable_in(2030)
        mckinsey_ids_2030 = [
            c.id for c in applicable_2030 if c.id == "us_live_commerce_mckinsey"
        ]
        assert len(mckinsey_ids_2030) == 0, "mckinsey should NOT apply in 2030"


# -----------------------------------------------------------------------------
# 验收 4: schema 校验拒绝缺字段
# -----------------------------------------------------------------------------

class TestSchemaValidation:
    def test_missing_id_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "calibers": [{
                "metric_type": "market_size",
                "entity": "x",
                "definition": "x" * 20,
                "canonical_source_type": "research_house",
            }],
        }
        with pytest.raises(ValidationError):
            CaliberRegistry(**raw)

    def test_missing_metric_type_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "calibers": [{
                "id": "test_x",
                "entity": "x",
                "definition": "x" * 20,
                "canonical_source_type": "research_house",
            }],
        }
        with pytest.raises(ValidationError):
            CaliberRegistry(**raw)

    def test_invalid_metric_type_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "calibers": [{
                "id": "test_x",
                "metric_type": "fictional_metric",
                "entity": "x",
                "definition": "x" * 20,
                "canonical_source_type": "research_house",
            }],
        }
        with pytest.raises(ValidationError):
            CaliberRegistry(**raw)

    def test_invalid_id_pattern_rejected(self) -> None:
        # id 必须以小写字母开头,只含小写字母数字下划线
        raw = {
            "vertical": "test",
            "calibers": [{
                "id": "Invalid-Id-With-Dashes",
                "metric_type": "market_size",
                "entity": "x",
                "definition": "x" * 20,
                "canonical_source_type": "research_house",
            }],
        }
        with pytest.raises(ValidationError):
            CaliberRegistry(**raw)

    def test_duplicate_id_rejected(self) -> None:
        raw = {
            "vertical": "test",
            "calibers": [
                {
                    "id": "dup_x",
                    "metric_type": "market_size",
                    "entity": "x",
                    "definition": "x" * 20,
                    "canonical_source_type": "research_house",
                },
                {
                    "id": "dup_x",
                    "metric_type": "share",
                    "entity": "y",
                    "definition": "y" * 20,
                    "canonical_source_type": "research_house",
                },
            ],
        }
        with pytest.raises(ValidationError):
            CaliberRegistry(**raw)


# -----------------------------------------------------------------------------
# 验收 5: loader 错误处理
# -----------------------------------------------------------------------------

class TestLoaderErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_caliber_registry("nonexistent", base_dir=tmp_path)


# -----------------------------------------------------------------------------
# Bonus: cross-check with F1 source registry(为 D2 准备)
# -----------------------------------------------------------------------------

class TestCrossRegistry:
    def test_canonical_source_type_matches_f1(self, registry: CaliberRegistry) -> None:
        """每个 caliber 的 canonical_source_type 应与 F1 源注册表对该 metric_type 的认知权威匹配。"""
        from open_deep_research.evidence.source_registry import (
            load_registry,
        )
        sr = load_registry("us_livecommerce", base_dir="data/registry/sources")
        for c in registry.calibers:
            tier_a_sources = sr.sources_by_metric(c.metric_type, min_tier="A")
            if not tier_a_sources:
                continue
            types = {s.source_type for s in tier_a_sources}
            # canonical_source_type 必须是 Literal 的字符串值之一
            assert c.canonical_source_type in types or c.canonical_source_type == "official", (
                f"caliber {c.id} canonical_source_type={c.canonical_source_type} "
                f"not represented by any tier-A source for metric {c.metric_type}"
            )
