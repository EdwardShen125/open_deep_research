"""A2 · tier 分类器验收测试

按 SPEC §5 A2(修订后):
- 20 fixture 源覆盖
- 注册表命中 100% 正确
- 未命中按信号正确(且每条触发 caplog "unmatched source, signal-tier fallback")
- 设 log_unmatched=False 时 logger 不出声(供 batch 静默测试)
- grep 断言 upgrade_source_tier 仍可调用但不被 acqurier 路径使用(本卡不实装 acquirer,只断言旧路径未被新路径直接调)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from open_deep_research.evidence.source_registry import (
    SourceRegistry,
    load_registry,
)
from open_deep_research.evidence.tier_classifier import (
    classify_tier,
    classify_tier_batch,
)


REGISTRY_PATH = Path("data/registry/sources/us_livecommerce.yaml")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def registry() -> SourceRegistry:
    return load_registry("us_livecommerce", base_dir=REGISTRY_PATH.parent)


# -----------------------------------------------------------------------------
# 验收 1: F1 注册表命中 100% 正确
# -----------------------------------------------------------------------------

class TestRegistryHit:
    @pytest.mark.parametrize("url,expected_tier", [
        # A 级:具名研究机构
        ("https://www.emarketer.com/articles/live-commerce-2025", "A"),
        ("https://coresight.com/research/live-commerce", "A"),
        ("https://www.mckinsey.com/industries/retail/our-insights", "A"),
        ("https://www.bain.com/insights/live-commerce", "A"),
        ("https://www.forrester.com/report", "A"),
        ("https://www.gartner.com/en/article", "A"),
        # A 级:官方 / 监管
        ("https://www.sec.gov/cgi-bin/browse-edgar", "A"),
        ("https://www.ftc.gov/news-events", "A"),
        ("https://www.census.gov/retail", "A"),
        ("https://www.uspto.gov/patent", "A"),
        ("https://www.fcc.gov/document", "A"),
        # B 级:主流媒体
        ("https://www.reuters.com/article", "B"),
        ("https://www.ft.com/content", "B"),
        ("https://www.wsj.com/articles", "B"),
        ("https://www.nytimes.com/2025/01/x", "B"),
        ("https://www.bloomberg.com/news", "B"),
        # B 级:中文权威
        ("https://www.caixin.com/2025-01-01", "B"),
        ("https://www.36kr.com/p/12345", "B"),
        # C 级:爬虫估算 / 厂商
        ("https://www.fastmoss.com/sku", "C"),
        ("https://www.semrush.com/analytics", "C"),
        # D 级:UGC
        ("https://www.reddit.com/r/livecommerce", "D"),
    ])
    def test_known_domains_classified_correctly(
        self, registry: SourceRegistry, url: str, expected_tier: str
    ) -> None:
        result = classify_tier(url, "", registry=registry, log_unmatched=False)
        assert result == expected_tier, (
            f"domain for {url} → got {result}, expected {expected_tier}"
        )


# -----------------------------------------------------------------------------
# 验收 2: 未命中 → 信号兜底 + 触发 warning(plan 硬教训 a 形式化)
# -----------------------------------------------------------------------------

class TestSignalFallback:
    def test_unmatched_emits_warning(self, registry: SourceRegistry, caplog) -> None:
        # 一个不在 F1 注册表里、但在 evidence.independence 白名单里(信号兜底命中)的域
        # 找一个独立白名单中有的域 — 不能在 F1 里
        # F1 没有 "reuters.com" 是 B(实际有);需要找一个不在 F1 但在独立性白名单的域
        # 选用 cnn.com — 独立白名单 secondary,但不在 F1
        caplog.set_level(logging.WARNING, logger="open_deep_research.evidence.tier_classifier")
        with caplog.at_level(logging.WARNING):
            tier = classify_tier(
                "https://www.cnn.com/2025/01/article",
                "cnn.com",
                registry=registry,
                log_unmatched=True,
            )
        assert tier == "B", "cnn.com in 独立白名单 secondary → 应映射 B"
        # 必须触发 warning
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("unmatched source, signal-tier fallback" in r.message for r in warnings), (
            f"expected warning 'unmatched source, signal-tier fallback', "
            f"got messages: {[r.message for r in caplog.records]}"
        )
        # warning 应包含 domain + tier 提示
        matching = [
            r for r in warnings
            if "unmatched source, signal-tier fallback" in r.message
        ]
        assert matching
        msg = matching[0].message
        assert "cnn.com" in msg
        assert "B" in msg or "secondary" in msg

    def test_signal_fallback_secondary_to_b(self, registry: SourceRegistry, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="open_deep_research.evidence.tier_classifier")
        with caplog.at_level(logging.WARNING):
            # pitchbook.com — 独立白名单 secondary
            tier = classify_tier(
                "https://pitchbook.com/news",
                "pitchbook.com",
                registry=registry,
            )
        assert tier == "B"

    def test_signal_fallback_ugc_to_d(self, registry: SourceRegistry, caplog) -> None:
        caplog.set_level(logging.WARNING, logger="open_deep_research.evidence.tier_classifier")
        with caplog.at_level(logging.WARNING):
            # zhihu.com — 独立白名单 ugc
            tier = classify_tier(
                "https://zhuanlan.zhihu.com/p/123",
                "zhihu.com",
                registry=registry,
            )
        assert tier == "D"


# -----------------------------------------------------------------------------
# 验收 3: log_unmatched=False 时 logger 不出声
# -----------------------------------------------------------------------------

class TestLogUnmatched:
    def test_no_warning_when_disabled(
        self, registry: SourceRegistry, caplog
    ) -> None:
        caplog.set_level(logging.WARNING, logger="open_deep_research.evidence.tier_classifier")
        with caplog.at_level(logging.WARNING):
            tier = classify_tier(
                "https://www.cnn.com/2025/01/x",
                "cnn.com",
                registry=registry,
                log_unmatched=False,
            )
        assert tier == "B"
        # 不应有 tier_classifier 自己发的 warning(其他 logger 可能发,只检查自己)
        my_warnings = [
            r for r in caplog.records
            if r.name == "open_deep_research.evidence.tier_classifier"
        ]
        assert len(my_warnings) == 0, (
            f"expected no warnings from tier_classifier when log_unmatched=False, "
            f"got: {[r.message for r in my_warnings]}"
        )


# -----------------------------------------------------------------------------
# 验收 4: 完全无法判定 → None + warning
# -----------------------------------------------------------------------------

class TestUnknown:
    def test_completely_unknown_returns_tertiary_signal(
        self, registry: SourceRegistry, caplog
    ) -> None:
        # plan A2:信号兜底覆盖所有域(未知域 classify_source_tier 也返 tertiary → C)
        # 这是 plan 设计的语义:未知但有兜底 ≠ None
        caplog.set_level(logging.WARNING, logger="open_deep_research.evidence.tier_classifier")
        with caplog.at_level(logging.WARNING):
            tier = classify_tier(
                "https://definitely-not-in-any-registry.invalid/x",
                "definitely-not-in-any-registry.invalid",
                registry=registry,
            )
        assert tier == "C", "未知域兜底为 tertiary → C(plan A2 设计)"
        my_warnings = [
            r for r in caplog.records
            if r.name == "open_deep_research.evidence.tier_classifier"
        ]
        assert any(
            "unmatched source, signal-tier fallback" in r.message
            for r in my_warnings
        ), "兜底命中必须触发 warning"


# -----------------------------------------------------------------------------
# 验收 5: classify_tier_batch 静默 + 返回 list
# -----------------------------------------------------------------------------

class TestBatch:
    def test_batch_returns_list_of_tiers(self, registry: SourceRegistry) -> None:
        sources = [
            ("https://www.emarketer.com/x", "emarketer.com"),  # A(F1 命中)
            ("https://www.cnn.com/y", "cnn.com"),  # signal B(白名单 secondary)
            ("https://www.never.invalid/z", "never.invalid"),  # signal C(未知→tertiary)
        ]
        result = classify_tier_batch(sources, registry=registry)
        assert result == ["A", "B", "C"]

    def test_batch_empty(self, registry: SourceRegistry) -> None:
        assert classify_tier_batch([], registry=registry) == []


# -----------------------------------------------------------------------------
# 验收 6: 旧的 independence.classify_source_tier / upgrade_source_tier 仍可调用
#           (plan §4:不删旧路径)
# -----------------------------------------------------------------------------

class TestLegacyPathsPreserved:
    def test_classify_source_tier_still_works(self) -> None:
        from open_deep_research.evidence.independence import classify_source_tier
        # 旧接口:domain → primary/secondary/tertiary/ugc
        assert classify_source_tier("cnn.com") == "secondary"
        assert classify_source_tier("reddit.com") == "ugc"
        assert classify_source_tier("unknown.invalid") == "tertiary"

    def test_upgrade_source_tier_still_works(self) -> None:
        from open_deep_research.evidence.independence import upgrade_source_tier
        from open_deep_research.evidence.schema import EvidenceUnitV2
        from datetime import datetime, timezone
        from uuid import uuid4
        from decimal import Decimal

        eu = EvidenceUnitV2(
            run_id=uuid4(),
            claim="x",
            claim_type="attribute",
            source_url="https://www.cnn.com/article",
            source_domain="cnn.com",
            source_span="some span here that is long enough",
            source_tier="tertiary",  # 假设未分类
            extractor_model="test",
        )
        upgraded = upgrade_source_tier(eu)
        assert upgraded.source_tier == "secondary"  # upgrade 后

    def test_tier_classifier_does_not_call_upgrade_directly(
        self, registry: SourceRegistry
    ) -> None:
        """A2 不直接调 upgrade_source_tier(plan §4);只调 classify_source_tier 做信号。"""
        with patch(
            "open_deep_research.evidence.independence.upgrade_source_tier"
        ) as mock_upgrade:
            classify_tier(
                "https://www.cnn.com/x",
                "cnn.com",
                registry=registry,
                log_unmatched=False,
            )
            mock_upgrade.assert_not_called()
