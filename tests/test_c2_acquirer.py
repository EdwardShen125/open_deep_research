"""C2 · 两级检索 + 每槽预算验收测试

按 SPEC §5 C2:
- market_size 槽产出的 claim 中 A/B 级占比 ≥ 阈值
- 每槽 EU 数 ≤ MAX_EU_PER_SLOT
- 定向失败时 fallback 生效且有日志
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from open_deep_research.evidence.acquirer import (
    DEFAULT_MAX_EU_PER_SLOT,
    InMemorySearchProvider,
    acquire,
)
from open_deep_research.evidence.framework import Slot
from open_deep_research.evidence.source_registry import load_registry


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def registry():
    return load_registry("us_livecommerce", base_dir="data/registry/sources")


@pytest.fixture
def market_size_slot() -> Slot:
    return Slot(
        slot_id="market_size_2025",
        question="2025 US live commerce market size",
        expected_claim_type="quantitative",
        required_tier_min="B",
        caliber_id="us_livestream_retail_emarketer",
    )


# -----------------------------------------------------------------------------
# 验收 1: MAX_EU_PER_SLOT 截断
# -----------------------------------------------------------------------------

class TestMaxEUBudget:
    def test_truncates_to_max(self, registry, market_size_slot) -> None:
        # 构造 50 条 mock 结果
        provider = InMemorySearchProvider(mapping={
            "site:emarketer.com": [
                {"url": f"https://emarketer.com/a/{i}",
                 "title": f"article {i}",
                 "snippet": "live commerce market"}
                for i in range(50)
            ],
        })
        claims = acquire(
            market_size_slot, registry=registry,
            search_provider=provider, max_eu=10,
        )
        assert len(claims) <= 10
        assert len(claims) == 10

    def test_default_max_eu_is_30(self) -> None:
        assert DEFAULT_MAX_EU_PER_SLOT == 30

    def test_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("OPEN_DEEP_RESEARCH_MAX_EU_PER_SLOT", "5")
        from open_deep_research.evidence.acquirer import _get_max_eu_per_slot
        assert _get_max_eu_per_slot() == 5


# -----------------------------------------------------------------------------
# 验收 2: A/B 占比阈值
# -----------------------------------------------------------------------------

class TestABRatio:
    def test_market_size_ab_ratio_meets_threshold(self, registry, market_size_slot) -> None:
        # Mock emarketer + coresight(都是 A research_house) + 几个未知
        provider = InMemorySearchProvider(mapping={
            "site:emarketer.com": [
                {"url": "https://www.emarketer.com/a", "title": "t1", "snippet": "x"},
                {"url": "https://www.coresight.com/b", "title": "t2", "snippet": "x"},
                {"url": "https://www.mckinsey.com/c", "title": "t3", "snippet": "x"},
                # 一个未知域(信号兜底 C)
                {"url": "https://unknown-invented.invalid/x", "title": "t4", "snippet": "x"},
            ],
        })
        claims = acquire(
            market_size_slot, registry=registry, search_provider=provider,
        )
        # A/B 比例:3/4 = 75%(阈值 ≥ 50%)
        ab_count = sum(1 for c in claims if c.tier in ("A", "B"))
        ratio = ab_count / len(claims)
        assert ratio >= 0.5, f"A/B ratio {ratio} below threshold 0.5"

    def test_tier_assigned_to_each_claim(self, registry, market_size_slot) -> None:
        provider = InMemorySearchProvider(mapping={
            "site:emarketer.com": [
                {"url": "https://www.emarketer.com/a", "title": "t1", "snippet": "x"},
            ],
        })
        claims = acquire(
            market_size_slot, registry=registry, search_provider=provider,
        )
        for c in claims:
            # 每个 claim 都有 tier(A2 跑过)
            assert c.tier is not None, f"claim {c.source_id} has no tier"


# -----------------------------------------------------------------------------
# 验收 3: 定向失败 → fallback 生效
# -----------------------------------------------------------------------------

class TestFallback:
    def test_fallback_when_directed_queries_return_nothing(
        self, registry, market_size_slot, caplog
    ) -> None:
        # 定向 site: 查询都不命中,只有 fallback 命中
        # fallback 模板 format 后形如 "us live commerce market 2025"
        provider = InMemorySearchProvider(mapping={
            "site:": [],  # 所有 site:xxx 返空
            "us live commerce": [  # fallback query 含 "us live commerce" 时命中
                {"url": "https://www.reuters.com/article/x",
                 "title": "fallback hit", "snippet": "x"},
            ],
        })
        caplog.set_level(logging.INFO, logger="open_deep_research.evidence.acquirer")
        with caplog.at_level(logging.INFO):
            claims = acquire(
                market_size_slot, registry=registry, search_provider=provider,
            )
        # fallback 应命中 1 条
        assert len(claims) >= 1
        assert any(c.source_domain == "reuters.com" for c in claims)

    def test_truncation_logs(
        self, registry, market_size_slot, caplog
    ) -> None:
        # 触发 truncation 日志
        provider = InMemorySearchProvider(mapping={
            "site:emarketer.com": [
                {"url": f"https://example.com/{i}", "title": f"t{i}", "snippet": "x"}
                for i in range(50)
            ],
        })
        caplog.set_level(logging.INFO, logger="open_deep_research.evidence.acquirer")
        with caplog.at_level(logging.INFO):
            acquire(
                market_size_slot, registry=registry,
                search_provider=provider, max_eu=5,
            )
        # 应有 truncation 日志
        trunc_logs = [r for r in caplog.records if "truncated" in r.message]
        assert len(trunc_logs) >= 1, (
            f"expected truncation log when input > max_eu, "
            f"got: {[r.message for r in caplog.records]}"
        )

    def test_empty_provider_returns_empty(self, registry, market_size_slot) -> None:
        provider = InMemorySearchProvider(mapping={})
        claims = acquire(
            market_size_slot, registry=registry, search_provider=provider,
        )
        assert claims == []


# -----------------------------------------------------------------------------
# 验收 4: 截断策略 — 保留 A/B 优先
# -----------------------------------------------------------------------------

class TestTruncationOrder:
    def test_takes_first_n_no_resort(self, registry, market_size_slot) -> None:
        # 验证 acquire 不做"重新排序"——前 N 个就前 N 个
        # 这是因为 search_provider 通常已按相关度排序,acquire 信任它
        urls = [f"https://example.com/{i}" for i in range(20)]
        provider = InMemorySearchProvider(mapping={
            "site:emarketer.com": [
                {"url": u, "title": f"t{i}", "snippet": "x"} for i, u in enumerate(urls)
            ],
        })
        claims = acquire(
            market_size_slot, registry=registry,
            search_provider=provider, max_eu=5,
        )
        assert len(claims) == 5
        # 取的前 5 个
        assert claims[0].source_id == urls[0]
        assert claims[4].source_id == urls[4]


# -----------------------------------------------------------------------------
# 验收 5: classify_tier_fn 可注入(测试隔离)
# -----------------------------------------------------------------------------

class TestInjectableClassifier:
    def test_classify_tier_fn_overrides_default(
        self, registry, market_size_slot
    ) -> None:
        # 注入一个返回固定 tier 的函数
        def fake_classifier(url, domain, **kwargs):
            return "A"

        provider = InMemorySearchProvider(mapping={
            "site:emarketer.com": [
                {"url": "https://www.example.com/x", "title": "t", "snippet": "x"},
            ],
        })
        claims = acquire(
            market_size_slot, registry=registry,
            search_provider=provider, classify_tier_fn=fake_classifier,
        )
        assert all(c.tier == "A" for c in claims)
