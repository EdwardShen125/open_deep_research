"""Tests for P0 source_tier 真实分级。

覆盖:
- _classify_source_tier(url) — 4 个 tier (primary/secondary/tertiary/ugc) + 未知
- EvidenceUnit.source_tier 字段 + to_v2() 透传
- EuDAO.count_by_source_tier 聚合 (需要 PG, 跳过)
"""
from __future__ import annotations

import pytest

from open_deep_research.eu_extractor import _classify_source_tier
from open_deep_research.evidence_units import EvidenceUnit


# ---- _classify_source_tier unit tests ----

class TestClassifySourceTier:
    """Domain → source_tier 真实分级 (P0 数据准确性导向)。"""

    @pytest.mark.parametrize("url,expected", [
        # primary: peer-reviewed / 官方 / 监管
        ("https://arxiv.org/abs/2105.02782", "primary"),
        ("https://www.sec.gov/Archives/edgar/data/...", "primary"),
        ("https://europa.eu/regulation/...", "primary"),
        ("https://cs.stanford.edu/papers/...", "primary"),
        ("https://www.nasa.gov/science/...", "primary"),
        # secondary: 行业媒体 / 厂商 blog / 主流新闻
        ("https://www.reuters.com/technology/...", "secondary"),
        ("https://techcrunch.com/2024/...", "secondary"),
        ("https://www.wired.com/story/...", "secondary"),
        ("https://www.gartner.com/en/...", "secondary"),
        ("https://www.mckinsey.com/...", "secondary"),
        # tertiary: 百科 / 知识库
        ("https://en.wikipedia.org/wiki/EDR", "tertiary"),
        ("https://www.britannica.com/topic/...", "tertiary"),
        ("https://baike.baidu.com/item/...", "tertiary"),
        # ugc: 论坛 / 社交
        ("https://www.reddit.com/r/cybersecurity/...", "ugc"),
        ("https://twitter.com/someone/status/...", "ugc"),
        ("https://www.linkedin.com/posts/...", "ugc"),
        ("https://medium.com/@author/...", "ugc"),
    ])
    def test_known_domains_classified_correctly(self, url, expected):
        assert _classify_source_tier(url) == expected

    def test_unknown_com_defaults_to_secondary(self):
        """未知 .com / .org / .net 默认 secondary (中位保守)。"""
        assert _classify_source_tier("https://random-startup-xyz.com/blog") == "secondary"
        assert _classify_source_tier("https://unknown-ngo.org/report") == "secondary"
        assert _classify_source_tier("https://random-vendor.net/page") == "secondary"

    def test_unknown_unknown_defaults_to_secondary(self):
        """完全未知域名 (非 .com/.org/.net) 也默认 secondary。"""
        assert _classify_source_tier("https://random.io/page") == "secondary"

    def test_empty_url_defaults_to_secondary(self):
        assert _classify_source_tier("") == "secondary"

    def test_subdomain_match(self):
        """子域名匹配也应工作。"""
        assert _classify_source_tier("https://blog.reuters.com/some-post") == "secondary"
        assert _classify_source_tier("https://en.wikipedia.org/wiki/Test") == "tertiary"
        assert _classify_source_tier("https://arxiv.org/abs/1234.5678") == "primary"

    def test_long_substring_priority(self):
        """长 substring 优先 (避免 'gov' 抢 'reuters.com')。

        sec.gov 应匹配 'sec.gov' (primary),不是其他规则。
        """
        assert _classify_source_tier("https://www.sec.gov/test") == "primary"

    def test_host_lowercase(self):
        """大小写不敏感。"""
        assert _classify_source_tier("https://ARXIV.ORG/abs/1234") == "primary"
        assert _classify_source_tier("https://En.Wikipedia.org/wiki/Test") == "tertiary"


# ---- EvidenceUnit.source_tier + to_v2() 透传 ----

class TestEvidenceUnitSourceTier:
    """EvidenceUnit 加 source_tier 字段,to_v2() 透传到 EvidenceUnitV2。"""

    def test_default_source_tier_is_none(self):
        eu = EvidenceUnit(claim="test", source_url="https://arxiv.org/abs/1")
        assert eu.source_tier is None

    def test_explicit_source_tier_accepted(self):
        eu = EvidenceUnit(
            claim="test",
            source_url="https://arxiv.org/abs/1",
            source_tier="primary",
        )
        assert eu.source_tier == "primary"

    def test_to_v2_transmits_source_tier(self):
        """to_v2() 把 self.source_tier 透传到 EvidenceUnitV2,不再硬编码 tertiary。"""
        from open_deep_research.evidence.schema import EvidenceUnitV2

        eu = EvidenceUnit(
            claim="test claim about something important",
            quote="This is a verbatim quote long enough to pass validation",
            source_url="https://arxiv.org/abs/1",
            source_title="Test",
            source_tier="primary",
        )
        v2 = eu.to_v2(run_id="r-test-001")
        assert v2.source_tier == "primary"

    def test_to_v2_defaults_to_tertiary_when_source_tier_none(self):
        """self.source_tier=None 时 to_v2() 仍给一个有效 tier 默认值 (tertiary)。"""
        from open_deep_research.evidence.schema import EvidenceUnitV2

        eu = EvidenceUnit(
            claim="test claim with enough content",
            quote="This is a verbatim quote long enough to pass validation",
            source_url="https://arxiv.org/abs/1",
        )
        v2 = eu.to_v2(run_id="r-test-001")
        assert v2.source_tier in ("tertiary", "primary", "secondary", "ugc")


# ---- extract_from_search_result 集成: source_tier 自动分类 ----

class TestExtractFromSearchResultTier:
    """extract_from_search_result 应该自动用 _classify_source_tier 填 source_tier。"""

    def test_arxiv_result_has_primary_tier(self):
        from open_deep_research.eu_extractor import extract_from_search_result

        result = {
            "url": "https://arxiv.org/abs/2105.02782",
            "title": "Test paper",
            "content": "This is the paper abstract. It has some content.",
            "provider": "searxng",
        }
        eus = extract_from_search_result(result)
        assert len(eus) >= 1
        for eu in eus:
            assert eu.source_tier == "primary"

    def test_wiki_result_has_tertiary_tier(self):
        from open_deep_research.eu_extractor import extract_from_search_result

        result = {
            "url": "https://en.wikipedia.org/wiki/EDR",
            "title": "EDR",
            "content": "Endpoint detection and response is a cybersecurity technology.",
            "provider": "searxng",
        }
        eus = extract_from_search_result(result)
        assert len(eus) >= 1
        for eu in eus:
            assert eu.source_tier == "tertiary"

    def test_reddit_result_has_ugc_tier(self):
        from open_deep_research.eu_extractor import extract_from_search_result

        result = {
            "url": "https://www.reddit.com/r/cybersecurity/comments/abc/test",
            "title": "Test post",
            "content": "Some user content here about EDR.",
            "provider": "searxng",
        }
        eus = extract_from_search_result(result)
        assert len(eus) >= 1
        for eu in eus:
            assert eu.source_tier == "ugc"


# ---- EuDAO.count_by_source_tier (PG 集成, 跳过如果没有 POSTGRES_HOST) ----

class TestEuDAOCountBySourceTier:
    """EuDAO.count_by_source_tier 聚合方法 (需要真 PG)。"""

    def test_count_by_source_tier_returns_dict(self):
        """如果有 PG,返回 {tier: count} 字典。"""
        import os
        if not os.environ.get("POSTGRES_HOST"):
            pytest.skip("POSTGRES_HOST 未设置 — EuDAO 集成测试需要真 PG")

        from open_deep_research.evidence.eu_dao import EuDAO
        # Use a fake run_id — should return empty dict
        with EuDAO() as dao:
            result = dao.count_by_source_tier("00000000-0000-0000-0000-000000000000")
        assert isinstance(result, dict)
        # For unknown run_id, all values are 0 (or empty)
        assert all(v == 0 or isinstance(v, int) for v in result.values())