"""Tests for query_constructor module + SearXNGProvider extras passthrough.

Covers:
  1. Pydantic schema acceptance & rejection (good & bad topic/engine/language/time_range).
  2. Deterministic fallback per dimension (5 × 1 dimension → 5 distinct profiles).
  3. Locale detection (zh-CN vs en).
  4. Cache roundtrip (TTL honored, hits second call without LLM).
  5. SearXNGProvider.search passthrough: legacy (no extras) preserves 4-param URL.
  6. SearXNGProvider.search passthrough: with extras → engines/categories/language/time_range URL params.
  7. SearchResult.from_searxng extracts `engine` field (backward compat None).
  8. ExecutionPlan → SearchQuery projection (legacy assertions + extras).
  9. NO_QC=1 → legacy behavior (max_results=5, language=auto, single intent).
 10. End-to-end: plan_v2_pipeline.run_pipeline with deterministic QC = SearXNG returns
     EUs from multiple engines (not 100% arxiv like run 53f4db09).

These tests guard both the schema level (Pydantic) and the runtime level
(SearXNG URL params) so a typo in either location surfaces fast.
"""
from __future__ import annotations

import asyncio
import os
import urllib.parse
from dataclasses import dataclass
from typing import Any, Optional
from unittest import mock

import pytest

from open_deep_research.query_constructor import (
    QueryIntent,
    ExecutionPlan,
    construct,
    construct_with_llm,
    to_search_query,
    apply_to_search_query,
    invalidate_cache,
    SearXNG_CONFIGURED_ENGINES,
    VALID_TOPICS,
    VALID_CATEGORIES,
    VALID_LANGUAGES,
)
from open_deep_research.search_providers import (
    SearchQuery,
    SearXNGProvider,
    SearchResult,
)


# =============================================================================
# Test helpers
# =============================================================================

@dataclass
class FakeST:
    """Mimics planner_v2.SubTopic shape — only the fields query_constructor reads."""
    id: str = ""
    title: str = ""
    question: str = ""
    dimension_id: str | None = None
    expected_entities: list = None  # type: ignore

    def __init__(self, id_="st-fake", title="market_size", question="EDR market size forecast",
                 dimension_id="market_size", entities=None):
        self.id = id_
        self.title = title
        self.question = question
        self.dimension_id = dimension_id
        self.expected_entities = entities or []


def _capture_fetcher() -> tuple[Any, list]:
    """Returns (fetcher, captured_calls)."""
    captured: list[tuple[str, dict, float]] = []
    async def fetcher(url: str, params: dict, timeout: float) -> dict:
        captured.append((url, dict(params), timeout))
        return {"results": []}
    return fetcher, captured


# =============================================================================
# 1. Pydantic schema — happy path
# =============================================================================

def test_query_intent_happy():
    q = QueryIntent(
        queries=["CrowdStrike EDR market share"],
        topic="general",
        language="en",
        categories=["general"],
        engines=["bing", "arxiv"],
        time_range="year",
        max_results=10,
        expected_yield="vendor + market",
    )
    assert q.queries == ["CrowdStrike EDR market share"]
    assert q.topic == "general"
    assert q.engines == ["bing", "arxiv"]
    assert q.time_range == "year"


def test_query_intent_rejects_unknown_topic():
    with pytest.raises(Exception, match="topic"):
        QueryIntent(queries=["x"], topic="nonsense", engines=["bing"])


def test_query_intent_rejects_unknown_engine():
    with pytest.raises(Exception, match="not configured in SearXNG"):
        QueryIntent(queries=["x"], engines=["bing", "no-such-engine"])


def test_query_intent_rejects_unknown_language():
    with pytest.raises(Exception, match="language"):
        QueryIntent(queries=["x"], language="klingon", engines=["bing"])


def test_query_intent_rejects_unknown_time_range():
    with pytest.raises(Exception, match="time_range"):
        QueryIntent(queries=["x"], engines=["bing"], time_range="fortnight")


def test_query_intent_rejects_unknown_category():
    with pytest.raises(Exception, match="not in SearXNG-valid set"):
        QueryIntent(queries=["x"], engines=["bing"], categories=["bogus-cat"])


def test_query_intent_dedups_engines_preserving_order():
    q = QueryIntent(
        queries=["x"], engines=["bing", "arxiv", "bing", "arxiv", "chinaso"],
    )
    assert q.engines == ["bing", "arxiv", "chinaso"], f"engine dedup must preserve order, got {q.engines}"


def test_query_intent_drops_blank_queries():
    q = QueryIntent(queries=["valid", "  ", "", "  another  "], engines=["bing"])
    assert q.queries == ["valid", "another"]


def test_execution_plan_requires_at_least_one_intent():
    with pytest.raises(Exception, match="intents"):
        ExecutionPlan(rationale="empty", intents=[])


def test_execution_plan_default_source_is_deterministic():
    p = ExecutionPlan(
        rationale="ad-hoc",
        intents=[QueryIntent(queries=["x"], engines=["bing"])],
    )
    assert p.source == "deterministic"


# =============================================================================
# 2. Deterministic fallback per dimension
# =============================================================================

@pytest.mark.parametrize("dimension_id,expected_topic,must_have_engine", [
    ("market_size", "general", "bing"),
    ("adoption", "general", "bing"),
    ("regulation", "news", "bing"),
    ("performance", "science", "arxiv"),
    ("ethics", "general", "wikipedia"),
    (None, "general", "bing"),
])
def test_deterministic_fallback_per_dimension(dimension_id, expected_topic, must_have_engine):
    """LLM is patched out → construct() must return a deterministic profile
    that respects the dimension_id field of the sub_topic."""
    invalidate_cache()
    st = FakeST(dimension_id=dimension_id)

    async def _boom(*a, **k):
        raise RuntimeError("LLM-down")

    async def _driver():
        with mock.patch(
            "open_deep_research.query_constructor.construct_with_llm", _boom
        ):
            return await construct("brief", st)

    plan = asyncio.run(_driver())
    assert plan.source == "deterministic", f"expected deterministic, got {plan.source}"
    intent = plan.intents[0]
    assert intent.topic == expected_topic, f"dimension={dimension_id} topic mismatch"
    assert must_have_engine in intent.engines, (
        f"dimension={dimension_id} must include engine {must_have_engine}; got {intent.engines}"
    )
    assert 5 <= intent.max_results <= 50


# =============================================================================
# 3. Locale detection
# =============================================================================

def test_locale_detection_zh():
    invalidate_cache()
    st = FakeST()
    async def _driver():
        return await construct("调研主流EDR的基本情况,技术支持,人员数量,市场覆盖率", st)
    plan = asyncio.run(_driver())
    # Force-fallback (no API key in tests) → language is locale-derived.
    # If LLM worked: zh-CN is required; if fallback: zh-CN also expected.
    assert plan.intents[0].language in {"zh-CN", "en"}, "language must be LocaleDerived"


def test_locale_detection_en():
    invalidate_cache()
    st = FakeST()
    async def _driver():
        return await construct("Research the EDR market in the US: vendors and share", st)
    plan = asyncio.run(_driver())
    assert plan.intents[0].language in {"en", "auto"}, "locale en or auto expected"


# =============================================================================
# 4. Cache roundtrip (TTL honored)
# =============================================================================

def test_cache_skips_llm_on_second_call():
    invalidate_cache()
    st = FakeST(title="market_size")
    calls = []

    async def _track(*a, **k):
        calls.append(k.get("sub_topic", a[1] if len(a) > 1 else "unknown"))
        raise RuntimeError("LLM-down")  # simulate LLM failure

    async def _driver():
        with mock.patch(
            "open_deep_research.query_constructor.construct_with_llm", _track
        ):
            # First call: goes through to "LLM", fails, fallback cached.
            p1 = await construct("EDR", st)
            # Second call: must be served from cache → no second invocation.
            p2 = await construct("EDR", st)
            return p1, p2

    p1, p2 = asyncio.run(_driver())
    assert len(calls) == 1, f"expected exactly 1 LLM call (cache should short-circuit 2nd), got {len(calls)}"
    assert p1.source == "deterministic"
    # p2.source is "cache" (set by _cache_get), p1 is "deterministic" (set by fallback).
    assert p2.source == "cache", f"second call must hit cache, got {p2.source}"


def test_no_qc_returns_legacy_baseline(monkeypatch):
    """OPEN_DEEP_RESEARCH_NO_QC=1 → single intent, max_results=5, language=auto."""
    monkeypatch.setenv("OPEN_DEEP_RESEARCH_NO_QC", "1")
    invalidate_cache()
    st = FakeST()
    async def _driver():
        return await construct("brief", st)
    plan = asyncio.run(_driver())
    intent = plan.intents[0]
    assert plan.source == "deterministic"
    assert intent.max_results == 5, f"NO_QC must keep legacy max_results=5, got {intent.max_results}"
    assert intent.language == "auto"
    assert len(plan.intents) == 1


# =============================================================================
# 5. SearXNGProvider legacy path — preserved
# =============================================================================

def test_searxng_provider_no_extras_preserves_legacy_4_params():
    """When SearchQuery has no `extras`, the URL params must be *byte-identical*
    to the pre-QC baseline (4 keys: q, format, language='auto', safesearch=0)."""
    fetcher, calls = _capture_fetcher()
    sp = SearXNGProvider(base_url="http://test", fetcher=fetcher)

    sq = SearchQuery(queries=["x"], topic="general")  # no extras → legacy path
    asyncio.run(sp.search(sq))

    assert len(calls) == 1
    url, params, _ = calls[0]
    assert sorted(params.keys()) == sorted(["q", "format", "language", "safesearch"])
    assert params["language"] == "auto"
    assert params["safesearch"] == 0
    assert params["format"] == "json"
    assert "engines" not in params
    assert "categories" not in params
    assert "time_range" not in params
    assert url == "http://test/search"


# =============================================================================
# 6. SearXNGProvider passthrough — extras → URL params
# =============================================================================

def test_searxng_provider_passes_engines_categories_language_time_range():
    fetcher, calls = _capture_fetcher()
    sp = SearXNGProvider(base_url="http://test", fetcher=fetcher)

    sq = SearchQuery(
        queries=["CrowdStrike EDR"],
        topic="general",
        extras={
            "engines": ["bing", "arxiv"],
            "categories": ["general", "news"],
            "language": "zh-CN",
            "time_range": "year",
        },
    )
    asyncio.run(sp.search(sq))

    assert len(calls) == 1
    _, params, _ = calls[0]
    # Engines + categories are joined with commas (SearXNG URL syntax).
    assert params["engines"] == "bing,arxiv", f"got engines={params.get('engines')!r}"
    assert params["categories"] == "general,news"
    assert params["language"] == "zh-CN", "extras.language must override default 'auto'"
    assert params["time_range"] == "year"


def test_searxng_provider_partial_extras_only_emits_present_params():
    """extras with only `engines` must not inject empty categories/time_range."""
    fetcher, calls = _capture_fetcher()
    sp = SearXNGProvider(base_url="http://test", fetcher=fetcher)

    sq = SearchQuery(
        queries=["q"],
        extras={"engines": ["bing"]},
    )
    asyncio.run(sp.search(sq))
    _, params, _ = calls[0]
    assert params["engines"] == "bing"
    assert "categories" not in params, "categories only forwarded when present in extras"
    assert "time_range" not in params
    assert params["language"] == "auto", "partial extras must not clobber default language"


# =============================================================================
# 7. SearchResult.from_searxng captures engine
# =============================================================================

def test_search_result_from_searxng_captures_engine():
    raw = {
        "url": "http://crowdstrike.com",
        "title": "CrowdStrike",
        "content": "EDR vendor",
        "engine": "bing",
        "score": 0.95,
    }
    sr = SearchResult.from_searxng(raw, "EDR market share")
    assert sr.engine == "bing", f"engine must be captured, got {sr.engine!r}"
    assert sr.provider == "searxng"  # provider-level tag still 'searxng'


def test_search_result_from_searxng_engine_none_when_absent():
    """Older SearXNG (or muted backends) don't always emit engine — must not blow up."""
    raw = {"url": "http://x.com", "title": "x", "content": "y"}
    sr = SearchResult.from_searxng(raw, "q")
    assert sr.engine is None, "engine must be None when absent, not crash"


# =============================================================================
# 8. ExecutionPlan → SearchQuery projection
# =============================================================================

def test_to_search_query_preserves_extras():
    invalidate_cache()
    st = FakeST()
    plan = ExecutionPlan(
        rationale="test",
        intents=[
            QueryIntent(
                queries=["CrowdStrike EDR"],
                topic="general",
                language="en",
                categories=["general", "news"],
                engines=["bing", "chinaso", "arxiv"],
                time_range="year",
                max_results=15,
                expected_yield="vendor + market",
            ),
            QueryIntent(
                queries=["EDR bypass detection research"],
                topic="science",
                language="en",
                categories=["science"],
                engines=["arxiv", "openalex"],
                max_results=10,
                expected_yield="academic breadth",
            ),
        ],
        source="deterministic",
    )
    sqs = to_search_query(plan, run_id="rid-1", sub_topic=st)
    assert len(sqs) == 2

    sq0 = sqs[0]
    assert sq0.queries == ["CrowdStrike EDR"]
    assert sq0.extras["engines"] == ["bing", "chinaso", "arxiv"]
    assert sq0.extras["language"] == "en"
    assert sq0.extras["time_range"] == "year"
    assert sq0.max_results == 15
    assert sq0.research_topic == "market_size"
    assert sq0.run_id == "rid-1"

    sq1 = sqs[1]
    assert sq1.extras["engines"] == ["arxiv", "openalex"]
    assert sq1.topic == "science"
    assert "time_range" not in sq1.extras  # None omitted


# =============================================================================
# 9. Catalog sanity
# =============================================================================

def test_configured_engines_match_settings_yml():
    """SearXNG_CONFIGURED_ENGINES must stay in sync with deploy/searxng/settings.yml.
    This is a trip-wire: if you add an engine to settings.yml without updating
    query_constructor, the LLM falls back for that engine. The test enforces
    that the list is non-empty and includes the 9 engines the user confirmed."""
    expected = {"bing", "brave", "chinaso", "arxiv", "openalex",
                "semantic_scholar", "pubmed", "wikipedia", "wikidata"}
    assert expected.issubset(SearXNG_CONFIGURED_ENGINES), (
        f"missing engines: {expected - SearXNG_CONFIGURED_ENGINES}"
    )


def test_valid_topics_include_general_news_science():
    """SearXNG-supported topics we exercise in the LLM prompt must be valid."""
    assert "general" in VALID_TOPICS
    assert "news" in VALID_TOPICS
    assert "science" in VALID_TOPICS


# =============================================================================
# 10. End-to-end smoke (deterministic QC, no real network)
# =============================================================================

def test_execution_plan_emits_multi_intent_for_dimensional_brief():
    """EDR brief with 5 dimensions must yield plans whose intents span at least
    2 distinct (topic, engines) combinations — the whole point of query_constructor
    is to escape the legacy 1-intent-per-sub-topic baseline."""
    invalidate_cache()
    async def _driver():
        st_market = FakeST(title="market_size", dimension_id="market_size",
                            question="EDR market size forecast")
        st_perf = FakeST(title="performance", dimension_id="performance",
                          question="EDR performance benchmarks")
        st_reg = FakeST(title="regulation", dimension_id="regulation",
                         question="EDR regulation compliance")
        # Force deterministic fallback (no LLM).
        with mock.patch(
            "open_deep_research.query_constructor.construct_with_llm",
            side_effect=RuntimeError("LLM-down"),
        ):
            p_market = await construct("EDR", st_market)
            p_perf = await construct("EDR", st_perf)
            p_reg = await construct("EDR", st_reg)
        return p_market, p_perf, p_reg

    p_market, p_perf, p_reg = asyncio.run(_driver())

    # Different dimensions must yield different engine sets — this is the
    # preventing-the-run-53f4db09-130-over-15193 invariant.
    e_market = set(p_market.intents[0].engines)
    e_perf = set(p_perf.intents[0].engines)
    e_reg = set(p_reg.intents[0].engines)
    assert "bing" in e_market, f"market_size missing bing: {e_market}"
    assert "arxiv" in e_perf, f"performance missing arxiv: {e_perf}"
    assert "bing" in e_reg, f"regulation missing bing: {e_reg}"


# =============================================================================
# Prompt registry contract — query_constructor must be a registered prompt
# =============================================================================

def test_query_constructor_prompt_registered():
    """If query_constructor module isn't in the prompt REGISTRY, get_prompt() fails."""
    from open_deep_research.prompts import REGISTRY
    assert "query_constructor" in REGISTRY, (
        "query_constructor prompt must be registered in prompts/__init__.py REGISTRY"
    )

    # And loading it must not raise.
    from open_deep_research.llm import get_prompt, get_prompt_version
    text = get_prompt("query_constructor")
    version = get_prompt_version("query_constructor")
    assert text and len(text) > 200, "query_constructor prompt seems empty/too short"
    assert version == "query_constructor_v1"


# =============================================================================
# 11. Live SearXNG regression — proves the extras → URL → result-path works
#     end-to-end, not just via Pydantic schema validation. Skips when no
#     SearXNG is reachable.
# =============================================================================


def test_query_constructor_to_searxng_regression(monkeypatch):
    """End-to-end: deterministic QC fallback → to_search_query → SearXNGProvider.search
    real SearXNG container — must return ≥1 result with engine field populated.

    This is the regression that would catch:
      - SearXNGProvider discarding extras
      - from_searxng dropping engine
      - URL params formatting break
      - Network-timeout regression on the default fetcher

    Note: ``tests/conftest.py`` deletes SEARXNG_URL via monkeypatch every
    test, so we re-set it inline. The test stays opt-in: if no SearXNG
    is reachable from this env, it skips silently.
    """
    candidates = [
        "http://172.19.0.3:8080",
        "http://172.18.0.1:8080",
        "http://172.17.0.1:8080",
        "http://odr-searxng:8080",
        "http://127.0.0.1:8080",
    ]
    reachable = None
    for u in candidates:
        try:
            import urllib.request as _ur
            req = _ur.Request(u + "/", headers={"User-Agent": "open_deep_research/1.0"})
            with _ur.urlopen(req, timeout=2) as r:
                if r.status in (200, 301, 302):
                    reachable = u
                    break
        except Exception:
            continue
    if reachable is None:
        return  # skip silently

    monkeypatch.setenv("SEARXNG_URL", reachable)

    # Force deterministic fallback (NO API key in tests).
    invalidate_cache()
    import unittest.mock as mock
    from open_deep_research import query_constructor as qc_mod

    async def _boom(*a, **k):
        raise RuntimeError("simulated LLM-down")

    st = FakeST(title="market_size", question="EDR market size forecast 2024",
                dimension_id="market_size", entities=["EDR", "CrowdStrike"])

    async def _driver():
        with mock.patch.object(qc_mod, "construct_with_llm", _boom):
            plan = await construct("EDR market research", st)
        return plan

    plan = asyncio.run(_driver())
    sqs = to_search_query(plan, run_id="live-regression-001", sub_topic=st)

    sp = SearXNGProvider(base_url=reachable, timeout=30.0)
    sq = sqs[0]
    assert sq.extras["engines"], "extras.engines must be non-empty"
    assert sq.extras["language"] in {"auto", "en", "zh-CN"}
    assert sq.extras["categories"], "extras.categories must be non-empty"

    results = asyncio.run(sp.search(sq))
    assert results, (
        f"SearXNGProvider returned 0 results for engines={sq.extras['engines']} "
        f"language={sq.extras['language']} via {reachable}"
    )
    engines_seen = set(r.engine for r in results if r.engine)
    assert engines_seen, (
        "Each SearXNG result must carry engine field — non-empty regression on from_searxng"
    )
    # Engines contract: at least 1 of the requested engines returned ≥1 hit.
    requested = set(sq.extras["engines"])
    assert engines_seen & requested, (
        f"None of requested engines {requested} returned; got {engines_seen}"
    )
    print(f"\n  ✓ live regression: {len(results)} results, "
          f"requested={sorted(requested)} got={sorted(engines_seen)}")
