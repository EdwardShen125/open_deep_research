"""Phase 1.3 — Search providers (UnifiedSearch + Tavily + SearXNG)."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.search_providers import (  # noqa: E402
    UnifiedSearch, TavilyProvider, SearXNGProvider,
    SearchQuery, SearchResult, SearchResponse, AllProvidersFailed,
)


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------

class FakeProvider:
    """Returns a configurable result set; can fail on demand."""
    def __init__(self, name, results=None, fail=False):
        self.name = name
        self._results = results or []
        self._fail = fail
        self.calls = []

    async def search(self, query: SearchQuery):
        self.calls.append(query)
        if self._fail:
            raise RuntimeError(f"{self.name} simulated failure")
        return list(self._results)


def _sr(url, title=None, score=0.5, provider="tavily"):
    return SearchResult(
        url=url, title=title, content="...", score=score,
        provider=provider, provider_query="Klue Crayon",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_search_result_from_tavily():
    raw = {"url": "https://klue.com/a", "title": "t", "content": "c", "score": 0.9, "raw_content": "raw"}
    r = SearchResult.from_tavily(raw, query="Klue")
    assert r.url == "https://klue.com/a"
    assert r.title == "t"
    assert r.score == 0.9
    assert r.provider == "tavily"
    assert r.provider_query == "Klue"
    assert r.raw_payload == raw


def test_search_result_from_searxng():
    raw = {"url": "https://crayon.co/b", "title": "t2", "content": "c2", "score": 0.8}
    r = SearchResult.from_searxng(raw, query="c2 query")
    assert r.url == "https://crayon.co/b"
    assert r.provider == "searxng"
    assert r.raw_content is None   # SearXNG doesn't ship raw_content
    print("  ✓ SearchResult.from_searxng normalizes fields")


def test_tavily_provider_missing_config_raises_at_call():
    p = TavilyProvider(api_key=None, client=None)
    try:
        asyncio.run(p.search(SearchQuery(queries=["x"])))
    except RuntimeError as e:
        assert "TavilyProvider is not configured" in str(e)
        print("  ✓ TavilyProvider refuses to run without API key")
        return
    raise AssertionError("expected RuntimeError")


def test_unified_search_uses_primary_when_results_returned():
    primary = FakeProvider("tavily", results=[
        _sr("https://klue.com/a", "Klue A"),
        _sr("https://klue.com/b", "Klue B"),
    ])
    us = UnifiedSearch(primary=primary)
    resp = asyncio.run(us.search(SearchQuery(queries=["Klue"])))
    assert resp.primary_used and not resp.fallback_used
    assert resp.source == "tavily"
    assert len(resp.results) == 2
    print("  ✓ UnifiedSearch returns primary when primary has results")


def test_unified_search_falls_back_when_primary_empty():
    primary = FakeProvider("tavily", results=[])
    fallback = FakeProvider("searxng", results=[
        _sr("https://crayon.co/x", "Crayon X"),
    ])
    us = UnifiedSearch(primary=primary, fallback=fallback)
    resp = asyncio.run(us.search(SearchQuery(queries=["x"])))
    assert not resp.primary_used
    assert resp.fallback_used
    assert resp.source == "searxng"
    assert len(resp.results) == 1
    assert primary.calls and fallback.calls
    print("  ✓ UnifiedSearch falls back to secondary when primary returns empty")


def test_unified_search_falls_back_when_primary_raises():
    primary = FakeProvider("tavily", fail=True)
    fallback = FakeProvider("searxng", results=[_sr("https://x.com/x", "X")])
    us = UnifiedSearch(primary=primary, fallback=fallback)
    resp = asyncio.run(us.search(SearchQuery(queries=["x"])))
    assert "tavily" in resp.failed_providers
    assert resp.fallback_used
    print("  ✓ UnifiedSearch swallows primary error → tries fallback")


def test_unified_search_all_providers_failed():
    primary = FakeProvider("tavily", fail=True)
    fallback = FakeProvider("searxng", fail=True)
    us = UnifiedSearch(primary=primary, fallback=fallback)
    try:
        asyncio.run(us.search(SearchQuery(queries=["x"])))
    except AllProvidersFailed as e:
        assert "tavily" in str(e) and "searxng" in str(e)
        print("  ✓ AllProvidersFailed when primary + fallback both fail")
        return
    raise AssertionError("expected AllProvidersFailed")


def test_unified_search_no_fallback_required_when_only_primary():
    primary = FakeProvider("tavily", results=[_sr("https://x.com/x", "X")])
    us = UnifiedSearch(primary=primary)   # no fallback configured
    resp = asyncio.run(us.search(SearchQuery(queries=["x"])))
    assert resp.primary_used and resp.source == "tavily"
    print("  ✓ UnifiedSearch runs without a fallback configured")


def test_unified_search_cache_hit_short_circuits():
    """If SearchCache has cached results, the providers are not called."""
    from open_deep_research.search_cache import SearchCache

    class _Counter(FakeProvider):
        pass

    counter = _Counter("tavily", results=[_sr("https://new.com/new", "fresh")])
    cache = SearchCache(ttl_seconds=60)
    cache.put("Klue CI market", {"results": [
        {"url": "https://cached.com/x", "title": "CACHED", "provider": "cache"}
    ]}, topic="general")
    us = UnifiedSearch(primary=counter, cache=cache)
    resp = asyncio.run(us.search(SearchQuery(queries=["Klue CI market"], topic="general")))
    assert resp.source == "cache"
    assert counter.calls == [], "primary should NOT be called on cache hit"
    assert resp.results[0].url == "https://cached.com/x"
    print("  ✓ cache hit short-circuits primary provider")


def test_unified_search_write_through_cache_on_miss():
    from open_deep_research.search_cache import SearchCache
    primary = FakeProvider("tavily", results=[_sr("https://x.com/x", "X")])
    cache = SearchCache(ttl_seconds=60)
    assert cache.get("Klue CI") is None
    us = UnifiedSearch(primary=primary, cache=cache)
    asyncio.run(us.search(SearchQuery(queries=["Klue CI"], topic="general")))
    # After the call, cache should have something
    cached = cache.get("Klue CI", topic="general")
    assert cached is not None
    assert cached["results"][0]["url"] == "https://x.com/x"
    print("  ✓ cache written-through after primary fetch")


def test_unified_search_registers_sources_via_dao():
    import json
    sys.path.insert(0, str(ROOT / "tests"))
    from test_sources_dao_sqlite import _SQLiteConnection, _DAOTest  # type: ignore
    primary = FakeProvider("tavily", results=[
        _sr("https://klue.com/a", "Klue A"),
        _sr("https://crayon.co/b", "Crayon B"),
    ])
    dao = _DAOTest(_SQLiteConnection())
    us = UnifiedSearch(primary=primary, sources_dao=dao)
    asyncio.run(us.search(SearchQuery(queries=["x"], run_id="r-1")))
    s = dao.stats()
    assert s["total"] >= 2
    print(f"  ✓ unified search registers source URLs into SourcesDAO (total={s['total']})")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("search_result_from_tavily", test_search_result_from_tavily),
        ("search_result_from_searxng", test_search_result_from_searxng),
        ("tavily_provider_missing_config_raises_at_call",
         test_tavily_provider_missing_config_raises_at_call),
        ("unified_search_uses_primary_when_results_returned",
         test_unified_search_uses_primary_when_results_returned),
        ("unified_search_falls_back_when_primary_empty",
         test_unified_search_falls_back_when_primary_empty),
        ("unified_search_falls_back_when_primary_raises",
         test_unified_search_falls_back_when_primary_raises),
        ("unified_search_all_providers_failed",
         test_unified_search_all_providers_failed),
        ("unified_search_no_fallback_required_when_only_primary",
         test_unified_search_no_fallback_required_when_only_primary),
        ("unified_search_cache_hit_short_circuits",
         test_unified_search_cache_hit_short_circuits),
        ("unified_search_write_through_cache_on_miss",
         test_unified_search_write_through_cache_on_miss),
        ("unified_search_registers_sources_via_dao",
         test_unified_search_registers_sources_via_dao),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} search-provider tests")
    print("=" * 70)
    failed = []
    for name, fn in tests:
        try:
            print(f"\n[{name}]")
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed.append(name)
    print("\n" + "=" * 70)
    if failed:
        print(f" {len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f" ALL {len(tests)} TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
