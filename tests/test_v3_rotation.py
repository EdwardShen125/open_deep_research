"""V3 rotation acceptance — engines rotate per query.

V3 spec: same engines list applied to every query lets SearXNG's
per-engine dedup collapse results (v34-v40 hit this — 7/40 cap
utilization). Rotation must:
  1. Issue each query with a different engines subset (defeat dedup)
  2. Ensure every engine reaches every query over the rotation
  3. Skip rotation when n_queries<3 or n_engines<3 (no point)
  4. Log per-variant result counts (always, not only with DEBUG)
  5. Raise AllProvidersFailed when all queries return 0 results
"""
import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from open_deep_research.search_providers import (
    SearXNGProvider, SearchQuery, AllProvidersFailed,
)


def _make_provider(mock_responses):
    """SearXNGProvider with a fake fetcher that returns canned responses."""
    p = SearXNGProvider(base_url="http://test:8080", timeout=5.0)
    iter_responses = iter(mock_responses)

    async def fake_fetcher(url, params, timeout):
        try:
            return next(iter_responses)
        except StopIteration:
            return {"results": []}

    p._default_fetcher = fake_fetcher  # type: ignore
    return p


def test_rotation_skipped_when_few_queries():
    """n_queries<3 → rotation off, all queries use the same engines."""
    p = _make_provider([{"results": []}, {"results": []}])
    # Build a query with 2 queries (below threshold)
    sq = SearchQuery(
        queries=["a", "b"],
        extras={"engines": ["bing", "360search", "chinaso news"]},
    )
    # We only check that rotation logic doesn't crash; the actual SearXNG
    # call is mocked to return 0 results which raises AllProvidersFailed.
    # So this test verifies the absence-of-crash precondition is met by
    # directly inspecting the rotate_engines gate logic via a side effect:
    # If rotation were ON with n_queries=2, it would still proceed — we
    # just want to confirm the gate suppresses it.
    # The simpler check: build with a generous mock that yields results.
    p2 = _make_provider([
        {"results": [{"url": "http://x", "title": "t", "content": ""}]},
        {"results": [{"url": "http://y", "title": "t", "content": ""}]},
    ])
    sq2 = SearchQuery(queries=["a", "b"], extras={"engines": ["bing", "360search", "chinaso news"]})
    # Capture log to verify rotation skipped message
    import logging
    rec = []
    class CaptureHandler(logging.Handler):
        def emit(self, record):
            rec.append(record)
    h = CaptureHandler(level=logging.INFO)
    logger = logging.getLogger("open_deep_research.search_providers")
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    try:
        rs = asyncio.run(p2.search(sq2))
        assert len(rs) == 2
    finally:
        logger.removeHandler(h)
    # Should see "rotation skipped" log line
    skip_logs = [r for r in rec if "rotation skipped" in r.getMessage()]
    assert skip_logs, f"Expected 'rotation skipped' log when n_queries<3; got: {[r.getMessage() for r in rec]}"


def test_rotation_engines_differ_per_query(caplog):
    """n_queries≥3 AND n_engines≥3 → each query gets a different slice."""
    async def go():
        # Track which engines were sent per query
        seen = []
        async def fetcher(url, params, timeout):
            seen.append(params.get("engines", ""))
            return {"results": [{"url": f"http://x{i}", "title": "t", "content": ""} for i in range(2)]}

        p = SearXNGProvider(base_url="http://test:8080", timeout=5.0)
        p._default_fetcher = fetcher  # type: ignore
        sq = SearchQuery(
            queries=["q1", "q2", "q3", "q4", "q5"],
            extras={"engines": ["bing", "360search", "chinaso news", "wikipedia", "sogou"]},
        )
        with caplog.at_level(logging.INFO, logger="open_deep_research.search_providers"):
            rs = await p.search(sq)
        return rs, seen

    rs, seen = asyncio.run(go())
    assert len(rs) == 10, f"expected 2 results × 5 queries = 10, got {len(rs)}"
    # All engines should be different
    assert len(set(seen)) >= 3, f"expected ≥3 unique engine slices, got {seen}"
    # First and last should not be identical (rotation moved)
    assert seen[0] != seen[-1], f"first/last engines identical — no rotation: {seen[0]}"


def test_rotation_per_variant_results_logged(caplog):
    """Per-variant result counts appear in INFO log."""
    async def go():
        counts = [16, 7, 0, 10, 17]
        async def fetcher(url, params, timeout):
            n = counts.pop(0) if counts else 0
            return {"results": [{"url": f"http://x{i}", "title": "t", "content": ""} for i in range(n)]}

        p = SearXNGProvider(base_url="http://test:8080", timeout=5.0)
        p._default_fetcher = fetcher  # type: ignore
        sq = SearchQuery(
            queries=["q1", "q2", "q3", "q4", "q5"],
            extras={"engines": ["bing", "360search", "chinaso news", "wikipedia", "sogou"]},
        )
        with caplog.at_level(logging.INFO, logger="open_deep_research.search_providers"):
            rs = await p.search(sq)
        return rs

    rs = asyncio.run(go())
    assert len(rs) == 50
    rot_logs = [r for r in caplog.records if "V3 rotation" in r.getMessage()]
    assert rot_logs, "Expected '[V3 rotation]' INFO log line"
    msg = rot_logs[0].getMessage()
    assert "per_variant_results=[16, 7, 0, 10, 17]" in msg, f"unexpected counts in log: {msg}"


def test_all_providers_failed_when_zero_results():
    """All 0-result queries must raise AllProvidersFailed (no silent [])."""
    async def go():
        async def fetcher(url, params, timeout):
            return {"results": []}
        p = SearXNGProvider(base_url="http://test:8080", timeout=5.0)
        p._default_fetcher = fetcher  # type: ignore
        sq = SearchQuery(
            queries=["q1", "q2", "q3"],
            extras={"engines": ["bing", "360search", "chinaso news"]},
        )
        with pytest.raises(AllProvidersFailed):
            await p.search(sq)

    asyncio.run(go())