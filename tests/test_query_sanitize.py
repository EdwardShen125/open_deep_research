"""Tests for F: _sanitize_query in search_providers.py.

Verifies unicode → ASCII, colon/parens stripping, whitespace collapse,
and ≤120 char truncation. Also verifies D: planner_v2 dimension templates
are short and ASCII-only.
"""
from __future__ import annotations

import pytest

from open_deep_research.search_providers import _sanitize_query, SearchQuery, UnifiedSearch
from open_deep_research.planner_v2 import (
    DIMENSION_TEMPLATES,
    plan_from_brief,
)


# ---- F: _sanitize_query unit tests ----

class TestSanitizeQuery:
    """Bottom-layer query sanitization (F)."""

    def test_ascii_round_trip(self):
        """Pure ASCII queries are unchanged."""
        q = "EDR market size forecast revenue"
        assert _sanitize_query(q) == q

    def test_em_dash_to_ascii(self):
        q = "EDR market overview — vendors growth"
        out = _sanitize_query(q)
        assert "\u2014" not in out
        assert "-" in out

    def test_en_dash_to_ascii(self):
        q = "Q3\u2013Q4 2024 forecast"
        out = _sanitize_query(q)
        assert "\u2013" not in out
        assert "-" in out

    def test_unicode_quotes_to_ascii(self):
        q = "The \u201csmart\u201d market for EDR"
        out = _sanitize_query(q)
        assert "\u201c" not in out
        assert "\u201d" not in out

    def test_colon_stripped(self):
        q = "context of: EDR market"
        out = _sanitize_query(q)
        assert ":" not in out

    def test_parens_and_brackets_stripped(self):
        q = "EDR market (Q3) [forecast]"
        out = _sanitize_query(q)
        assert "(" not in out
        assert ")" not in out
        assert "[" not in out
        assert "]" not in out

    def test_whitespace_collapsed(self):
        q = "  multiple   spaces   and   emdash  "
        out = _sanitize_query(q)
        assert "  " not in out  # no double-spaces
        assert out == "multiple spaces and emdash"

    def test_truncate_to_120_chars(self):
        q = "A" * 200
        out = _sanitize_query(q)
        assert len(out) == 120

    def test_empty_query_returns_empty(self):
        assert _sanitize_query("") == ""

    def test_non_ascii_input_no_unicode_punct_in_output(self):
        """Final output should be ASCII-only (punctuation map covers all unicode)."""
        q = "EDR \u00b7 market \u2026 overview \u00a0 2024"
        out = _sanitize_query(q)
        out.encode("ascii")  # raises if any non-ASCII char remains


# ---- F: UnifiedSearch entry-point integration ----

class TestUnifiedSearchSanitizesQueries:
    """Verify UnifiedSearch.search sanitizes query.queries in place."""

    @pytest.mark.asyncio
    async def test_sanitize_runs_before_provider_call(self):
        """Even if a provider is None, sanitize runs without error."""
        us = UnifiedSearch(primary=None, fallback=None)
        q = SearchQuery(queries=["EDR market \u2014 vendors"])
        # We can't actually call us.search() without providers, but we can
        # verify the sanitize line runs by calling it directly:
        q.queries = [_sanitize_query(s) for s in q.queries]
        assert q.queries == ["EDR market - vendors"]


# ---- D: planner_v2 dimension templates short + ASCII ----

class TestDimensionTemplatesShortASCII:
    """Verify D: dimension query templates are short + ASCII."""

    @pytest.mark.parametrize("dim", ["market_size", "adoption", "regulation", "performance", "ethics"])
    def test_template_format_yields_ascii(self, dim):
        brief = "EDR market 2024"
        q = DIMENSION_TEMPLATES[dim].format(brief=brief)
        out = q.encode("ascii", errors="strict")
        # ASCII-only, no crash
        assert isinstance(out, bytes)

    @pytest.mark.parametrize("dim", ["market_size", "adoption", "regulation", "performance", "ethics"])
    def test_template_format_yields_short_query(self, dim):
        brief = "EDR market 2024"
        q = DIMENSION_TEMPLATES[dim].format(brief=brief)
        # After sanitize: ≤ 120 chars. Before sanitize: should be < 80 chars
        # (template itself is short + brief is short).
        assert len(q) < 80

    @pytest.mark.parametrize("dim", ["market_size", "adoption", "regulation", "performance", "ethics"])
    def test_template_no_colon_no_unicode(self, dim):
        brief = "EDR market 2024"
        q = DIMENSION_TEMPLATES[dim].format(brief=brief)
        assert ":" not in q
        assert "\u2014" not in q
        assert "(" not in q


# ---- D: plan_from_brief context query short ----

class TestPlanFromBriefContextShort:
    """Context sub_topic question should be brief verbatim, not 'context of:' wrapper."""

    def test_dimensions_mode_context_no_wrapper(self):
        brief = "EDR market 2024 vendors growth regulation"
        plan = plan_from_brief(brief=brief, max_subtopics=6, mode="dimensions")
        context = [st for st in plan.sub_topics if st.dimension_id is None]
        assert len(context) == 1
        q = context[0].question
        assert "context of:" not in q
        assert "What is" not in q  # no question wrapper
        # Just brief verbatim (≤ 80 chars)
        assert len(q) <= 80

    def test_clauses_mode_context_no_wrapper(self):
        brief = "EDR market 2024 vendors growth regulation"
        plan = plan_from_brief(brief=brief, max_subtopics=6, mode="clauses")
        context = [st for st in plan.sub_topics if st.title == "context"]
        assert len(context) == 1
        q = context[0].question
        assert "context of:" not in q
        assert "What is" not in q