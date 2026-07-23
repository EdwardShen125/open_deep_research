"""Phase 1.3: Unified search provider interface + Tavily / SearXNG implementations.

## Why

v1's `tavily_search` is hard-wired to the Tavily SDK and depends on the
LangGraph runtime — it can't be tested offline, doesn't fall back to
SearXNG when Tavily rate-limits, and doesn't integrate with the Phase 1.1
SourcesDAO / Phase 1.2 SearchCache.

Plan v2 introduction: a `SearchProvider` protocol + `TavilyProvider` /
`SearXNGProvider` implementations + a `UnifiedSearch` orchestrator that
routes through primary, falls back to secondary, caches via Phase 1.2,
and registers hits via Phase 1.1.

## API

    from open_deep_research.search_providers import (
        UnifiedSearch, TavilyProvider, SearXNGProvider,
        SearchQuery, SearchResult,
    )

    us = UnifiedSearch(
        primary=TavilyProvider(),
        fallback=SearXNGProvider(base_url=os.environ["SEARXNG_URL"]),
        cache=SearchCache(sources_dao=...),
    )
    out = await us.search(SearchQuery(queries=["Klue CI market"], topic="ci"))
    # out.results: list[SearchResult] — provider-agnostic shape

Each `SearchResult` is a typed dict mapped onto Phase 1.1
`SourceRecord.from_raw()` so the data flow stays consistent.
"""

from __future__ import annotations

import logging

import asyncio
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from open_deep_research.search_cache import SearchCache
from open_deep_research.sources_dao import SourcesDAO, SourceRecord


# =============================================================================
# Query sanitization (F: bottom-layer insurance for ALL providers)
# =============================================================================
# SearXNG forwards query directly to upstream HTTP endpoints (arxiv.org etc.)
# which reject: em-dash (U+2014), colon (`:`), unicode quotes, parens, etc.
# Tavily tolerates more, but stripping still helps. We replace unicode
# punctuation with ASCII equivalents, collapse whitespace, truncate to ≤120
# chars. Pure-ASCII queries round-trip unchanged. Defensive fallback returns
# the original query if sanitization yields empty.
_MAX_QUERY_LEN = 120
_UNICODE_PUNCT_MAP = {
    "\u2014": "-",  # em dash —
    "\u2013": "-",  # en dash –
    "\u2018": "'",  # left single quote '
    "\u2019": "'",  # right single quote '
    "\u201C": '"',  # left double quote "
    "\u201D": '"',  # right double quote "
    "\u00B7": ".",  # middle dot ·
    "\u2026": "...",  # ellipsis …
    "\u00A0": " ",  # non-breaking space
}


def _sanitize_query(q: str) -> str:
    """Strip unicode punctuation + collapse whitespace + truncate to ≤120 chars.

    Pure-ASCII queries round-trip unchanged (length permitting). Defensive
    fallback returns the original query if sanitization yields empty.
    """
    if not q:
        return q
    out = q
    for src, dst in _UNICODE_PUNCT_MAP.items():
        out = out.replace(src, dst)
    # Drop remaining punctuation that arxiv/SearXNG choke on: `:`, `(`, `)`,
    # `[`, `]`, `{`, `}`, `\`, `/`. Keep `-`, `_`, `.`, `'`, `,`, `?`, `!`.
    out = re.sub(r"[:\(\)\[\]\{\}\\/]", " ", out)
    # Collapse whitespace runs.
    out = re.sub(r"\s+", " ", out).strip()
    # Truncate to ≤120 chars (arxiv sweet spot).
    if len(out) > _MAX_QUERY_LEN:
        out = out[:_MAX_QUERY_LEN].rstrip()
    return out or q


# =============================================================================
# Public types
# =============================================================================

@dataclass
class SearchQuery:
    """One logical search operation (may carry multiple queries)."""
    queries: list[str]
    topic: str = "general"               # 'general' / 'news' / 'finance'
    max_results: int = 5
    include_raw_content: bool = True
    run_id: Optional[str] = None
    research_topic: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Provider-agnostic search hit — same shape as a Tavily result."""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    raw_content: Optional[str] = None
    score: Optional[float] = None
    provider: str = ""                  # 'tavily' / 'searxng'
    provider_query: Optional[str] = None
    http_status: Optional[int] = None
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_tavily(cls, raw: dict, query: str) -> "SearchResult":
        return cls(
            url=raw.get("url", ""),
            title=raw.get("title"),
            content=raw.get("content"),
            raw_content=raw.get("raw_content"),
            score=raw.get("score"),
            provider="tavily",
            provider_query=query,
            raw_payload=raw,
        )

    @classmethod
    def from_searxng(cls, raw: dict, query: str) -> "SearchResult":
        # SearXNG returns slightly different field names; normalize.
        return cls(
            url=raw.get("url", ""),
            title=raw.get("title"),
            content=raw.get("content"),
            raw_content=None,                  # SearXNG doesn't ship raw_content
            score=raw.get("score"),
            provider="searxng",
            provider_query=query,
            raw_payload=raw,
        )


@dataclass
class SearchResponse:
    """Aggregate output of `UnifiedSearch.search(...)`."""
    results: list[SearchResult]
    source: str                            # 'cache' / 'tavily' / 'searxng' / 'mixed'
    primary_used: bool = False
    fallback_used: bool = False
    failed_providers: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Provider protocol
# =============================================================================

@runtime_checkable
class SearchProvider(Protocol):
    """A search backend. Implementations must be async and side-effect-free."""

    name: str

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute a search and return a flat list of `SearchResult`."""
        ...


# =============================================================================
# TavilyProvider
# =============================================================================

class TavilyProvider:
    """Search backend backed by Tavily.

    Configuration:
      - `api_key` (env: TAVILY_API_KEY) — required at construction.
    """

    name = "tavily"

    def __init__(self, api_key: Optional[str] = None, *, client: Any = None) -> None:
        # Lazy import — the module should still importable in environments
        # where `tavily` is not installed (e.g. minimal CI).
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self._client = client
        if client is None and self._api_key:
            try:
                from tavily import AsyncTavilyClient  # type: ignore
                self._client = AsyncTavilyClient(api_key=self._api_key)
            except Exception:
                # Leave _client as None → all calls will raise at call-time.
                self._client = None

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if self._client is None:
            raise RuntimeError(
                "TavilyProvider is not configured: pass api_key=... or set "
                "TAVILY_API_KEY env var + ensure `tavily-python` is installed"
            )
        results: list[SearchResult] = []
        tasks = [
            self._client.search(
                q,
                max_results=query.max_results,
                include_raw_content=query.include_raw_content,
                topic=query.topic,
            )
            for q in query.queries
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for q, resp in zip(query.queries, responses):
            if isinstance(resp, Exception):
                # Skip but record; UnifiedSearch decides whether to retry.
                continue
            for raw in resp.get("results", []) or []:
                results.append(SearchResult.from_tavily(raw, q))
        return results


# =============================================================================
# SearXNGProvider
# =============================================================================

class SearXNGProvider:
    """Search backend backed by SearXNG (HTTP JSON API).

    Configuration:
      - `base_url` (env: SEARXNG_URL, default http://127.0.0.1:8888)
      - Optional `fetcher` override for tests (signature: async (url, params) -> dict)
    """

    name = "searxng"

    DEFAULT_BASE_URL = "http://127.0.0.1:8888"

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        fetcher: Any = None,
        timeout: float = 5.0,
    ) -> None:
        self._base_url = (base_url or os.environ.get("SEARXNG_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self._fetcher = fetcher
        self._timeout = timeout

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        fetcher = self._fetcher or self._default_fetcher
        results: list[SearchResult] = []
        for q in query.queries:
            params = {
                "q": q,
                "format": "json",
                "language": "auto",
                "safesearch": 0,
            }
            try:
                resp = await fetcher(self._base_url + "/search", params, self._timeout)
            except Exception:
                continue
            for raw in resp.get("results", []) or []:
                results.append(SearchResult.from_searxng(raw, q))
        return results

    async def _default_fetcher(self, url: str, params: dict, timeout: float) -> dict:
        """Use urllib in a thread — keeps the dependency surface tiny."""
        import urllib.parse
        import urllib.request
        full = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            full,
            headers={"User-Agent": "open_deep_research/1.0"},
        )
        # Run in a worker thread so we don't block the event loop.
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=timeout)
        )
        import json
        return json.loads(resp.read().decode("utf-8"))


# =============================================================================
# UnifiedSearch orchestrator
# =============================================================================

class UnifiedSearch:
    """Routes a SearchQuery through primary → cache → fallback → register.

    Behavior:
      1. Try cache hit (Phase 1.2 SearchCache, keyed on query + topic).
         If cache returns ≥1 results, return them with `source='cache'`.
      2. Run primary.search(query). If results returned, optionally write
         through to cache, register SourceRecords via SourcesDAO, return.
      3. On primary failure OR empty primary results, run fallback.
         Same write-through + registration logic.
      4. If both primary and fallback fail: raise `AllProvidersFailed`
         carrying the underlying exceptions.
    """

    def __init__(
        self,
        *,
        primary: Optional[SearchProvider] = None,
        fallback: Optional[SearchProvider] = None,
        cache: Optional[SearchCache] = None,
        sources_dao: Optional[SourcesDAO] = None,
        register_to_sources: bool = True,
        write_through_cache: bool = True,
        clock: Any = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cache = cache
        self.sources_dao = sources_dao
        self._register = register_to_sources
        self._write_through = write_through_cache
        self._clock = clock or time.monotonic

    async def search(self, query: SearchQuery) -> SearchResponse:
        # F: Sanitize queries at the entry point — strip unicode punctuation,
        # collapse whitespace, truncate to ≤120 chars. Protects all providers
        # (Tavily / SearXNG / future) from upstream rejects on em-dash / colon /
        # quotes / parentheses. Original semantics preserved for ASCII queries.
        query.queries = [_sanitize_query(q) for q in query.queries]
        started = self._clock()
        resp = SearchResponse(results=[], source="")

        # ---- 1. cache hit ----
        cached = self._read_cache(query) if self.cache else None
        if cached and cached.get("results"):
            resp.results = [_dict_to_result(r) for r in cached["results"]]
            resp.source = "cache"
            resp.cache_hits += 1
            resp.latency_ms = int((self._clock() - started) * 1000)
            return resp
        if self.cache:
            resp.cache_misses += 1

        # ---- 2. primary ----
        primary_failed = False
        if self.primary is not None:
            try:
                results = await self.primary.search(query)
            except Exception as e:
                primary_failed = True
                resp.failed_providers.append(self.primary.name)
                logger.warning("primary=%s failed: %s", self.primary.name, e)
            else:
                if results:
                    resp.results = results
                    resp.primary_used = True
                    resp.source = self.primary.name

        # ---- 3. fallback ----
        # 当 primary 没给出结果 / 没 primary / primary 失败时,fallback 接管
        if not resp.results and self.fallback is not None:
            try:
                results = await self.fallback.search(query)
            except Exception as e:
                resp.failed_providers.append(self.fallback.name)
                logger.warning("fallback=%s failed: %s", self.fallback.name, e)
            else:
                if results:
                    resp.results = results
                    resp.fallback_used = True
                    if resp.source:
                        resp.source += "+" + self.fallback.name
                    else:
                        resp.source = self.fallback.name

        if not resp.results:
            raise AllProvidersFailed(
                f"All providers failed for queries={query.queries!r}; "
                f"failed={resp.failed_providers}"
            )

        # ---- 4. side effects: registration + write-through cache ----
        self._register_sources(query, resp)
        if self._write_through and self.cache:
            self._write_cache(query, resp)

        resp.latency_ms = int((self._clock() - started) * 1000)
        return resp

    # ------- helpers -------
    def _read_cache(self, query: SearchQuery) -> Optional[dict]:
        # Cache per query; we just probe the first one in a multi-query batch.
        if not query.queries:
            return None
        return self.cache.get(query.queries[0], topic=query.topic)

    def _write_cache(self, query: SearchQuery, resp: SearchResponse) -> None:
        if not query.queries:
            return
        self.cache.put(
            query.queries[0],
            {"results": [r.to_dict() for r in resp.results]},
            topic=query.topic,
            urls=[r.to_dict() for r in resp.results],
        )

    def _register_sources(self, query: SearchQuery, resp: SearchResponse) -> None:
        if not (self._register and self.sources_dao is not None):
            return
        for r in resp.results:
            try:
                rec = SourceRecord.from_raw(
                    url=r.url,
                    title=r.title,
                    provider=r.provider,
                    provider_query=r.provider_query or (query.queries[0] if query.queries else None),
                    provider_score=r.score,
                    provider_payload=r.raw_payload,
                    run_id=query.run_id,
                    research_topic=query.research_topic,
                )
                self.sources_dao.upsert(rec)
            except Exception:
                # Best-effort; failure to register is not fatal.
                continue


# =============================================================================
# Errors
# =============================================================================

class AllProvidersFailed(Exception):
    """Both primary and fallback providers returned 0 results."""


# =============================================================================
# Internal helpers
# =============================================================================

def _dict_to_result(d: dict) -> SearchResult:
    return SearchResult(
        url=d.get("url", ""),
        title=d.get("title"),
        content=d.get("content"),
        raw_content=d.get("raw_content"),
        score=d.get("score"),
        provider=d.get("provider", ""),
        provider_query=d.get("provider_query"),
        http_status=d.get("http_status"),
        raw_payload=d.get("raw_payload", {}) or {},
    )
