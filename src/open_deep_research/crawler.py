"""Phase 1.4: Crawl4AI integration — page-level URL resolution + content fetch.

## Why

Plan v2 has a specific anchor category (B-class):
  - 8/59 references are domain-only URLs (no specific page)

When a search provider returns a domain-only URL like `https://klue.com`
(no path), v1 just stamp-cites it; the reader can land on the homepage
but not the specific statement.

Phase 1.4 introduces `Crawl4AIProvider.fetch(url)`:
  - For a domain-only URL → tries the homepage + sitemap or known entry
    points to find a page-level URL that *likely* matches the topic.
  - For a page-level URL → fetches the markdown body and stores it in
    `evidence.sources.raw_content` + `raw_content_hash`.

Design:
  - The crawler is a Protocol-shaped interface — the same SearchProvider
    pattern as `search_providers.py`. We expose:
      `Crawl4AIProvider` (real implementation, lazy-imports `crawl4ai`)
      `MockCrawlProvider` (test-time deterministic substitution)
  - The "domain-only → page-level" promotion is heuristic:
    1. Try fetching the root page.
    2. If the root page links to topic-relevant sections, pick the most
       semantically similar match.
    3. Otherwise, leave the URL alone and the verifier flags it.

The crawler is wired into the `enforce_page_level()` resolver callback
(Phase 3b) so the publish-time URL audit benefits from the upgraded
page-level URLs without callers having to do anything special.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from open_deep_research.sources_dao import (
    SourcesDAO, SourceRecord, canonicalize_url, classify_page_level, PageLevel,
)


# =============================================================================
# Public types
# =============================================================================

@dataclass
class CrawlResponse:
    url: str
    final_url: Optional[str] = None    # when redirected
    markdown: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    http_status: Optional[int] = None
    error: Optional[str] = None
    promoted_to_page_level: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class CrawlProvider(Protocol):
    name: str
    async def fetch(self, url: str, *, prompt_hint: Optional[str] = None) -> CrawlResponse: ...


# =============================================================================
# Crawl4AI provider (real)
# =============================================================================

class Crawl4AIProvider:
    """Async crawler backed by crawl4ai (when installed).

    Usage:
        p = Crawl4AIProvider()
        resp = await p.fetch("https://klue.com")

    The crawler best-effort normalizes a domain-only URL into a
    page-level URL by inspecting the homepage's links and choosing one
    that best matches `prompt_hint` (a short topic/keyword).
    """

    name = "crawl4ai"

    def __init__(self, *, timeout: float = 15.0, fetcher: Any = None) -> None:
        self._timeout = timeout
        self._fetcher = fetcher  # async fn(url) -> {markdown, links, http_status}

    async def fetch(self, url: str, *, prompt_hint: Optional[str] = None) -> CrawlResponse:
        """Return a CrawlResponse.

        For domain-only URLs we attempt to find a child page whose text
        looks like the topic. For page-level URLs we fetch and store.
        """
        if not url:
            return CrawlResponse(url=url, error="empty url")
        cls = classify_page_level(url)
        if cls is PageLevel.UNKNOWN:
            return CrawlResponse(url=url, error="unparseable url")
        fetcher = self._fetcher or self._default_fetcher
        try:
            blob = await fetcher(url, self._timeout)
        except Exception as e:
            return CrawlResponse(url=url, error=f"fetch error: {type(e).__name__}: {e}")

        if cls is PageLevel.PAGE:
            return CrawlResponse(
                url=url,
                final_url=blob.get("final_url") or url,
                markdown=blob.get("markdown") or blob.get("raw_content"),
                metadata=blob.get("metadata", {}) or {},
                http_status=blob.get("http_status"),
                promoted_to_page_level=False,
            )

        # Domain-only: try to find a page-level URL via topic matching.
        links = blob.get("links") or []
        body_md = blob.get("markdown") or blob.get("raw_content") or ""
        promoted = _best_child_url(url, links, body_md, prompt_hint)
        if not promoted:
            return CrawlResponse(
                url=url, error="no topic-relevant child page found",
                metadata={"candidate_count": len(links)},
                promoted_to_page_level=False,
            )
        # Re-fetch the promoted page.
        try:
            blob2 = await fetcher(promoted, self._timeout)
            md2 = blob2.get("markdown") or blob2.get("raw_content")
        except Exception as e:
            return CrawlResponse(
                url=url, error=f"promoted fetch failed: {e}",
                promoted_to_page_level=False,
            )
        return CrawlResponse(
            url=url,
            final_url=promoted,
            markdown=md2,
            metadata={"promoted_from": url, "original_links": len(links)},
            http_status=blob2.get("http_status"),
            promoted_to_page_level=True,
        )

    async def _default_fetcher(self, url: str, timeout: float) -> dict:
        """Default fetcher: try `crawl4ai` if available, else urllib fallback."""
        try:
            from crawl4ai import AsyncWebCrawler  # type: ignore
            async with AsyncWebCrawler() as crawler:
                r = await crawler.arun(url=url, timeout=timeout)
            return {
                "final_url": getattr(r, "url", None) or url,
                "markdown": getattr(r, "markdown", None) or getattr(r, "cleaned_html", None),
                "links": [ln.get("href") for ln in (getattr(r, "links", {}) or {}).get("internal", []) or []],
                "metadata": getattr(r, "metadata", {}) or {},
                "http_status": getattr(r, "status_code", None),
            }
        except ImportError:
            # Fallback: urllib for body only; no link extraction in raw mode.
            import urllib.request
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(url, timeout=timeout)
            )
            body = resp.read().decode("utf-8", errors="ignore")
            return {
                "final_url": resp.geturl(),
                "markdown": body,
                "links": _extract_href(body, base=url),
                "metadata": {"Content-Type": resp.headers.get("Content-Type", "")},
                "http_status": resp.status,
            }


_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)


def _extract_href(html: str, base: str) -> list[str]:
    """Pull hrefs from a raw HTML body, normalize to absolute URLs."""
    out: list[str] = []
    for h in _HREF_RE.findall(html or ""):
        if h.startswith("#") or h.startswith("mailto:"):
            continue
        if h.startswith("//"):
            out.append("https:" + h)
        elif h.startswith("/"):
            # join with base
            from urllib.parse import urlsplit, urlunsplit
            parts = urlsplit(base)
            out.append(urlunsplit((parts.scheme, parts.netloc, h, "", "")))
        elif h.startswith("http"):
            out.append(h)
    return list(dict.fromkeys(out))  # de-dup, preserve order


def _best_child_url(
    root_url: str,
    links: Iterable[str],
    body_md: str,
    prompt_hint: Optional[str],
) -> Optional[str]:
    """Choose the link whose body matches `prompt_hint` most strongly."""
    if not links:
        return None
    hint = (prompt_hint or "").lower().strip()
    if not hint:
        return None
    candidates = []
    for ln in links:
        if classify_page_level(ln) is not PageLevel.PAGE:
            continue
        if not _same_domain(root_url, ln):
            continue
        score = _topic_match(ln + " " + body_md, hint)
        candidates.append((score, ln))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _same_domain(a: str, b: str) -> bool:
    from urllib.parse import urlsplit
    ha = (urlsplit(a).hostname or "").lower()
    hb = (urlsplit(b).hostname or "").lower()
    if not ha or not hb:
        return False
    # Compare registrable suffix loosely: last 2 components.
    return ha.split(".")[-2:] == hb.split(".")[-2:]


def _topic_match(blob: str, hint: str) -> int:
    blob_l = blob.lower()
    if not hint:
        return 0
    hint_l = hint.lower()
    # crude: count occurrences of hint substrings (case-insensitive)
    count = blob_l.count(hint_l)
    # boost when path itself contains the hint words
    for w in hint_l.split():
        if len(w) < 3:
            continue
        if w in blob_l:
            count += 1
    return count


# =============================================================================
# Mock provider (deterministic — for tests + offline runs)
# =============================================================================

class MockCrawlProvider:
    """A simple dict-keyed CrawlProvider used by tests and offline runs.

    Configure `responses: dict[url, dict]` where each value is the
    inner `blob` shape returned by Crawl4AIProvider's fetcher.
    """

    name = "mock"

    def __init__(self, responses: dict[str, dict] = None) -> None:
        self._responses: dict[str, dict] = responses or {}

    def set(self, url: str, blob: dict) -> None:
        self._responses[url] = blob

    async def fetch(self, url: str, *, prompt_hint: Optional[str] = None) -> CrawlResponse:
        if not url:
            return CrawlResponse(url=url, error="empty url")
        cls = classify_page_level(url)
        if cls is PageLevel.UNKNOWN:
            return CrawlResponse(url=url, error="unparseable url")
        # Try to match canonical URL form for resilience to hash-stripped variants.
        canon = canonicalize_url(url)
        blob = (
            self._responses.get(url)
            or self._responses.get(canon)
            or self._responses.get(_any_host_match(self._responses, url))
        )
        if blob is None:
            return CrawlResponse(url=url, error="mock: no response configured")
        if cls is PageLevel.PAGE:
            return CrawlResponse(
                url=url,
                final_url=blob.get("final_url") or url,
                markdown=blob.get("markdown") or blob.get("raw_content"),
                metadata=blob.get("metadata", {}) or {},
                http_status=blob.get("http_status"),
                promoted_to_page_level=False,
            )
        # domain-only path — promote via topic match.
        links = blob.get("links") or []
        body_md = blob.get("markdown") or blob.get("raw_content") or ""
        promoted = _best_child_url(url, links, body_md, prompt_hint)
        if not promoted:
            return CrawlResponse(
                url=url, error="no topic-relevant child page (mock)",
                metadata={"candidate_count": len(links)},
                promoted_to_page_level=False,
            )
        sub = (
            self._responses.get(promoted)
            or self._responses.get(canonicalize_url(promoted))
        )
        if sub is None:
            return CrawlResponse(
                url=url, error="no mock for promoted child",
                promoted_to_page_level=False,
            )
        return CrawlResponse(
            url=url,
            final_url=promoted,
            markdown=sub.get("markdown") or sub.get("raw_content"),
            metadata={"promoted_from": url},
            http_status=sub.get("http_status"),
            promoted_to_page_level=True,
        )


def _any_host_match(responses: dict, url: str) -> Optional[str]:
    """Match response keys against canonical-url if exact miss."""
    canon = canonicalize_url(url)
    for k in responses:
        try:
            if canonicalize_url(k) == canon:
                return k
        except Exception:
            continue
    return None


# =============================================================================
# Convenience: integrate Crawl4AIProvider into SourcesDAO + Phase 3b resolver
# ---------------------------------------------------------------------------

async def crawl_and_register(
    url: str,
    *,
    crawler: CrawlProvider,
    sources_dao: SourcesDAO,
    run_id: Optional[str] = None,
    research_topic: Optional[str] = None,
    prompt_hint: Optional[str] = None,
) -> Optional[int]:
    """Fetch the URL and write the result back into `evidence.sources`.

    Returns the row id if a row was inserted/updated, None on failure.
    """
    resp = await crawler.fetch(url, prompt_hint=prompt_hint)
    target_url = resp.final_url if resp.final_url and resp.final_url != resp.url else url
    rec = SourceRecord.from_raw(
        url=target_url,
        provider=crawler.name,
        provider_query=prompt_hint,
        provider_payload=resp.metadata,
        run_id=run_id,
        research_topic=research_topic,
    )
    if resp.markdown:
        rec.raw_content = resp.markdown[:100_000]   # cap at 100 KB
        import hashlib
        rec.raw_content_hash = hashlib.sha256(resp.markdown.encode("utf-8", errors="ignore")).hexdigest()
    rec.fetch_status = "fetched" if resp.error is None else "failed"
    rec.http_status = resp.http_status
    return sources_dao.upsert(rec)


# =============================================================================
# Resolver adapter for Phase 3b
# ---------------------------------------------------------------------------

class CrawlResolver:
    """Adapter so that `enforce_page_level(resolver=...)` can use a CrawlProvider.

    Caches results in-process so repeated audit passes don't re-fetch.
    """

    def __init__(self, crawler: CrawlProvider) -> None:
        self._crawler = crawler
        self._cache: dict[str, str] = {}

    async def __call__(self, url: str) -> str:
        # Synchronous-looking callable — wraps the async crawler.
        if url in self._cache:
            return self._cache[url]
        resp = await self._crawler.fetch(url)
        result = resp.final_url if resp.final_url and resp.final_url != resp.url else ""
        self._cache[url] = result
        return result

    def call_sync(self, url: str, *, hint: Optional[str] = None) -> str:
        """Sync entrypoint used by enforce_page_level (which is sync)."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # Use a fresh loop if needed
        return loop.run_until_complete(self.__call__(url))
