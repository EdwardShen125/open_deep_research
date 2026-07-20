"""Phase 1.2: Search results TTL cache (L1 in-process + L2 Postgres `sources`).

Design
------
Two-tier cache keyed by canonical query string (and topic, optionally).

- L1: an in-process LRU keyed by `(query, topic)`. Cheap, deterministic. Used
  inside a single run to short-circuit when the same query is repeated
  (e.g. parallel researchers ask the same domain).
- L2: PG-backed, per-URL `expires_at`. Used across runs and processes.
  Hit semantics: if `expires_at > now()`, the cached `raw_content`/`title`/
  `provider_payload` are still valid; we still re-issue the search API call
  if `query` is new — but the *page-level validation* and *crawl* layer can
  reuse stored content via `get_by_url`.

API surface
-----------
    from open_deep_research.search_cache import SearchCache
    cache = SearchCache(ttl_seconds=3600)        # 1 hour default
    payload = cache.get("Klue Crayon", topic="ci")
    if payload is None:
        payload = search_api.search("Klue Crayon")
        cache.put("Klue Crayon", payload, topic="ci", urls=[...])

When `SourcesDAO` is None, only L1 is used; cache is process-local and
disappears on restart.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


# =============================================================================
# Cache key derivation
# =============================================================================

def query_key(query: str, topic: str = "general") -> str:
    """Stable cache key for a (query, topic) pair.

    We lowercase + collapse whitespace so trivial rephrasings dedup.
    """
    qn = " ".join((query or "").lower().split())
    tn = (topic or "general").lower()
    raw = f"q={qn}|t={tn}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# L1 entry record
# =============================================================================

@dataclass
class _L1Entry:
    payload: dict[str, Any]
    expire_at: float  # monotonic — `time.monotonic()` seconds
    inserted_at: datetime  # for stats


# =============================================================================
# SearchCache
# =============================================================================

class SearchCache:
    """Two-tier TTL cache for search results."""

    DEFAULT_TTL = 3600          # 1 hour
    DEFAULT_L1_MAX_ENTRIES = 64  # LRU capacity per process

    def __init__(
        self,
        sources_dao: Any = None,
        *,
        ttl_seconds: int = DEFAULT_TTL,
        l1_max_entries: int = DEFAULT_L1_MAX_ENTRIES,
        clock: Any = None,
    ) -> None:
        """`sources_dao` is an `optional SourcesDAO` instance for L2.

        `clock` is a callable returning monotonic seconds (default `time.monotonic`).
        Injectable for tests.
        """
        self._dao = sources_dao
        self._ttl = ttl_seconds
        self._l1_max = max(1, l1_max_entries)
        self._clock = clock or time.monotonic
        self._l1: "OrderedDict[str, _L1Entry]" = OrderedDict()
        # Stats
        self.l1_hits = 0
        self.l1_misses = 0
        self.l1_invalidations = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.puts = 0

    # ---------- L1 ----------
    def _l1_get(self, key: str) -> Optional[dict[str, Any]]:
        e = self._l1.get(key)
        if e is None:
            self.l1_misses += 1
            return None
        if e.expire_at <= self._clock():
            # Expired — remove and miss.
            del self._l1[key]
            self.l1_invalidations += 1
            self.l1_misses += 1
            return None
        # LRU touch: move to end.
        self._l1.move_to_end(key)
        self.l1_hits += 1
        return e.payload

    def _l1_put(self, key: str, payload: dict[str, Any]) -> None:
        # If over capacity, drop oldest.
        while len(self._l1) >= self._l1_max:
            self._l1.popitem(last=False)
        self._l1[key] = _L1Entry(
            payload=payload,
            expire_at=self._clock() + self._ttl,
            inserted_at=_now(),
        )

    # ---------- L2 ----------
    def _l2_get(self, key: str) -> Optional[dict[str, Any]]:
        if self._dao is None:
            return None
        # We treat L2 as keyed on `provider_query` (the raw query) so we can
        # recover it via a focused read. We persist a derived key in the
        # `run_id` slot to keep the dependency surface small: this is a
        # search-result cache, not a column-level extension.
        # (Phase 1.2 trades off: in-PG key-tracking is added in 1.2.1.)
        try:
            # Reuse SourcesDAO.get_by_url via the first URL in payload? No —
            # search results aren't 1:1 with a URL. Instead we persist the
            # query key into a small `key` field stored in `provider_payload`
            # for the row we created at put-time. Look it up below.
            from open_deep_research.sources_dao import canonicalize_url  # local import
        except Exception:
            return None
        return None  # L2 lookup is implemented via a dedicated method in v1.2.1

    # ---------- public ----------
    def get(self, query: str, topic: str = "general") -> Optional[dict[str, Any]]:
        """Try L1 → L2. Returns a dict payload or None on miss."""
        key = query_key(query, topic)
        v = self._l1_get(key)
        if v is not None:
            return v
        v = self._l2_get(key)
        if v is not None:
            # Re-warm L1 so subsequent reads stay in-process.
            self._l1_put(key, v)
            self.l2_hits += 1
            return v
        self.l2_misses += 1
        return None

    def put(
        self,
        query: str,
        payload: dict[str, Any],
        *,
        topic: str = "general",
        urls: Optional[list] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Persist `payload` in L1 (always) and L2 (when dao is set + urls given).

        `urls` is the list of returned URLs (each will be upserted with the
        matching `expires_at` if a DAO is provided).
        """
        key = query_key(query, topic)
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl
        if ttl != self._ttl:
            # Per-entry TTL override
            payload = {**payload, "_ttl_override_s": ttl}
        self._l1_put(key, payload)
        self.puts += 1

        # L2 = per-URL expiry. We don't store the whole payload keyed on
        # query in PG (that's a future migration) — what we *do* store is
        # the URL→expires_at TTL, so a later Crawl4AI / re-fetch can
        # decide whether to skip the network call.
        if self._dao is not None and urls:
            from open_deep_research.sources_dao import SourceRecord
            expire_at = _now() + timedelta(seconds=ttl)
            for u in urls:
                if not isinstance(u, dict):
                    continue
                url = u.get("url")
                if not url:
                    continue
                rec = SourceRecord.from_raw(
                    url=url,
                    title=u.get("title"),
                    provider=u.get("provider", "cache"),
                    provider_query=query,
                    provider_score=u.get("score"),
                    provider_payload=u.get("payload", {}),
                )
                rec.expires_at = expire_at
                self._dao.upsert(rec)

    def invalidate(self, query: str, topic: str = "general") -> bool:
        """Drop an L1 entry. Returns True if anything was removed."""
        key = query_key(query, topic)
        return self._l1.pop(key, None) is not None

    def clear_l1(self) -> int:
        n = len(self._l1)
        self._l1.clear()
        return n

    def stats(self) -> dict[str, Any]:
        return {
            "l1_size": len(self._l1),
            "l1_max": self._l1_max,
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l1_invalidations": self.l1_invalidations,
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "puts": self.puts,
            "ttl_seconds": self._ttl,
        }


# =============================================================================
# TTL helpers for content
# =============================================================================

def compute_expires_at(ttl_seconds: int, now: Optional[datetime] = None) -> datetime:
    """Helper used by callers that want to pass `expires_at` to SourcesDAO directly."""
    return (now or _now()) + timedelta(seconds=ttl_seconds)


def is_fresh(
    expires_at: Optional[datetime],
    now: Optional[datetime] = None,
    *,
    grace_seconds: int = 0,
) -> bool:
    """True iff `expires_at` (UTC) is strictly in the future with optional grace."""
    if expires_at is None:
        return False
    cur = now or _now()
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return (expires_at - timedelta(seconds=grace_seconds)) > cur
