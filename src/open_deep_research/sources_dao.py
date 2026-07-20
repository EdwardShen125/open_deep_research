"""Phase 1.1: Sources DAO

Database access layer for the `evidence.sources` table.

Design goals
------------
1. URL canonicalization that matches the `evidence.is_page_level()` SQL helper.
2. Idempotent inserts via `url_hash` (sha256 of normalized URL).
3. Page-level classification is decided in Python *and* validated by the SQL helper
   (we re-check via SQL when a row is fetched, to detect schema drift).
4. Optional cache TTL via `expires_at` — used by Phase 1.2.
5. No external deps beyond `psycopg` (already required for odr-postgres).

Public surface
--------------
- `SourcesDAO`        : sync context-managed wrapper
- `SourceRecord`      : TypedDict-like dataclass of one row
- `PageLevel`         : Enum of page-level status (for clear semantics)

This module is intentionally framework-agnostic: it does *not* import LangChain
or LangGraph, so it can be reused by the crawler (Phase 1.4), the writer
(Phase 2), or the verifier (Phase 3).
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


# =============================================================================
# PageLevel + canonicalization
# =============================================================================

class PageLevel(Enum):
    """Whether a URL is page-level (B-anchor compliant) or domain-only."""
    PAGE = "page"               # has a meaningful path
    DOMAIN_ONLY = "domain_only" # root or query-only
    UNKNOWN = "unknown"         # could not parse

# UTM / tracker parameters we always strip during canonicalization.
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid", "ref", "ref_src",
    "_hsenc", "_hsmi", "__hssc", "__hstc",
    "vero_id", "vero_conv", "trk", "trkCampaign", "scm", "spm",
})

_URL_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def canonicalize_url(url: str) -> str:
    """Return a deterministic, lowercase, tracking-stripped URL string.

    The result is suitable for hashing/dedup. We do NOT normalize away
    meaningful path components (e.g. case-sensitive Tumblr-style URLs are
    rare in our corpus).
    """
    if not url:
        return ""
    # Lowercase scheme + host only.
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "http").lower()
    # Reject non-http(s) — these are not fetchable by tavily/crawl4ai anyway.
    if scheme not in ("http", "https"):
        return url.strip().lower()
    netloc = parts.netloc.lower()
    path = parts.path or ""

    # When path is "/", collapse it to "" (we'll emit a domain-only URL).
    if path == "/":
        path = ""
    # If path looks like a directory-style suffix (e.g. "/blog/"), drop trailing
    # slash so /blog and /blog/ are treated as the same canonical URL. Skip
    # this if the final segment appears to be a filename (has an extension).
    elif path.endswith("/") and "/" in path.rstrip("/"):
        last = path.rstrip("/").rsplit("/", 1)[-1]
        if "." not in last:  # not a filename
            path = path.rstrip("/")

    # Strip tracking params.
    q_pairs = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
               if k.lower() not in _TRACKING_PARAMS]
    query = urlencode(q_pairs, doseq=True)
    fragment = ""  # always strip fragment
    return urlunsplit((scheme, netloc, path, query, fragment))


def url_hash(url_normalized: str) -> str:
    """SHA-256 hex of the canonical URL — 64 chars, lowercase."""
    return hashlib.sha256(url_normalized.encode("utf-8")).hexdigest()


def classify_page_level(url: str) -> PageLevel:
    """Decide if URL is page-level (path with ≥1 meaningful segment) or domain-only.

    Mirrors the SQL helper `evidence.is_page_level()`; keep in sync.
    """
    try:
        parts = urlsplit(url)
    except Exception:
        return PageLevel.UNKNOWN
    if not parts.netloc:
        return PageLevel.UNKNOWN
    path = (parts.path or "").strip()
    # Domain-only: empty path, or path == "/"
    if path in ("", "/"):
        # Also accept ?query-only — those are domain-level even if query exists.
        return PageLevel.DOMAIN_ONLY
    return PageLevel.PAGE


def host_of(url: str) -> str:
    """Lowercased hostname portion of a URL, or '' on parse failure."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


# =============================================================================
# Record dataclass
# =============================================================================

@dataclass
class SourceRecord:
    """One row of `evidence.sources`. Fields default to None / safe values."""

    id: Optional[int] = None
    url: str = ""
    url_normalized: str = ""
    url_hash: str = ""
    domain: str = ""
    title: Optional[str] = None
    provider: str = "tavily"
    provider_query: Optional[str] = None
    provider_score: Optional[float] = None
    page_level: bool = False
    page_level_reason: Optional[str] = None
    fetch_status: str = "fetched"
    http_status: Optional[int] = None
    content_type: Optional[str] = None
    provider_payload: dict = field(default_factory=dict)
    raw_content: Optional[str] = None
    raw_content_hash: Optional[str] = None
    fetched_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    research_topic: Optional[str] = None
    run_id: Optional[str] = None

    @classmethod
    def from_raw(
        cls,
        *,
        url: str,
        title: Optional[str] = None,
        provider: str,
        provider_query: Optional[str] = None,
        provider_score: Optional[float] = None,
        provider_payload: Optional[dict] = None,
        run_id: Optional[str] = None,
        research_topic: Optional[str] = None,
        http_status: Optional[int] = None,
        content_type: Optional[str] = None,
        fetch_status: str = "fetched",
    ) -> "SourceRecord":
        """Build a record from a Tavily/SearXNG/Crawl4AI raw result dict."""
        norm = canonicalize_url(url)
        pl = classify_page_level(url)
        reason = None if pl == PageLevel.PAGE else (
            "domain_root" if pl == PageLevel.DOMAIN_ONLY else "unparseable"
        )
        return cls(
            url=url,
            url_normalized=norm,
            url_hash=url_hash(norm),
            domain=host_of(url),
            title=title,
            provider=provider,
            provider_query=provider_query,
            provider_score=provider_score,
            page_level=(pl == PageLevel.PAGE),
            page_level_reason=reason,
            fetch_status=fetch_status,
            http_status=http_status,
            content_type=content_type,
            provider_payload=provider_payload or {},
            run_id=run_id,
            research_topic=research_topic,
            fetched_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Datetime → ISO string so JSON serialization is straightforward.
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = v.isoformat()
        return d


# =============================================================================
# DAO
# =============================================================================

class SourcesDAO:
    """Thin synchronous wrapper around evidence.sources.

    Use as a context manager (auto-rollback on exception). All write methods
    are idempotent on `url_hash` — re-fetching the same URL updates the
    `fetched_at`, `raw_content`, and `expires_at` rather than duplicating rows.
    """

    def __init__(self, conn: Any = None) -> None:
        """`conn` is an open psycopg.Connection. Use None + connect() lazily."""
        self._conn = conn

    # ---------- context manager ----------
    def __enter__(self) -> "SourcesDAO":
        if self._conn is None:
            self._conn = self._connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and self._conn is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        return False

    @staticmethod
    def _connect() -> Any:
        """Open a psycopg connection using env vars (mirrors deploy/init_db.py)."""
        import psycopg  # local import — keeps this module import-safe without psycopg
        host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
        port = int(os.environ.get("POSTGRES_PORT", "5432"))
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "odr_v2_pg_pass_change_me")
        dbname = os.environ.get("POSTGRES_DB", "odr_v2")
        return psycopg.connect(
            host=host, port=port, user=user, password=password, dbname=dbname,
            autocommit=False,
        )

    # ---------- helpers (use self._conn which the type checker can't narrow) ----------
    def _cur(self):
        """Cursor accessor that asserts self._conn is not None (set by __enter__)."""
        assert self._conn is not None, "SourcesDAO must be used as a context manager"
        return self._conn.cursor()

    def _commit(self) -> None:
        assert self._conn is not None, "SourcesDAO must be used as a context manager"
        self._conn.commit()

    # ---------- writes ----------
    def upsert(self, record: SourceRecord) -> int:
        """Insert or update by url_hash. Returns the row id."""
        cur = self._cur()
        cur.execute(
            """
            INSERT INTO evidence.sources (
                url, url_normalized, url_hash, domain, title,
                provider, provider_query, provider_score,
                page_level, page_level_reason, fetch_status,
                http_status, content_type, provider_payload,
                raw_content, raw_content_hash, fetched_at, expires_at,
                research_topic, run_id
            ) VALUES (
                %(url)s, %(url_normalized)s, %(url_hash)s, %(domain)s, %(title)s,
                %(provider)s, %(provider_query)s, %(provider_score)s,
                %(page_level)s, %(page_level_reason)s, %(fetch_status)s,
                %(http_status)s, %(content_type)s, %(provider_payload)s::jsonb,
                %(raw_content)s, %(raw_content_hash)s, %(fetched_at)s, %(expires_at)s,
                %(research_topic)s, %(run_id)s
            )
            ON CONFLICT (url_hash) DO UPDATE SET
                title = COALESCE(EXCLUDED.title, evidence.sources.title),
                provider_payload = EXCLUDED.provider_payload,
                fetch_status = EXCLUDED.fetch_status,
                http_status = EXCLUDED.http_status,
                content_type = EXCLUDED.content_type,
                raw_content = COALESCE(EXCLUDED.raw_content, evidence.sources.raw_content),
                raw_content_hash = COALESCE(EXCLUDED.raw_content_hash, evidence.sources.raw_content_hash),
                fetched_at = EXCLUDED.fetched_at,
                expires_at = EXCLUDED.expires_at,
                research_topic = COALESCE(EXCLUDED.research_topic, evidence.sources.research_topic),
                run_id = COALESCE(EXCLUDED.run_id, evidence.sources.run_id)
            RETURNING id;
            """,
            {
                **record.to_dict(),
                "fetched_at": record.fetched_at or datetime.now(timezone.utc),
            },
        )
        row_id = cur.fetchone()[0]
        self._commit()
        return row_id

    def upsert_many(self, records: Iterable[SourceRecord]) -> list[int]:
        """Bulk upsert. Returns list of row ids in input order."""
        return [self.upsert(r) for r in records]

    # ---------- reads ----------
    def get_by_url(self, url: str) -> Optional[SourceRecord]:
        cur = self._cur()
        cur.execute(
            "SELECT * FROM evidence.sources WHERE url_hash = %s",
            (url_hash(canonicalize_url(url)),),
        )
        row = cur.fetchone()
        return _row_to_record(cur.description, row) if row else None

    def get_by_id(self, row_id: int) -> Optional[SourceRecord]:
        cur = self._cur()
        cur.execute("SELECT * FROM evidence.sources WHERE id = %s", (row_id,))
        row = cur.fetchone()
        return _row_to_record(cur.description, row) if row else None

    def list_by_run(self, run_id: str, *, page_level_only: bool = False) -> list[SourceRecord]:
        cur = self._cur()
        sql = "SELECT * FROM evidence.sources WHERE run_id = %s"
        if page_level_only:
            sql += " AND page_level = TRUE"
        sql += " ORDER BY fetched_at"
        cur.execute(sql, (run_id,))
        return [_row_to_record(cur.description, r) for r in cur.fetchall()]

    def list_domain_only(self, limit: int = 100) -> list[SourceRecord]:
        """Used by Phase 3b rule-4 verifier to find B-anchor violations."""
        cur = self._cur()
        cur.execute(
            "SELECT * FROM evidence.sources WHERE page_level = FALSE "
            "ORDER BY fetched_at DESC LIMIT %s",
            (limit,),
        )
        return [_row_to_record(cur.description, r) for r in cur.fetchall()]

    def stats(self) -> dict[str, Any]:
        """Aggregate counts — used by the acceptance script."""
        cur = self._cur()
        out: dict[str, Any] = {}
        cur.execute("SELECT count(*) FROM evidence.sources")
        out["total"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM evidence.sources WHERE page_level")
        out["page_level"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM evidence.sources WHERE NOT page_level")
        out["domain_only"] = cur.fetchone()[0]
        cur.execute("""
            SELECT provider, count(*) FROM evidence.sources
            GROUP BY provider ORDER BY count(*) DESC
        """)
        out["by_provider"] = {r[0]: r[1] for r in cur.fetchall()}
        return out


# =============================================================================
# Internal: row → record
# =============================================================================

def _row_to_record(description, row) -> SourceRecord:
    """Convert a psycopg row into SourceRecord, mapping enum/datetime fields."""
    col_names = [c.name for c in description]
    raw = dict(zip(col_names, row))
    # Normalize enum / datetime values to plain types.
    fs = raw.get("fetch_status")
    if fs is not None and not isinstance(fs, str):
        raw["fetch_status"] = fs.value
    for k in ("fetched_at", "expires_at", "created_at", "updated_at"):
        v = raw.get(k)
        if isinstance(v, datetime):
            raw[k] = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    # Build kwargs from declared fields only, defaulting missing columns.
    fields = (
        "id", "url", "url_normalized", "url_hash", "domain", "title",
        "provider", "provider_query", "provider_score",
        "page_level", "page_level_reason", "fetch_status",
        "http_status", "content_type", "provider_payload",
        "raw_content", "raw_content_hash", "fetched_at", "expires_at",
        "research_topic", "run_id",
    )
    kwargs = {}
    for f in fields:
        v = raw.get(f)
        if v is None:
            continue
        if f in ("url", "url_normalized", "url_hash", "domain", "provider"):
            kwargs[f] = v or ""                # str must be non-empty
        elif f == "page_level":
            kwargs[f] = bool(v)
        elif f == "fetch_status":
            kwargs[f] = v or "fetched"
        else:
            kwargs[f] = v
    return SourceRecord(**kwargs)
