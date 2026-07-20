"""Unit tests for sources_dao using an in-memory SQLite database.

Why SQLite here:
- SourcesDAO is the most-referenced module of Phase 1. It must be testable
  without a Postgres dependency (the sandbox can't reach odr-postgres).
- We mock psycopg's `cursor.execute` behavior with sqlite3 + a SQL adapter,
  or — more pragmatically — use sqlite3 directly and reproduce the schema.

Strategy:
- Implement a parallel SourcesDAOSQLite that implements the same interface
  but uses sqlite3. This lets us prove the SQL semantics and the Python
  layer separately. The Postgres version is exercised end-to-end via
  `tests/test_phase1_accept.py` once Docker is reachable.
"""
import os
import re
import sys
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.sources_dao import (  # noqa: E402
    SourcesDAO, SourceRecord, canonicalize_url, url_hash,
    classify_page_level, host_of, PageLevel,
)


# ---------------------------------------------------------------------------
# Constants used by both schema and DAO test adapter
# ---------------------------------------------------------------------------

SQLITE_SCHEMA = """
CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    url_normalized TEXT NOT NULL,
    url_hash TEXT NOT NULL UNIQUE,
    domain TEXT NOT NULL,
    title TEXT,
    provider TEXT NOT NULL,
    provider_query TEXT,
    provider_score REAL,
    page_level INTEGER NOT NULL DEFAULT 0,
    page_level_reason TEXT,
    fetch_status TEXT NOT NULL DEFAULT 'fetched',
    http_status INTEGER,
    content_type TEXT,
    provider_payload TEXT NOT NULL DEFAULT '{}',
    raw_content TEXT,
    raw_content_hash TEXT,
    fetched_at TEXT NOT NULL,
    expires_at TEXT,
    research_topic TEXT,
    run_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_sources_url_hash ON sources(url_hash);
CREATE INDEX idx_sources_page_level ON sources(page_level);
"""

# SQLite-friendly mirror of the PostgreSQL upsert in DAO.upsert().
# Parameter style: dict (named), matching psycopg%(name)s ↔ sqlite3 :name.
SQLITE_UPSERT = """
INSERT INTO sources (
    url, url_normalized, url_hash, domain, title,
    provider, provider_query, provider_score,
    page_level, page_level_reason, fetch_status,
    http_status, content_type, provider_payload,
    raw_content, raw_content_hash, fetched_at, expires_at,
    research_topic, run_id
) VALUES (
    :url, :url_normalized, :url_hash, :domain, :title,
    :provider, :provider_query, :provider_score,
    :page_level, :page_level_reason, :fetch_status,
    :http_status, :content_type, :provider_payload,
    :raw_content, :raw_content_hash, :fetched_at, :expires_at,
    :research_topic, :run_id
)
ON CONFLICT(url_hash) DO UPDATE SET
    title = COALESCE(excluded.title, sources.title),
    provider_payload = excluded.provider_payload,
    fetch_status = excluded.fetch_status,
    http_status = excluded.http_status,
    content_type = excluded.content_type,
    raw_content = COALESCE(excluded.raw_content, sources.raw_content),
    raw_content_hash = COALESCE(excluded.raw_content_hash, sources.raw_content_hash),
    fetched_at = excluded.fetched_at,
    expires_at = excluded.expires_at,
    research_topic = COALESCE(excluded.research_topic, sources.research_topic),
    run_id = COALESCE(excluded.run_id, sources.run_id)
RETURNING id;
"""


def _adapt_sql(sql: str) -> str:
    """Translate a (subset of) psycopg SQL into SQLite-friendly form.

    Handles two cases:
      1. The DAO upsert, identified by header `INSERT INTO evidence.sources`.
         → returns the SQLite-equivalent statement with `:name` params.
      2. Read queries that reference `evidence.sources` and `%s` params.
         → rewrites to `sources` + `?` positional params.
    """
    head = re.sub(r"\s+", " ", sql[:60]).strip().upper()
    if head.startswith("INSERT INTO") and "EVIDENCE.SOURCES" in head:
        return SQLITE_UPSERT
    out = sql.replace("evidence.sources", "sources")
    # Convert psycopg `%(name)s` → sqlite3 `:name`.
    out = re.sub(r"%\(([a-zA-Z_][a-zA-Z0-9_]*)\)s", r":\1", out)
    # Convert psycopg positional `%s` → `?` while leaving LIKE '...%...' patterns alone.
    out = re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])", "?", out)
    return out


def _materialize_params(sql: str, params):
    """Switch dict params to positional list for queries that need `?` style.

    After `_adapt_sql`, queries that use `?` need params as a tuple; queries
    that use `:name` need params as a dict. Both stay valid here.
    """
    return params


# ---------------------------------------------------------------------------
# sqlite3-side adapters implementing a subset of psycopg's Connection API
# ---------------------------------------------------------------------------

class _Col:
    def __init__(self, name):
        self.name = name


class _Cur:
    """Cursor wrapper that translates psycopg SQL → sqlite SQL on the fly."""
    def __init__(self, raw):
        self._raw = raw

    def execute(self, sql, params=None):
        sql = _adapt_sql(sql)
        if params is None:
            return self._raw.execute(sql)
        if isinstance(params, dict):
            # If we still have :name placeholders, use sqlite3 named-binding.
            # Note: the `:name` syntax supports dict parameters in sqlite3 ≥ 3.7.
            if ":" in sql and "?" not in sql:
                # provider_payload is a dict → serialize to JSON for sqlite.
                if isinstance(params.get("provider_payload"), dict):
                    params = {**params, "provider_payload": json.dumps(params["provider_payload"])}
                return self._raw.execute(sql, params)
            return self._raw.execute(sql, _dict_to_positional(sql, params))
        return self._raw.execute(sql, params)

    @property
    def description(self):
        if self._raw.description is None:
            return None
        return [_Col(d[0]) for d in self._raw.description]

    def fetchone(self):
        return self._raw.fetchone()

    def fetchall(self):
        return self._raw.fetchall()


def _dict_to_positional(sql: str, params: dict):
    """Extract `:name` placeholders in declaration order, return tuple of values."""
    names = re.findall(r":([a-zA-Z_][a-zA-Z0-9_]*)", sql)
    # de-dup while preserving order
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return tuple(params.get(n) for n in ordered)


class _SQLiteConnection:
    """Subset of psycopg Connection — cursor/commit/rollback/close."""
    def __init__(self):
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(SQLITE_SCHEMA)

    def cursor(self):
        return _Cur(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


class _DAOTest(SourcesDAO):
    """SourcesDAO wired to SQLite — focuses tests on Python layer correctness."""


# ---------------------------------------------------------------------------
# canonicalization tests
# ---------------------------------------------------------------------------

def test_canonicalize_url_strips_tracking():
    samples = [
        ("https://Klue.com/?utm_source=x&page=1", "https://klue.com?page=1"),
        ("HTTPS://CRAYON.CO/path/", "https://crayon.co/path"),
        ("https://example.com/a/b?ref=x#frag", "https://example.com/a/b"),
    ]
    for raw, expect in samples:
        got = canonicalize_url(raw)
        assert got == expect, f"{raw!r} → {got!r}, expected {expect!r}"
    print("  ✓ canonicalize strips tracking + lowercase + trailing /")


def test_classify_page_level():
    assert classify_page_level("https://klue.com") is PageLevel.DOMAIN_ONLY
    assert classify_page_level("https://klue.com/") is PageLevel.DOMAIN_ONLY
    assert classify_page_level("https://klue.com/?ref=1") is PageLevel.DOMAIN_ONLY
    assert classify_page_level("https://klue.com/product/battlecards") is PageLevel.PAGE
    assert classify_page_level("https://klue.com/product/battlecards/") is PageLevel.PAGE
    assert classify_page_level("https://example.com/file.html") is PageLevel.PAGE
    assert classify_page_level("not a url") is PageLevel.UNKNOWN
    print("  ✓ page_level classifier: domain-only vs page split correctly")


def test_url_hash_deterministic_and_64():
    h = url_hash("https://klue.com/path")
    assert len(h) == 64, f"hash length = {len(h)}"
    assert all(c in "0123456789abcdef" for c in h)
    assert url_hash("https://klue.com/PATH") == url_hash("https://klue.com/PATH")
    print("  ✓ url_hash: 64-char lowercase hex, deterministic")


def test_host_of():
    assert host_of("https://www.Klue.com/path") == "www.klue.com"
    assert host_of("https://KLUE.com") == "klue.com"
    assert host_of("garbage") == ""
    print("  ✓ host_of: lowercased host extraction")


# ---------------------------------------------------------------------------
# DAO round-trip tests (sqlite-parity)
# ---------------------------------------------------------------------------

def test_dao_upsert_idempotent():
    conn = _SQLiteConnection()
    dao = _DAOTest(conn=conn)
    r = SourceRecord.from_raw(
        url="https://klue.com/a?ref=x",
        title="Klue vs Crayon",
        provider="tavily",
        provider_query="Klue Crayon",
        provider_score=0.81,
        provider_payload={"raw": {"title": "Klue vs Crayon"}},
        run_id="run-1",
        research_topic="competitors",
    )
    id1 = dao.upsert(r)
    id2 = dao.upsert(r)
    assert id1 == id2, "url_hash UNIQUE → upsert must update not insert"
    out = dao.get_by_url("https://klue.com/a?ref=x")
    assert out is not None
    assert out.title == "Klue vs Crayon"
    assert out.provider_score == 0.81
    assert out.page_level is True
    assert out.run_id == "run-1"
    print(f"  ✓ upsert idempotent on url_hash (id={id1})")


def test_dao_distinguishes_page_level_vs_domain_only():
    conn = _SQLiteConnection()
    dao = _DAOTest(conn=conn)

    for u in [
        "https://crayon.co/crayon-vs-klue",
        "https://klue.com/product/battlecards",
        "https://www.crayon.co",                # domain-only
        "https://klue.com/",                    # domain-only
    ]:
        dao.upsert(SourceRecord.from_raw(url=u, provider="tavily"))

    domain_only = dao.list_domain_only(limit=10)
    flagged = {r.url for r in domain_only}
    assert "https://www.crayon.co" in flagged
    assert "https://klue.com/" in flagged
    assert "https://crayon.co/crayon-vs-klue" not in flagged
    assert "https://klue.com/product/battlecards" not in flagged
    print(f"  ✓ B-classifier: {len(domain_only)} domain-only URLs flagged")


def test_dao_canonicalizes_dedup_inputs():
    conn = _SQLiteConnection()
    dao = _DAOTest(conn=conn)
    r1 = SourceRecord.from_raw(url="https://klue.com/a", provider="tavily", title="A")
    r2 = SourceRecord.from_raw(
        url="HTTPS://KLUE.COM/a?utm_source=x",
        provider="tavily",
        title="A",
    )
    id1 = dao.upsert(r1)
    id2 = dao.upsert(r2)
    assert id1 == id2, "Case + UTM should still dedup via canonicalization"
    print(f"  ✓ canonical dedup across casing + tracking params")


def test_dao_stats():
    conn = _SQLiteConnection()
    dao = _DAOTest(conn=conn)
    for u, p in [
        ("https://klue.com/a", "tavily"),
        ("https://klue.com/b", "tavily"),
        ("https://crayon.co/x", "searxng"),
        ("https://crayon.co", "searxng"),
    ]:
        dao.upsert(SourceRecord.from_raw(url=u, provider=p))
    s = dao.stats()
    assert s["total"] == 4, s
    assert s["by_provider"]["tavily"] == 2
    assert s["by_provider"]["searxng"] == 2
    assert s["domain_only"] == 1
    assert s["page_level"] == 3
    print(f"  ✓ stats: {s}")


def test_dao_module_imports():
    import open_deep_research.sources_dao as m
    assert hasattr(m, "SourcesDAO")
    assert hasattr(m, "SourceRecord")
    assert hasattr(m, "canonicalize_url")
    assert hasattr(m, "PageLevel")
    print("  ✓ module imports clean + exports stable surface")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("canonicalize_url_strips_tracking", test_canonicalize_url_strips_tracking),
        ("classify_page_level", test_classify_page_level),
        ("url_hash_deterministic_and_64", test_url_hash_deterministic_and_64),
        ("host_of", test_host_of),
        ("dao_upsert_idempotent", test_dao_upsert_idempotent),
        ("dao_distinguishes_page_level_vs_domain_only",
         test_dao_distinguishes_page_level_vs_domain_only),
        ("dao_canonicalizes_dedup_inputs", test_dao_canonicalizes_dedup_inputs),
        ("dao_stats", test_dao_stats),
        ("dao_module_imports", test_dao_module_imports),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} sources_dao tests")
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
            print(f"  ✗ ERROR: {e!r}")
            failed.append(name)
    print("\n" + "=" * 70)
    if failed:
        print(f" {len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f" ALL {len(tests)} TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
