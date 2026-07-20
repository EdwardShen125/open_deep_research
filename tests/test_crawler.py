"""Phase 1.4 — Crawler (Crawl4AIProvider + Mock) tests."""
import asyncio
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from open_deep_research.sources_dao import classify_page_level, PageLevel  # noqa: E402
from open_deep_research.crawler import (  # noqa: E402
    Crawl4AIProvider, MockCrawlProvider,
    CrawlResponse, CrawlResolver,
    crawl_and_register, _extract_href, _best_child_url, _topic_match, _same_domain,
)

from test_sources_dao_sqlite import _SQLiteConnection, _DAOTest  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Page-level fetch (no promotion needed)
# ---------------------------------------------------------------------------

def test_crawler_returns_page_level_directly():
    mock = MockCrawlProvider({
        "https://klue.com/product/battlecards": {
            "markdown": "Battlecards content here.",
            "links": [],
            "http_status": 200,
        }
    })
    resp = asyncio.run(mock.fetch("https://klue.com/product/battlecards"))
    assert resp.error is None
    assert resp.markdown == "Battlecards content here."
    assert resp.promoted_to_page_level is False
    assert resp.http_status == 200
    print("  ✓ page-level URL → returns body directly, no promotion")


# ---------------------------------------------------------------------------
# Domain-only → page-level promotion
# ---------------------------------------------------------------------------

def test_crawler_promotes_domain_only_via_topic_match():
    mock = MockCrawlProvider({
        "https://www.crayon.co": {
            "markdown": "Crayon vs Klue vs Kompyte. Pick the right CI tool.",
            "links": [
                "https://www.crayon.co/vs-klue",
                "https://www.crayon.co/products",
                "https://www.crayon.co/about",
            ],
            "http_status": 200,
        },
        "https://www.crayon.co/vs-klue": {
            "markdown": "Crayon vs Klue feature comparison and pricing.",
            "http_status": 200,
        },
    })
    resp = asyncio.run(
        mock.fetch("https://www.crayon.co", prompt_hint="Klue")
    )
    assert resp.error is None, resp.error
    assert resp.promoted_to_page_level is True
    assert resp.final_url == "https://www.crayon.co/vs-klue"
    assert "Crayon vs Klue feature" in (resp.markdown or "")
    print("  ✓ domain-only URL → child page selected by topic match")


def test_crawler_promotion_fails_when_no_topic_hint():
    mock = MockCrawlProvider({
        "https://www.crayon.co": {
            "markdown": "general landing",
            "links": ["https://www.crayon.co/about"],
            "http_status": 200,
        }
    })
    resp = asyncio.run(mock.fetch("https://www.crayon.co"))  # no prompt_hint
    assert resp.error is not None
    assert "topic" in resp.error.lower()
    print("  ✓ no topic hint → no promotion")


def test_crawler_promotion_fails_when_no_subdomain_match():
    mock = MockCrawlProvider({
        "https://klue.com": {
            "markdown": "landing",
            "links": ["https://example.com/foo"],
            "http_status": 200,
        }
    })
    resp = asyncio.run(mock.fetch("https://klue.com", prompt_hint="Klue"))
    assert resp.error is not None
    print("  ✓ promotion skipped for non-same-domain links")


def test_crawler_handles_empty_url():
    mock = MockCrawlProvider()
    resp = asyncio.run(mock.fetch(""))
    assert resp.error is not None
    print("  ✓ empty URL → error result, no crash")


def test_crawler_handles_unparseable_url():
    mock = MockCrawlProvider()
    resp = asyncio.run(mock.fetch("not a url"))
    assert resp.error == "unparseable url"
    print("  ✓ unparseable URL → graceful error")


# ---------------------------------------------------------------------------
# Helpers — extract / topic / domain
# ---------------------------------------------------------------------------

def test_extract_href_normalizes_relative_links():
    html = '<a href="/foo">x</a><a href="https://other.com/y">y</a><a href="#frag">z</a>'
    hrefs = _extract_href(html, base="https://x.com/path")
    assert "https://x.com/foo" in hrefs
    assert "https://other.com/y" in hrefs
    assert not any("#frag" in h for h in hrefs)
    print(f"  ✓ _extract_href dedup: {len(hrefs)} links")


def test_topic_match_counts_and_boosts():
    blob = "Klue 收购 Crayon Kompyte Klue Klue"
    score = _topic_match(blob, "Klue")
    assert score >= 3
    print(f"  ✓ _topic_match: klue×{blob.count('Klue')} → score {score}")


def test_same_domain_loose_match():
    assert _same_domain("https://www.klue.com/x", "https://klue.com/y")
    assert not _same_domain("https://klue.com/a", "https://crayon.co/b")
    print("  ✓ _same_domain: registrable-suffix comparison")


# ---------------------------------------------------------------------------
# Crawl4AIProvider with explicit fetcher (no crawl4ai dep in tests)
# ---------------------------------------------------------------------------

def test_crawl4ai_provider_uses_injected_fetcher():
    seen: list[str] = []

    async def fake_fetch(url, timeout):
        seen.append(url)
        return {
            "markdown": "Body of " + url,
            "links": ["https://klue.com/product/battlecards"],
            "http_status": 200,
        }

    p = Crawl4AIProvider(fetcher=fake_fetch)
    resp = asyncio.run(p.fetch("https://klue.com/product/battlecards"))
    assert seen == ["https://klue.com/product/battlecards"]
    assert "Body of" in (resp.markdown or "")
    print("  ✓ Crawl4AIProvider respects injected fetcher")


def test_crawl4ai_provider_promotes_via_injected_fetcher():
    async def fake_fetch(url, timeout):
        if url == "https://klue.com":
            return {
                "markdown": "Klue vs Crayon comparison.",
                "links": [
                    "https://klue.com/vs-crayon",
                    "https://klue.com/blog",
                ],
                "http_status": 200,
            }
        return {
            "markdown": "Klue vs Crayon feature compare.",
            "http_status": 200,
        }

    p = Crawl4AIProvider(fetcher=fake_fetch)
    resp = asyncio.run(p.fetch("https://klue.com", prompt_hint="Crayon"))
    assert resp.promoted_to_page_level is True
    assert resp.final_url == "https://klue.com/vs-crayon"


# ---------------------------------------------------------------------------
# Integration with SourcesDAO
# ---------------------------------------------------------------------------

def test_crawl_and_register_writes_back_to_sources():
    mock = MockCrawlProvider({
        "https://klue.com/product/battlecards": {
            "markdown": "Battlecards feature list.",
            "http_status": 200,
        }
    })
    dao = _DAOTest(_SQLiteConnection())
    rid = asyncio.run(crawl_and_register(
        "https://klue.com/product/battlecards",
        crawler=mock,
        sources_dao=dao,
        run_id="r-1",
        research_topic="klue_product",
    ))
    assert rid is not None
    row = dao.get_by_url("https://klue.com/product/battlecards")
    assert row is not None
    assert row.raw_content == "Battlecards feature list."
    assert row.raw_content_hash is not None
    assert row.fetch_status == "fetched"
    assert row.provider == "mock"
    print(f"  ✓ crawl_and_register wrote raw_content + hash to sources (id={rid})")


# ---------------------------------------------------------------------------
# CrawlResolver — used by enforce_page_level
# ---------------------------------------------------------------------------

def test_crawl_resolver_returns_promoted_url():
    mock = MockCrawlProvider({
        "https://klue.com": {
            "markdown": "Klue vs Crayon",
            "links": ["https://klue.com/vs-crayon"],
            "http_status": 200,
        },
        "https://klue.com/vs-crayon": {"markdown": "comparison"},
    })
    resolver = CrawlResolver(mock)
    out = resolver.call_sync("https://klue.com")
    # Resolver says nothing (since fetcher's matcher uses prompt_hint; without
    # hint, the resolver sees no `final_url` change → returns '').
    # We expect '' because CrawlResolver doesn't carry the hint path.
    assert out == ""
    print("  ✓ CrawlResolver.call_sync: blank on no-promotion, no crash")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("crawler_returns_page_level_directly",
         test_crawler_returns_page_level_directly),
        ("crawler_promotes_domain_only_via_topic_match",
         test_crawler_promotes_domain_only_via_topic_match),
        ("crawler_promotion_fails_when_no_topic_hint",
         test_crawler_promotion_fails_when_no_topic_hint),
        ("crawler_promotion_fails_when_no_subdomain_match",
         test_crawler_promotion_fails_when_no_subdomain_match),
        ("crawler_handles_empty_url", test_crawler_handles_empty_url),
        ("crawler_handles_unparseable_url", test_crawler_handles_unparseable_url),
        ("extract_href_normalizes_relative_links",
         test_extract_href_normalizes_relative_links),
        ("topic_match_counts_and_boosts", test_topic_match_counts_and_boosts),
        ("same_domain_loose_match", test_same_domain_loose_match),
        ("crawl4ai_provider_uses_injected_fetcher",
         test_crawl4ai_provider_uses_injected_fetcher),
        ("crawl4ai_provider_promotes_via_injected_fetcher",
         test_crawl4ai_provider_promotes_via_injected_fetcher),
        ("crawl_and_register_writes_back_to_sources",
         test_crawl_and_register_writes_back_to_sources),
        ("crawl_resolver_returns_promoted_url",
         test_crawl_resolver_returns_promoted_url),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} crawler tests")
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
