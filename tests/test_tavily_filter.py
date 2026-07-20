"""Phase 2.5 — Tavily observation noise filter tests.

The runtime e2e run surfaced several classes of Tavily noise:
  - Off-topic domains (social media, entertainment, marketing aggregators)
  - Pages whose raw markdown is full of `![img](data:...)` tokens

These tests pin the filter behavior so the blacklist / patterns can't
accidentally regress (e.g. dropping real sources) and so the noise
classification logic is exercised in CI without needing a live Tavily call.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.deep_researcher import (  # noqa: E402
    _chunk_is_low_quality,
    _filter_tavily_chunks,
    _host_of,
    _is_noise_domain,
    _MIN_CHUNK_CONTENT_CHARS,
)


def test_host_of_extracts_clean_host() -> None:
    assert _host_of("https://techcrunch.com/2021/12/x") == "techcrunch.com"
    assert _host_of("http://WWW.BetaKit.com/") == "www.betakit.com"
    assert _host_of("https://m.facebook.com/x") == "m.facebook.com"
    assert _host_of("not a url") == ""
    assert _host_of("") == ""
    print("  ✓ host extraction")


def test_noise_domain_matches_known_blacklist() -> None:
    # Direct hits
    assert _is_noise_domain("https://facebook.com/post/123")
    assert _is_noise_domain("https://www.instagram.com/p/abc")
    assert _is_noise_domain("https://m.facebook.com/x")
    assert _is_noise_domain("https://reddit.com/r/vancouver")
    assert _is_noise_domain("https://www.linkedin.com/posts/klue_x")
    assert _is_noise_domain("https://worldofreel.com/blog/2025/x")
    assert _is_noise_domain("https://www.throughthesilverscreen.com/tag/weapons-2025")
    assert _is_noise_domain("https://topstartups.io?funding_round=Series+C")
    # Real research sources MUST survive
    assert not _is_noise_domain("https://techcrunch.com/2021/12/01/klue")
    assert not _is_noise_domain("https://www.betakit.com/klue-funding")
    assert not _is_noise_domain("https://klue.com/blog/klue-raises-62m")
    assert not _is_noise_domain("https://www.theglobeandmail.com/business/article-x")
    assert not _is_noise_domain("https://www.prnewswire.com/news-releases/x")
    print("  ✓ noise-domain classification")


def test_low_quality_chunk_detects_markdown_noise() -> None:
    """BusinessInsider-style page: lots of `![alt](url)` tokens."""
    noisy_chunk = (
        "![Klue](https://i.insider.com/5f5b89227ed0ee001e25ec3c?width) "
        + "![Klue](https://i.insider.com/5f5b37697ed0ee001e25eb59?width) " * 50
        + "<svg xmlns='http://www.w3.org/2000/svg'>"
        + "![Logo](/public/assets/logos/stacked-black.svg) "
    )
    assert len(noisy_chunk) > _MIN_CHUNK_CONTENT_CHARS
    assert _chunk_is_low_quality(noisy_chunk), "should be flagged low quality"

    # A real prose chunk should survive
    real_chunk = (
        "Klue, a Vancouver-based AI-powered competitive enablement "
        "platform, has raised $62 million in Series B funding led by "
        "Tiger Global, with participation from Salesforce Ventures. "
        "Founded in 2017, Klue combines competitive intelligence "
        "collection with content distribution. The platform serves "
        "nearly 400 enterprise clients and more than 110,000 users."
    )
    assert not _chunk_is_low_quality(real_chunk)
    print("  ✓ markdown/HTML noise detection")


def test_low_quality_chunk_short_content() -> None:
    assert _chunk_is_low_quality("")
    assert _chunk_is_low_quality("hi")
    assert _chunk_is_low_quality("x" * (_MIN_CHUNK_CONTENT_CHARS - 1))
    assert not _chunk_is_low_quality("x" * _MIN_CHUNK_CONTENT_CHARS)
    print("  ✓ content length floor")


def test_filter_drops_blacklisted_and_low_quality() -> None:
    """End-to-end of _filter_tavily_chunks on a representative mix."""
    raws = [
        # Good real source
        {
            "url": "https://techcrunch.com/2021/12/01/klue",
            "title": "Klue raises $62M",
            "content": "Klue raised $62M Series B led by Tiger Global " * 20,
        },
        # Noise domain
        {
            "url": "https://www.facebook.com/post/123",
            "title": "Some FB post",
            "content": "x" * 1000,
        },
        # Subdomain of blacklisted domain
        {
            "url": "https://m.instagram.com/p/abc",
            "title": "IG post",
            "content": "y" * 1000,
        },
        # Real domain but junk content
        {
            "url": "https://www.businessinsider.com/x",
            "title": "BI pitch deck",
            "content": "![Klue](https://i.insider.com/x) " * 30
            + "<svg xmlns='http://www.w3.org/2000/svg'>",
        },
        # Real domain, real content — keep
        {
            "url": "https://www.betakit.com/klue-funding",
            "title": "BetaKit Klue article",
            "content": "Vancouver-based Klue secured $79M CAD " * 20,
        },
    ]
    kept = _filter_tavily_chunks(raws)
    urls = [r["url"] for r in kept]
    assert "https://techcrunch.com/2021/12/01/klue" in urls
    assert "https://www.betakit.com/klue-funding" in urls
    assert "https://www.facebook.com/post/123" not in urls
    assert "https://m.instagram.com/p/abc" not in urls
    assert "https://www.businessinsider.com/x" not in urls
    assert len(kept) == 2, f"expected 2 kept, got {len(kept)}: {urls}"
    print(f"  ✓ filter kept {len(kept)}/5 chunks (2 noise domains, 1 low-quality)")


def main() -> int:
    tests = [
        ("host_of_extracts_clean_host", test_host_of_extracts_clean_host),
        ("noise_domain_matches_known_blacklist", test_noise_domain_matches_known_blacklist),
        ("low_quality_chunk_detects_markdown_noise", test_low_quality_chunk_detects_markdown_noise),
        ("low_quality_chunk_short_content", test_low_quality_chunk_short_content),
        ("filter_drops_blacklisted_and_low_quality", test_filter_drops_blacklisted_and_low_quality),
    ]
    print("=" * 70)
    print(f" Phase 2.5 Tavily noise filter tests ({len(tests)} tests)")
    print("=" * 70)
    failed: list[str] = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
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
        return 1
    print(f" ALL {len(tests)} TAVILY FILTER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())