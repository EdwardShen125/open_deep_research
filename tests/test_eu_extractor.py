"""Phase 2.2 — EU extractor tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.eu_extractor import (  # noqa: E402
    split_sentences, mine_entities, extract_from_search_result,
    extract_from_search_results,
)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def test_split_sentences_english():
    s = "Crayon leads enterprise CI. Klue targets battlecards. Kompyte is the budget option."
    out = split_sentences(s)
    assert len(out) == 3, out
    assert "Crayon" in out[0]
    print(f"  ✓ English sentences: {len(out)}")


def test_split_sentences_chinese():
    s = "Klue 2022 年共融资约 1.5 亿美元。Crayon 估值约 5 亿美元。Kompyte 已被 Crayon 收购。"
    out = split_sentences(s)
    assert len(out) >= 2, out
    print(f"  ✓ CJK sentences: {len(out)}")


def test_split_sentences_mixed():
    s = "Klue launched in 2014. Klue 2022 年收入达 5000 万美元."
    out = split_sentences(s)
    assert len(out) >= 2
    print(f"  ✓ mixed CJK/ASCII: {len(out)}")


# ---------------------------------------------------------------------------
# Entity mining
# ---------------------------------------------------------------------------

def test_mine_entities_lexicon_hit():
    ents = mine_entities("Crayon acquired Kompyte in 2022 for $35M.")
    names = sorted(e.name for e in ents)
    assert "Crayon" in names
    assert "Kompyte" in names
    # Acquisition hint
    by = [e for e in ents if e.name == "Kompyte"]
    if by:
        assert by[0].extra.get("acquired_by") in ("Crayon", None)
    print(f"  ✓ mine_entities: {names}")


def test_mine_entities_no_match():
    ents = mine_entities("Some random blog post about gardening with no vendors named.")
    assert ents == []
    print(f"  ✓ mine_entities: 0 hits on off-topic text")


# ---------------------------------------------------------------------------
# Extractor: page-level vs domain-only
# ---------------------------------------------------------------------------

def test_extractor_one_result_page_level():
    r = {
        "url": "https://klue.com/product/battlecards",
        "title": "Klue Battlecards",
        "content": "Crayon leads enterprise CI. Klue targets battlecards. Kompyte is budget.",
        "raw_content": "Crayon leads enterprise CI programs. Klue targets battlecards. Kompyte is the budget option at $300/yr.",
        "score": 0.78,
    }
    eus = extract_from_search_result(r, run_id="test-run")
    assert len(eus) >= 3, f"expected ≥3 EUs, got {len(eus)}"
    # Each EU has a non-empty quote and url
    for eu in eus:
        assert eu.claim
        assert eu.quote
        assert eu.source_url == "https://klue.com/product/battlecards"
        assert eu.run_id == "test-run"
    # The Kompyte sentence should have $300 number binding or Kompyte entity
    kompyte_eus = [e for e in eus if "Kompyte" in e.claim]
    assert len(kompyte_eus) >= 1
    keu = kompyte_eus[0]
    has_number = any("300" in n.text for n in keu.numbers)
    has_entity = any(e.name == "Kompyte" for e in keu.entities)
    assert has_number or has_entity, f"Kompyte EU should bind number or entity: {keu.numbers}, {keu.entities}"
    print(f"  ✓ extractor: {len(eus)} EUs from 1 page-level result, Kompyte grounding: {has_number=} {has_entity=}")


def test_extractor_domain_only_downgrades_confidence():
    """A result whose URL is domain-only should lower EU confidence.
    Must pass a DAO so the page_level flag is sourced from PG.
    """
    sys.path.insert(0, str(ROOT / "tests"))
    from test_sources_dao_sqlite import _SQLiteConnection, _DAOTest  # type: ignore
    dao = _DAOTest(_SQLiteConnection())
    r = {
        "url": "https://www.crayon.co",
        "title": "Crayon home",
        "content": "Crayon is enterprise competitive intelligence.",
    }
    eus = extract_from_search_result(r, run_id="test-run", sources_dao=dao)
    assert len(eus) >= 1
    for eu in eus:
        assert eu.confidence <= 0.55, f"domain-only EU should be downgraded (got {eu.confidence})"
        assert eu.extraction_method == "domain_only"
    print(f"  ✓ domain-only result: confidence ≤ 0.55, method=domain_only")


def test_extractor_no_url_is_silent():
    r = {"title": "no url", "content": "anything"}
    assert extract_from_search_result(r) == []
    print("  ✓ extractor: no URL → 0 EUs (no crash)")


def test_extractor_handles_raw_content():
    """raw_content preferred over content/summary; longer chunks give more EUs."""
    r_short = {
        "url": "https://klue.com/a",
        "content": "Klue makes battlecards.",
    }
    r_long = {
        "url": "https://klue.com/b",
        "raw_content": " ".join(["Klue makes battlecards." for _ in range(10)]),
    }
    short = extract_from_search_result(r_short)
    long_ = extract_from_search_result(r_long)
    assert len(long_) >= len(short)
    print(f"  ✓ raw_content usage: long={len(long_)} ≥ short={len(short)}")


def test_extractor_dedupes_across_results():
    """Same source URL → unique EU IDs."""
    r1 = {"url": "https://klue.com/a", "content": "Klue is a battlecards tool. Pricing starts at $20K."}
    r2 = {"url": "https://klue.com/a", "content": "Klue is a battlecards tool. Different angle on pricing $20K."}
    eus = extract_from_search_results([r1, r2])
    ids = [e.id for e in eus]
    # At minimum, the dup sentence "Klue is a battlecards tool." must dedup.
    assert len(ids) == len(set(ids)), "dedup by content_hash failed"
    print(f"  ✓ dedup across results: {len(eus)} unique EUs")


def test_extractor_integration_with_sources_dao_sqlite():
    """End-to-end: extractor + sqlite-parity DAO exercise the full path."""
    import json
    sys.path.insert(0, str(ROOT / "tests"))
    from test_sources_dao_sqlite import _SQLiteConnection, _DAOTest  # type: ignore

    conn = _SQLiteConnection()
    dao = _DAOTest(conn=conn)

    r = {
        "url": "https://klue.com/product/battlecards",
        "title": "Klue Battlecards",
        "content": "Klue is the battlecards leader. Pricing from $20K-$40K/yr.",
        "score": 0.81,
    }
    eus = extract_from_search_result(r, run_id="r-1", sources_dao=dao)
    assert len(eus) >= 1
    # source_id should be set from PG insert
    assert eus[0].source_id is not None
    # DAO stats should see the registered source
    s = dao.stats()
    assert s["total"] >= 1
    print(f"  ✓ extractor wired to DAO: source_id={eus[0].source_id}, dao total={s['total']}")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("split_sentences_english", test_split_sentences_english),
        ("split_sentences_chinese", test_split_sentences_chinese),
        ("split_sentences_mixed", test_split_sentences_mixed),
        ("mine_entities_lexicon_hit", test_mine_entities_lexicon_hit),
        ("mine_entities_no_match", test_mine_entities_no_match),
        ("extractor_one_result_page_level", test_extractor_one_result_page_level),
        ("extractor_domain_only_downgrades_confidence",
         test_extractor_domain_only_downgrades_confidence),
        ("extractor_no_url_is_silent", test_extractor_no_url_is_silent),
        ("extractor_handles_raw_content", test_extractor_handles_raw_content),
        ("extractor_dedupes_across_results", test_extractor_dedupes_across_results),
        ("extractor_integration_with_sources_dao_sqlite",
         test_extractor_integration_with_sources_dao_sqlite),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} extractor tests")
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


# =============================================================================
# P1 regression tests — keyword guard for cyber-focused briefs
# =============================================================================

class TestCyberGuard:
    """P1 fix: drop EU whose claim lacks cybersecurity context when the brief
    is about cybersecurity. This filters arxiv's EDR-disambiguation noise
    (Early Data Release / Energy Demand Reduction / Event Data Recorder)."""

    def test_cyber_guard_drops_arxiv_edr_astronomy_pollution(self):
        from open_deep_research.eu_extractor import extract_from_search_result
        # Simulate an arxiv hit on "Early Data Release Quasars" — a paper
        # that has nothing to do with cybersecurity but mentions "EDR".
        arxiv_result = {
            "url": "https://arxiv.org/abs/2308.15586v1",
            "title": "Improved Redshifts for DESI EDR Quasars",
            "content": (
                "We address the challenges of interpreting endpoint data by "
                "segmenting narratives into windows and emit X-ray stacking "
                "techniques. The majority of the DESI pipeline redshifts are reliable. "
                "We use very accurate photometric redshifts to measure AGN absorbing columns."
            ),
            "score": 1.0,
            "provider": "searxng",
            "engine": "arxiv",
        }
        # With cyber_guard enabled + cyber topic → no EU emitted
        eus = extract_from_search_result(
            arxiv_result,
            research_topic="EDR endpoint detection response market 2024",
            cyber_guard=True,
        )
        # All sentences mention "endpoint" or "EDR" but NOT in cybersecurity
        # context (it's astronomy). The guard should drop them all.
        assert len(eus) == 0, (
            f"cyber_guard should drop astronomy EDR, got {len(eus)} EU"
        )

    def test_cyber_guard_keeps_genuine_cybersecurity_content(self):
        from open_deep_research.eu_extractor import extract_from_search_result
        genuine = {
            "url": "https://arxiv.org/abs/2408.01993",
            "title": "Towards Automatic Hands-on-Keyboard Attack Detection Using LLMs in EDR Solutions",
            "content": (
                "This paper presents a machine learning model for endpoint detection and response. "
                "We address the challenges of detecting malware in real-time. "
                "Our approach improves threat hunting accuracy by 23%."
            ),
            "score": 1.0,
            "provider": "searxng",
            "engine": "arxiv",
        }
        eus = extract_from_search_result(
            genuine,
            research_topic="EDR endpoint detection response market 2024",
            cyber_guard=True,
        )
        # 3 sentences, all cybersecurity-relevant, all kept
        assert len(eus) >= 2, f"genuine EDR paper should pass guard, got {len(eus)} EU"

    def test_cyber_guard_disabled_keeps_everything(self):
        from open_deep_research.eu_extractor import extract_from_search_result
        arxiv_result = {
            "url": "https://arxiv.org/abs/2308.15586v1",
            "title": "Improved Redshifts for DESI EDR Quasars",
            "content": "The majority of the DESI pipeline redshifts are reliable.",
            "score": 1.0,
            "provider": "searxng",
            "engine": "arxiv",
        }
        # cyber_guard=False → all EU pass through (no filtering)
        eus = extract_from_search_result(
            arxiv_result,
            research_topic="EDR endpoint detection response market 2024",
            cyber_guard=False,
        )
        assert len(eus) >= 1, f"cyber_guard=False should keep all, got {len(eus)} EU"

    def test_cyber_guard_keeps_vendor_pages_even_without_cyber_keyword(self):
        """Vendor pages (qihoo/sangfor/crowdstrike etc.) should be exempt from
        the guard — their claim may be generic product copy that doesn't
        contain cyber keywords but is still highly relevant."""
        from open_deep_research.eu_extractor import extract_from_search_result
        vendor_page = {
            "url": "https://www.qihoo.com/products",
            "title": "Qihoo 360 Enterprise Security Suite",
            "content": (
                "Our flagship product delivers comprehensive protection. "
                "Trusted by 10,000+ enterprises across China."
            ),
            "score": 1.0,
            "provider": "searxng",
            "engine": "bing",
        }
        eus = extract_from_search_result(
            vendor_page,
            research_topic="EDR endpoint detection response market 2024",
            cyber_guard=True,
        )
        # Vendor page exempted → keep all
        assert len(eus) >= 1, f"vendor page should pass guard, got {len(eus)} EU"

    def test_cyber_guard_no_op_for_non_cyber_topic(self):
        """For non-cyber briefs (e.g. market research on agriculture), the
        guard should be a no-op. This prevents accidentally dropping
        legitimate content when the brief isn't about cybersecurity."""
        from open_deep_research.eu_extractor import extract_from_search_result
        non_cyber = {
            "url": "https://example.com/agriculture-report",
            "title": "Global Wheat Market 2024",
            "content": "Wheat production grew by 5% this year due to favorable weather.",
            "score": 1.0,
            "provider": "searxng",
            "engine": "bing",
        }
        eus = extract_from_search_result(
            non_cyber,
            research_topic="Global wheat market analysis 2024",
            cyber_guard=True,
        )
        assert len(eus) >= 1, f"non-cyber topic should not trigger guard, got {len(eus)} EU"
