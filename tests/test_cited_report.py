"""Phase 2.3 — chain-of-citation schema, parser, and validator tests."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.cited_report import (  # noqa: E402
    CitedClaim, CitedSection, CitedReport,
    parse_cited_report, validate_cited_report, render_eu_pool,
    CITED_REPORT_PROMPT, _extract_json,
)
from open_deep_research.evidence_units import (  # noqa: E402
    EvidenceUnit, NumberBinding,
)


def _mk_eu(eid: str, claim: str, url: str, numbers=None, entities=None,
           quote: str = ""):
    return EvidenceUnit(
        id=eid,
        claim=claim,
        quote=quote or claim[:180],
        source_url=url,
        numbers=numbers or [],
        entities=entities or [],
    )


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def test_extract_json_from_fenced_block():
    raw = "Here is the JSON:\n```json\n{\"k\": 1}\n```\nThanks!"
    assert _extract_json(raw) == '{"k": 1}'


def test_extract_json_from_unfenced_braces():
    raw = "Some prose. {\"a\": 1, \"b\": 2} and more."
    out = _extract_json(raw)
    assert out and json.loads(out) == {"a": 1, "b": 2}


def test_extract_json_returns_none_when_none():
    assert _extract_json("") is None
    assert _extract_json("nothing to extract") is None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_valid_report():
    raw = json.dumps({
        "title": "CI Market 2026",
        "sections": [{
            "heading": "Overview",
            "claims": [
                {
                    "text": "Crayon leads the enterprise CI market.",
                    "eu_ids": ["eu-aaaa", "eu-bbbb"],
                    "numbers": [],
                    "confidence": 0.8,
                    "rationale": "two independent sources"
                }
            ]
        }]
    })
    rep, warns = parse_cited_report(raw)
    assert rep.title == "CI Market 2026"
    assert len(rep.sections) == 1
    assert len(rep.sections[0].claims) == 1
    assert rep.sections[0].claims[0].eu_ids == ["eu-aaaa", "eu-bbbb"]
    assert warns == []
    print(f"  ✓ parse valid report: {rep.title} / {len(rep.sections[0].claims)} claims")


def test_parse_invalid_report_returns_warnings():
    rep, warns = parse_cited_report("no JSON anywhere")
    assert warns
    assert "no JSON" in warns[0].lower() or "found" in warns[0].lower()
    print(f"  ✓ parse invalid: {warns}")


# ---------------------------------------------------------------------------
# Validator — gap-A (single-source relation), gap-C (unsourced)
# ---------------------------------------------------------------------------

def test_validator_flags_no_citations():
    eu = _mk_eu("eu-a1", "Crayon leads CI", "https://klue.com/a")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S1", claims=[CitedClaim(
            text="Crayon leads CI.",
            eu_ids=[], confidence=0.7
        )])]
    )
    issues = validate_cited_report(rep, [eu])
    assert any("NO eu_ids" in i for i in issues)
    assert len(rep.orphan_claim_text) == 1
    print(f"  ✓ validator flags orphan claim")


def test_validator_flags_relation_with_single_source():
    eu = _mk_eu("eu-only", "Kompyte was acquired by Crayon", "https://klue.com/k")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("Relations", claims=[CitedClaim(
            text="Kompyte was acquired by Crayon.",
            eu_ids=["eu-only"],
            confidence=0.7,
        )])],
    )
    issues = validate_cited_report(rep, [eu])
    assert any("gap-A" in i or "ownership" in i for i in issues), issues
    print(f"  ✓ validator flags single-source relation claim")


def test_validator_allows_relation_with_two_sources():
    eu1 = _mk_eu("eu-x", "Kompyte", "https://klue.com/a")
    eu2 = _mk_eu("eu-y", "Kompyte now under Crayon", "https://crayon.co/b")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="Kompyte was acquired by Crayon.",
            eu_ids=["eu-x", "eu-y"],
            confidence=0.85,
        )])],
    )
    issues = validate_cited_report(rep, [eu1, eu2])
    rel_issues = [i for i in issues if "ownership" in i]
    assert not rel_issues, rel_issues
    print(f"  ✓ relation OK with ≥2 EU citations")


def test_validator_numeric_mismatch():
    """Claim quotes a number not backed by the cited EU's NumberBinding."""
    eu = _mk_eu(
        "eu-a", "Crayon leads CI",
        "https://klue.com/a",
        numbers=[NumberBinding(text="$20K", value_min=20000, value_max=20000, unit="USD")],
    )
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="Crayon is $999M.",
            eu_ids=["eu-a"],
            numbers=[NumberBinding(text="$999M", value_min=999e6, value_max=999e6, unit="USD")],
            confidence=0.7,
        )])],
    )
    issues = validate_cited_report(rep, [eu])
    assert any("matching NumberBinding" in i for i in issues), issues
    print(f"  ✓ validator catches numeric mismatch")


def test_validator_numeric_match_textual():
    eu = _mk_eu(
        "eu-a", "Crayon 30-60 亿美元",
        "https://klue.com/a",
        numbers=[NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD")],
    )
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="TAM 30-60 亿美元.",
            eu_ids=["eu-a"],
            numbers=[NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD")],
            confidence=0.7,
        )])],
    )
    issues = validate_cited_report(rep, [eu])
    assert not any("matching NumberBinding" in i for i in issues), issues
    print(f"  ✓ validator accepts matching NumberBinding")


def test_validator_unresolved_eu_ids():
    eu = _mk_eu("eu-a", "claim", "https://klue.com/a")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="A claim.", eu_ids=["eu-bogus"],
            confidence=0.7,
        )])],
    )
    issues = validate_cited_report(rep, [eu])
    assert any("unknown EU" in i for i in issues), issues
    assert "eu-bogus" in rep.unresolved_eu_ids
    print(f"  ✓ validator records unresolved eu_ids")


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def test_render_eu_pool():
    eu = _mk_eu("eu-1", "Test", "https://klue.com/a",
                numbers=[NumberBinding(text="$1M", value_min=1e6, value_max=1e6, unit="USD")])
    block = render_eu_pool([eu])
    assert "eu-1" in block
    assert "$1M" in block
    print(f"  ✓ render_eu_pool: {len(block)} bytes")


def test_cited_report_to_markdown():
    rep = CitedReport(
        title="Demo",
        sections=[CitedSection("Intro", claims=[CitedClaim(
            text="Klue leads.", eu_ids=["eu-1"], confidence=0.7,
        )])],
    )
    md = rep.to_markdown()
    assert "# Demo" in md
    assert "## Intro" in md
    assert "[eu-1]" in md
    print(f"  ✓ to_markdown includes heading + citation")


def test_prompt_is_json_only():
    """The prompt template instructs JSON-only and forbids Markdown."""
    assert "JSON" in CITED_REPORT_PROMPT
    assert "Do NOT write Markdown" in CITED_REPORT_PROMPT
    print(f"  ✓ CITED_REPORT_PROMPT forbids Markdown output")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("extract_json_from_fenced_block", test_extract_json_from_fenced_block),
        ("extract_json_from_unfenced_braces", test_extract_json_from_unfenced_braces),
        ("extract_json_returns_none_when_none", test_extract_json_returns_none_when_none),
        ("parse_valid_report", test_parse_valid_report),
        ("parse_invalid_report_returns_warnings", test_parse_invalid_report_returns_warnings),
        ("validator_flags_no_citations", test_validator_flags_no_citations),
        ("validator_flags_relation_with_single_source",
         test_validator_flags_relation_with_single_source),
        ("validator_allows_relation_with_two_sources",
         test_validator_allows_relation_with_two_sources),
        ("validator_numeric_mismatch", test_validator_numeric_mismatch),
        ("validator_numeric_match_textual", test_validator_numeric_match_textual),
        ("validator_unresolved_eu_ids", test_validator_unresolved_eu_ids),
        ("render_eu_pool", test_render_eu_pool),
        ("cited_report_to_markdown", test_cited_report_to_markdown),
        ("prompt_is_json_only", test_prompt_is_json_only),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} cited-report tests")
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
