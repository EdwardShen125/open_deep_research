"""Phase 3b tests — ReportDataObject + Rule 4 enforcement."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.report_data import (  # noqa: E402
    DataRow, ReportSection, ReportDataObject,
    enforce_page_level,
)


# ---------------------------------------------------------------------------
# DataRow
# ---------------------------------------------------------------------------

def test_data_row_render_prose_from_template():
    r = DataRow(
        key="klue_rank",
        label="Klue",
        category="ranking",
        values={"rank": 2, "category": "CI"},
        prose_template="{label} ranks #{rank} in {category}",
    )
    assert r.render_prose() == "Klue ranks #2 in CI"
    print(f"  ✓ DataRow.render_prose: '{r.render_prose()}'")


# ---------------------------------------------------------------------------
# ReportSection.table
# ---------------------------------------------------------------------------

def test_section_markdown_table_columns():
    sec = ReportSection("CI Vendors")
    sec.add_row(DataRow(
        key="klue", label="Klue", category="ranking",
        values={"rank": 2, "price": "$20K-$40K"},
        table_columns=["rank", "price"],
    ))
    sec.add_row(DataRow(
        key="crayon", label="Crayon", category="ranking",
        values={"rank": 1, "price": "$20K-$40K"},
        table_columns=["rank", "price"],
    ))
    out = sec.to_markdown_table()
    assert "| --- | --- | --- |" in out
    assert "**Klue**" in out
    assert "2" in out
    assert "**Crayon**" in out
    print(f"  ✓ section.markdown_table columns unioned correctly")


def test_section_markdown_table_empty_columns():
    sec = ReportSection("Empty")
    assert sec.to_markdown_table() == ""
    print(f"  ✓ section.to_markdown_table returns empty string for empty section")


# ---------------------------------------------------------------------------
# ReportDataObject
# ---------------------------------------------------------------------------

def test_rdo_prose_and_table_share_same_source():
    rdo = ReportDataObject(title="CI Market")
    sec = rdo.add_section("Vendors", prose_lead="Three vendors stand out.")
    rdo.add_row(sec, DataRow(
        key="klue", label="Klue", category="ranking",
        values={"rank": 2, "category": "CI"},
        prose_template="{label} ranks #{rank} in {category}",
        table_columns=["rank"],
    ))
    rdo.add_row(sec, DataRow(
        key="crayon", label="Crayon", category="ranking",
        values={"rank": 1, "category": "CI"},
        prose_template="{label} ranks #{rank} in {category}",
        table_columns=["rank"],
    ))
    md = rdo.to_markdown()
    assert "Crayon ranks #1" in md
    assert "Klue ranks #2" in md
    assert "| rank |" in md     # markdown table column header
    assert "**Crayon**" in md
    assert "**Klue**" in md
    # Both views must include the same row data
    prose_vendors = sum(1 for line in md.split("\n") if "ranks #" in line)
    table_vendors = sum(1 for line in md.split("\n") if line.startswith("| **"))
    assert prose_vendors == 2 and table_vendors == 2, (prose_vendors, table_vendors)
    print(f"  ✓ RDO prose + table derive from the SAME DataRow")


def test_rdo_get_row():
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section("S")
    rdo.add_row(sec, DataRow(key="x", label="X", category="r", values={"a": 1}))
    assert rdo.get_row("x") is not None
    assert rdo.get_row("nope") is None


def test_rdo_to_dict_roundtrip():
    rdo = ReportDataObject(title="Dict test")
    sec = rdo.add_section("Sec")
    rdo.add_row(sec, DataRow(key="k", label="L", category="c", values={"v": 1}))
    d = rdo.to_dict()
    assert d["title"] == "Dict test"
    assert len(d["sections"]) == 1
    assert d["sections"][0]["rows"][0]["key"] == "k"


# ---------------------------------------------------------------------------
# Rule 4 — page-level URL enforcement
# ---------------------------------------------------------------------------

def test_rule_4_flags_domain_only_in_source_url():
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section("Sec")
    rdo.add_row(sec, DataRow(
        key="r1", label="X", category="c", values={"a": 1},
        source_url="https://www.crayon.co",       # domain-only
        source_id=42,
    ))
    issues = enforce_page_level(rdo)
    assert len(issues) == 1
    assert issues[0].severity == "high"
    assert issues[0].new_url_or_label == "[UNVERIFIED_DOMAIN_ONLY]"
    # Mutation applied
    assert rdo.sections[0].rows[0].source_url == "[UNVERIFIED_DOMAIN_ONLY]"
    print(f"  ✓ rule_4 flags + replaces domain-only URLs in source_url")


def test_rule_4_resolver_promotes_to_page_level():
    def resolver(url):
        # "Simulated" resolver for https://www.crayon.co → /vs-klue
        if url == "https://www.crayon.co":
            return "https://www.crayon.co/vs-klue"
        return url
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section("Sec")
    rdo.add_row(sec, DataRow(
        key="r1", label="X", category="c", values={"a": 1},
        source_url="https://www.crayon.co",
    ))
    issues = enforce_page_level(rdo, resolver=resolver)
    assert len(issues) == 1
    assert rdo.sections[0].rows[0].source_url == "https://www.crayon.co/vs-klue"
    print(f"  ✓ rule_4 resolver promotes to page-level via resolver callback")


def test_rule_4_flags_prose_field_url():
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section(
        "Sec",
        prose_lead="See https://klue.com for context.",
    )
    issues = enforce_page_level(rdo)
    # Note: klue.com without path → domain-only? Actually it's domain root.
    # classify_page_level('https://klue.com') → DOMAIN_ONLY
    assert len(issues) == 1
    assert issues[0].where.startswith("prose")
    assert "klue.com" in issues[0].raw_url
    print(f"  ✓ rule_4 flags domain-only URLs in prose text")


def test_rule_4_passes_for_page_level_url():
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section("Sec")
    rdo.add_row(sec, DataRow(
        key="r1", label="X", category="c", values={"a": 1},
        source_url="https://klue.com/product/battlecards",  # page-level
    ))
    issues = enforce_page_level(rdo)
    assert len(issues) == 0
    print(f"  ✓ rule_4 passes for page-level URLs (no issue)")


def test_rule_4_passes_for_subdomain_page():
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section("Sec")
    rdo.add_row(sec, DataRow(
        key="r1", label="X", category="c", values={"a": 1},
        source_url="https://blog.klue.com/2026/ci-market",
    ))
    issues = enforce_page_level(rdo)
    assert len(issues) == 0
    print(f"  ✓ rule_4 accepts subdomain page-level URL")


# ---------------------------------------------------------------------------
# Phase E2E-runtime edge-case fixtures (Plan v2 §未修 #3 closure)
#
# These cover runtime-shaped payloads where Plan v2's writer synthesises
# prose alongside structured DataRows. They run offline (no LangGraph server).
# ---------------------------------------------------------------------------

def test_rule_4_audits_multiple_sections_simultaneously():
    """Multi-section RDO: each section gets its own audit pass, all issues
    surface in a single return so the runtime can pack them into
    state.url_compliance in one call.
    """
    rdo = ReportDataObject(title="Multi-section runtime payload")
    # Section A — domain-only
    sec_a = rdo.add_section("Section A: ownership", prose_lead="See https://crayon.co for context.")
    rdo.add_row(sec_a, DataRow(
        key="a1", label="Crayon", category="claim", values={"x": 1},
        source_url="https://crayon.co",  # domain-only
    ))
    # Section B — page-level
    sec_b = rdo.add_section("Section B: pricing")
    rdo.add_row(sec_b, DataRow(
        key="b1", label="Pricing", category="claim", values={"x": 2},
        source_url="https://crayon.co/pricing/competitive-intelligence",  # page-level
    ))
    # Section C — domain-only in prose_lead
    sec_c = rdo.add_section("Section C: market", prose_lead="Source: https://klue.com")
    rdo.add_row(sec_c, DataRow(
        key="c1", label="Klue", category="claim", values={"x": 3},
        source_url="https://klue.com/vs-crayon",  # page-level
    ))

    issues = enforce_page_level(rdo)
    # Expect: 1 issue from sec A source_url + 1 issue from sec C prose_lead
    # Sec B is page-level → no issue.
    raws = sorted(i.raw_url for i in issues)
    assert "https://crayon.co" in raws, "sec A domain-only must be flagged"
    assert "https://klue.com" in raws, "sec C prose domain-only must be flagged"
    assert "https://crayon.co/pricing/competitive-intelligence" not in raws, "sec B page-level must NOT be flagged"
    print(f"  ✓ rule_4 audited {len(issues)} issues across 3 sections (expected 2)")


def test_rule_4_flags_prose_lead_and_prose_footer_independently():
    """Both prose_lead and prose_footer are scanned; each URL gets its own
    issue with the right `where` field. Runtime synthesises both fields
    depending on writer prompt structure.

    Note: `where` is formatted as ``"prose (section \\"<heading>\\")"``.
    """
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section(
        "S",
        prose_lead="Read more at https://kompyte.com",
    )
    sec.prose_footer = "Archive: https://kompyte.com"
    rdo.add_row(sec, DataRow(
        key="r1", label="X", category="c", values={"a": 1},
        source_url="https://kompyte.com/pricing",  # page-level
    ))
    issues = enforce_page_level(rdo)
    assert len(issues) == 2, f"expected 2 prose hits, got {len(issues)}: {[i.where for i in issues]}"
    assert all(i.where.startswith("prose") for i in issues), (
        f"both issues should be in prose fields, got: {[i.where for i in issues]}"
    )
    assert all("kompyte.com" in i.raw_url for i in issues)
    print(f"  ✓ rule_4 flags both prose_lead and prose_footer separately")


def test_rule_4_emits_per_row_issue_for_repeated_raw_url():
    """Known runtime quirk: if the same domain-only URL appears in N rows
    of the same section (writer reuses an EU for multiple claims),
    `enforce_page_level` currently emits N issues — one per row — instead
    of deduplicating to a single (section, raw_url) issue.

    This is a deliberate gap: dedup would require changing the
    UrlComplianceIssue schema (e.g. carry an `occurrences` counter), and
    runtime callers today either log every issue or wrap them in
    `{by_severity, by_url}` — both paths tolerate N identical entries.

    TODO(plan-v2-7): dedup `enforce_page_level` output to (section, raw_url)
    once the consumer side signals it would rather see aggregated counts.
    """
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section("Duplicated")
    for i in range(3):
        rdo.add_row(sec, DataRow(
            key=f"r{i}", label=f"L{i}", category="claim", values={"v": i},
            source_url="https://crayon.co",  # domain-only, repeated
        ))
    issues = enforce_page_level(rdo)
    # Current behaviour: one issue per row, all flagged identically.
    assert len(issues) == 3, (
        f"expected 3 per-row issues (current behaviour), got {len(issues)}: "
        f"{[i.raw_url for i in issues]}"
    )
    assert all(i.raw_url == "https://crayon.co" for i in issues)
    # All 3 rows still get the placeholder replacement.
    new_urls = [r.source_url for r in sec.rows]
    assert all(u == "[UNVERIFIED_DOMAIN_ONLY]" for u in new_urls), (
        f"all rows should be replaced, got: {new_urls}"
    )
    print(f"  ⚠ rule_4 emits per-row issue for repeated URL (3 issues, no dedup) — TODO plan-v2-7")


def test_rule_4_handles_empty_rdo_and_empty_section_gracefully():
    """Runtime may call enforce_page_level on a partially-built RDO
    (e.g. writer emitted 0 sections). Must return [] without crashing.
    """
    rdo_empty = ReportDataObject(title="Empty")
    assert enforce_page_level(rdo_empty) == []

    rdo_section_empty = ReportDataObject(title="Has empty section")
    rdo_section_empty.add_section("Nothing here")
    assert enforce_page_level(rdo_section_empty) == []
    print(f"  ✓ rule_4 no-op on empty RDO / empty section")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("data_row_render_prose_from_template",
         test_data_row_render_prose_from_template),
        ("section_markdown_table_columns", test_section_markdown_table_columns),
        ("section_markdown_table_empty_columns",
         test_section_markdown_table_empty_columns),
        ("rdo_prose_and_table_share_same_source",
         test_rdo_prose_and_table_share_same_source),
        ("rdo_get_row", test_rdo_get_row),
        ("rdo_to_dict_roundtrip", test_rdo_to_dict_roundtrip),
        ("rule_4_flags_domain_only_in_source_url",
         test_rule_4_flags_domain_only_in_source_url),
        ("rule_4_resolver_promotes_to_page_level",
         test_rule_4_resolver_promotes_to_page_level),
        ("rule_4_flags_prose_field_url", test_rule_4_flags_prose_field_url),
        ("rule_4_passes_for_page_level_url",
         test_rule_4_passes_for_page_level_url),
        ("rule_4_passes_for_subdomain_page",
         test_rule_4_passes_for_subdomain_page),
        ("rule_4_audits_multiple_sections_simultaneously",
         test_rule_4_audits_multiple_sections_simultaneously),
        ("rule_4_flags_prose_lead_and_prose_footer_independently",
         test_rule_4_flags_prose_lead_and_prose_footer_independently),
        ("rule_4_emits_per_row_issue_for_repeated_raw_url",
         test_rule_4_emits_per_row_issue_for_repeated_raw_url),
        ("rule_4_handles_empty_rdo_and_empty_section_gracefully",
         test_rule_4_handles_empty_rdo_and_empty_section_gracefully),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} ReportDataObject + Rule 4 tests")
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
