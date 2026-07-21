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
    """Both prose_lead and prose_footer are scanned; when they contain
    DIFFERENT domain-only URLs each gets its own issue. Runtime
    synthesises both fields depending on writer prompt structure.

    Note: `where` is formatted as ``"prose (section \\"<heading>\\")"``.
    """
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section(
        "S",
        prose_lead="Read more at https://kompyte.com",
    )
    sec.prose_footer = "Archive: https://crayon.co"  # different URL from lead
    rdo.add_row(sec, DataRow(
        key="r1", label="X", category="c", values={"a": 1},
        source_url="https://klue.com/pricing",  # page-level
    ))
    issues = enforce_page_level(rdo)
    assert len(issues) == 2, (
        f"expected 2 prose hits (one per distinct URL), got {len(issues)}: "
        f"{[i.where for i in issues]}"
    )
    assert all(i.where.startswith("prose") for i in issues), (
        f"both issues should be in prose fields, got: {[i.where for i in issues]}"
    )
    raws = sorted(i.raw_url for i in issues)
    assert "https://kompyte.com" in raws
    assert "https://crayon.co" in raws
    # Both prose fields are rewritten with the placeholder
    assert "[UNVERIFIED_DOMAIN_ONLY]" in sec.prose_lead
    assert "[UNVERIFIED_DOMAIN_ONLY]" in sec.prose_footer
    print(f"  ✓ rule_4 flags distinct domain-only URLs in prose_lead + prose_footer")


def test_rule_4_dedupes_same_url_across_prose_lead_and_footer():
    """When the SAME domain-only URL appears in both prose_lead and
    prose_footer (rare but possible if writer repeats a citation),
    dedup collapses to one issue. The placeholder still replaces both
    occurrences in the prose text.
    """
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section(
        "S",
        prose_lead="See https://kompyte.com for context",
    )
    sec.prose_footer = "Reiterating https://kompyte.com here"
    issues = enforce_page_level(rdo)
    # Both occurrences dedup to a single issue, but `by_url` replacement
    # still hits the second occurrence in the footer (we re-read the
    # blob after each replacement).
    assert len(issues) == 1, (
        f"expected 1 deduped issue, got {len(issues)}: {[i.raw_url for i in issues]}"
    )
    assert issues[0].raw_url == "https://kompyte.com"
    # Both prose fields should now contain the placeholder
    assert "[UNVERIFIED_DOMAIN_ONLY]" in sec.prose_lead
    assert "[UNVERIFIED_DOMAIN_ONLY]" in sec.prose_footer
    print(f"  ✓ rule_4 dedupes same URL across prose_lead+footer (1 issue, both replaced)")
def test_rule_4_dedupes_repeated_raw_url_with_occurrences_counter():
    """Same domain-only URL appearing in N rows of one section collapses
    into ONE issue with `occurrences == N`. The in-place replacement
    still visits every row so all 3 rows end up with the placeholder.
    This closes TODO plan-v2-7.
    """
    rdo = ReportDataObject(title="T")
    sec = rdo.add_section("Duplicated")
    for i in range(3):
        rdo.add_row(sec, DataRow(
            key=f"r{i}", label=f"L{i}", category="claim", values={"v": i},
            source_url="https://crayon.co",  # domain-only, repeated
        ))
    issues = enforce_page_level(rdo)
    # ONE issue per (section, raw_url), with occurrences counting repeats.
    assert len(issues) == 1, (
        f"expected 1 deduped issue, got {len(issues)}: "
        f"{[i.raw_url for i in issues]}"
    )
    assert issues[0].occurrences == 3, (
        f"expected occurrences=3 (3 rows reused the URL), got {issues[0].occurrences}"
    )
    assert issues[0].raw_url == "https://crayon.co"
    # All 3 rows still point at the placeholder — replacement is per-row.
    new_urls = [r.source_url for r in sec.rows]
    assert all(u == "[UNVERIFIED_DOMAIN_ONLY]" for u in new_urls), (
        f"all rows should be replaced, got: {new_urls}"
    )
    print(f"  ✓ rule_4 dedupes repeated URL → 1 issue with occurrences={issues[0].occurrences}, "
          f"{len(sec.rows)} rows replaced")


def test_rule_4_dedup_respects_section_boundary():
    """The same domain-only URL in two DIFFERENT sections is two distinct
    issues (because the writer rendered different content referencing it).
    """
    rdo = ReportDataObject(title="T")
    sec_a = rdo.add_section("Section A")
    sec_b = rdo.add_section("Section B")
    for sec in (sec_a, sec_b):
        rdo.add_row(sec, DataRow(
            key=f"r-{sec.heading}", label="X", category="claim", values={"a": 1},
            source_url="https://crayon.co",  # same URL, different sections
        ))
    issues = enforce_page_level(rdo)
    assert len(issues) == 2, (
        f"expected 2 issues (one per section), got {len(issues)}: "
        f"{[(i.raw_url, i.where) for i in issues]}"
    )
    assert all(i.occurrences == 1 for i in issues), "each section's issue should have occurrences=1"
    # Both sections' rows replaced
    assert sec_a.rows[0].source_url == "[UNVERIFIED_DOMAIN_ONLY]"
    assert sec_b.rows[0].source_url == "[UNVERIFIED_DOMAIN_ONLY]"
    print(f"  ✓ rule_4 dedup respects section boundary (2 issues, 1 per section)")


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
        ("rule_4_dedupes_same_url_across_prose_lead_and_footer",
         test_rule_4_dedupes_same_url_across_prose_lead_and_footer),
        ("rule_4_emits_per_row_issue_for_repeated_raw_url",
         test_rule_4_dedupes_repeated_raw_url_with_occurrences_counter),
        ("rule_4_dedup_respects_section_boundary",
         test_rule_4_dedup_respects_section_boundary),
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
