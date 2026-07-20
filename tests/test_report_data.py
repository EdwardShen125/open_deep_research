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
