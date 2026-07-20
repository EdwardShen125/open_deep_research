"""Phase 3a — verifier engine tests."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.cited_report import (  # noqa: E402
    CitedClaim, CitedSection, CitedReport,
)
from open_deep_research.evidence_units import (  # noqa: E402
    EvidenceUnit, NumberBinding, EntityRef,
)
from open_deep_research.verifier import (  # noqa: E402
    verify, rule_1_numeric_binding, rule_2_entity_relation,
    rule_3_high_risk_xsource, rule_c_chinese_numbers,
    VerificationIssue,
)


def _eu(eid: str, claim: str, url: str, numbers=None, entities=None, conf=0.7):
    return EvidenceUnit(
        id=eid,
        claim=claim,
        quote=claim[:160],
        source_url=url,
        numbers=numbers or [],
        entities=entities or [],
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# Rule 1 — numeric binding
# ---------------------------------------------------------------------------

def test_rule_1_no_cited_eus_with_matching_number_triggers_issue():
    eu = _eu("eu-x", "earnings $20M",
             "https://klue.com/a",
             numbers=[NumberBinding(text="$20M", value_min=20e6, value_max=20e6, unit="USD")])
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="Crayon is $999M.",
            eu_ids=["eu-x"],
            numbers=[NumberBinding(text="$999M", value_min=999e6, value_max=999e6, unit="USD")],
            confidence=0.7,
        )])],
    )
    issues = rule_1_numeric_binding(rep, [eu])
    assert len(issues) == 1
    assert issues[0].rule_id == "rule_1"
    assert issues[0].severity == "high"
    assert "$999M" in issues[0].detail
    print(f"  ✓ rule_1 fires when no overlap")


def test_rule_1_match_within_5pct_passes():
    eu = _eu("eu-x", "TAM 30-60 亿美元",
             "https://klue.com/a",
             numbers=[NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD")])
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="TAM 30-60 亿美元.",
            eu_ids=["eu-x"],
            numbers=[NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD")],
            confidence=0.7,
        )])],
    )
    issues = rule_1_numeric_binding(rep, [eu])
    assert len(issues) == 0
    print(f"  ✓ rule_1 passes on verbatim match")


def test_rule_1_textual_substring_passes():
    eu = _eu("eu-x", "Klue 2022 Series C 估值 8-10 亿美元",
             "https://klue.com/a",
             numbers=[NumberBinding(text="8-10 亿美元", value_min=8, value_max=10, unit="USD")])
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="Klue Series C 估值 8-10 亿美元.",
            eu_ids=["eu-x"],
            numbers=[NumberBinding(text="8-10 亿美元", value_min=8, value_max=10, unit="USD")],
            confidence=0.7,
        )])],
    )
    issues = rule_1_numeric_binding(rep, [eu])
    assert len(issues) == 0
    print(f"  ✓ rule_1 textual substring passes")


# ---------------------------------------------------------------------------
# Rule 2 — entity relation / ownership
# ---------------------------------------------------------------------------

def test_rule_2_ownership_single_source_triggers_critical():
    eu = _eu("eu-only", "Kompyte acquired by Crayon.",
             "https://klue.com/a")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("Rel", claims=[CitedClaim(
            text="Kompyte was acquired by Crayon.",
            eu_ids=["eu-only"],
            confidence=0.8,
        )])],
    )
    issues = rule_2_entity_relation(rep, [eu])
    assert len(issues) == 1
    assert issues[0].rule_id == "rule_2"
    assert issues[0].severity == "critical"
    assert issues[0].anchor_id == "A1_kompyte_ownership"
    print(f"  ✓ rule_2 fires for single-source ownership (Kompyte/Crayon A1)")


def test_rule_2_ownership_two_domains_passes():
    eu1 = _eu("eu-x", "Kompyte", "https://klue.com/a")
    eu2 = _eu("eu-y", "Kompyte under Crayon", "https://crayon.co/b")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("Rel", claims=[CitedClaim(
            text="Kompyte was acquired by Crayon.",
            eu_ids=["eu-x", "eu-y"],
            confidence=0.85,
        )])],
    )
    issues = rule_2_entity_relation(rep, [eu1, eu2])
    assert len(issues) == 0
    print(f"  ✓ rule_2 passes on 2-domain relation")


def test_rule_2_same_domain_two_urls_still_triggers():
    """If the 2 EUs come from the same domain, they don't count as independent."""
    eu1 = _eu("eu-x", "Kompyte", "https://klue.com/page1")
    eu2 = _eu("eu-y", "Kompyte", "https://klue.com/page2")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("Rel", claims=[CitedClaim(
            text="Kompyte was acquired by Crayon.",
            eu_ids=["eu-x", "eu-y"],
            confidence=0.85,
        )])],
    )
    issues = rule_2_entity_relation(rep, [eu1, eu2])
    assert len(issues) == 1, "single-domain reuse is NOT independent"
    print(f"  ✓ rule_2 still fires when 2 EUs share domain")


# ---------------------------------------------------------------------------
# Rule 3 — high-risk cross-source
# ---------------------------------------------------------------------------

def test_rule_3_high_confidence_claim_with_no_cross_domain_triggers():
    eu = _eu("eu-x", "TAM $30B",
             "https://klue.com/a",
             numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")],
             conf=0.7)
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="TAM is $30B.",
            eu_ids=["eu-x"],
            numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")],
            confidence=0.8,
        )])],
    )
    issues = rule_3_high_risk_xsource(rep, [eu])
    assert len(issues) == 1
    assert issues[0].rule_id == "rule_3"
    assert issues[0].severity == "medium"
    print(f"  ✓ rule_3 fires for high-confidence w/o cross-domain")


def test_rule_3_skips_low_confidence_claims():
    eu = _eu("eu-x", "TAM $30B",
             "https://klue.com/a",
             conf=0.5)
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="TAM is $30B.",
            eu_ids=["eu-x"],
            numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")],
            confidence=0.4,         # below threshold
        )])],
    )
    issues = rule_3_high_risk_xsource(rep, [eu])
    assert len(issues) == 0
    print(f"  ✓ rule_3 skips low-confidence claims")


def test_rule_3_passes_when_two_domains():
    e1 = _eu("e1", "TAM $30B", "https://klue.com/a",
             conf=0.7,
             numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")])
    e2 = _eu("e2", "TAM $30B", "https://crayon.co/b",
             conf=0.7,
             numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")])
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="TAM is $30B.",
            eu_ids=["e1", "e2"],
            numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")],
            confidence=0.85,
        )])],
    )
    issues = rule_3_high_risk_xsource(rep, [e1, e2])
    assert len(issues) == 0
    print(f"  ✓ rule_3 passes when 2-domain backing exists")


# ---------------------------------------------------------------------------
# Rule C — CN numbers
# ---------------------------------------------------------------------------

def test_rule_c_flags_missing_cn_binding():
    eu = _eu("eu-x", "TAM is roughly 30 billion.",   # no CN
             "https://klue.com/a")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="全球 B2B SaaS TAM 30-60 亿美元.",
            eu_ids=["eu-x"],
            numbers=[NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD")],
            confidence=0.7,
        )])],
    )
    issues = rule_c_chinese_numbers(rep, [eu])
    assert len(issues) == 1
    assert issues[0].rule_id == "rule_c"
    print(f"  ✓ rule_c fires when cited EU lacks CN numerals")


def test_rule_c_passes_when_eu_has_cn():
    eu = _eu("eu-x", "全球 SaaS 约 30-60 亿美元",  # has CN
             "https://klue.com/a",
             numbers=[NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD")])
    rep = CitedReport(
        title="t",
        sections=[CitedSection("S", claims=[CitedClaim(
            text="TAM 30-60 亿美元.",
            eu_ids=["eu-x"],
            numbers=[NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD")],
            confidence=0.7,
        )])],
    )
    issues = rule_c_chinese_numbers(rep, [eu])
    assert len(issues) == 0
    print(f"  ✓ rule_c passes when EU has CN text")


# ---------------------------------------------------------------------------
# Top-level verify() aggregator
# ---------------------------------------------------------------------------

def test_verify_aggregates_by_rule_and_severity():
    eu_x = _eu("eu-x", "Kompyte acquired by Crayon", "https://klue.com/a")
    eu_y = _eu("eu-y", "TAM $30B", "https://klue.com/b",
               numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")])
    rep = CitedReport(
        title="t",
        sections=[
            CitedSection("Rel", claims=[CitedClaim(
                text="Kompyte was acquired by Crayon.",
                eu_ids=["eu-x"], confidence=0.8,
            )]),
            CitedSection("TAM", claims=[CitedClaim(
                text="TAM is $999B.",
                eu_ids=["eu-y"],
                numbers=[NumberBinding(text="$999B", value_min=999e9, value_max=999e9, unit="USD")],
                confidence=0.7,
            )]),
        ],
    )
    r = verify(rep, [eu_x, eu_y])
    assert not r.passes
    assert r.by_rule.get("rule_2", 0) >= 1
    assert r.by_rule.get("rule_1", 0) >= 1
    assert r.by_severity.get("critical", 0) >= 1
    assert r.by_severity.get("high", 0) >= 1
    assert r.anchors_triggered.get("A1_kompyte_ownership") == 1
    print(f"  ✓ verify(): by_rule={r.by_rule}, by_sev={r.by_severity}, passes={r.passes}")


def test_verify_clean_report_passes():
    eu_a = _eu("a", "Kompyte", "https://klue.com/a")
    eu_b = _eu("b", "Kompyte now under Crayon", "https://crayon.co/b")
    eu_c = _eu("c", "Klue TAM $30B", "https://klue.com/c",
               numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")])
    eu_d = _eu("d", "Klue TAM $30B", "https://crayon.co/d",
               numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")])
    rep = CitedReport(
        title="good",
        sections=[
            CitedSection("Rel", claims=[CitedClaim(
                text="Kompyte was acquired by Crayon.",
                eu_ids=["a", "b"], confidence=0.85,
            )]),
            CitedSection("TAM", claims=[CitedClaim(
                text="TAM is $30B.",
                eu_ids=["c", "d"],
                numbers=[NumberBinding(text="$30B", value_min=30e9, value_max=30e9, unit="USD")],
                confidence=0.7,
            )]),
        ],
    )
    r = verify(rep, [eu_a, eu_b, eu_c, eu_d])
    assert r.passes
    assert r.by_severity.get("critical", 0) == 0
    print(f"  ✓ verify clean report: passes")


# ---------------------------------------------------------------------------
# Issue to_dict / serialization
# ---------------------------------------------------------------------------

def test_issue_to_dict_roundtrip():
    issue = VerificationIssue(
        rule_id="rule_2",
        severity="critical",
        anchor_id="A1_kompyte_ownership",
        section="Rel",
        claim_text="Kompyte was acquired by Crayon.",
        detail="single domain",
        affected_eu_ids=["eu-x"],
        context={"domains": ["klue.com"]},
    )
    d = issue.to_dict()
    assert d["rule_id"] == "rule_2"
    assert d["anchor_id"] == "A1_kompyte_ownership"
    assert d["context"] == {"domains": ["klue.com"]}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("rule_1_no_cited_eus_with_matching_number_triggers_issue",
         test_rule_1_no_cited_eus_with_matching_number_triggers_issue),
        ("rule_1_match_within_5pct_passes", test_rule_1_match_within_5pct_passes),
        ("rule_1_textual_substring_passes", test_rule_1_textual_substring_passes),
        ("rule_2_ownership_single_source_triggers_critical",
         test_rule_2_ownership_single_source_triggers_critical),
        ("rule_2_ownership_two_domains_passes",
         test_rule_2_ownership_two_domains_passes),
        ("rule_2_same_domain_two_urls_still_triggers",
         test_rule_2_same_domain_two_urls_still_triggers),
        ("rule_3_high_confidence_claim_with_no_cross_domain_triggers",
         test_rule_3_high_confidence_claim_with_no_cross_domain_triggers),
        ("rule_3_skips_low_confidence_claims",
         test_rule_3_skips_low_confidence_claims),
        ("rule_3_passes_when_two_domains",
         test_rule_3_passes_when_two_domains),
        ("rule_c_flags_missing_cn_binding", test_rule_c_flags_missing_cn_binding),
        ("rule_c_passes_when_eu_has_cn", test_rule_c_passes_when_eu_has_cn),
        ("verify_aggregates_by_rule_and_severity",
         test_verify_aggregates_by_rule_and_severity),
        ("verify_clean_report_passes", test_verify_clean_report_passes),
        ("issue_to_dict_roundtrip", test_issue_to_dict_roundtrip),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} verifier tests")
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
