"""Phase 3a.4 acceptance — verify against v1 baseline anchors.

For each v1 baseline anchor (A1-A4 ownership/relation anchors; B is rule-4
covered in Phase 3b), construct a *pathological* `CitedReport` that
mirrors the v1 failure mode and assert the verifier fires the matching rule.

These tests stand in for the real end-to-end LangGraph run that we cannot
execute until Docker / LangGraph server is reachable.
"""
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
from open_deep_research.verifier import verify  # noqa: E402


def _eu(eid, claim, url, numbers=None, entities=None, conf=0.7):
    return EvidenceUnit(
        id=eid, claim=claim, quote=claim[:160], source_url=url,
        numbers=numbers or [], entities=entities or [], confidence=conf,
    )


# ---------------------------------------------------------------------------
# A1 — Kompyte ownership
# ---------------------------------------------------------------------------

def test_a1_kompyte_ownership_reproduces_v1_failure():
    """v1 baseline: Kompyte described as independent, ownership absent.
    Construction: 1 EU from a single domain asserts ownership,
    plus another EU from the SAME domain (insufficient cross-domain)."""
    eu_a = _eu("a", "Kompyte is the leading CI product.",
               "https://klue.com/blog/kompyte")
    eu_b = _eu("b", "Crayon acquired Kompyte in 2022.",
               "https://klue.com/blog/crayon-acquired-kompyte")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("Rel", claims=[CitedClaim(
            text="Kompyte 独立 PMM 垂直产品",
            eu_ids=["a", "b"],
            confidence=0.8,
        )])],
    )
    r = verify(rep, [eu_a, eu_b])
    assert "A1_kompyte_ownership" in r.anchors_triggered, (
        f"verifier should fire A1 Kompyte ownership: {r.anchors_triggered}"
    )
    print(f"  ✓ A1 Kompyte ownership fires anchor (rule_2 critical)")


# ---------------------------------------------------------------------------
# A2 — Klue / Algorithmia 虚构关系
# ---------------------------------------------------------------------------

def test_a2_klue_algorithmia_fabrication():
    """v1 baseline: Klue acquired Algorithmia win/loss. False claim.
    Construction: claim cites only a SINGLE source for this acquisition."""
    eu = _eu("eu-only",
             "Klue acquired Algorithmia's win/loss business.",
             "https://klue.com/page-x")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("Rel", claims=[CitedClaim(
            text="Klue 收购了 Algorithmia 旗下的 win/loss 分析业务",
            eu_ids=["eu-only"],
            confidence=0.7,
        )])],
    )
    r = verify(rep, [eu])
    assert r.by_rule.get("rule_2", 0) >= 1
    assert r.by_severity.get("critical", 0) >= 1
    print(f"  ✓ A2 Klue/Algorithmia triggers rule_2 critical")


# ---------------------------------------------------------------------------
# A3 — Klue 估值 / Crayon 估值
# ---------------------------------------------------------------------------

def test_a3_valuation_high_risk_no_xdomain():
    """v1 baseline: valuation stated as fact with no source.
    Construction: high-confidence claim quotes $1B but cited EU
    doesn't contain a matching NumberBinding."""
    eu = _eu("eu-x", "Klue ranks #1 in enterprise CI.",
             "https://klue.com/a")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("Val", claims=[CitedClaim(
            text="Klue 估值约 10 亿美元,Crayon 估值约 5 亿美元",
            eu_ids=["eu-x"],
            numbers=[
                NumberBinding(text="10 亿美元", value_min=10, value_max=10, unit="USD"),
                NumberBinding(text="5 亿美元", value_min=5, value_max=5, unit="USD"),
            ],
            confidence=0.85,
        )])],
    )
    r = verify(rep, [eu])
    assert r.by_rule.get("rule_1", 0) >= 2
    assert r.by_rule.get("rule_3", 0) >= 1
    print(f"  ✓ A3 valuation: rule_1 + rule_3 fire")


# ---------------------------------------------------------------------------
# A4 — TAM 估算链
# ---------------------------------------------------------------------------

def test_a4_tam_chain_unsourced():
    """v1 baseline: 3000亿 × 1-2% = 30-60亿 stated as fact.
    Construction: claim quotes $30B but cited EU doesn't include the value."""
    eu = _eu("eu-x", "Global SaaS market is large.",
             "https://klue.com/blog/saas")
    rep = CitedReport(
        title="t",
        sections=[CitedSection("TAM", claims=[CitedClaim(
            text="全球 B2B SaaS 市场约 3000 亿美元,其中 PMM/CI 工具约占 1-2%,即 30-60 亿美元 TAM",
            eu_ids=["eu-x"],
            numbers=[
                NumberBinding(text="3000 亿美元", value_min=3000, value_max=3000, unit="USD"),
                NumberBinding(text="1-2%", value_min=1.0, value_max=2.0),
                NumberBinding(text="30-60 亿美元", value_min=30, value_max=60, unit="USD"),
            ],
            confidence=0.85,
        )])],
    )
    r = verify(rep, [eu])
    # rule_1 fires at least once (no matching EU), rule_3 fires (high confidence)
    assert r.by_rule.get("rule_1", 0) >= 1
    assert r.by_rule.get("rule_3", 0) >= 1
    print(f"  ✓ A4 TAM chain: rule_1 + rule_3 fire")


# ---------------------------------------------------------------------------
# Composite — fix the anchors and re-verify (sanity check)
# ---------------------------------------------------------------------------

def test_fixed_report_passes_composite():
    """Same topics, but with proper cross-domain EUs + matching numbers."""
    eu = [
        _eu("a1", "Kompyte was acquired by Crayon in 2022 for ~$35M.",
            "https://klue.com/blog",
            numbers=[NumberBinding(text="约 $35M", value_min=35e6, value_max=35e6, unit="USD")]),
        _eu("a2", "Crayon announced the Kompyte acquisition.",
            "https://crayon.co/news/acquisition",
            numbers=[NumberBinding(text="$35M deal", value_min=35e6, value_max=35e6, unit="USD")]),
        _eu("b1", "Klue 2022 Series C 估值约 8-10 亿美元",
            "https://klue.com/funding",
            numbers=[NumberBinding(text="8-10 亿美元", value_min=8, value_max=10, unit="USD")]),
        _eu("b2", "Crayon 估值约 5 亿美元",
            "https://crayon.co/company",
            numbers=[NumberBinding(text="5 亿美元", value_min=5, value_max=5, unit="USD")]),
        _eu("c1", "全球 B2B SaaS 市场约 3000 亿美元",
            "https://klue.com/tam",
            numbers=[NumberBinding(text="3000 亿美元", value_min=3000, value_max=3000, unit="USD")]),
        _eu("c2", "PMM 工具约占 1-2%",
            "https://crayon.co/tam-breakdown",
            numbers=[NumberBinding(text="1-2%", value_min=1.0, value_max=2.0)]),
    ]
    rep = CitedReport(
        title="t",
        sections=[
            CitedSection("Rel", claims=[CitedClaim(
                text="Kompyte was acquired by Crayon in 2022.",
                eu_ids=["a1", "a2"],
                numbers=[NumberBinding(text="约 $35M", value_min=35e6, value_max=35e6, unit="USD")],
                confidence=0.85,
            )]),
            CitedSection("Val", claims=[CitedClaim(
                text="Klue 估值约 8-10 亿美元,Crayon 估值约 5 亿美元",
                eu_ids=["b1", "b2"],
                numbers=[
                    NumberBinding(text="8-10 亿美元", value_min=8, value_max=10, unit="USD"),
                    NumberBinding(text="5 亿美元", value_min=5, value_max=5, unit="USD"),
                ],
                confidence=0.7,
            )]),
            CitedSection("TAM", claims=[CitedClaim(
                text=(
                    "全球 B2B SaaS 市场约 3000 亿美元,其中 PMM 工具约占 1-2%, "
                    "即估算 TAM (3000 × 1-2% = 30-60 亿美元). 这是估算链,非直接来源"
                ),
                eu_ids=["c1", "c2"],
                numbers=[
                    NumberBinding(text="3000 亿美元", value_min=3000, value_max=3000, unit="USD"),
                    NumberBinding(text="1-2%", value_min=1.0, value_max=2.0),
                ],
                confidence=0.6,  # lower confidence — this is a derived estimate
            )]),
        ],
    )
    r = verify(rep, eu)
    assert r.passes, f"expected clean report to pass, got issues: {[i.to_dict() for i in r.issues]}"
    print(f"  ✓ fixed composite report passes all 4 anchor categories")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("a1_kompyte_ownership", test_a1_kompyte_ownership_reproduces_v1_failure),
        ("a2_klue_algorithmia", test_a2_klue_algorithmia_fabrication),
        ("a3_valuation", test_a3_valuation_high_risk_no_xdomain),
        ("a4_tam_chain", test_a4_tam_chain_unsourced),
        ("fixed_composite_passes", test_fixed_report_passes_composite),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} Phase-3a acceptance tests")
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
    print(f" ALL {len(tests)} ANCHOR ACCEPTANCE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
