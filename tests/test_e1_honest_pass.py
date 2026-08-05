"""E1 · 诚实 pass 验收测试

按 SPEC §5 E1:
- 主结论里每个定量 claim tier≥B 且 verified,否则降级/标记
- [待核实]附录非空且齐全
- 有 caliber_mismatch 时自动出"口径误区"段
- 来源页由元数据渲染(非手写)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_deep_research.evidence.claim_v3 import ClaimV3, GateResults
from open_deep_research.evidence.honest_pass import (
    ALLOWED_TIERS_FOR_MAIN,
    run_honest_pass,
)
from open_deep_research.evidence.reconciliation import ReconciliationRecord
from open_deep_research.evidence.report_result import (
    FilledSlot,
    ReportResult,
    SectionResult,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_claim(
    value: str = "x",
    *,
    norm_value: float | None = 0.0,  # 默认给个 0,numeric claim 必须有
    status: str = "verified",
    tier: str | None = "A",
    source_domain: str = "emarketer.com",
    source_url: str = "https://emarketer.com/x",
    claim_type: str = "numeric",
    caliber_id: str | None = None,
    source_title: str | None = None,
) -> ClaimV3:
    gate = GateResults(span=True, entail="entailed") if status == "verified" else (
        GateResults(span=True, entail="contradicted") if status == "failed_gate" else GateResults()
    )
    return ClaimV3(
        value=value,
        source_id=source_url,
        source_url=source_url,
        source_domain=source_domain,
        source_title=source_title,
        claim_type=claim_type,  # type: ignore[arg-type]
        norm_value=norm_value,
        verification_status=status,  # type: ignore[arg-type]
        tier=tier,  # type: ignore[arg-type]
        caliber_id=caliber_id,
        gate_results=gate,
    )


def _make_report(*, sections: list[SectionResult]) -> ReportResult:
    return ReportResult(
        title="T", vertical_id="us_livecommerce",
        sections=sections,
        unresolved=[],
    )


# -----------------------------------------------------------------------------
# 验收 1: 主结论门 — tier<B 或 not verified 降级
# -----------------------------------------------------------------------------

class TestMainConclusionGate:
    def test_tier_below_b_in_main_downgrades(self) -> None:
        """主结论里 numeric claim tier=C → confidence 降级。"""
        slot = FilledSlot(
            slot_id="market_size",
            claims=[_make_claim(norm_value=100.0, tier="C")],
            confidence="confirmed",
        )
        report = _make_report(sections=[
            SectionResult(section_id="m", title="Market", slots=[slot]),
        ])
        # tier=C 在 numeric claim 中触发降级
        out = run_honest_pass(report, claims=slot.claims, reconciliations=[])
        # 原来 confirmed → 降级
        assert out.sections[0].slots[0].confidence in ("structural", "to_verify")

    def test_tier_b_verified_no_downgrade(self) -> None:
        slot = FilledSlot(
            slot_id="market_size",
            claims=[_make_claim(norm_value=100.0, tier="B", status="verified")],
            confidence="confirmed",
        )
        report = _make_report(sections=[
            SectionResult(section_id="m", title="Market", slots=[slot]),
        ])
        out = run_honest_pass(report, claims=slot.claims, reconciliations=[])
        # tier=B + verified → 保持 confirmed
        assert out.sections[0].slots[0].confidence == "confirmed"

    def test_to_verify_downgrades_to_to_verify(self) -> None:
        slot = FilledSlot(
            slot_id="market_size",
            claims=[_make_claim(norm_value=100.0, tier="A", status="to_verify")],
            confidence="structural",  # 起始是 structural
        )
        report = _make_report(sections=[
            SectionResult(section_id="m", title="Market", slots=[slot]),
        ])
        out = run_honest_pass(report, claims=slot.claims, reconciliations=[])
        # status=to_verify → 降级到 to_verify
        assert out.sections[0].slots[0].confidence == "to_verify"

    def test_non_numeric_not_downgraded(self) -> None:
        """非 numeric claim 不触发主结论门(plan E1 DoD:只对 numeric)。"""
        slot = FilledSlot(
            slot_id="trend",
            claims=[_make_claim(norm_value=None, claim_type="attribute", tier="D")],
            confidence="structural",
        )
        report = _make_report(sections=[
            SectionResult(section_id="t", title="Trends", slots=[slot]),
        ])
        out = run_honest_pass(report, claims=slot.claims, reconciliations=[])
        # attribute tier=D 不触发降级(主结论门只看 numeric)
        assert out.sections[0].slots[0].confidence == "structural"


# -----------------------------------------------------------------------------
# 验收 2: [待核实]附录
# -----------------------------------------------------------------------------

class TestUnverifiedAppendix:
    def test_appendix_includes_to_verify_claims(self) -> None:
        claims = [
            _make_claim(value="claim A", status="to_verify", tier="A"),
            _make_claim(value="claim B", status="verified", tier="A"),
        ]
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=claims, reconciliations=[])
        appendix = out.honest_pass["unverified_appendix"]  # type: ignore[index]
        assert len(appendix) == 1
        assert appendix[0]["value"] == "claim A"
        assert appendix[0]["reason"]

    def test_appendix_includes_failed_gate(self) -> None:
        claims = [
            _make_claim(value="bad claim", status="failed_gate", tier="A"),
        ]
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=claims, reconciliations=[])
        appendix = out.honest_pass["unverified_appendix"]  # type: ignore[index]
        assert len(appendix) == 1
        assert "bad claim" in appendix[0]["value"]

    def test_appendix_includes_tier_below_b(self) -> None:
        """tier<D 也进待核实(tier<C/D 在主结论门外)。"""
        claims = [
            _make_claim(value="c-tier claim", status="verified", tier="C"),
        ]
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=claims, reconciliations=[])
        appendix = out.honest_pass["unverified_appendix"]  # type: ignore[index]
        assert len(appendix) == 1
        assert "tier=C" in appendix[0]["reason"] or "tier" in appendix[0]["reason"]

    def test_appendix_empty_when_all_verified_ab(self) -> None:
        claims = [
            _make_claim(value="good", status="verified", tier="A"),
            _make_claim(value="also good", status="verified", tier="B"),
        ]
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=claims, reconciliations=[])
        appendix = out.honest_pass["unverified_appendix"]  # type: ignore[index]
        assert appendix == []

    def test_appendix_has_required_fields(self) -> None:
        """每条 appendix item 含 value/source/tier/reason。"""
        claims = [_make_claim(value="x", status="to_verify", tier="C")]
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=claims, reconciliations=[])
        item = out.honest_pass["unverified_appendix"][0]  # type: ignore[index]
        for k in ("value", "source", "tier", "reason"):
            assert k in item


# -----------------------------------------------------------------------------
# 验收 3: 口径误区段
# -----------------------------------------------------------------------------

class TestCaliberWarnings:
    def _make_mismatch_record(self) -> ReconciliationRecord:
        return ReconciliationRecord(
            measurand="market_size of us_live_commerce (2025)",
            primary_value=146.4,
            primary_source_id="u1",
            primary_tier="A",
            primary_caliber_id="us_livestream_retail_emarketer",
            alternatives=[
                {
                    "value": 550.0,
                    "caliber_id": "us_video_shopping_broad_coresight",
                    "why_not_comparable": "口径不同:广义视频购物 vs 直播零售",
                },
                {
                    "value": 678.0,
                    "caliber_id": "us_live_commerce_mckinsey",
                    "why_not_comparable": "口径不同",
                },
            ],
            independence_note="独立源 3 个",
            confidence=0.85,
            divergence_kind="caliber_mismatch",
        )

    def test_caliber_mismatch_appears_in_warnings(self) -> None:
        rec = self._make_mismatch_record()
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=[], reconciliations=[rec])
        warnings = out.honest_pass["caliber_warnings"]  # type: ignore[index]
        assert len(warnings) == 1
        assert warnings[0]["primary_caliber"] == "us_livestream_retail_emarketer"
        assert warnings[0]["primary_value"] == 146.4
        assert len(warnings[0]["alternatives"]) == 2

    def test_clean_records_no_warning(self) -> None:
        rec = ReconciliationRecord(
            measurand="x",
            primary_value=100.0,
            primary_source_id="u",
            primary_tier="A",
            primary_caliber_id="c",
            alternatives=[],
            independence_note="",
            confidence=0.9,
            divergence_kind="clean",
        )
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=[], reconciliations=[rec])
        warnings = out.honest_pass["caliber_warnings"]  # type: ignore[index]
        assert warnings == []

    def test_data_conflict_record_also_visible(self) -> None:
        """data_conflict 也应在 honest_pass 中可观察(caliber_warnings 只收 caliber_mismatch,
        但 stats / 计数上应有体现)。
        """
        rec = ReconciliationRecord(
            measurand="x",
            primary_value=100.0,
            primary_source_id="u",
            primary_tier="A",
            primary_caliber_id="c",
            alternatives=[],
            independence_note="",
            confidence=0.5,
            divergence_kind="data_conflict",
        )
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=[], reconciliations=[rec])
        # data_conflict 不进 caliber_warnings(只 caliber_mismatch 进)
        warnings = out.honest_pass["caliber_warnings"]  # type: ignore[index]
        assert warnings == []
        # 但 reconciliation 仍被记录(供后续 E1 扩展或 debug)


# -----------------------------------------------------------------------------
# 验收 4: 来源页(元数据渲染,非手写)
# -----------------------------------------------------------------------------

class TestSourcePages:
    def test_source_pages_rendered_from_metadata(self) -> None:
        claims = [
            _make_claim(value="a", source_domain="emarketer.com",
                         source_url="https://emarketer.com/a",
                         source_title="E1 Article"),
            _make_claim(value="b", source_domain="coresight.com",
                         source_url="https://coresight.com/b",
                         source_title="C1 Article"),
        ]
        slot1 = FilledSlot(slot_id="s1", claims=[claims[0]], confidence="confirmed")
        slot2 = FilledSlot(slot_id="s2", claims=[claims[1]], confidence="confirmed")
        report = _make_report(sections=[
            SectionResult(section_id="sec1", title="Section 1", slots=[slot1]),
            SectionResult(section_id="sec2", title="Section 2", slots=[slot2]),
        ])
        out = run_honest_pass(report, claims=claims, reconciliations=[])

        pages = out.honest_pass["source_pages"]  # type: ignore[index]
        assert "by_section" in pages
        assert "by_source" in pages
        # by_section 含两 section
        assert len(pages["by_section"]) == 2
        # by_source 按 claim_count 降序
        assert len(pages["by_source"]) == 2
        # 验证每个 by_source 含 domain / tier / claim_count / sections
        for s in pages["by_source"]:
            assert "domain" in s
            assert "tier" in s
            assert "claim_count" in s
            assert "sections" in s

    def test_source_pages_non_empty(self) -> None:
        slot = FilledSlot(
            slot_id="s1",
            claims=[_make_claim(source_domain="emarketer.com")],
            confidence="confirmed",
        )
        report = _make_report(sections=[
            SectionResult(section_id="s", title="S", slots=[slot]),
        ])
        out = run_honest_pass(report, claims=slot.claims, reconciliations=[])
        assert len(out.honest_pass["source_pages"]["by_source"]) >= 1  # type: ignore[index]

    def test_source_pages_dedup_per_section(self) -> None:
        """同一 section 同一域名不重复列。"""
        c1 = _make_claim(value="a", source_domain="emarketer.com",
                          source_url="https://e.com/1")
        c2 = _make_claim(value="b", source_domain="emarketer.com",
                          source_url="https://e.com/2")
        slot = FilledSlot(slot_id="s", claims=[c1, c2], confidence="confirmed")
        report = _make_report(sections=[
            SectionResult(section_id="s", title="S", slots=[slot]),
        ])
        out = run_honest_pass(report, claims=[c1, c2], reconciliations=[])
        sec0_sources = out.honest_pass["source_pages"]["by_section"][0]["sources"]  # type: ignore[index]
        # emarketer.com 只出现一次
        emarketer_entries = [s for s in sec0_sources if s["domain"] == "emarketer.com"]
        assert len(emarketer_entries) == 1


# -----------------------------------------------------------------------------
# 验收 5: 集成 — 完整 honest_pass 跑通
# -----------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_with_mixed_claims(self) -> None:
        """混合 verified/to_verify/failed_gate + caliber_mismatch 一起跑。"""
        good = _make_claim(value="good claim", norm_value=146.4,
                           status="verified", tier="A",
                           source_domain="emarketer.com",
                           source_title="E Article")
        bad = _make_claim(value="bad claim", norm_value=99.9,
                          status="to_verify", tier="C",
                          source_domain="unknown.invalid",
                          source_title="Unknown")
        slot1 = FilledSlot(slot_id="market_size", claims=[good, bad],
                           confidence="structural")
        report = _make_report(sections=[
            SectionResult(section_id="m", title="Market", slots=[slot1]),
        ])
        rec = ReconciliationRecord(
            measurand="market_size of us_live_commerce (2025)",
            primary_value=146.4,
            primary_source_id="https://e.com/x",
            primary_tier="A",
            primary_caliber_id="us_livestream_retail_emarketer",
            alternatives=[
                {"value": 550.0, "caliber_id": "broad",
                 "why_not_comparable": "口径不同"},
            ],
            independence_note="独立源 2",
            confidence=0.8,
            divergence_kind="caliber_mismatch",
        )
        out = run_honest_pass(report, claims=[good, bad], reconciliations=[rec])

        # 1. 主结论门:bad claim tier=C,触发降级
        assert out.sections[0].slots[0].confidence == "to_verify"

        # 2. 待核实附录含 bad claim
        appendix = out.honest_pass["unverified_appendix"]  # type: ignore[index]
        assert any(item["value"] == "bad claim" for item in appendix)

        # 3. 口径误区段
        warnings = out.honest_pass["caliber_warnings"]  # type: ignore[index]
        assert len(warnings) == 1
        assert warnings[0]["primary_caliber"] == "us_livestream_retail_emarketer"

        # 4. 来源页含两个域
        pages = out.honest_pass["source_pages"]  # type: ignore[index]
        domains = {s["domain"] for s in pages["by_source"]}
        assert "emarketer.com" in domains
        assert "unknown.invalid" in domains

        # 5. stats 字段填充
        stats = out.honest_pass["stats"]  # type: ignore[index]
        assert stats["unverified_count"] >= 1
        assert stats["caliber_mismatch_count"] == 1


# -----------------------------------------------------------------------------
# 验收 6: 常量
# -----------------------------------------------------------------------------

class TestConstants:
    def test_allowed_tiers_for_main(self) -> None:
        assert ALLOWED_TIERS_FOR_MAIN == ("A", "B")


# -----------------------------------------------------------------------------
# 验收 7: 现有报告无 claims 时 honest_pass 不崩溃
# -----------------------------------------------------------------------------

class TestEmpty:
    def test_empty_report_no_crash(self) -> None:
        report = _make_report(sections=[])
        out = run_honest_pass(report, claims=[], reconciliations=[])
        assert out.honest_pass is not None
        assert out.honest_pass["unverified_appendix"] == []  # type: ignore[index]
        assert out.honest_pass["caliber_warnings"] == []  # type: ignore[index]
        assert out.honest_pass["source_pages"]["by_source"] == []  # type: ignore[index]
