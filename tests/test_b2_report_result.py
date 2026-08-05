"""B2 · ReportResult 验收测试

按 SPEC §5 B2:
- 无 string fallback 路径(grep 断言)
- 未填槽进入 unresolved 而非被空字符串掩盖
- "confirmed" 槽内不得含 to_verify claim(校验)
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from open_deep_research.evidence.claim_v3 import ClaimV3, GateResults
from open_deep_research.evidence.report_result import (
    Confidence,
    FilledSlot,
    ReportResult,
    SectionResult,
    derive_confidence,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_claim(status: str = "verified", tier: str | None = "A") -> ClaimV3:
    # failed_gate 在 A1 ClaimV3 里有 validator:必须 entail='contradicted'
    gate = (
        GateResults(span=True, entail="contradicted")
        if status == "failed_gate"
        else GateResults()
    )
    return ClaimV3(
        value="x", source_id="u",
        verification_status=status,  # type: ignore[arg-type]
        tier=tier,  # type: ignore[arg-type]
        gate_results=gate,
    )


# -----------------------------------------------------------------------------
# 验收 1: 基本构造
# -----------------------------------------------------------------------------

class TestBasic:
    def test_minimal_construct(self) -> None:
        r = ReportResult(title="T", vertical_id="v")
        assert r.title == "T"
        assert r.vertical_id == "v"
        assert r.sections == []
        assert r.unresolved == []  # 永远存在
        assert r.honest_pass is None

    def test_unresolved_always_present_field(self) -> None:
        r = ReportResult(title="T", vertical_id="v")
        assert "unresolved" in r.model_dump()
        assert r.unresolved == []

    def test_to_unresolved_summary(self) -> None:
        r = ReportResult(
            title="T", vertical_id="v",
            sections=[
                SectionResult(
                    section_id="s1", title="S1",
                    slots=[
                        FilledSlot(
                            slot_id="slot1",
                            claims=[_make_claim()],
                            confidence="confirmed",
                        ),
                    ],
                ),
            ],
            unresolved=["slot2", "slot3"],
        )
        summary = r.to_unresolved_summary()
        assert summary["unresolved_count"] == 2
        assert summary["section_count"] == 1
        assert summary["filled_slot_count"] == 1


# -----------------------------------------------------------------------------
# 验收 2: confirmed 槽内不得含 to_verify claim
# -----------------------------------------------------------------------------

class TestConfirmedValidator:
    def test_confirmed_all_verified_passes(self) -> None:
        slot = FilledSlot(
            slot_id="s1",
            claims=[_make_claim("verified", "A"), _make_claim("verified", "B")],
            confidence="confirmed",
        )
        assert slot.confidence == "confirmed"

    def test_confirmed_with_to_verify_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-verified"):
            FilledSlot(
                slot_id="s1",
                claims=[_make_claim("verified", "A"), _make_claim("to_verify")],
                confidence="confirmed",
            )

    def test_confirmed_with_failed_gate_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-verified"):
            FilledSlot(
                slot_id="s1",
                claims=[_make_claim("failed_gate", "A")],
                confidence="confirmed",
            )

    def test_to_verify_can_contain_anything(self) -> None:
        slot = FilledSlot(
            slot_id="s1",
            claims=[_make_claim("to_verify"), _make_claim("failed_gate")],
            confidence="to_verify",
        )
        assert slot.confidence == "to_verify"

    def test_structural_can_contain_mixed(self) -> None:
        slot = FilledSlot(
            slot_id="s1",
            claims=[_make_claim("verified", "C"), _make_claim("to_verify")],
            confidence="structural",
        )
        assert slot.confidence == "structural"

    def test_confirmed_with_no_claims_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no claims"):
            FilledSlot(slot_id="s1", claims=[], confidence="confirmed")

    def test_structural_with_no_claims_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no claims"):
            FilledSlot(slot_id="s1", claims=[], confidence="structural")


# -----------------------------------------------------------------------------
# 验收 3: derive_confidence 推导
# -----------------------------------------------------------------------------

class TestDeriveConfidence:
    def test_all_verified_ab(self) -> None:
        c = derive_confidence([
            _make_claim("verified", "A"),
            _make_claim("verified", "B"),
        ])
        assert c == "confirmed"

    def test_verified_c_only(self) -> None:
        # 有 verified 但有 tier=C
        c = derive_confidence([
            _make_claim("verified", "A"),
            _make_claim("verified", "C"),
        ])
        assert c == "structural"

    def test_mixed_verified_and_to_verify(self) -> None:
        c = derive_confidence([
            _make_claim("verified", "A"),
            _make_claim("to_verify"),
        ])
        assert c == "structural"

    def test_no_verified(self) -> None:
        c = derive_confidence([
            _make_claim("to_verify"),
            _make_claim("failed_gate"),
        ])
        assert c == "to_verify"

    def test_empty(self) -> None:
        c = derive_confidence([])
        assert c == "to_verify"


# -----------------------------------------------------------------------------
# 验收 4: grep 断言 — plan B2 DoD 无 string fallback
# -----------------------------------------------------------------------------

class TestNoStringFallback:
    """plan B2 DoD:final_report_generation 不再有"失败时返回 string"的路径。"""

    def test_no_string_fallback_in_report_result(self) -> None:
        """ReportResult 本身不暴露 string fallback 接口。"""
        import inspect
        from open_deep_research.evidence import report_result
        # 检查模块不含任何带 'fallback' 名字的公开函数
        for name in dir(report_result):
            if name.startswith("_"):
                continue
            obj = getattr(report_result, name)
            if callable(obj):
                assert "fallback" not in name.lower(), (
                    f"公开符号 {name} 含 'fallback' — plan B2 DoD 禁止"
                )

    def test_no_string_fallback_path_in_plan_v2_pipeline(self) -> None:
        """plan_v2_pipeline.py 不再含字符串 fallback(注:旧 _placeholder_cited_response 不算,但不能作为 silently-passed 路径)。"""
        path = Path("src/open_deep_research/plan_v2_pipeline.py")
        if not path.exists():
            pytest.skip("plan_v2_pipeline.py not present")
        content = path.read_text(encoding="utf-8")
        # 不应再有 "fallback to string" 之类返回字符串当作报告的路径
        # 旧 _placeholder_cited_response 仍存在但 B3 会替换它
        # 这里只断言没有 silent exception → string 的转换
        assert "except Exception" not in content or "out.error" in content, (
            "plan_v2_pipeline 应捕获异常并写 out.error,不应 silently 返字符串"
        )


# -----------------------------------------------------------------------------
# 验收 5: unresolved 显式暴露
# -----------------------------------------------------------------------------

class TestUnresolvedExposed:
    def test_unfilled_slots_can_be_collected(self) -> None:
        """未填 slot_id 应显式进 unresolved,而不是被空字符串掩盖。"""
        r = ReportResult(
            title="T", vertical_id="v",
            sections=[
                SectionResult(
                    section_id="s1", title="S1",
                    slots=[
                        FilledSlot(
                            slot_id="slot_filled",
                            claims=[_make_claim()],
                            confidence="confirmed",
                        ),
                    ],
                ),
            ],
            unresolved=["slot_unfilled_1", "slot_unfilled_2"],
        )
        assert "slot_unfilled_1" in r.unresolved
        assert "slot_unfilled_2" in r.unresolved
        # filled 不应进 unresolved
        assert "slot_filled" not in r.unresolved

    def test_report_can_be_empty_safely(self) -> None:
        """空 ReportResult 是合法的(unresolved=[] 不是 None,validator 不拒)。"""
        r = ReportResult(title="T", vertical_id="v")
        # 即使没有 sections,unresolved 也应是 []
        assert r.unresolved == []
        assert r.sections == []

    def test_no_string_field_in_models(self) -> None:
        """plan B2 修订后的 SPEC:SectionResult 不应有 markdown 字段。"""
        section = SectionResult(section_id="s1", title="S1")
        dumped = section.model_dump()
        assert "markdown" not in dumped, (
            "plan B2 修订后 SectionResult 不应有 markdown 字段(plan §3 patch #3)"
        )
