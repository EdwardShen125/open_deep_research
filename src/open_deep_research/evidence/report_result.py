"""B2 · ReportResult 模型(替换 string fallback)

按 plan B2:上次 final_report_generation 用字符串 fallback,
失败时 fallback 文本冒充成功报告。改成结构化模型,失败即结构性可见。

接口:
    FilledSlot:
        slot_id
        claims: list[ClaimV3]
        confidence: "confirmed" | "structural" | "to_verify"
        校验:confidence="confirmed" 时不得含 to_verify claim

    SectionResult:
        section_id
        title
        slots: list[FilledSlot]
        (渲染产物 markdown 由 B3 / E1 输出端构造,不进 schema)

    ReportResult:
        title
        vertical_id
        sections: list[SectionResult]
        unresolved: list[str]  # 未填/待核实槽,显式暴露
        honest_pass: Optional[dict]  # E1 填

禁止:
    - 任何 fallback 路径返回 unresolved=["all_filled"] 假装成功
    - "confirmed" 槽内含 to_verify claim(validator 拒)
    - markdown 字段(渲染耦合,plan B2 未要求)
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from open_deep_research.evidence.claim_v3 import ClaimV3


# =============================================================================
# Literal type aliases
# =============================================================================

Confidence = Literal["confirmed", "structural", "to_verify"]


# =============================================================================
# FilledSlot
# =============================================================================

class FilledSlot(BaseModel):
    """一个已填的槽(plan B2)。

    confidence 语义:
        - "confirmed" : 该槽内所有 claim 都是 verified + tier >= B
        - "structural": 槽有数据,但至少一个 claim 是 to_verify 或 tier < B
                        (报告里需降级表述或加注)
        - "to_verify" : 槽内没有 verified claim(全部 to_verify / failed_gate)
    """

    slot_id: str = Field(min_length=1, max_length=128)
    claims: list[ClaimV3] = Field(default_factory=list)
    confidence: Confidence

    @model_validator(mode="after")
    def _confirmed_no_unverified(self) -> "FilledSlot":
        """plan B2 DoD:'confirmed' 槽内不得含 to_verify claim(plan B2 validator 强制)。"""
        if self.confidence == "confirmed":
            bad = [c for c in self.claims if c.verification_status != "verified"]
            if bad:
                raise ValueError(
                    f"FilledSlot {self.slot_id!r} confidence='confirmed' "
                    f"but contains {len(bad)} non-verified claim(s); "
                    f"first bad claim status: {bad[0].verification_status}"
                )
        return self

    @model_validator(mode="after")
    def _nonempty_claims_for_confidence(self) -> "FilledSlot":
        if not self.claims and self.confidence in ("confirmed", "structural"):
            raise ValueError(
                f"FilledSlot {self.slot_id!r} confidence='{self.confidence}' "
                f"but has no claims"
            )
        return self


# =============================================================================
# SectionResult
# =============================================================================

class SectionResult(BaseModel):
    """一个 section 的填槽结果。

    注:plan B2 未定义 markdown 字段,渲染产物由 B3/E1 在输出端构造,不进 schema。
    """

    section_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    slots: list[FilledSlot] = Field(default_factory=list)


# =============================================================================
# ReportResult
# =============================================================================

class ReportResult(BaseModel):
    """最终报告结构化产物(plan B2)。

    unresolved 永远存在(plan DoD:未填/待核实槽显式暴露,不沉默)。
    即使没有未填槽,unresolved=[] 也保留字段(validator 不拒)。
    """

    title: str = Field(min_length=1, max_length=500)
    vertical_id: str = Field(min_length=1, max_length=64)
    sections: list[SectionResult] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)  # 未填/待核实 slot_id 列表
    honest_pass: Optional[dict] = None  # E1 填

    @model_validator(mode="after")
    def _unresolved_always_present(self) -> "ReportResult":
        # plan DoD:unresolved 必须存在,缺失 = validator 拒
        # Pydantic default_factory=list 已保证字段存在;此校验仅为显式语义
        if self.unresolved is None:
            raise ValueError("ReportResult.unresolved must always be present (use [] if none)")
        return self

    def to_unresolved_summary(self) -> dict[str, int]:
        """返回未填/待核实槽数(供诊断)。"""
        return {
            "unresolved_count": len(self.unresolved),
            "section_count": len(self.sections),
            "filled_slot_count": sum(len(s.slots) for s in self.sections),
        }


# =============================================================================
# Helper: confidence 推导(供 B3 调用)
# =============================================================================

def derive_confidence(claims: list[ClaimV3]) -> Confidence:
    """根据槽内 claim 推导 confidence。

    规则:
        - 空 → to_verify(兜底)
        - 所有 claim verified 且 tier ≥ B → "confirmed"
        - 至少一个 verified → "structural"(可能含 to_verify,需要降级表述)
        - 没有 verified claim → "to_verify"
    """
    if not claims:
        return "to_verify"
    verified = [c for c in claims if c.verification_status == "verified"]
    if not verified:
        return "to_verify"
    # 至少有一个 verified,但可能有 to_verify
    if all(c.verification_status == "verified" and c.tier in ("A", "B") for c in claims):
        return "confirmed"
    return "structural"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Confidence",
    "FilledSlot",
    "SectionResult",
    "ReportResult",
    "derive_confidence",
]
