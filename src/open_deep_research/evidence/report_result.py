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
    """一个 section 的填槽结果(L0 填槽层)。

    注:plan B2 未定义 markdown 字段,渲染产物由 B3/E1 在输出端构造,不进 schema。

    W9-L1 扩展:本结构每个 FilledSlot 自带 slot_id 与 claim ids(ClaimV3.value
    作 claim_id 兜底),综合层溯源全靠它。
    """

    section_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    slots: list[FilledSlot] = Field(default_factory=list)

    def all_slot_ids(self) -> list[str]:
        return [s.slot_id for s in self.slots]

    def all_claim_ids(self) -> list[str]:
        # ClaimV3 没有 stable id 字段;用 (source_url, value, tier) 元组 dedup,
        # 综合层溯源到 "claim 引用点" 而非 claim_id 字符串。
        ids: list[str] = []
        for s in self.slots:
            for c in s.claims:
                ids.append(f"{c.source_url}::{c.value[:40]}")
        return ids


# =============================================================================
# W9-L1: SectionSummary — 章节综合层(L1)
# =============================================================================


class SectionSummary(BaseModel):
    """L1 章节小结层。

    综合自 L0 一节的所有 FilledSlot,**不重新拉 EU 池**。
    summary_md 必须显式标 "(summarizing N slots)" 或嵌入 source_slot_ids。

    is_stub 字段:W9/W10 兜底路径标记 — LLM 返空时为 True,plan_v2_pipeline
    据此设 degraded=True / passed=False(任务书 §0.2 不糊弄)。
    """

    section_id: str = Field(min_length=1, max_length=128)
    summary_md: str = Field(min_length=1, max_length=20000)
    source_slot_ids: list[str] = Field(default_factory=list)
    key_claim_ids: list[str] = Field(default_factory=list)
    is_stub: bool = False

    @model_validator(mode="after")
    def _source_slots_nonempty(self) -> "SectionSummary":
        if not self.source_slot_ids:
            raise ValueError(
                f"SectionSummary {self.section_id!r} must declare source_slot_ids "
                "(W9-L1 non-negotiable: every summary must trace to slots)"
            )
        return self


# =============================================================================
# W9-L2: CrossCut — 跨切综合层(L2)
# =============================================================================

CrossCutKind = Literal["comparison_table", "analysis"]


class CrossCut(BaseModel):
    """L2 跨切综合层。

    kind=comparison_table:必须挂 reconciliation_ids(口径感知,跨口径标"不可比")
    kind=analysis:可挂 structural_judgment 标(超出 claim 支撑的定论必须显式标注)
    """

    crosscut_id: str = Field(min_length=1, max_length=128)
    kind: CrossCutKind
    md: str = Field(min_length=1, max_length=20000)
    source_section_ids: list[str] = Field(default_factory=list)
    reconciliation_ids: list[str] = Field(default_factory=list)
    is_structural_judgment: bool = False
    is_stub: bool = False  # W9/W10 兜底路径标记(L2 LLM 返空时为 True)

    @model_validator(mode="after")
    def _kind_invariants(self) -> "CrossCut":
        # 非交涉 1:comparison_table 必须挂对账记录;analysis 标注要走 structural_judgment
        if self.kind == "comparison_table" and not self.reconciliation_ids:
            raise ValueError(
                f"CrossCut {self.crosscut_id!r} kind='comparison_table' "
                "must declare reconciliation_ids (caliber-aware invariant)"
            )
        if self.is_structural_judgment and "结构性判断" not in self.md and "structural" not in self.md.lower():
            raise ValueError(
                f"CrossCut {self.crosscut_id!r} is_structural_judgment=True "
                "must explicitly mark '结构性判断' in md (render-time guard)"
            )
        return self


# =============================================================================
# W10: ExecSummary — 执行摘要层(L3)
# =============================================================================


class ExecSummary(BaseModel):
    """L3 执行摘要。

    整篇报告的最顶层产物。生成在 L1/L2 之后,只能从摘要合成。
    """

    one_liner: str = Field(min_length=1, max_length=1000)
    key_points: list[str] = Field(default_factory=list)
    source_section_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _source_sections_nonempty(self) -> "ExecSummary":
        if not self.source_section_ids:
            raise ValueError(
                "ExecSummary must declare source_section_ids "
                "(W10: every key_point must trace back)"
            )
        if not self.key_points:
            raise ValueError(
                "ExecSummary must have at least one key_point"
            )
        return self


# =============================================================================
# ReportResult
# =============================================================================


class ReportResult(BaseModel):
    """最终报告结构化产物(plan B2 + W9/W10 扩展)。

    unresolved 永远存在(plan DoD:未填/待核实槽显式暴露,不沉默)。
    即使没有未填槽,unresolved=[] 也保留字段(validator 不拒)。

    W9/W10 扩展:加 L1 / L2 / L3 三层综合层字段 — 渲染端按
    [exec_summary, sections(L0), section_summaries(L1), crosscuts(L2)] 顺序输出。
    """

    title: str = Field(min_length=1, max_length=500)
    vertical_id: str = Field(min_length=1, max_length=64)
    sections: list[SectionResult] = Field(default_factory=list)
    # L1 章节小结
    section_summaries: list[SectionSummary] = Field(default_factory=list)
    # L2 跨切(对照表 + 分析)
    crosscuts: list[CrossCut] = Field(default_factory=list)
    # L3 执行摘要(整篇最前)
    exec_summary: Optional[ExecSummary] = None
    unresolved: list[str] = Field(default_factory=list)  # 未填/待核实 slot_id 列表
    honest_pass: Optional[dict] = None  # E1 填
    # 长报告装配产物(由 plan_v2_pipeline L5 装配步骤写入)
    assembled_markdown: Optional[str] = None
    degraded: bool = False  # W5 降级保留

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
