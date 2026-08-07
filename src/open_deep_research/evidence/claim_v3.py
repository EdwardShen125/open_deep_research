"""A1 · ClaimV3(单源原子观察 + 验证元数据)

按 plan A1:扩展 Pydantic + 写 PG migration。
"Claim" 在 plan 语义下实际上是"单源原子观察 + 验证状态",
对应现有 EvidenceUnitV2 的扩展形态——不是跨源归并结论(那是 ClaimV2 / D3 ReconciliationRecord)。

新增字段:
    tier:                 认知权威(A/B/C/D),A2 填
    caliber_id:           口径 id(F2 注册表),C2 填
    verification_status:  verified / to_verify / failed_gate
    gate_results:         {span, drift, entail} 三门结果
    origin_chain:         来源链(供 D2 独立性)
    embedding_model:      embedding 版本列(非运行时路由,只记录)

与 EvidenceUnitV2 的关系:
    ClaimV3.from_eu(eu) 把 EU 字段映射过来,verification_status='to_verify' 默认。
    现有 12,598 EU 不丢,新列 nullable + default,回填由 default 满足。
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from open_deep_research.evidence.schema import EvidenceUnitV2


# =============================================================================
# Literal type aliases
# =============================================================================

Tier = Literal["A", "B", "C", "D"]
VerificationStatus = Literal["verified", "to_verify", "failed_gate"]
EntailVerdict = Literal["entailed", "contradicted", "unknown"]


# =============================================================================
# GateResults
# =============================================================================

class GateResults(BaseModel):
    """A3a + A3b 三门结果聚合。

    span:  A3a 字面命中(bool;None=未跑)
    drift: A3a 数值漂移(float;0.0=完美匹配,None=未跑)
    entail:A3b LLM 蕴含(entailed/contradicted/unknown;None=未跑)
    """

    span: Optional[bool] = None
    drift: Optional[float] = Field(default=None, ge=0.0)
    entail: Optional[EntailVerdict] = None


# =============================================================================
# ClaimV3
# =============================================================================

class ClaimV3(BaseModel):
    """单源原子观察 + 验证元数据(plan 所谓 'Claim')。

    用于:
        - B2 报告填槽(plan B2 FilledSlot.claims: list[ClaimV3])
        - D1 claim 聚簇
        - D2 来源独立性(origin_chain)
        - E1 诚实过(verification_status + tier)
    """

    # ---- 核心值 ----
    value: str = Field(min_length=1, max_length=2000)
    source_id: str = Field(min_length=1, max_length=2048)
    eu_id: Optional[UUID] = None

    # ---- 计划 A1 要求字段 ----
    tier: Optional[Tier] = None
    caliber_id: Optional[str] = Field(default=None, max_length=128)
    verification_status: VerificationStatus = "to_verify"
    gate_results: GateResults = Field(default_factory=GateResults)
    origin_chain: list[str] = Field(default_factory=list)
    embedding_model: Optional[str] = Field(default=None, max_length=64)

    # ---- 配套元数据(报告渲染需要) ----
    source_url: str = Field(default="", max_length=2048)
    source_domain: str = Field(default="", max_length=253)
    source_title: Optional[str] = Field(default=None, max_length=500)
    published_at: Optional[datetime] = None
    claim_type: Literal["numeric", "event", "attribute", "relation", "opinion"] = "attribute"
    norm_value: Optional[float] = None
    unit: Optional[str] = Field(default=None, max_length=32)
    value_as_of: Optional[datetime] = None

    # R9 新增:metric_type(market_size/share/regulation 等标准化键)与 entity 主语
    # 给 reconciliation.cluster_key 用(替代之前用 claim_type / source_domain 的弱代理)。
    metric_type: Optional[str] = Field(
        default=None, max_length=32,
        description="标准化 metric key: market_size/share/regulation/...",
    )
    entity: Optional[str] = Field(
        default=None, max_length=128,
        description="EU.entities[0] 的主语(对账 entity 维度)",
    )

    # ---- D2 预留(plan D2 严格限 3 条;本字段默认 None 不参与判定) ----
    _attribution_chain: Optional[list[str]] = None

    @model_validator(mode="after")
    def _numeric_requires_norm_value(self) -> "ClaimV3":
        if self.claim_type == "numeric" and self.norm_value is None:
            raise ValueError("numeric claim must have norm_value")
        return self

    @model_validator(mode="after")
    def _failed_gate_consistency(self) -> "ClaimV3":
        # failed_gate 时 entail 必须是 contradicted
        if self.verification_status == "failed_gate":
            if self.gate_results.entail != "contradicted":
                raise ValueError(
                    "failed_gate requires gate_results.entail == 'contradicted'"
                )
        return self

    # -----------------------------------------------------------------
    # 构造
    # -----------------------------------------------------------------

    @classmethod
    def from_eu(
        cls,
        eu: EvidenceUnitV2,
        *,
        tier: Optional[Tier] = None,
        caliber_id: Optional[str] = None,
        verification_status: Optional[VerificationStatus] = None,
        origin_chain: Optional[list[str]] = None,
        embedding_model: Optional[str] = None,
    ) -> "ClaimV3":
        """从现有 EU 构造 ClaimV3。

        默认 verification_status='to_verify',其余 plan 字段默认 None / 空。
        调用方(A2 / A3 / D)负责后续填入。
        """
        # 把 EU 现有的 span_verified / numeric_drift / entailment_verdict 映射到 gate_results
        # EU verdict ∈ {entailed, partial, contradicted, unverifiable}
        # GateResults.entail ∈ {entailed, contradicted, unknown}
        entail_mapped: Optional[EntailVerdict]
        if eu.entailment_verdict == "entailed":
            entail_mapped = "entailed"
        elif eu.entailment_verdict == "partial":
            entail_mapped = "entailed"  # 部分蕴含 → 视为蕴含
        elif eu.entailment_verdict == "contradicted":
            entail_mapped = "contradicted"
        elif eu.entailment_verdict == "unverifiable":
            entail_mapped = "unknown"
        else:
            entail_mapped = None
        gate = GateResults(
            span=eu.span_verified if eu.span_verified else None,
            drift=None,  # EU 没存 drift 数值
            entail=entail_mapped,
        )
        # norm_value: EU 是 Decimal,ClaimV3 是 float
        norm_value: Optional[float] = None
        if eu.norm_value is not None:
            try:
                norm_value = float(eu.norm_value)
            except (TypeError, ValueError):
                norm_value = None

        # verification_status 推断:三门都过 → verified
        inferred_vs: VerificationStatus = verification_status or "to_verify"
        if verification_status is None and eu.usable:
            inferred_vs = "verified"
        elif verification_status is None and eu.entailment_verdict == "contradicted":
            inferred_vs = "failed_gate"

        return cls(
            value=eu.claim,
            source_id=str(eu.eu_id) if eu.eu_id else eu.source_url,
            eu_id=eu.eu_id,
            tier=tier,
            caliber_id=caliber_id,
            verification_status=inferred_vs,
            gate_results=gate,
            origin_chain=list(origin_chain or []),
            embedding_model=embedding_model,
            source_url=eu.source_url,
            source_domain=eu.source_domain,
            source_title=eu.source_title,
            published_at=eu.published_at,
            claim_type=eu.claim_type,
            norm_value=norm_value,
            unit=eu.unit,
            value_as_of=datetime.combine(eu.value_as_of, datetime.min.time()) if eu.value_as_of else None,
            # R9:从 V2 EU 透传 metric_type / entity 主语
            metric_type=eu.metric_type,
            entity=(eu.entities[0] if eu.entities else None),
        )

    # -----------------------------------------------------------------
    # PG 序列化(对齐 EU 表的列)
    # -----------------------------------------------------------------

    def to_pg_update_dict(self) -> dict[str, Any]:
        """把 ClaimV3 转为 PG UPDATE 参数字典(只含 A1 新加的 6 列)。"""
        import json
        return {
            "tier": self.tier,
            "caliber_id": self.caliber_id,
            "verification_status": self.verification_status,
            "gate_results": json.dumps(
                {
                    "span": self.gate_results.span,
                    "drift": self.gate_results.drift,
                    "entail": self.gate_results.entail,
                },
                ensure_ascii=False,
            ),
            "origin_chain": list(self.origin_chain),
            "embedding_model": self.embedding_model,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Tier",
    "VerificationStatus",
    "EntailVerdict",
    "GateResults",
    "ClaimV3",
]
