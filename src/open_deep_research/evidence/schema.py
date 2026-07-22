"""Phase 3 (= Runbook v1 阶段 1): EvidenceUnit + Claim 双层 Pydantic schema.

设计依据: notes/evidence-pipeline-runbook-v1.md 1.1 节。

双层模型:
    EvidenceUnit (EU): 单源原子观察。一条 EU = 一个来源说的一件事。
                      不可变,只增。闸 1/2/3 (阶段 2) 的入口。
    Claim:            跨源归并后的结论。报告只消费 Claim,不消费 EU。

迁移注意:
- 与现有 src/open_deep_research/evidence_units.py 中的 dataclass 关系:
    - dataclass EvidenceUnit (v1) 保留,作为 in-process / legacy 兼容层
    - Pydantic EvidenceUnitV2 (本文件) 是 PG 持久化 / 跨模块传输的"硬"schema
- v1 dataclass 字段到 v2 Pydantic 的映射:
    claim     → claim (语义同)
    quote     → source_span (语义:不再 ≤200 字符截断,改为完整逐字片段)
    numbers[] → norm_value / unit (单列存储,冲突靠结构化字段检测)
    entities[]→ entities (同)
    confidence→ (删除;改由 span_verified + numeric_drift + entailment_verdict 联合决定)
    extraction_method → extractor_model (string 命名变)
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


# -----------------------------------------------------------------------------
# Literal type aliases (与 migrations/002_*.sql 列定义对齐)
# -----------------------------------------------------------------------------
ClaimType = Literal["numeric", "event", "attribute", "relation", "opinion"]
SourceTier = Literal["primary", "secondary", "tertiary", "ugc"]
Verdict = Literal["entailed", "partial", "contradicted", "unverifiable"]
Grade = Literal["A", "B", "C", "D"]


# =============================================================================
# EvidenceUnit v2
# =============================================================================

class EvidenceUnitV2(BaseModel):
    """单源原子观察:一条 EU = 一个来源说的一件事。不做跨源合并。

    不可变。写入 PG 后 id 立即生成;后续任何字段变更都应通过
    `eu_dao.update_verification()` 之类的窄通道,不直接 mutate。
    """

    eu_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    dimension_id: Optional[str] = Field(
        None,
        description="阶段 1 允许 None(下游 EU 不知 dimension);阶段 3 强制非空",
    )

    claim: str = Field(
        max_length=300,
        description="自足陈述句,不含指代,主语写全称",
    )
    claim_type: ClaimType
    entities: list[str] = Field(default_factory=list)

    # 数值单列存储:冲突检测靠结构化字段,不靠文本比对
    norm_value: Optional[Decimal] = None
    unit: Optional[str] = None
    value_as_of: Optional[date] = Field(
        None, description="数据所属时点,非发布时点",
    )

    source_url: str
    source_domain: str
    source_title: Optional[str] = None
    published_at: Optional[datetime] = None
    source_tier: SourceTier

    source_span: str = Field(
        min_length=10,
        description="原文逐字片段。阶段 2 闸 1 校验它是否在 source_url 的正文中",
    )
    span_start: Optional[int] = Field(None, ge=0)
    span_end: Optional[int] = Field(None, ge=0)

    extractor_model: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 阶段 2 三道闸的回填结果
    span_verified: bool = False
    numeric_drift: bool = False
    entailment_verdict: Optional[Verdict] = None
    entailment_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    # 阶段 3 归并后回填
    claim_id: Optional[UUID] = None

    # 跨运行 dedup 锚(对应旧 dataclass 的 content_hash;可选)
    content_hash: Optional[str] = Field(None, min_length=64, max_length=64)

    # -------------------------------------------------------------------------
    # 校验器
    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def _numeric_requires_value(self) -> "EvidenceUnitV2":
        if self.claim_type == "numeric" and self.norm_value is None:
            raise ValueError("numeric claim 必须提供 norm_value")
        return self

    @model_validator(mode="after")
    def _span_bounds_consistent(self) -> "EvidenceUnitV2":
        if self.span_start is not None and self.span_end is not None:
            if self.span_end < self.span_start:
                raise ValueError("span_end 必须 ≥ span_start")
        if self.span_start is not None and self.span_start != len(self.source_span) - 1:
            # 仅在 source_span 是原始片段(不是截断副本)时检查;
            # 阶段 1 允许 None / 不一致,留给阶段 2 闸 1 修正。
            pass
        return self

    # -------------------------------------------------------------------------
    # 派生属性
    # -------------------------------------------------------------------------
    @property
    def usable(self) -> bool:
        """是否可被归并 / 写入报告。

        Runbook 阶段 1.1 定义:span_verified && !numeric_drift &&
        entailment_verdict in ('entailed', 'partial')。
        """
        return (
            self.span_verified
            and not self.numeric_drift
            and self.entailment_verdict in ("entailed", "partial")
        )

    @property
    def has_span_offsets(self) -> bool:
        return self.span_start is not None and self.span_end is not None

    # -------------------------------------------------------------------------
    # (反)序列化
    # -------------------------------------------------------------------------
    def to_pg_row(self) -> dict[str, Any]:
        """转换为 PG INSERT 参数字典(None 字段显式保留,NULL 入库)。

        注意:UUID 列在 psycopg/SQLAlchemy 中通常会自动适配 str/UUID,
        这里统一转 str 以避免 asyncpg + 自定义类型踩坑。
        """
        return {
            "eu_id": str(self.eu_id),
            "run_id": str(self.run_id),
            "dimension_id": self.dimension_id,
            "claim": self.claim,
            "claim_type": self.claim_type,
            "entities": list(self.entities),
            "norm_value": self.norm_value,
            "unit": self.unit,
            "value_as_of": self.value_as_of,
            "source_url": self.source_url,
            "source_domain": self.source_domain,
            "source_title": self.source_title,
            "published_at": self.published_at,
            "source_tier": self.source_tier,
            "source_span": self.source_span,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "extractor_model": self.extractor_model,
            "extracted_at": self.extracted_at,
            "span_verified": self.span_verified,
            "numeric_drift": self.numeric_drift,
            "entailment_verdict": self.entailment_verdict,
            "entailment_score": self.entailment_score,
            "claim_id": str(self.claim_id) if self.claim_id else None,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_pg_row(cls, row: dict[str, Any]) -> "EvidenceUnitV2":
        """从 PG SELECT 行构造。空字符串视为 None(避免 numeric 列解析失败)。"""
        clean: dict[str, Any] = dict(row)
        for k in ("dimension_id", "source_title", "unit", "claim_id",
                  "entailment_verdict", "content_hash"):
            if clean.get(k) == "":
                clean[k] = None
        # UUID 列
        for k in ("eu_id", "run_id", "claim_id"):
            v = clean.get(k)
            if isinstance(v, str):
                try:
                    clean[k] = UUID(v)
                except ValueError:
                    clean[k] = None
        return cls(**clean)


# =============================================================================
# Claim
# =============================================================================

class ClaimV2(BaseModel):
    """跨源归并后的结论。报告只消费 Claim,不消费 EU。

    写入 PG 后由归并算法 (阶段 3) 负责生成。Phase 1 schema 只定义形状,
    生成逻辑后续阶段。
    """

    claim_id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    dimension_id: str

    canonical_claim: str
    claim_type: ClaimType
    entities: list[str] = Field(default_factory=list)

    norm_value: Optional[Decimal] = None
    unit: Optional[str] = None
    value_as_of: Optional[date] = None
    value_spread: Optional[float] = Field(
        None, ge=0.0, description="各源数值最大相对偏差",
    )

    eu_count: int = Field(ge=1)
    independent_source_count: int = Field(ge=0)
    primary_source_count: int = Field(ge=0)
    earliest_published_at: Optional[datetime] = None

    has_conflict: bool = False
    conflicting_values: list[dict[str, Any]] = Field(default_factory=list)

    grade: Grade
    grade_reason: str

    @model_validator(mode="after")
    def _grade_d_requires_no_entailed(self) -> "ClaimV2":
        # D 级只能用于"无可用 EU",eu_count 必须为 0
        # (虽然 Field(ge=1) 强制 ≥1,所以 D 级不可能存在;留作 Phase 3 前的 sanity check)
        if self.grade == "D" and self.eu_count > 0:
            raise ValueError("D 级 claim 的 eu_count 必须为 0")
        if self.grade != "D" and self.eu_count == 0:
            raise ValueError(f"{self.grade} 级 claim 必须至少 1 个 EU")
        return self

    def to_pg_row(self) -> dict[str, Any]:
        return {
            "claim_id": str(self.claim_id),
            "run_id": str(self.run_id),
            "dimension_id": self.dimension_id,
            "canonical_claim": self.canonical_claim,
            "claim_type": self.claim_type,
            "entities": list(self.entities),
            "norm_value": self.norm_value,
            "unit": self.unit,
            "value_as_of": self.value_as_of,
            "value_spread": self.value_spread,
            "eu_count": self.eu_count,
            "independent_source_count": self.independent_source_count,
            "primary_source_count": self.primary_source_count,
            "earliest_published_at": self.earliest_published_at,
            "has_conflict": self.has_conflict,
            "conflicting_values": self.conflicting_values,
            "grade": self.grade,
            "grade_reason": self.grade_reason,
        }

    @classmethod
    def from_pg_row(cls, row: dict[str, Any]) -> "ClaimV2":
        clean: dict[str, Any] = dict(row)
        for k in ("unit", "value_spread", "earliest_published_at"):
            if clean.get(k) == "":
                clean[k] = None
        for k in ("claim_id", "run_id"):
            v = clean.get(k)
            if isinstance(v, str):
                clean[k] = UUID(v)
        return cls(**clean)


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "ClaimType",
    "SourceTier",
    "Verdict",
    "Grade",
    "EvidenceUnitV2",
    "ClaimV2",
]