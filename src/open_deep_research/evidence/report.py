"""阶段 5 = Runbook v1 阶段 5.1-5.3: ReportResult 结构化输出。

设计依据: notes/evidence-pipeline-runbook-v1.md 阶段 5 节。

P0 整改:fallback 伪装成功。
原行为: writer LLM 失败时,把错误信息写到 final_report (str),调用方以为是合法调研报告。
整改: ReportResult.ok 字段硬信号 — ok=False 时调用方一定知道是 degraded / failed。

设计:
- ReportResult 是顶级结果容器
- ReportSection 是单 section(每个 claim / 一段消化 EU digest)
- ResearchStatus 枚举: ok / partial / fallback_used / failed
- 渲染层仍用 markdown 字串向后兼容,但 ok / status / failures 都在结构化字段里
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ResearchStatus:Stage 输出语义
# - ok: writer 全程成功,EU digest 没有被用过
# - partial: writer 部分成功 / 用了 EU digest 兜底但内容完整
# - fallback_used: writer 失败,靠 _render_eu_digest 兜底(EU digest 仍可读)
# - failed: 严重错误(write_section 也失败,或上游 stage 没跑过)
ResearchStatus = Literal["ok", "partial", "fallback_used", "failed"]


class ClaimStats(BaseModel):
    """Claim / EU 统计 metadata。

    阶段 5.3:把 EU metrics 一等公民化 — ReportResult 必须带。
    """
    total_eus: int = 0
    usable_eus: int = 0          # 通过闸 1+2+3 验证
    rejected_eus: int = 0         # 没通过闸而被拒
    total_claims: int = 0
    primary_claims: int = 0       # A grade
    secondary_claims: int = 0     # B grade
    tertiary_claims: int = 0      # C grade
    unverified_claims: int = 0    # D grade
    grade_dist_pct: dict[str, float] = Field(default_factory=dict)  # {"A": 0.2, "B": 0.3, ...}
    unique_sources: int = 0
    unique_primary_sources: int = 0
    has_conflict: int = 0         # has_conflict=True 的 claim 数

    @classmethod
    def from_claim_list(
        cls,
        claims: list[Any],
        *,
        eus: Optional[list[Any]] = None,
    ) -> "ClaimStats":
        """从 claims 列表推导 stats。

        claims 是 ClaimV2 列表(或同构字段的 dataclass)。
        eus 是可选 EvidenceUnitV2 列表 — 用于计算 EU 维度统计。
        """
        dist: dict[str, int] = {}
        indep_total = 0
        prim_total = 0
        conflict = 0
        for c in claims:
            grade = getattr(c, "grade", None) or "D"
            dist[grade] = dist.get(grade, 0) + 1
            if getattr(c, "has_conflict", False):
                conflict += 1
            indep_total += getattr(c, "independent_source_count", 0) or 0
            prim_total += getattr(c, "primary_source_count", 0) or 0

        total_c = sum(dist.values()) or 1  # 防 0 除
        grade_pct = {g: round(c / total_c, 3) for g, c in dist.items()}

        # EU 维度
        total_eus = 0
        usable_eus = 0
        rejected_eus = 0
        unique_sources: set[str] = set()
        unique_primary_sources: set[str] = set()
        if eus:
            total_eus = len(eus)
            for eu in eus:
                if getattr(eu, "usable", False):
                    usable_eus += 1
                u = getattr(eu, "source_url", None)
                if u:
                    unique_sources.add(u)
                tier = getattr(eu, "source_tier", None)
                if u and tier == "primary":
                    unique_primary_sources.add(u)

        # rejected = 总数 - 可用 - (unknown/未跑闸)
        # 这里保守估算:total - usable = rejected (假定没拒绝的是 usable)
        if total_eus:
            rejected_eus = max(0, total_eus - usable_eus)

        return ClaimStats(
            total_eus=total_eus,
            usable_eus=usable_eus,
            rejected_eus=rejected_eus,
            total_claims=len(claims),
            primary_claims=dist.get("A", 0),
            secondary_claims=dist.get("B", 0),
            tertiary_claims=dist.get("C", 0),
            unverified_claims=dist.get("D", 0),
            grade_dist_pct=grade_pct,
            unique_sources=len(unique_sources),
            unique_primary_sources=len(unique_primary_sources),
            has_conflict=conflict,
        )


class ReportSection(BaseModel):
    """单 section — 一个 topic / 一个 claim / 一段 EU digest。"""
    section_id: str
    title: str
    body_markdown: str = Field(default="", description="本 section 的 markdown 正文")
    claim_ids: list[str] = Field(default_factory=list, description="本 section 引用的 Claim ID")
    eu_ids: list[str] = Field(default_factory=list, description="本 section 引用的 EU ID")
    grade: Optional[Literal["A", "B", "C", "D"]] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class Failure(BaseModel):
    """记录 stage 失败信息(用于 partial / failed 状态诊断)。"""
    stage: str
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReportResult(BaseModel):
    """顶层调研结果 — 替换 final_report: str。

    关键字段:
    - ok: 硬信号。True = 写报告成功(可能用了 fallback 但仍有完整输出);False = 严重失败
    - status: 详细分类(ok / partial / fallback_used / failed)
    - body_markdown: 最终报告的 markdown 字串(向后兼容 final_report 字段)
    - sections: 结构化 sections (按 dimension / claim 拆分)
    - claim_stats: EU / Claim 统计 metadata
    - failures: 失败列表(每个失败 stage + error)
    """
    # ---- 硬信号 ----
    ok: bool
    status: ResearchStatus

    # ---- 内容 ----
    body_markdown: str = Field(default="", description="拼起来当作 final_report 用,向后兼容")
    sections: list[ReportSection] = Field(default_factory=list)

    # ---- 元信息 ----
    run_id: Optional[str] = None
    research_brief: Optional[str] = None
    claim_stats: Optional[ClaimStats] = None

    # ---- 错误链 ----
    failures: list[Failure] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # ---- 时间戳 ----
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_duration_ms: Optional[float] = None

    @classmethod
    def from_markdown_and_status(
        cls,
        body: str,
        *,
        status: ResearchStatus,
        sections: Optional[list[ReportSection]] = None,
        claim_stats: Optional[ClaimStats] = None,
        failures: Optional[list[Failure]] = None,
        warnings: Optional[list[str]] = None,
        run_id: Optional[str] = None,
        research_brief: Optional[str] = None,
        pipeline_duration_ms: Optional[float] = None,
    ) -> "ReportResult":
        """便利构造: 根据 status 自动算 ok。"""
        ok = status in ("ok", "partial", "fallback_used")
        return cls(
            ok=ok,
            status=status,
            body_markdown=body,
            sections=sections or [],
            claim_stats=claim_stats,
            failures=failures or [],
            warnings=warnings or [],
            run_id=run_id,
            research_brief=research_brief,
            pipeline_duration_ms=pipeline_duration_ms,
        )

    @property
    def has_failures(self) -> bool:
        return bool(self.failures)

    @property
    def fallback_used(self) -> bool:
        return self.status in ("partial", "fallback_used")

    def to_markdown_with_warnings(self) -> str:
        """渲染 body + 顶部 + 尾部 warnings(给 operator 看的诊断信息)。

        调用方拿到 ok=True 仍可只 body_markdown;
        拿到 ok=False 仍能看见 body(可能为空)+ failures 诊断。
        """
        parts: list[str] = []
        if self.warnings:
            parts.append("## ⚠️ Warnings\n")
            for w in self.warnings:
                parts.append(f"- {w}")
            parts.append("")
        if self.failures:
            parts.append("## ❌ Failures\n")
            for f in self.failures:
                parts.append(
                    f"- **{f.stage}** [{f.error_type}]: {f.error_message}"
                )
            parts.append("")
        if self.claim_stats is not None:
            s = self.claim_stats
            parts.append("## 📊 Evidence Stats\n")
            parts.append(f"- Total Claims: {s.total_claims} (A:{s.primary_claims} B:{s.secondary_claims} C:{s.tertiary_claims} D:{s.unverified_claims})")
            parts.append(f"- Total EUs: {s.total_eus} (usable: {s.usable_eus}, rejected: {s.rejected_eus})")
            parts.append(f"- Unique sources: {s.unique_sources} (primary: {s.unique_primary_sources})")
            parts.append(f"- Has conflict: {s.has_conflict}")
            parts.append("")
        parts.append(self.body_markdown)
        return "\n".join(parts)


def is_report_success(result: ReportResult) -> bool:
    """给上层(higher level)用:只看 status 是不是 ok。

    失败/兜底时返回 False — 调用方必须显式处理。
    """
    return result.ok and result.status == "ok"
