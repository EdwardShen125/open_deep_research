"""E1 · 诚实 pass(自动附录 + 警告 + 来源页)

按 plan E1:把你已有的"手写诚实声明"形式化成自动 pass。
从 claim 元数据和对账记录生成[待核实]清单、口径误区警告、逐章 + 主来源页。

接口:
    run_honest_pass(
        report: ReportResult,
        claims: list[ClaimV3],
        reconciliations: list[ReconciliationRecord],
    ) -> ReportResult

四件事:
    1. 主结论里每个 numeric claim tier≥B 且 verified,否则降级或标记
    2. 从所有 to_verify claim 生成[待核实]附录
    3. 从 D3 的 caliber_mismatch 记录生成"口径误区"段
    4. 从 claim 的 source_id/tier 渲染逐章 + 主来源页
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from open_deep_research.evidence.claim_v3 import ClaimV3
from open_deep_research.evidence.reconciliation import ReconciliationRecord
from open_deep_research.evidence.report_result import (
    Confidence,
    CrossCut,  # W9-L2
    ExecSummary,  # W10
    FilledSlot,
    ReportResult,
    SectionResult,
    SectionSummary,  # W9-L1
    derive_confidence,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

ALLOWED_TIERS_FOR_MAIN = ("A", "B")  # tier<B 不进主结论


# =============================================================================
# 1. 主结论门 — 数值 claim tier≥B 且 verified
# =============================================================================

def _downgrade_unverified_main_claims(report: ReportResult) -> ReportResult:
    """主结论里 numeric claim tier<B 或 not verified → confidence 降级。"""
    for section in report.sections:
        for slot in section.slots:
            for c in slot.claims:
                # 只对 numeric claim 强制
                if c.claim_type != "numeric":
                    continue
                if c.tier not in ALLOWED_TIERS_FOR_MAIN or c.verification_status != "verified":
                    # 降级:structural → to_verify;confirmed → structural
                    if slot.confidence == "confirmed":
                        slot.confidence = "structural"
                    elif slot.confidence == "structural":
                        slot.confidence = "to_verify"
    return report


# =============================================================================
# 2. [待核实]附录
# =============================================================================

def _build_unverified_appendix(
    claims: list[ClaimV3],
) -> list[dict]:
    """从所有"非 verified+ab"claim 生成 [待核实] 附录。

    触发条件(任一):
        - verification_status == "to_verify"
        - verification_status == "failed_gate"
        - tier 不在 A/B(tier=C/D 即使 verified 也进待核实)

    每条: {"value": str, "source": str, "tier": str, "reason": str}
    """
    items: list[dict] = []
    for c in claims:
        reasons: list[str] = []
        if c.verification_status == "to_verify":
            reasons.append("verification_status=to_verify")
        elif c.verification_status == "failed_gate":
            reasons.append("verification_status=failed_gate")
        if c.tier and c.tier not in ALLOWED_TIERS_FOR_MAIN:
            reasons.append(f"tier={c.tier} (低于 B,不入主结论)")

        if not reasons:
            continue

        items.append({
            "value": c.value,
            "source": c.source_url or c.source_id,
            "tier": c.tier,
            "verification_status": c.verification_status,
            "reason": "; ".join(reasons),
        })
    return items


# =============================================================================
# 3. 口径误区警告
# =============================================================================

def _build_caliber_mismatch_warnings(
    reconciliations: list[ReconciliationRecord],
) -> list[dict]:
    """从 D3 的 caliber_mismatch 记录生成"口径误区"段。"""
    warnings: list[dict] = []
    for rec in reconciliations:
        if rec.divergence_kind != "caliber_mismatch":
            continue
        warnings.append({
            "measurand": rec.measurand,
            "primary_caliber": rec.primary_caliber_id,
            "primary_value": rec.primary_value,
            "alternatives": [
                {
                    "caliber_id": a.get("caliber_id"),
                    "value": a.get("value"),
                    "why_not_comparable": a.get("why_not_comparable"),
                }
                for a in rec.alternatives
            ],
            "warning": (
                f"{rec.measurand}:primary 取 {rec.primary_caliber_id}="
                f"{rec.primary_value},其他 caliber 列 alternatives "
                f"(口径不可比,绝不取平均)"
            ),
        })
    return warnings


# =================================================================-
# 4. 来源页
# =================================================================-

def _build_source_pages(report: ReportResult) -> dict[str, list[dict]]:
    """逐章 + 主来源页(由 claim 元数据渲染,非手写)。

    Returns:
        {
            "by_section": [{section_id, title, sources: [...]}],
            "by_source": [{source_domain, tier, claim_count, sections: [...]}],
        }
    """
    by_section: list[dict] = []
    for section in report.sections:
        sources: list[dict] = []
        seen_domains: set[str] = set()
        for slot in section.slots:
            for c in slot.claims:
                if c.source_domain and c.source_domain not in seen_domains:
                    seen_domains.add(c.source_domain)
                    sources.append({
                        "domain": c.source_domain,
                        "url": c.source_url,
                        "tier": c.tier,
                        "title": c.source_title,
                    })
        by_section.append({
            "section_id": section.section_id,
            "title": section.title,
            "sources": sources,
        })

    # 主来源页:按 source_domain 聚合
    domain_agg: dict[str, dict] = defaultdict(lambda: {
        "claim_count": 0, "sections": set(), "tier": None, "title": None,
    })
    for section in report.sections:
        for slot in section.slots:
            for c in slot.claims:
                d = c.source_domain
                if not d:
                    continue
                agg = domain_agg[d]
                agg["claim_count"] += 1
                agg["sections"].add(section.section_id)
                # 取最高 tier(简易)
                if not agg["tier"] or (c.tier and _tier_rank(c.tier) < _tier_rank(agg["tier"])):
                    agg["tier"] = c.tier
                if not agg["title"] and c.source_title:
                    agg["title"] = c.source_title

    by_source = [
        {
            "domain": d,
            "tier": agg["tier"],
            "title": agg["title"],
            "claim_count": agg["claim_count"],
            "sections": sorted(agg["sections"]),
        }
        for d, agg in sorted(domain_agg.items(), key=lambda x: -x[1]["claim_count"])
    ]

    return {"by_section": by_section, "by_source": by_source}


def _tier_rank(tier: Optional[str]) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(tier or "D", 3)


# =============================================================================
# §6 · 任务书:综合层溯源校验 + structural_judgment 标注校验
# =============================================================================


def _build_unsourced_synthesis_audit(
    section_summaries: list[SectionSummary],
    crosscuts: list[CrossCut],
    exec_summary: Optional[ExecSummary],
) -> list[dict]:
    """§6 · 任务书:综合层每句须有 source_*_ids 支撑,否则进未溯源清单。

    非交涉 1(任务书 §0.1):
      "综合层每一句要么溯源到 claim/slot,要么显式标注为'结构性判断'。
       无引用注水 = 毁掉整条流水线的护城河。"

    实现策略(粗粒度启发式,非全 NLP):
      - 句切分(中英文句号 / 问号)
      - 每句至少有:一个 [slot_id] / [source_...] 引用 → OK
                    一个 "结构性判断"/"structural_judgment" 标 → OK
                    否则 → 进未溯源清单
    """
    audit: list[dict] = []

    def _audit_sentences(scope: str, md: str, sources: list[str]) -> None:
        import re
        sents = re.split(r"[。！？!?\.]+", md)
        for s in sents:
            s = s.strip()
            if not s or len(s) < 4:
                continue
            # 显式溯源:[xxx] 形式 → OK
            has_inline_ref = bool(re.search(r"\[[a-z_0-9\-]+\]", s, re.IGNORECASE))
            # 显式标结构性判断
            has_structural = (
                "结构性判断" in s or
                "[structural_judgment]" in s.lower() or
                "structural_judgment" in s.lower()
            )
            # 显式列 source_section_ids → OK(整篇套用)
            if not has_inline_ref and not has_structural and not sources:
                audit.append({
                    "scope": scope,
                    "sentence": s[:200],
                    "issue": "no source / no structural_judgment marker",
                })

    for sm in section_summaries:
        _audit_sentences(
            f"section_summary:{sm.section_id}",
            sm.summary_md,
            sm.source_slot_ids,
        )
    for cc in crosscuts:
        # crosscut 已有 is_structural_judgment 字段 → 跳过检测
        if cc.is_structural_judgment:
            continue
        _audit_sentences(
            f"crosscut:{cc.crosscut_id}",
            cc.md,
            cc.source_section_ids,
        )
    if exec_summary is not None:
        for kp in exec_summary.key_points:
            _audit_sentences(
                f"exec_summary:key_point",
                kp,
                exec_summary.source_section_ids,
            )
    return audit


def _collect_structural_judgments(
    crosscuts: list[CrossCut],
) -> list[dict]:
    """§6 · 任务书:渲染端校验 — is_structural_judgment=True 必在正文显式标'结构性判断'。"""
    items: list[dict] = []
    for cc in crosscuts:
        if not cc.is_structural_judgment:
            continue
        if "结构性判断" not in cc.md and "structural_judgment" not in cc.md.lower():
            items.append({
                "crosscut_id": cc.crosscut_id,
                "issue": (
                    "is_structural_judgment=True but md lacks '结构性判断' marker — "
                    "render guard requires explicit marking"
                ),
            })
    return items


def run_honest_pass(
    report: ReportResult,
    claims: list[ClaimV3],
    reconciliations: list[ReconciliationRecord],
    *,
    section_summaries: Optional[list[SectionSummary]] = None,
    crosscuts: Optional[list[CrossCut]] = None,
    exec_summary: Optional[ExecSummary] = None,
) -> ReportResult:
    """终检 pass(plan E1 + 任务书 §6 扩展)。返回新的 ReportResult,honest_pass 字段填好。

    任务书 §6 扩展:接 L1/L2/L3 综合层字段,做溯源审计 + structural_judgment 标注审计。
    """
    # 1. 主结论门
    report = _downgrade_unverified_main_claims(report)

    # 2. [待核实]附录
    unverified = _build_unverified_appendix(claims)

    # 3. 口径误区
    caliber_warnings = _build_caliber_mismatch_warnings(reconciliations)

    # 4. 来源页
    source_pages = _build_source_pages(report)

    # 5. §6 · 综合层溯源审计(L1 / L2 / L3)
    section_summaries = section_summaries if section_summaries is not None else report.section_summaries
    crosscuts = crosscuts if crosscuts is not None else report.crosscuts
    exec_summary = exec_summary if exec_summary is not None else report.exec_summary

    unsourced = _build_unsourced_synthesis_audit(
        section_summaries, crosscuts, exec_summary,
    )
    structural_items = _collect_structural_judgments(crosscuts)

    # 构造 honest_pass dict(plan E1 + §6)
    honest_pass = {
        "unverified_appendix": unverified,
        "caliber_warnings": caliber_warnings,
        "source_pages": source_pages,
        # §6 任务书扩展字段
        "unsourced_synthesis": unsourced,
        "structural_judgments": structural_items,
        "stats": {
            "unverified_count": len(unverified),
            "caliber_mismatch_count": len(caliber_warnings),
            "section_count": len(report.sections),
            "unique_source_domains": len(source_pages["by_source"]),
            "unsourced_synthesis_count": len(unsourced),
            "structural_judgment_count": len(structural_items),
        },
    }
    report.honest_pass = honest_pass
    return report


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ALLOWED_TIERS_FOR_MAIN",
    "run_honest_pass",
    "_build_unverified_appendix",
    "_build_caliber_mismatch_warnings",
    "_build_source_pages",
    "_build_unsourced_synthesis_audit",  # §6
    "_collect_structural_judgments",  # §6
]
