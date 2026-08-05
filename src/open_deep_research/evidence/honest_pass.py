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
    FilledSlot,
    ReportResult,
    SectionResult,
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
# 公开 API
# =============================================================================

def run_honest_pass(
    report: ReportResult,
    claims: list[ClaimV3],
    reconciliations: list[ReconciliationRecord],
) -> ReportResult:
    """终检 pass(plan E1)。返回新的 ReportResult,honest_pass 字段填好。"""
    # 1. 主结论门
    report = _downgrade_unverified_main_claims(report)

    # 2. [待核实]附录
    unverified = _build_unverified_appendix(claims)

    # 3. 口径误区
    caliber_warnings = _build_caliber_mismatch_warnings(reconciliations)

    # 4. 来源页
    source_pages = _build_source_pages(report)

    # 构造 honest_pass dict(plan E1:从元数据渲染,不手写)
    honest_pass = {
        "unverified_appendix": unverified,
        "caliber_warnings": caliber_warnings,
        "source_pages": source_pages,
        "stats": {
            "unverified_count": len(unverified),
            "caliber_mismatch_count": len(caliber_warnings),
            "section_count": len(report.sections),
            "unique_source_domains": len(source_pages["by_source"]),
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
]
