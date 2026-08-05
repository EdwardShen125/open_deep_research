"""C1 · 源意图路由

按 plan C1:不同槽该去不同类源(市场规模→具名研究机构+官方统计;
法规→.gov;财务→filings)。把 slot 的 expected_claim_type + caliber 映射成
"源意图 + 查询策略"。

接口:
    SourceIntent:
        target_source_types: list[str]   # 期望源类型
        query_list: list[str]            # 生成的查询,**前 N 条强制 site-scoped**

    route_sources(slot, *, registry, calibers) -> SourceIntent

查询生成规则(plan C1 "site-scoped 优先"形式化):
    1. target_source_types = slot.expected_claim_type 映射表
    2. site-scoped 前 N 条:N = len(target_source_types)
       从 F1 注册表筛该 vertical 下 source_type ∈ target 的所有 A/B 级源,
       逐源生成 query_template.format(topic=..., year=..., domain=domain)
    3. 兜底:F1 中该 metric_type 的广义 query_templates(无 site-scoped),排后段
    4. 未知 claim_type → 只走通用模板,query_list 全无 site-scoped
       (明示"无定向意图")

示例:
    market_size slot → target=['research_house','official']
    → query_list 前 2 条: site:emarketer.com / site:coresight.com ...
    → query_list 后段: {topic} market size {year}(广义)

"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from open_deep_research.evidence.caliber_registry import CaliberRegistry
from open_deep_research.evidence.framework import Slot
from open_deep_research.evidence.source_registry import (
    MetricType,
    SourceRegistry,
    SourceType,
)


logger = logging.getLogger(__name__)


# =============================================================================
# claim_type → target_source_types 映射表(plan C1)
# =============================================================================

# plan C1 期望:
#   market_size / penetration / share → research_house / official
#   regulation                       → official
#   financial / m_and_a              → official
#   trends / gmv_estimate            → research_house / industry_media
#   sku_rank                         → crawler_estimate / industry_media

_CLAIM_TYPE_TO_SOURCES: dict[str, list[str]] = {
    "quantitative_market": ["research_house", "official"],
    "quantitative_share": ["research_house", "official"],
    "quantitative_penetration": ["research_house", "official"],
    "regulation": ["official"],
    "financial": ["official"],
    "m_and_a": ["official", "research_house"],
    "trends": ["research_house", "industry_media", "mainstream_media"],
    "gmv_estimate": ["crawler_estimate", "industry_media"],
    "sku_rank": ["crawler_estimate", "industry_media"],
    "comparative": ["research_house", "official"],
    "attribute": ["research_house", "mainstream_media", "industry_media"],
    "event": ["mainstream_media", "industry_media", "official"],
    "qualitative": ["research_house", "mainstream_media", "industry_media"],
}


def claim_type_to_target_sources(claim_type: str) -> list[str]:
    """slot.expected_claim_type → target_source_types 列表。

    未知 claim_type → [] (明示"无定向意图",query_list 全无 site-scoped)
    """
    # 简化:quantitative_xxx 子型用同一组
    key = claim_type
    if claim_type == "quantitative":
        key = "quantitative_market"  # 默认 quantitative = market_size 类
    return list(_CLAIM_TYPE_TO_SOURCES.get(key, []))


# =============================================================================
# SourceIntent
# =============================================================================

class SourceIntent(BaseModel):
    """路由产物。

    target_source_types 在前,query_list 在后(强制顺序)。
    query_list 前 N 条 site-scoped(N = len(target_source_types))。
    """

    target_source_types: list[str] = Field(default_factory=list)
    query_list: list[str] = Field(default_factory=list)

    def site_scoped_count(self) -> int:
        return sum(1 for q in self.query_list if "site:" in q)

    def first_n_site_scoped(self, n: int) -> list[str]:
        """前 N 条强制 site-scoped。"""
        return [q for q in self.query_list[:n] if "site:" in q]


# =============================================================================
# Query template instantiation
# =============================================================================

def _format_query(template: str, *, topic: str, year: Optional[int], domain: Optional[str]) -> str:
    """format query template;domain=None 时跳过 site: 部分。"""
    try:
        if domain:
            return template.format(topic=topic, year=year or "", domain=domain)
        else:
            # 模板含 site:{domain} 时去掉
            return template.replace("site:{domain}", "").strip().format(topic=topic, year=year or "")
    except (KeyError, IndexError):
        return template


def _extract_topic(slot: Slot) -> tuple[str, Optional[int]]:
    """从 slot.question 中提取 topic 和 year(简化版)。"""
    import re
    q = slot.question
    # 找 4 位年份
    year_match = re.search(r"\b(20\d{2}|19\d{2})\b", q)
    year = int(year_match.group(1)) if year_match else None
    # topic:去掉年份和问号,首词
    topic = re.sub(r"\b(20\d{2}|19\d{2})\b", "", q).rstrip("?.").strip()
    return topic or "topic", year


# =============================================================================
# route_sources
# =============================================================================

def route_sources(
    slot: Slot,
    *,
    registry: SourceRegistry,
    calibers: Optional[CaliberRegistry] = None,
) -> SourceIntent:
    """plan C1:slot → SourceIntent(site-scoped 强制前 N 条)。

    Args:
        slot: 目标槽
        registry: F1 源注册表
        calibers: F2 口径注册表(可选,用于 caliber_id 命中特定源类型)
    """
    target_types = claim_type_to_target_sources(slot.expected_claim_type)
    if not target_types:
        # 未知 claim_type → 无定向意图(明示),query_list 全无 site-scoped
        templates = _fallback_templates(slot, registry)
        return SourceIntent(
            target_source_types=[],
            query_list=templates,
        )

    # 1. site-scoped 前 N 条
    topic, year = _extract_topic(slot)
    site_scoped_queries: list[str] = []
    target_type_set = set(target_types)

    # 从 F1 筛 source_type ∈ target_types 的所有 A/B 级源
    for source_type in target_types:
        matching_sources = registry.sources_by_type(source_type, min_tier="B")  # type: ignore[arg-type]
        # 取 metric_type 匹配的优先,否则取全部
        if slot.caliber_id:
            # caliber → metric_type 推断(简化:用 slot.expected_claim_type)
            metric_match = [s for s in matching_sources if _matches_metric(s, slot, registry)]
            if metric_match:
                matching_sources = metric_match

        # 取每个 source_type 至少 1 个代表源(避免 query_list 过长)
        for s in matching_sources[:2]:  # 每 source_type 至多 2 个
            template = _pick_template(slot, registry)
            q = _format_query(template, topic=topic, year=year, domain=s.domain)
            if "site:" in q:
                site_scoped_queries.append(q)

    # 2. 兜底:广义 query templates(无 site-scoped)
    fallback_queries = _fallback_templates(slot, registry)

    # 3. 拼装:site-scoped 前 N 条(N = len(target_types)) + 兜底
    #    如果 site_scoped 比 N 多,只取前 N
    n = len(target_types)
    site_scoped_first_n = site_scoped_queries[:n]

    return SourceIntent(
        target_source_types=target_types,
        query_list=site_scoped_first_n + fallback_queries,
    )


def _matches_metric(source, slot: Slot, registry: SourceRegistry) -> bool:
    """简化:source.covers 含 slot 隐含的 metric_type。"""
    if not slot.caliber_id:
        return True
    # slot.expected_claim_type='quantitative' → 隐含 market_size 类
    inferred_metric = "market_size"
    if slot.expected_claim_type == "regulation":
        inferred_metric = "regulation"
    elif slot.expected_claim_type == "financial":
        inferred_metric = "financial"
    elif slot.expected_claim_type == "trends":
        inferred_metric = "trends"
    return inferred_metric in source.covers


def _pick_template(slot: Slot, registry: SourceRegistry) -> str:
    """取 F1 中适配该 metric_type 的模板。"""
    metric = "market_size"  # 默认
    if slot.expected_claim_type == "regulation":
        metric = "regulation"
    elif slot.expected_claim_type == "financial":
        metric = "financial"
    elif slot.expected_claim_type == "m_and_a":
        metric = "m_and_a"
    elif slot.expected_claim_type == "trends":
        metric = "trends"
    templates = registry.get_templates(metric)  # type: ignore[arg-type]
    if templates:
        # 优先 site-scoped 模板
        for t in templates:
            if "site:" in t:
                return t
        return templates[0]
    # 兜底:通用模板
    return "{topic} {year}"


def _fallback_templates(slot: Slot, registry: SourceRegistry) -> list[str]:
    """取 F1 中不带 site-scoped 的通用模板。"""
    metric = "market_size"
    if slot.expected_claim_type == "regulation":
        metric = "regulation"
    elif slot.expected_claim_type == "financial":
        metric = "financial"
    elif slot.expected_claim_type == "m_and_a":
        metric = "m_and_a"
    elif slot.expected_claim_type == "trends":
        metric = "trends"
    templates = registry.get_templates(metric)  # type: ignore[arg-type]
    fallback = [t for t in templates if "site:" not in t]
    return fallback or ["{topic} {year}"]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SourceIntent",
    "route_sources",
    "claim_type_to_target_sources",
]
