"""C2 · 两级检索 + 每槽预算

按 plan C2:治"EU 堆积病"——先定向命中 A 级源,广度爬只兜底;
每槽 EU 有硬上界(要每槽最好的几条,不是几千条)。
tier 在落库时经 A2 赋值。

接口:
    MAX_EU_PER_SLOT: int = 30 (可由 env 调整)
    acquire(slot, *, registry, search_provider=None, classify_tier_fn=None) -> list[ClaimV3]

流程:
    1. 跑 C1 定向查询 → A/B 命中
    2. 不足 → 广义兜底(由 search_provider.execute 跑所有 query)
    3. MAX_EU_PER_SLOT 截断
    4. 每条经 A2 赋 tier(由 classify_tier_fn)

背压:每槽 EU ≤ 30
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from open_deep_research.evidence.claim_v3 import ClaimV3
from open_deep_research.evidence.framework import Slot
from open_deep_research.evidence.source_registry import SourceRegistry, Tier
from open_deep_research.evidence.source_router import SourceIntent, route_sources


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_MAX_EU_PER_SLOT = 30


def _get_max_eu_per_slot() -> int:
    env = os.environ.get("OPEN_DEEP_RESEARCH_MAX_EU_PER_SLOT")
    if env and env.isdigit():
        return int(env)
    return DEFAULT_MAX_EU_PER_SLOT


# =============================================================================
# Search provider interface
# =============================================================================

class SearchProvider:
    """极简 search provider 接口(供测试 mock / SearXNG / Tavily)。

    execute(query) -> list[dict]:
        返回 [{"url": ..., "title": ..., "snippet": ...}, ...]
    """

    def execute(self, query: str) -> list[dict[str, Any]]:
        raise NotImplementedError


class InMemorySearchProvider(SearchProvider):
    """测试用:预置 query -> URLs 的映射。"""

    def __init__(self, mapping: Optional[dict[str, list[dict[str, Any]]]] = None):
        self._mapping = mapping or {}

    def execute(self, query: str) -> list[dict[str, Any]]:
        q_lower = query.lower()
        for key, results in self._mapping.items():
            k_lower = key.lower()
            if k_lower in q_lower or q_lower in k_lower:
                return list(results)
        return []


# =============================================================================
# acquire
# =============================================================================

def acquire(
    slot: Slot,
    *,
    registry: SourceRegistry,
    calibers: Optional[Any] = None,
    search_provider: Optional[SearchProvider] = None,
    classify_tier_fn: Optional[Callable[..., Optional[Tier]]] = None,
    max_eu: Optional[int] = None,
) -> list[ClaimV3]:
    """plan C2:定向 → 兜底 → 截断 → 赋 tier。

    Args:
        slot: 目标槽
        registry: F1 源注册表
        calibers: F2 口径注册表(可选)
        search_provider: 检索提供方;None 时用 InMemorySearchProvider(空)
        classify_tier_fn: tier 分类器;None 时 evidence.tier_classifier.classify_tier
        max_eu: 硬上界;None 时读 env 或默认 30
    """
    if max_eu is None:
        max_eu = _get_max_eu_per_slot()
    if search_provider is None:
        search_provider = InMemorySearchProvider()
    if classify_tier_fn is None:
        from open_deep_research.evidence.tier_classifier import classify_tier
        classify_tier_fn = classify_tier  # type: ignore[assignment]

    # 1. C1 定向查询(intent.query_list 已 format,直接喂 search provider)
    intent = route_sources(slot, registry=registry, calibers=calibers)

    all_results: list[dict[str, Any]] = []
    for q in intent.query_list:
        results = search_provider.execute(q)
        all_results.extend(results)

    # 2. 去重 by URL
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in all_results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            deduped.append(r)

    # 3. 截断
    truncated = deduped[:max_eu]

    if len(deduped) > max_eu:
        logger.info(
            "acquire: slot=%s truncated %d → %d (MAX_EU_PER_SLOT=%d)",
            slot.slot_id, len(deduped), len(truncated), max_eu,
        )

    # 4. 赋 tier + 构造 ClaimV3
    claims: list[ClaimV3] = []
    for r in truncated:
        url = r.get("url", "")
        domain = _extract_domain(url)
        tier = classify_tier_fn(url, domain, registry=registry)
        claim = ClaimV3(
            value=r.get("title", "") or r.get("snippet", "")[:200] or url,
            source_id=url,
            source_url=url,
            source_domain=domain,
            source_title=r.get("title"),
            tier=tier,  # type: ignore[arg-type]
            verification_status="to_verify",  # 默认 to_verify,后续 A3 跑门
        )
        claims.append(claim)

    return claims


def _extract_domain(url: str) -> str:
    from urllib.parse import urlsplit
    host = (urlsplit(url).hostname or "").lower()
    # 去掉 www. 前缀(让 F1 get_tier 的 eTLD+1 匹配生效)
    if host.startswith("www."):
        host = host[4:]
    return host


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DEFAULT_MAX_EU_PER_SLOT",
    "SearchProvider",
    "InMemorySearchProvider",
    "acquire",
]
