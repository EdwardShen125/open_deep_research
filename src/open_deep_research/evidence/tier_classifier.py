"""A2 · tier 分类器(替换域名白名单机械降级)

按 plan A2(硬教训 a):分级必须按"认知权威"判定,
**注册表命中优先**,未命中按信号兜底且强制 log warning(防退化)。

接口:
    classify_tier(source_url, source_domain, *, registry=None, log_unmatched=True)
    -> Optional[Tier]   # A/B/C/D 或 None

行为:
    1. F1 注册表命中 → 直接返回 tier(唯一权威路径)
    2. 未命中 → 用 evidence.independence.classify_source_tier 做信号判定 → 映射:
         primary/secondary/tertiary/ugc → A/B/C/D
    3. 信号兜底命中时强制 logger.warning("unmatched source, signal-tier fallback")
       (log_unmatched=True 默认;关闭仅供 batch 静默测试)

不删:
    evidence.independence.classify_source_tier / upgrade_source_tier 保留,
    仅作信号输入,不被 A2 主路径直接调。
"""
from __future__ import annotations

import logging
from typing import Optional

from open_deep_research.evidence.source_registry import (
    SourceRegistry,
    Tier,
    load_registry,
)


logger = logging.getLogger(__name__)


# =============================================================================
# 信号映射:旧 source_tier → 新 tier(A/B/C/D)
# =============================================================================

# primary/secondary/tertiary/ugc 映射:
#   primary  → A  (具名研究机构 / .gov / IR / SEC / 法院 / 监管)
#   secondary→ B  (老牌主流媒体 / 行业垂直 / 券商研报)
#   tertiary → C  (聚合 / 爬虫估算 / 厂商二手)
#   ugc      → D  (自媒体 / UGC)

_SOURCE_TIER_TO_TIER = {
    "primary": "A",
    "secondary": "B",
    "tertiary": "C",
    "ugc": "D",
}


def _signal_tier_to_tier(signal: str) -> Optional[Tier]:
    """信号兜底:旧 source_tier 字符串 → 新 A/B/C/D Tier Literal。"""
    return _SOURCE_TIER_TO_TIER.get(signal)  # type: ignore[return-value]


# =============================================================================
# 公开 API
# =============================================================================

def classify_tier(
    source_url: str,
    source_domain: str,
    *,
    registry: Optional[SourceRegistry] = None,
    log_unmatched: bool = True,
) -> Optional[Tier]:
    """按认知权威分级。

    严格两步:
        1. F1 注册表命中 → 取 tier(唯一权威路径)
        2. 未命中 → 信号兜底(evidence.independence.classify_source_tier)
           + logger.warning("unmatched source, signal-tier fallback: ...")
           **除非显式 log_unmatched=False**

    Args:
        source_url:  完整 URL,用于 F1 注册表命中
        source_domain: 域,fallback 时用于 classify_source_tier
        registry:    F1 注册表实例;None 时按需 load("us_livecommerce")
                     (生产环境应显式传入,避免每次重新读 YAML)
        log_unmatched: True(默认)= 信号兜底时记 warning;
                       False= 静默(仅 batch 测试场景)

    Returns:
        Tier (A/B/C/D) 或 None(全部失败时)
    """
    # ---- Step 1: F1 注册表命中(唯一权威路径) ----
    if registry is None:
        # 默认 vertical;调用方应显式传 registry
        try:
            registry = load_registry("us_livecommerce")
        except FileNotFoundError:
            registry = None

    if registry is not None:
        tier = registry.get_tier(source_url) or registry.get_tier(source_domain)
        if tier is not None:
            return tier

    # ---- Step 2: 信号兜底(plan 硬教训 a 的临时出口) ----
    # 用 evidence.independence 的旧白名单做信号判定
    from open_deep_research.evidence.independence import classify_source_tier

    signal = classify_source_tier(source_domain or source_url)
    tier = _signal_tier_to_tier(signal)

    if tier is not None:
        if log_unmatched:
            # plan 硬教训 (a) 的形式化:兜底命中必触发 warning,留"被人工接管"出口
            logger.warning(
                "unmatched source, signal-tier fallback: domain=%s signal=%s -> tier=%s; "
                "consider adding to F1 registry (data/registry/sources/<vertical>.yaml)",
                source_domain or source_url, signal, tier,
            )
        return tier

    # 完全无法判定
    if log_unmatched:
        logger.warning(
            "unmatched source, no signal: domain=%s; cannot classify tier",
            source_domain or source_url,
        )
    return None


# =============================================================================
# Batch helper(显式禁用 log,供大批量静默分类)
# =============================================================================

def classify_tier_batch(
    sources: list[tuple[str, str]],
    *,
    registry: Optional[SourceRegistry] = None,
) -> list[Optional[Tier]]:
    """批量分类。log_unmatched 强制 False,调用方自己处理 unknown。"""
    return [
        classify_tier(url, domain, registry=registry, log_unmatched=False)
        for url, domain in sources
    ]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "classify_tier",
    "classify_tier_batch",
    "Tier",
]
