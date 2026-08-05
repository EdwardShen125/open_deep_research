"""Track D · 对账(皇冠差异化)

D1 + D2 + D3 一并实现,共享 Pydantic 模型。

设计原则(plan 硬约束):
    - 跨口径绝不取中位数/平均(plan D3 DoD)
    - 不同 period 不聚(D1 DoD)
    - 同一 metric + entity + time 但 caliber 不同 → 1 簇(交给 D2 区分)
    - 独立源数:同 origin_chain / 共享措辞 → 降权(D2)

接口:
    cluster_claims(claims) -> list[ClaimCluster]
    detect_caliber_divergence(cluster) -> Literal["caliber_mismatch","data_conflict","clean"]
    independence(cluster) -> int
    reconcile_cluster(cluster, *, calibers=None) -> ReconciliationRecord
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from open_deep_research.evidence.claim_v3 import ClaimV3, Tier


logger = logging.getLogger(__name__)


# =============================================================================
# D1 · ClaimCluster
# =============================================================================

ClusterKey = tuple[str, str, str, str]  # (metric_type, entity, time_window, caliber_id)


class ClaimCluster(BaseModel):
    """一个聚簇(plan D1)。

    字段:
        cluster_id:  簇 id
        key:         (metric_type, entity, time_window, caliber_id)
        claims:      簇内 ClaimV3
    """

    cluster_id: str = Field(min_length=1, max_length=128)
    key: ClusterKey
    claims: list[ClaimV3] = Field(default_factory=list)

    @property
    def measurand(self) -> str:
        return f"{self.key[0]} of {self.key[1]} ({self.key[2]})"


def _cluster_key(c: ClaimV3) -> ClusterKey:
    """构造聚簇 key(plan D1:必须含 caliber_id)。

    metric_type: c.claim_type 或 'unknown'(未分类)
    entity: c.source_domain 或 'unknown'
    time_window: c.value_as_of.year (或 'unknown')
    caliber_id: c.caliber_id 或 'unknown'
    """
    metric_type = getattr(c, "claim_type", None) or "unknown"
    entity = c.source_domain or "unknown"
    if c.value_as_of:
        time_window = str(c.value_as_of.year if isinstance(c.value_as_of, datetime) else c.value_as_of.year)
    else:
        time_window = "unknown"
    caliber_id = c.caliber_id or "unknown"
    return (metric_type, entity, time_window, caliber_id)


def cluster_claims(claims: list[ClaimV3]) -> list[ClaimCluster]:
    """按 (metric_type, entity, time_window, caliber_id) 聚簇(plan D1)。

    注:这是简化版(纯 key 匹配,不依赖 embedding 模糊合并)。
    Embedding 模糊合并由现有 evidence/merge.py 处理。
    """
    groups: dict[ClusterKey, list[ClaimV3]] = defaultdict(list)
    for c in claims:
        groups[_cluster_key(c)].append(c)

    clusters: list[ClaimCluster] = []
    for i, (key, members) in enumerate(groups.items(), start=1):
        cluster_id = f"cluster_{i:04d}"
        clusters.append(ClaimCluster(
            cluster_id=cluster_id,
            key=key,
            claims=members,
        ))
    return clusters


# =============================================================================
# D2 · 口径分歧检测 + 来源独立性
# =============================================================================

DriftKind = Literal["caliber_mismatch", "data_conflict", "clean"]

DRIFT_THRESHOLD = 0.05


def detect_caliber_divergence(
    cluster: ClaimCluster,
    *,
    drift_threshold: float = DRIFT_THRESHOLD,
) -> DriftKind:
    """plan D2:判定簇内分歧类型。

    判定规则:
        - cluster 内不同 caliber_id → 标 caliber_mismatch
          (但 plan D1 key 已含 caliber,所以同 cluster 同 caliber;
           此函数假设 key 是允许跨 caliber 合并后的,即 cluster 实际可能含不同 caliber_id 的 claim)
        - 数值超 drift 阈值 且 caliber 不同 → caliber_mismatch
        - caliber 相同却差 → data_conflict
        - 否则 → clean
    """
    if not cluster.claims:
        return "clean"

    # 数值化:只取 norm_value 不为 None 的
    numeric_claims = [c for c in cluster.claims if c.norm_value is not None]
    if len(numeric_claims) < 2:
        return "clean"  # 单条无冲突

    # 1. 不同 caliber → caliber_mismatch
    calibers = {c.caliber_id for c in numeric_claims}
    if len(calibers - {None, "unknown"}) > 1:
        return "caliber_mismatch"

    # 2. 同 caliber 但数值差 → data_conflict
    values = [float(c.norm_value) for c in numeric_claims if c.norm_value is not None]
    if not values:
        return "clean"
    abs_max = max(abs(v) for v in values)
    if abs_max == 0:
        return "clean"
    spread = (max(values) - min(values)) / abs_max
    if spread > drift_threshold:
        return "data_conflict"

    return "clean"


def independence(
    cluster: ClaimCluster,
    *,
    text_sim_threshold: float = 0.9,
) -> int:
    """plan D2:独立源数(降权同源 chain)。

    规则:
        - origin_chain 共享:同 chain 算 1 个独立源
        - 共享措辞:text_sim > 0.9 → 算同源(简化:整段包含)
        - _attribution_chain 默认 None,不参与判定(plan D2 严格限 3 条)
    """
    if not cluster.claims:
        return 0

    # 简化版:按 source_domain + origin_chain
    seen_chains: set[tuple[str, ...]] = set()
    unique: set[str] = set()
    for c in cluster.claims:
        chain_key = tuple(c.origin_chain) if c.origin_chain else (c.source_domain,)
        if chain_key in seen_chains:
            continue
        # 共享措辞(简化):value 子串
        if any(c.value[:30] in other.value for other in cluster.claims
               if other.source_id != c.source_id):
            continue
        seen_chains.add(chain_key)
        unique.add(c.source_domain or "unknown")
    return len(unique)


# =============================================================================
# D3 · ReconciliationRecord(锁死 · 皇冠)
# =============================================================================

class ReconciliationRecord(BaseModel):
    """plan D3:对账记录(跨口径绝不取平均)。

    divergence_kind:
        - "caliber_mismatch": 不同 caliber,primary + alternatives
        - "data_conflict": 同 caliber 矛盾,需 LLM 推理
        - "clean": 无冲突
    """

    measurand: str = Field(min_length=1, max_length=500)
    primary_value: float
    primary_source_id: str = Field(min_length=1, max_length=2048)
    primary_tier: Tier
    primary_caliber_id: str = Field(min_length=1, max_length=128)
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    independence_note: str = Field(default="")
    confidence: float = Field(ge=0.0, le=1.0)
    divergence_kind: DriftKind = "clean"

    @model_validator(mode="after")
    def _no_averaging(self) -> "ReconciliationRecord":
        """plan D3 DoD:防止 upstream 不小心取平均。

        规则:primary_value == alternatives 均值时(误差 1e-6),报错。
        """
        if not self.alternatives:
            return self
        vals = [
            a.get("value") for a in self.alternatives
            if isinstance(a.get("value"), (int, float))
        ]
        if len(vals) < 2:
            return self
        mean = sum(vals) / len(vals)
        if abs(self.primary_value - mean) < 1e-6:
            raise ValueError(
                "primary_value equals mean of alternatives — "
                "suggests averaging across calibers (forbidden)"
            )
        return self


def reconcile_cluster(
    cluster: ClaimCluster,
    *,
    calibers: Optional[Any] = None,
) -> ReconciliationRecord:
    """plan D3:从 cluster 产出 ReconciliationRecord。

    策略:
        1. detect_caliber_divergence 判定分歧类型
        2. 按 caliber 选 primary(calibers 注册表提供 canonical 优先级;否则按 tier A→D)
        3. 其他 caliber 进 alternatives 标 "why_not_comparable"
        4. confidence 按 independence(tier 高的独立源多 → confidence 高)
    """
    if not cluster.claims:
        raise ValueError("empty cluster cannot be reconciled")

    divergence = detect_caliber_divergence(cluster)

    # 按 caliber 分组
    by_caliber: dict[str, list[ClaimV3]] = defaultdict(list)
    for c in cluster.claims:
        cal = c.caliber_id or "unknown"
        by_caliber[cal].append(c)

    # 选 primary caliber:优先 A/B tier 的 claim;同 tier 取先出现的
    primary_caliber: Optional[str] = None
    primary_claim: Optional[ClaimV3] = None
    for cal_id, members in by_caliber.items():
        # 取该 caliber 内 best(verified + tier 最高)
        for c in members:
            if primary_claim is None:
                primary_claim = c
                primary_caliber = cal_id
            elif _rank_tier(c.tier) < _rank_tier(primary_claim.tier):
                primary_claim = c
                primary_caliber = cal_id

    if primary_claim is None or primary_caliber is None:
        raise ValueError("could not determine primary claim")

    # 构造 alternatives(异 caliber 进)
    alternatives: list[dict[str, Any]] = []
    for cal_id, members in by_caliber.items():
        if cal_id == primary_caliber:
            continue
        for c in members:
            if c.norm_value is None:
                continue
            alternatives.append({
                "value": float(c.norm_value),
                "caliber_id": cal_id,
                "source_id": c.source_id,
                "tier": c.tier,
                "why_not_comparable": (
                    f"口径 {cal_id} 与 primary {primary_caliber} 不可比:"
                    f" excludes/includes 定义不同"
                ),
            })

    # independence
    indep_count = independence(cluster)
    independence_note = (
        f"簇内独立源 {indep_count} 个(按 source_domain + origin_chain 去重)"
    )

    # confidence:独立源多 + tier 高 → 高
    avg_tier = sum(_rank_tier(c.tier) for c in cluster.claims if c.tier) / len(cluster.claims)
    # tier A=0,B=1,C=2,D=3 → 越低越好;反转:confidence = 1 - avg/3
    confidence = max(0.0, min(1.0, 1.0 - (avg_tier / 3.0) + (indep_count - 1) * 0.05))

    if primary_claim.norm_value is None:
        raise ValueError("primary claim has no norm_value")

    return ReconciliationRecord(
        measurand=cluster.measurand,
        primary_value=float(primary_claim.norm_value),
        primary_source_id=primary_claim.source_id,
        primary_tier=primary_claim.tier or "D",  # type: ignore[arg-type]
        primary_caliber_id=primary_caliber,
        alternatives=alternatives,
        independence_note=independence_note,
        confidence=confidence,
        divergence_kind=divergence,
    )


def _rank_tier(tier: Optional[str]) -> int:
    return {"A": 0, "B": 1, "C": 2, "D": 3}.get(tier or "D", 3)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ClusterKey",
    "ClaimCluster",
    "DriftKind",
    "ReconciliationRecord",
    "cluster_claims",
    "detect_caliber_divergence",
    "independence",
    "reconcile_cluster",
]
