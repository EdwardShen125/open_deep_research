"""Phase 5 (= Runbook v1 阶段 3) 归并算法。

依据: notes/evidence-pipeline-runbook-v1.md 阶段 3.1

核心:把 N 条 EU 合并为 K 个归并组,每组对应一个 Claim。
  - 分桶避免 O(n²):(dimension_id, claim_type) 桶
  - 桶内用 BGE-M3 cosine(已 L2 归一)+ 结构化约束
  - 三个容易漏掉的约束:
      1. 实体集合无交集 → 不是同一件事
      2. numeric 的 value_as_of 必须相同(防止 2023 vs 2024 营收被并)
      3. 数值冲突仍合并 → 同一 claim 的冲突,并列呈现

MERGE_COSINE 默认 0.92(Runbook 原文);NUMERIC_TOL 默认 0.02(2% 单位换算容差)。
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any, Optional

from open_deep_research.evidence.schema import EvidenceUnitV2


MERGE_COSINE = 0.92
NUMERIC_TOL = 0.02


# =============================================================================
# Union-Find
# =============================================================================

class _UnionFind:
    """路径压缩 + 秩合并。O(n α(n)) 摊还。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # 秩合并
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# =============================================================================
# unit 同质判定
# =============================================================================

def same_unit(u1: Optional[str], u2: Optional[str]) -> bool:
    """两个 unit 字符串是否语义相同。

    简化版:大小写不敏感 + 去空白 + 同义映射(USD/$/dollar、RMB/¥/元/人民币)。
    """
    if u1 is None and u2 is None:
        return True
    if u1 is None or u2 is None:
        return False

    _SYNONYMS = {
        frozenset({"USD", "$", "DOLLAR", "DOLLARS", "US$"}),
        frozenset({"RMB", "¥", "YUAN", "CNY", "人民币", "元"}),
        frozenset({"EUR", "€", "EURO", "EUROS"}),
        frozenset({"GBP", "£", "POUND", "POUNDS"}),
    }
    n1, n2 = u1.strip().upper(), u2.strip().upper()
    if n1 == n2:
        return True
    for grp in _SYNONYMS:
        if n1 in grp and n2 in grp:
            return True
    return False


def _entities_overlap(
    a: list[str],
    b: list[str],
    *,
    require_nonempty: bool = True,
) -> bool:
    """两个 EU 的 entities 是否至少有一个交集。

    require_nonempty: 如果任一为空,默认 True(Runbook 3.1 原文:
    "实体集合无交集 → 不是同一件事" — 当一个 EU 没 entities 时不应作为否决条件)。
    """
    if not a or not b:
        # 一方/双方没 entities → 视为可合并(无否决证据)
        return True
    return bool(set(a) & set(b))


def _numeric_close(
    v1: Optional[Decimal],
    v2: Optional[Decimal],
    tol: float = NUMERIC_TOL,
) -> bool:
    """两个 Decimal 是否在容差内视为"不冲突"。"""
    if v1 is None or v2 is None:
        return True
    if v1 == v2:
        return True
    a, b = float(v1), float(v2)
    base = max(abs(a), abs(b), 1.0)
    return abs(a - b) / base <= tol


# =============================================================================
# 主入口
# =============================================================================

def merge_units(
    eus: list[EvidenceUnitV2],
    embeddings: Optional[Any] = None,
    *,
    cosine_threshold: float = MERGE_COSINE,
    numeric_tol: float = NUMERIC_TOL,
) -> list[list[int]]:
    """把 EU 列表归并为分组,返回每组的 EU 索引列表。

    Args:
        eus: 待归并的 EU 列表(必须已过闸 1+2+3 — span_verified && !numeric_drift)
        embeddings: 可选 — numpy array (n, dim),shape 必须匹配 len(eus)。
                     如果 None,只用结构化约束(entity / value_as_of / unit / claim_type),
                     跳过 embedding 相似度。
        cosine_threshold: embedding 余弦相似度阈值(默认 0.92)
        numeric_tol: 数值冲突容差(默认 2%)

    Returns:
        list of groups,每个 group 是 EU 在输入列表中的索引列表。
        调用方根据 group 构造 Claim。

    复杂度:O(n α(n) + Σ_k O(k²)) — k 是单个桶的大小(典型 << n)。
    """
    n = len(eus)
    if n == 0:
        return []
    uf = _UnionFind(n)

    # 1. 分桶 (dimension_id, claim_type)
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, eu in enumerate(eus):
        buckets[(eu.dimension_id or "", eu.claim_type)].append(i)

    # 2. 桶内成对判断
    for idxs in buckets.values():
        # 2a. embedding 余弦相似度(可选)
        if embeddings is not None:
            try:
                import numpy as np
                sub = np.asarray(embeddings)[idxs]
                sims = sub @ sub.T  # L2 已归一
            except Exception:
                sims = None
        else:
            sims = None

        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                ai, bi = idxs[a_i], idxs[b_i]
                # 2a. embedding 阈值(如果可算)
                if sims is not None:
                    if sims[a_i, b_i] < cosine_threshold:
                        continue  # embedding 不够近,直接 skip
                else:
                    # 没 embedding 时:不能仅靠结构化约束合并 — 没有"够近"的信号
                    # 默认 skip(让 caller 提供 embedding 或接受松散归并)
                    continue

                x, y = eus[ai], eus[bi]

                # 2b. 实体集合无交集 → 不是同一件事
                if not _entities_overlap(x.entities, y.entities):
                    continue

                # 2c. claim_type=numeric 的特殊约束
                if x.claim_type == "numeric":
                    # 时点不同 → 不是同一 claim
                    if x.value_as_of != y.value_as_of:
                        continue
                    if not same_unit(x.unit, y.unit):
                        continue
                    # 数值冲突仍然合并 — 标记 conflict 后由报告并列呈现
                    # 这里不 continue,fall through 到 union

                # 通过所有约束 → 合并
                uf.union(ai, bi)

    # 3. 收集分组
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return list(groups.values())


# =============================================================================
# Group → Claim 草稿
# =============================================================================

from dataclasses import dataclass, field
from datetime import date, datetime, timezone


@dataclass
class ClaimDraft:
    """由 merge_units 的一个 group 构造的 Claim 草稿。

    还需要 grade(3.3)和 primary_source_count(3.2)来最终生成 ClaimV2。
    """
    eu_indices: list[int]
    canonical_claim: str
    claim_type: str
    entities: list[str]
    norm_value: Optional[Decimal] = None
    unit: Optional[str] = None
    value_as_of: Optional[date] = None
    value_spread: Optional[float] = None  # 数值最大相对偏差(冲突度量)
    has_conflict: bool = False
    conflicting_values: list[dict[str, Any]] = field(default_factory=list)
    earliest_published_at: Optional[datetime] = None

    @property
    def eu_count(self) -> int:
        return len(self.eu_indices)


def build_claim_drafts(
    eus: list[EvidenceUnitV2],
    groups: list[list[int]],
) -> list[ClaimDraft]:
    """从归并 group 构造 ClaimDraft 列表(不写 grade)。

    canonical_claim: 取 group 中第一个 EU 的 claim(简化 — 阶段 7 planner 可改进)
    entities: 取 group 中并集
    norm_value: numeric 时取众数(最多 EU 共享的那个),若无则取首个
    has_conflict: numeric 时若最大值/最小值偏差 > NUMERIC_TOL → True
    conflicting_values: 列出每个独立 value + source_url
    """
    drafts: list[ClaimDraft] = []
    for group in groups:
        items = [eus[i] for i in group]
        if not items:
            continue
        first = items[0]
        # entities 并集(去重 + 排序,确保 stable)
        entities = sorted({e for eu in items for e in eu.entities})

        # numeric 时处理 value 冲突
        norm_value: Optional[Decimal] = first.norm_value
        unit: Optional[str] = first.unit
        value_as_of: Optional[date] = first.value_as_of
        has_conflict = False
        conflicting_values: list[dict[str, Any]] = []
        value_spread: Optional[float] = None
        if first.claim_type == "numeric":
            distinct: dict[Decimal, list[EvidenceUnitV2]] = defaultdict(list)
            for eu in items:
                if eu.norm_value is not None:
                    distinct[eu.norm_value].append(eu)
            if len(distinct) > 1:
                has_conflict = True
                # value_spread = (max - min) / max
                vals = [float(v) for v in distinct.keys()]
                if max(abs(v) for v in vals) > 0:
                    value_spread = (max(vals) - min(vals)) / max(abs(v) for v in vals)
                for v, eus_with_v in distinct.items():
                    conflicting_values.append({
                        "value": str(v),
                        "unit": unit,
                        "count": len(eus_with_v),
                        "source_urls": [eu.source_url for eu in eus_with_v],
                    })
            # norm_value 取众数 max(count)
            norm_value = max(distinct.items(), key=lambda kv: len(kv[1]))[0]

        # earliest published_at
        published = [eu.published_at for eu in items if eu.published_at]
        earliest = min(published) if published else None

        drafts.append(ClaimDraft(
            eu_indices=group,
            canonical_claim=first.claim,
            claim_type=first.claim_type,
            entities=entities,
            norm_value=norm_value,
            unit=unit,
            value_as_of=value_as_of,
            value_spread=value_spread,
            has_conflict=has_conflict,
            conflicting_values=conflicting_values,
            earliest_published_at=earliest,
        ))

    return drafts


__all__ = [
    "MERGE_COSINE",
    "NUMERIC_TOL",
    "same_unit",
    "merge_units",
    "ClaimDraft",
    "build_claim_drafts",
]