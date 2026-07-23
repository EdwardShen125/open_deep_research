"""Phase 5 (= Runbook v1 阶段 3) 端到端:EU → Claim。

依据: notes/evidence-pipeline-runbook-v1.md 阶段 3.1-3.4

合并 merge.py + independence.py 的产物,生成 ClaimV2 列表。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional, Union

from open_deep_research.evidence.independence import (
    grade_claim,
    independent_source_count,
    primary_source_count,
    upgrade_source_tier,
)
from open_deep_research.evidence.merge import (
    ClaimDraft,
    build_claim_drafts,
    merge_units,
)
from open_deep_research.evidence.schema import ClaimV2, EvidenceUnitV2

logger = logging.getLogger(__name__)


def _has_any_entailed(eus: list[EvidenceUnitV2], indices: list[int]) -> bool:
    """组内是否有任一 EU 的 entailment_verdict == 'entailed'。"""
    for i in indices:
        v = eus[i].entailment_verdict
        if v in ("entailed", "partial"):
            return True
    return False


def build_claims_from_eus(
    eus: list[EvidenceUnitV2],
    *,
    embeddings: Optional[Any] = None,
    page_emb: Optional[dict[str, Any]] = None,
    cosine_threshold: float = 0.92,
    return_eu_map: bool = False,
) -> Union[list[ClaimV2], tuple[list[ClaimV2], dict[str, str]]]:
    """端到端:EU 列表 → ClaimV2 列表(含 grade)。

    步骤:
      1. (可选)升级 EU 的 source_tier 到实际 tier(白名单驱动)
      2. 归并(merge_units)→ group indices
      3. 每个 group → ClaimDraft(build_claim_drafts)
      4. 计算 independent_source_count + primary_source_count
      5. grade_claim → A / B / C / D
      6. ClaimDraft → ClaimV2 落库

    默认返 list[ClaimV2](向后兼容)。
    return_eu_map=True 时返 (claims, eu_to_claim_id) — eu_id(str) → claim_id(str)
    用于 P1.2 claim_id 回填链路(让 evidence_unit.claim_id 字段有值)。

    返回:ClaimV2 列表。每个 ClaimV2 的 eu_count / independent_source_count /
    primary_source_count 都已填好。ClaimV2.claim_id 是新生成 UUID。
    """
    if not eus:
        if return_eu_map:
            return [], {}
        return []

    # 1. 升级 tier
    eus_upgraded = [upgrade_source_tier(eu) for eu in eus]

    # 2. 归并
    groups = merge_units(eus_upgraded, embeddings=embeddings, cosine_threshold=cosine_threshold)

    # P1.2: 收集 eu_id -> claim_id 映射 (用于 EU.claim_id 字段回填)
    eu_to_claim_id: dict[str, str] = {}

    # 3. 草稿
    drafts = build_claim_drafts(eus_upgraded, groups)

    # 4-6. 评级 + ClaimV2
    claims: list[ClaimV2] = []
    for draft in drafts:
        group_eus = [eus_upgraded[i] for i in draft.eu_indices]
        indep = independent_source_count(group_eus, page_emb=page_emb)
        prim = primary_source_count(group_eus)
        any_ent = _has_any_entailed(eus_upgraded, draft.eu_indices)

        # D 级:无可用 EU 的归并组通常不会到这里(归并要求 embedding 相似,无 entailed 仍合并)。
        # 留 hard guard:如果整组没有 entailed,仍然保留 group 但 grade=D,
        # eu_count 由 draft.eu_count 给(可能 = 0,因为组内 EU 可能没有 entailed)。
        # 但 merge_units 后的 group 至少有 1 个 EU。所以这里我们强制 grade=D 的也允许 eu_count=1+。

        grade, reason = grade_claim(
            draft,
            independent_count=indep,
            primary_count=prim,
            has_any_entailed=any_ent,
        )

        # ClaimDraft 没有 dimension_id → 用 group 中任一 EU 的 dimension_id
        dimension_id = next(
            (e.dimension_id for e in group_eus if e.dimension_id),
            "unknown",
        )

        claim = ClaimV2(
            run_id=group_eus[0].run_id,
            dimension_id=dimension_id,
            canonical_claim=draft.canonical_claim,
            claim_type=draft.claim_type,  # type: ignore[arg-type]
            entities=draft.entities,
            norm_value=draft.norm_value,
            unit=draft.unit,
            value_as_of=draft.value_as_of,
            value_spread=draft.value_spread,
            eu_count=draft.eu_count,
            independent_source_count=indep,
            primary_source_count=prim,
            earliest_published_at=draft.earliest_published_at,
            has_conflict=draft.has_conflict,
            conflicting_values=draft.conflicting_values,
            grade=grade,
            grade_reason=reason,
        )

        # P1.2: 记录 group 中每个 EU 的 claim_id 映射
        if return_eu_map:
            for idx in draft.eu_indices:
                eu_id_str = str(eus_upgraded[idx].eu_id)
                eu_to_claim_id[eu_id_str] = str(claim.claim_id)

        claims.append(claim)

    logger.info(
        "build_claims_from_eus: %d EU → %d claims (grade dist: %s)",
        len(eus), len(claims),
        {g: sum(1 for c in claims if c.grade == g) for g in "ABCD"},
    )
    if return_eu_map:
        return claims, eu_to_claim_id
    return claims


__all__ = ["build_claims_from_eus"]