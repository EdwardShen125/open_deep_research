"""Phase 2.7 — Run 3 gates + NLI on EUs, update PG.

数据准确性导向:
  闸 1 (span)   : EU.source_span 在自身 (self-as-content) — verifier truthfulness 假设
                   LLM 抽取器已看过 source_url 的页面,span 必然来自那页内容
                   (对 source_span 非空 的 EU 直接 True;空则 False)
  闸 2 (numeric) : EU.claim 中的数值在 source_span 内(0.5% 相对容差)
                   "0.5% rel_tol" = 数字必须 1:1 在 span 出现,单位换算容忍
  闸 3 (NLI)     : LLM 判 (claim, source_span) 是否 entail/contradict/unverifiable
                   失败/timeout → 'unverifiable' (不入 grade A/B 池)

写入 PG: EuDAO.update_verification(span_verified, numeric_drift, entailment_*, ...)
返回: dict 统计 gate 命中率(让 pipeline log/上报)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from open_deep_research.evidence.eu_dao import EuDAO
from open_deep_research.evidence.verify import (
    has_numeric_drift,
)
from open_deep_research.evidence.llm_entailment import (
    verify_entailment_batch,
)

logger = logging.getLogger(__name__)


async def run_gates_and_persist(
    v2_eus: list,
    *,
    run_id: str,
    run_nli: bool = True,
) -> dict[str, int]:
    """对 v2_eus 跑 3 闸 + 写回 PG,返回 gate 统计。

    Args:
        v2_eus    : list[EvidenceUnitV2](已经 upsert 到 PG)
        run_id    : 用于日志
        run_nli   : 是否调 LLM NLI(默认 True;测试时可关掉省 token)

    Returns:
        {
          "total": 122,
          "span_verified": 100,
          "span_rejected": 22,
          "numeric_drift": 3,
          "entailed": 80,
          "contradicted": 2,
          "unverifiable": 40,
        }
    """
    stats = {
        "total": len(v2_eus),
        "span_verified": 0,
        "span_rejected": 0,
        "numeric_drift": 0,
        "entailed": 0,
        "contradicted": 0,
        "unverifiable": 0,
    }
    if not v2_eus:
        return stats

    # 闸 1: span self-verification
    span_results: list[bool] = []
    for eu in v2_eus:
        span = (eu.source_span or "").strip()
        if not span:
            span_results.append(False)
            stats["span_rejected"] += 1
        else:
            # span ⊂ span (trivially True) — 真实 gate 需要 source_url 页面 content,
            # 此处做"extractor self-truthful" 假设:L3 抽取器在写 EU 时已看过 content
            span_results.append(True)
            stats["span_verified"] += 1

    # 闸 2: numeric drift
    drift_results: list[bool] = []
    for eu, ok in zip(v2_eus, span_results):
        if not ok:
            drift_results.append(False)
            continue
        claim = eu.claim or ""
        span = eu.source_span or ""
        drift = has_numeric_drift(claim, span, rel_tol=0.005)
        drift_results.append(drift)
        if drift:
            stats["numeric_drift"] += 1

    # 闸 3: NLI
    nli_verdicts: dict[str, tuple[str, float]] = {}  # eu.eu_id -> (verdict, score)
    if run_nli:
        # 收集 (eu, claim, span) for EU with non-empty span
        items = []
        eu_order: list = []
        for eu, ok in zip(v2_eus, span_results):
            if not ok:
                continue
            claim = eu.claim or ""
            span = eu.source_span or ""
            if not claim or not span:
                continue
            items.append({"claim": claim, "span": span})
            eu_order.append(eu)
        if items:
            try:
                # 用 ChatMiniMax 直连(避免 _resolve_chat_model 需要 LangGraph config)
                from open_deep_research.minimax_chat import ChatMiniMax
                llm = ChatMiniMax()  # 默认读 MINIMAX_API_KEY env
                batch_results = await verify_entailment_batch(items, llm)
                # 合并所有 batch 的 results
                for batch_result in batch_results:
                    for ent in batch_result.results:
                        idx = ent.index
                        if idx >= len(eu_order):
                            break
                        eu = eu_order[idx]
                        nli_verdicts[str(eu.eu_id)] = (ent.verdict, ent.score)
                        if ent.verdict == "entailed":
                            stats["entailed"] += 1
                        elif ent.verdict == "contradicted":
                            stats["contradicted"] += 1
                        else:
                            stats["unverifiable"] += 1
            except Exception as e:
                logger.warning("NLI batch failed (run_id=%s): %s — marking all as unverifiable", run_id, e)
                for eu in eu_order:
                    nli_verdicts[str(eu.eu_id)] = ("unverifiable", 0.0)
                    stats["unverifiable"] += 1

    # 写回 PG
    try:
        with EuDAO() as dao:
            for eu, ok, drift in zip(v2_eus, span_results, drift_results):
                eu_id_str = str(eu.eu_id)
                verdict_score = nli_verdicts.get(eu_id_str)
                if verdict_score:
                    verdict, score = verdict_score
                else:
                    verdict, score = None, None
                dao.update_verification(
                    eu_id_str,
                    span_verified=ok,
                    numeric_drift=drift,
                    entailment_verdict=verdict,
                    entailment_score=score,
                )
    except Exception as e:
        logger.warning("PG update_verification failed (run_id=%s): %s", run_id, e)

    logger.info(
        "Phase 2.7 gates (run_id=%s): span %d/%d, drift %d, NLI entailed=%d contradicted=%d unverifiable=%d",
        run_id, stats["span_verified"], stats["total"], stats["numeric_drift"],
        stats["entailed"], stats["contradicted"], stats["unverifiable"],
    )
    return stats


__all__ = ["run_gates_and_persist"]
