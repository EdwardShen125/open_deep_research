"""B3 · 分节检索 + 写作(修 context 溢出 P0)

按 plan B3:上次把 19,955+ EU 一次塞给 final_report_generation → context 溢出 →
MiniMax HTTP 400。改成每槽只检索该槽的 claim 再写,bounded context。

接口:
    retrieve_for_slot(slot, *, k=10, claims=None) -> list[ClaimV3]
        按 slot_id + caliber + (可选 pgvector 相似度)取 top-k,k 小。
        claims=None 时返回空(测试场景)。
        **任意单次返回值 ≤ k**(bounded context)。

    write_section(slot, claims, *, llm=None) -> SectionResult
        经 llm.py:get_llm() 调 LLM,prompt 只含该 slot 的 claim(不超 k)。
        生成 SectionResult(FilledSlot + 上下文)。

    assemble_report(framework, *, claims_per_slot=None, llm=None) -> ReportResult
        逐节聚合。每节调 write_section(单次 LLM context 有上界)。
        未填槽显式进 ReportResult.unresolved。

P0 修复:任何 retrieve_for_slot 不返回 > k 条,任何 write_section prompt 不含
> k 条 claim 的 JSON 序列化文本。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from open_deep_research.evidence.claim_v3 import ClaimV3
from open_deep_research.evidence.framework import Framework, Slot
from open_deep_research.evidence.report_result import (
    FilledSlot,
    ReportResult,
    SectionResult,
    derive_confidence,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TOP_K = 10  # plan B3:每槽最多 10 条 claim


# =============================================================================
# retrieve_for_slot
# =============================================================================

def retrieve_for_slot(
    slot: Slot,
    *,
    k: int = DEFAULT_TOP_K,
    claims: Optional[list[ClaimV3]] = None,
) -> list[ClaimV3]:
    """从给定 claim 池(测试 / in-memory 路径)按 slot 过滤,返回 top-k。

    Args:
        slot: 目标槽
        k: 上界(plan B3:k ≤ 10,bound context)
        claims: 候选 claim 池;None 时返回 []

    Returns:
        最多 k 条 ClaimV3,按"是否 verified + tier 降序"排序。

    注:这是测试 / in-memory 版本。生产环境(pgvector + EU 表)
    应另有 pg 路径;本接口抽象不依赖存储,只依赖输入池。
    """
    if claims is None:
        return []

    # 过滤:caliber_id 匹配 + value 与 slot.question 文本相似(简化用包含)
    relevant: list[ClaimV3] = []
    for c in claims:
        # 1. caliber 匹配(若有)
        if slot.caliber_id and c.caliber_id and c.caliber_id != slot.caliber_id:
            continue
        # 2. 文本相似:claim value 含 slot.question 中的关键词(简化)
        if _soft_match(c.value, slot.question):
            relevant.append(c)

    # 排序:verified + tier 高的优先
    def _rank(c: ClaimV3) -> tuple[int, int]:
        status_rank = 0 if c.verification_status == "verified" else 1
        tier_rank = {"A": 0, "B": 1, "C": 2, "D": 3}.get(c.tier or "D", 3)
        return (status_rank, tier_rank)

    relevant.sort(key=_rank)
    return relevant[:k]


def _soft_match(claim_value: str, question: str) -> bool:
    """简化版文本相似:question 中 ≥3 字符 token 出现在 claim_value。"""
    claim_lower = claim_value.lower()
    for token in question.split():
        if len(token) >= 3 and token.lower() in claim_lower:
            return True
    return False


# =============================================================================
# write_section
# =============================================================================

# 估算 token 数:1 token ≈ 4 字符(粗略;实际由 tiktoken 算)
CHARS_PER_TOKEN = 4


def _build_section_prompt(slot: Slot, claims: list[ClaimV3]) -> tuple[str, str]:
    """构造 section 写作 prompt。

    返回 (system, user)。user 中含该 slot 的 claim(已限制数量)。
    """
    claims_json = json.dumps(
        [
            {
                "value": c.value,
                "tier": c.tier,
                "caliber_id": c.caliber_id,
                "verification_status": c.verification_status,
                "source_url": c.source_url,
                "source_domain": c.source_domain,
                "value_as_of": c.value_as_of.isoformat() if c.value_as_of else None,
            }
            for c in claims
        ],
        ensure_ascii=False,
    )

    system = (
        "You are a research report section writer. "
        "Given a slot question and a list of verified claims, "
        "write a concise answer section that cites each claim inline. "
        "Do NOT introduce claims not in the list. "
        "Return strict JSON: {\"markdown\": \"...\", \"rationale\": \"...\"}"
    )
    user = f"""## Slot Question
{slot.question}

## Slot Type
{slot.expected_claim_type}
{f'(caliber: {slot.caliber_id})' if slot.caliber_id else ''}

## Claims Available ({len(claims)} total)
{claims_json}

## Output (strict JSON, no markdown)
{{"markdown": "...", "rationale": "..."}}
"""
    return system, user


async def write_section_async(
    slot: Slot,
    claims: list[ClaimV3],
    *,
    llm: Any = None,
) -> SectionResult:
    """异步写一个 section。

    单次 LLM 调用只看到该 slot 的 claim,bounded context。
    """
    if llm is None:
        from open_deep_research.llm import get_llm
        llm = get_llm()

    system, user = _build_section_prompt(slot, claims)

    # 估算 prompt 大小,记 log 便于诊断
    approx_tokens = (len(system) + len(user)) // CHARS_PER_TOKEN
    logger.info(
        "write_section: slot=%s claims=%d approx_tokens=%d",
        slot.slot_id, len(claims), approx_tokens,
    )

    ainvoke = getattr(llm, "ainvoke", None)
    if ainvoke is not None:
        response = await ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    else:
        invoke = getattr(llm, "invoke", None)
        if invoke is None:
            raise RuntimeError(
                f"llm object {type(llm).__name__} has neither ainvoke nor invoke"
            )
        response = invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    content = getattr(response, "content", None) or str(response)
    parsed = _parse_section_response(content)

    # FilledSlot 推导 confidence
    confidence = derive_confidence(claims)
    filled = FilledSlot(slot_id=slot.slot_id, claims=claims, confidence=confidence)
    return SectionResult(
        section_id=slot.slot_id,
        title=slot.question,
        slots=[filled],
    )


def write_section(
    slot: Slot,
    claims: list[ClaimV3],
    *,
    llm: Any = None,
) -> SectionResult:
    """同步版本(供非 async 调用方)。"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return _sync_write_section(slot, claims, llm)
        return loop.run_until_complete(write_section_async(slot, claims, llm=llm))
    except RuntimeError:
        return asyncio.run(write_section_async(slot, claims, llm=llm))


def _sync_write_section(slot: Slot, claims: list[ClaimV3], llm: Any) -> SectionResult:
    """同步包装,不走 asyncio.run(避免在已有 loop 中死循环)。"""
    # 实际单次 LLM 调用,在已有 loop 中调 invoke(同步)
    if llm is None:
        from open_deep_research.llm import get_llm
        llm = get_llm()
    system, user = _build_section_prompt(slot, claims)
    invoke = getattr(llm, "invoke", None)
    if invoke is None:
        raise RuntimeError("llm has no invoke in sync context")
    response = invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    content = getattr(response, "content", None) or str(response)
    confidence = derive_confidence(claims)
    filled = FilledSlot(slot_id=slot.slot_id, claims=claims, confidence=confidence)
    return SectionResult(
        section_id=slot.slot_id,
        title=slot.question,
        slots=[filled],
    )


def _parse_section_response(text: str) -> dict[str, str]:
    """解析 LLM 返回的 JSON(markdown + rationale)。"""
    if not text:
        return {"markdown": "", "rationale": ""}
    s = text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {
                "markdown": str(obj.get("markdown", "")),
                "rationale": str(obj.get("rationale", "")),
            }
    except (json.JSONDecodeError, ValueError):
        pass
    # 围栏
    import re
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return {
                    "markdown": str(obj.get("markdown", "")),
                    "rationale": str(obj.get("rationale", "")),
                }
        except (json.JSONDecodeError, ValueError):
            pass
    return {"markdown": "", "rationale": text[:200]}


# =============================================================================
# assemble_report
# =============================================================================

async def assemble_report_async(
    framework: Framework,
    *,
    claims_per_slot: Optional[dict[str, list[ClaimV3]]] = None,
    llm: Any = None,
    k: int = DEFAULT_TOP_K,
) -> ReportResult:
    """逐节聚合(plan B3:每节单独 LLM,bounded context)。

    claims_per_slot: {slot_id: list[ClaimV3]} 显式提供每槽的 claim
                    None 时视为空,该槽进 unresolved
    """
    sections: list[SectionResult] = []
    unresolved: list[str] = []

    claims_per_slot = claims_per_slot or {}

    for section in framework.sections:
        sec_slots: list[FilledSlot] = []
        for slot in section.slots:
            slot_claims = retrieve_for_slot(
                slot, k=k, claims=claims_per_slot.get(slot.slot_id),
            )
            if not slot_claims:
                # 未填槽显式进 unresolved
                unresolved.append(slot.slot_id)
                continue
            section_result = await write_section_async(slot, slot_claims, llm=llm)
            sec_slots.extend(section_result.slots)
        sections.append(SectionResult(
            section_id=section.section_id,
            title=section.title,
            slots=sec_slots,
        ))

    return ReportResult(
        title=framework.title,
        vertical_id=framework.vertical_id,
        sections=sections,
        unresolved=unresolved,
    )


def assemble_report(
    framework: Framework,
    *,
    claims_per_slot: Optional[dict[str, list[ClaimV3]]] = None,
    llm: Any = None,
    k: int = DEFAULT_TOP_K,
) -> ReportResult:
    """同步版本。"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("assemble_report called in running loop; use assemble_report_async")
        return loop.run_until_complete(
            assemble_report_async(framework, claims_per_slot=claims_per_slot, llm=llm, k=k)
        )
    except RuntimeError:
        return asyncio.run(
            assemble_report_async(framework, claims_per_slot=claims_per_slot, llm=llm, k=k)
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DEFAULT_TOP_K",
    "retrieve_for_slot",
    "write_section",
    "write_section_async",
    "assemble_report",
    "assemble_report_async",
]
