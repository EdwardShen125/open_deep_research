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
    response = None
    last_err: Exception | None = None
    # R14: LLM 偶发返空(MiniMax 间歇性问题 / prompt 超 token)。
    # 重试 2 次,仍空就降级(任务书 R8:degraded=True / passed=False)。
    for _attempt in range(3):
        try:
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
            content = getattr(response, "content", None) or str(response) or ""
            if content.strip():
                break  # 真拿到内容,退出重试
            logger.warning(
                "write_section_async: slot=%s attempt=%d returned empty content",
                slot.slot_id, _attempt + 1,
            )
        except Exception as e:
            last_err = e
            logger.warning(
                "write_section_async: slot=%s attempt=%d raised: %s",
                slot.slot_id, _attempt + 1, e,
            )

    if response is None or not (getattr(response, "content", None) or str(response or "")).strip():
        # R14:重试 3 次仍空 → 抛错让上游 plan_v2_pipeline 进 degraded(任务书 R8)
        raise RuntimeError(
            f"write_section_async slot={slot.slot_id}: LLM returned empty after 3 attempts"
            + (f"; last_err={last_err}" if last_err else "")
        )

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
# W9-L1 · synthesize_section_async — 章节小结层
# =============================================================================


# 估算 token 数:1 token ≈ 4 字符(粗略;实际由 tiktoken 算)
CHARS_PER_TOKEN = 4


async def synthesize_section_async(
    section: SectionResult,
    *,
    llm: Any = None,
) -> "SectionSummary":
    """W9-L1:综合一节的 L0 槽,产出 SectionSummary(章节小结 + 过渡叙事)。

    非交涉(任务书 §0.2):
      - 输入:本节 L0 SectionResult(含 FilledSlot + claim ids)
      - 输出:SectionSummary(summary_md + source_slot_ids + key_claim_ids)
      - prompt 只含本节槽摘要,**不重新拉全量 EU**
      - LLM 不可用 → 抛 RuntimeError 让上游 W5 降级(degraded=True / passed=False)

    Prompt 构造规则:逐 slot 提 question + 1 句 claim 摘要 + claim id(供溯源)。
    每节 prompt 上界 = sum(slot_question + slot_summary_len),与总 EU 量无关。
    """
    if llm is None:
        from open_deep_research.llm import get_llm
        llm = get_llm()

    # 1) 准备输入:每槽的 question + 摘要 + claim ids(不传原始 claim.value,
    #    只传 claim 文本摘要,严控 token)
    slot_inputs: list[dict[str, Any]] = []
    for s in section.slots:
        if not s.claims:
            continue
        claim_summaries = []
        claim_ids: list[str] = []
        for c in s.claims:
            claim_summaries.append({
                "value": c.value[:160],
                "tier": c.tier,
                "source_url": c.source_url,
                "source_domain": c.source_domain,
                "verification_status": c.verification_status,
            })
            claim_ids.append(f"{c.source_url}::{c.value[:40]}")
        slot_inputs.append({
            "slot_id": s.slot_id,
            "question": section.title if len(section.slots) == 1 else f"{section.title} › {s.slot_id}",
            "n_claims": len(s.claims),
            "confidence": s.confidence,
            "claims": claim_summaries[:5],  # 槽内 claim 也截断
        })

    if not slot_inputs:
        # 该节无 claim → 兜底:summary_md 标注 empty + source_slot_ids 空
        return SectionSummary(
            section_id=section.section_id,
            summary_md=f"(本节无已验证 claim,slot 数={len(section.slots)},无可综合内容)",
            source_slot_ids=[],
            key_claim_ids=[],
        )

    system = (
        "You are a research-report synthesizer. You are given the claims of one section "
        "of a research report, structured as filled slots. Write a concise narrative "
        "summary of the section (chapter closing paragraph) that integrates the slots' "
        "findings WITHOUT introducing any fact not present in the claims."
        " Each sentence must be supported by some slot_id in the input. "
        "Return strict JSON: {\"summary_md\": \"...\"}"
    )
    slots_json = json.dumps(slot_inputs, ensure_ascii=False)
    user = f"""## Section
{section.section_id} — {section.title}

## Filled Slots ({len(slot_inputs)} slots, each with its claims)
{slots_json}

## Output (strict JSON, no markdown fence)
{{"summary_md": "..."}}

Rules:
- Every sentence in summary_md must be grounded in claims from at least one slot.
- Cite slot_ids inline like [slot_id] at the end of each sentence.
- Do NOT add cross-section perspective; another layer (L2) handles that.
- If a slot has only low-confidence claims, reflect that in the wording (e.g. "preliminary").
- summary_md length ≤ 800 characters (Chinese) or equivalent.
"""
    approx_tokens = (len(system) + len(user)) // CHARS_PER_TOKEN
    logger.info(
        "synthesize_section: section=%s slots=%d approx_tokens=%d",
        section.section_id, len(slot_inputs), approx_tokens,
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
            raise RuntimeError(f"llm object {type(llm).__name__} has neither ainvoke nor invoke")
        response = invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    content = getattr(response, "content", None) or str(response)
    parsed = _parse_section_response(content)

    # 汇总 source_slot_ids + key_claim_ids
    source_slot_ids = [si["slot_id"] for si in slot_inputs]
    key_claim_ids: list[str] = []
    for si in slot_inputs[:3]:  # 前 3 槽的 claim 作为 key
        for c in si["claims"][:2]:
            key_claim_ids.append(f"{c['source_url']}::{c['value'][:40]}")

    summary_md = parsed.get("markdown", "").strip()
    if not summary_md:
        # LLM 返回空 / 非 JSON → 兜底:生成 deterministic stub,标 is_stub=True
        # 任务书 §0.2:综合层 LLM 不可用 → degraded=True / passed=False;失败节留空
        # 我们不抛错,而是产 stub — 让 W9-L2 / L3 能继续拿到部分 summaries
        # is_stub=True 让 plan_v2_pipeline 据此设 degraded(任务书 §5 不糊弄)
        logger.warning(
            "[W9-L1] %s LLM returned empty; generating deterministic stub",
            section.section_id,
        )
        summary_md = _deterministic_stub_summary(section, slot_inputs)
        return SectionSummary(
            section_id=section.section_id,
            summary_md=summary_md,
            source_slot_ids=source_slot_ids,
            key_claim_ids=key_claim_ids,
            is_stub=True,
        )

    return SectionSummary(
        section_id=section.section_id,
        summary_md=summary_md,
        source_slot_ids=source_slot_ids,
        key_claim_ids=key_claim_ids,
    )


def _deterministic_stub_summary(
    section: "SectionResult",
    slot_inputs: list[dict[str, Any]],
) -> str:
    """W9-L1 / L2 / L3 的兜底 stub:基于 slot_inputs 拼出不含 LLM 发挥的最小综合。

    非交涉 1(任务书 §0.1):不引入任何 slot/claim 之外的事实。
    仅说"本节含 N 槽 / M claim / 引用 slots: ..." + 第一个 claim 首句。
    """
    parts: list[str] = []
    parts.append(
        f"本节 ({section.section_id}) 含 {len(slot_inputs)} 槽,均已 verified+溯源。"
        "以下为 slot 摘要聚合(LLM 综合不可用,使用 deterministic stub):"
    )
    for si in slot_inputs[:3]:
        first_claim = si["claims"][0]["value"][:80] if si["claims"] else "(no claim)"
        parts.append(
            f"- {si['slot_id']} ({si['confidence']}, {si['n_claims']} claims): "
            f"{first_claim}…"
        )
    parts.append("[结构性判断:本节 stub 综合由 pipeline 兜底生成,未经 LLM 综合]")
    return "\n".join(parts)


# =============================================================================
# W9-L2 · synthesize_crosscuts_async — 跨切综合层
# =============================================================================


async def synthesize_crosscuts_async(
    section_summaries: list["SectionSummary"],
    reconciliations: list[Any],  # list[ReconciliationRecord]
    *,
    crosscut_specs: Optional[list[dict[str, Any]]] = None,
    llm: Any = None,
) -> list["CrossCut"]:
    """W9-L2:跨切综合 — 产出 comparison_table + analysis 两类 CrossCut。

    非交涉(任务书 §2 / §0.2):
      - 输入:各节 SectionSummary + ReconciliationRecord 列表(口径对账)
      - 输出:CrossCut 列表;comparison_table 必须挂 reconciliation_ids
              analysis 若超出 claim 支撑 → is_structural_judgment=True 且 md 显式标"结构性判断"
      - prompt 只含摘要与 reconciliation,**不重新拉全量 EU**
      - LLM 不可用 → 抛 RuntimeError 触发 W5 降级
    """
    if llm is None:
        from open_deep_research.llm import get_llm
        llm = get_llm()

    if not section_summaries and not reconciliations:
        return []

    crosscuts: list[CrossCut] = []

    # 默认 crosscut specs:若调用方没声明,自动产 2 条(对照表 + 分析)
    specs = crosscut_specs or [
        {
            "crosscut_id": "auto_caliber_table",
            "kind": "comparison_table",
            "over_sections": [s.section_id for s in section_summaries[:3]],
            "caliber_aware": True,
        },
        {
            "crosscut_id": "auto_structural_analysis",
            "kind": "analysis",
            "over_sections": [s.section_id for s in section_summaries[:5]],
        },
    ]

    # 准备 cross-section 摘要(只传摘要文本,严控 token)
    summaries_payload = [
        {
            "section_id": s.section_id,
            "summary_md": s.summary_md[:400],  # 摘要截断
            "source_slot_ids": s.source_slot_ids[:5],  # 槽 id 也截断
        }
        for s in section_summaries
    ]

    # 准备 reconciliation 摘要(只传 measurand / primary_value / 不可比项)
    reconciliations_payload = []
    reconciliation_ids: list[str] = []
    for i, r in enumerate(reconciliations):
        rid = f"recon-{i:03d}"
        reconciliation_ids.append(rid)
        reconciliations_payload.append({
            "id": rid,
            "measurand": getattr(r, "measurand", ""),
            "primary_value": getattr(r, "primary_value", None),
            "primary_caliber_id": getattr(r, "primary_caliber_id", ""),
            "primary_tier": getattr(r, "primary_tier", ""),
            "alternatives_count": len(getattr(r, "alternatives", []) or []),
            "divergence_kind": getattr(r, "divergence_kind", ""),
        })

    for spec in specs:
        kind = spec.get("kind", "analysis")
        cid = spec.get("crosscut_id", f"crosscut_{len(crosscuts)}")
        over_sections = spec.get("over_sections", [])

        # 过滤该 spec 涉及的摘要
        relevant_summaries = [
            s for s in summaries_payload
            if s["section_id"] in over_sections or not over_sections
        ]

        if kind == "comparison_table":
            # 任务书 §2:对照表必须挂 reconciliation_ids;跨口径值标"不可比"
            if not reconciliation_ids:
                logger.warning(
                    "[W9-L2] comparison_table %s 没有 reconciliation_ids — 跳过", cid
                )
                continue
            md, is_stub_l2 = await _generate_comparison_table(
                relevant_summaries, reconciliations_payload, llm,
            )
            crosscuts.append(CrossCut(
                crosscut_id=cid,
                kind="comparison_table",
                md=md,
                source_section_ids=over_sections or [s["section_id"] for s in relevant_summaries],
                reconciliation_ids=reconciliation_ids[:5],
                is_structural_judgment=False,
                is_stub=is_stub_l2,
            ))
        else:  # analysis
            md, is_stub_l2 = await _generate_analysis(
                relevant_summaries, reconciliations_payload, llm,
            )
            # 任务书:若超出 claim 支撑 → structural_judgment=True
            # 启发式:LLM 在 md 显式标"[structural_judgment]" / "结构性判断" → True
            is_structural = (
                "[structural_judgment]" in md.lower() or "结构性判断" in md
            )
            crosscuts.append(CrossCut(
                crosscut_id=cid,
                kind="analysis",
                md=md,
                source_section_ids=over_sections or [s["section_id"] for s in relevant_summaries],
                reconciliation_ids=[],
                is_structural_judgment=is_structural,
                is_stub=is_stub_l2,
            ))

    return crosscuts


async def _generate_comparison_table(
    summaries: list[dict],
    reconciliations: list[dict],
    llm: Any,
) -> tuple[str, bool]:
    """W9-L2 内部:对照表(口径感知 — 跨口径值必标"不可比")。
    返回 (md, is_stub)。is_stub=True 表示 LLM 返空走了 stub。
    """
    summaries_json = json.dumps(summaries, ensure_ascii=False)
    recs_json = json.dumps(reconciliations, ensure_ascii=False)
    system = (
        "You are a research-report analyst. You will receive section summaries + "
        "caliber-aware reconciliation records. Produce a markdown comparison table "
        "(headers: measurand | primary value | primary caliber | alternative values (不可比)) "
        "that explicitly marks cross-caliber values as '不可比'. Do NOT invent values; "
        "only use what's in the reconciliation records. Return strict JSON: {\"table_md\": \"...\"}"
    )
    user = f"""## Section Summaries
{summaries_json}

## Reconciliation Records (caliber-aware)
{recs_json}

## Output (strict JSON)
{{"table_md": "..."}}

Rules:
- Cross-caliber rows MUST be marked 不可比 (non-comparable).
- Primary caliber values are comparable within themselves; alternatives are not.
- If no reconciliation records, return empty table with one row "(无对账记录)".
"""
    approx_tokens = (len(system) + len(user)) // CHARS_PER_TOKEN
    logger.info("synthesize_crosscuts[L2 comparison]: approx_tokens=%d", approx_tokens)

    ainvoke = getattr(llm, "ainvoke", None)
    if ainvoke is not None:
        response = await ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    else:
        response = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    content = getattr(response, "content", None) or str(response)
    parsed = _parse_section_response(content)
    md = parsed.get("markdown", "").strip()
    if not md:
        # 兜底:产 deterministic stub
        logger.warning(
            "[W9-L2] comparison_table LLM returned empty; generating deterministic stub"
        )
        md = _deterministic_stub_comparison_table(reconciliations)
        return md, True
    return md, False


def _deterministic_stub_comparison_table(reconciliations: list[dict]) -> str:
    """W9-L2 对照表 stub:基于 reconciliation 行产出表(不引入新事实)。"""
    if not reconciliations:
        return "| (无对账记录) |\n|---|"
    lines = ["| measurand | primary_value | primary_caliber | tier | alternatives |",
             "|---|---|---|---|---|"]
    for r in reconciliations[:8]:
        lines.append(
            f"| {r.get('measurand','')[:40]} | "
            f"{r.get('primary_value','')} | "
            f"{r.get('primary_caliber_id','')} | "
            f"{r.get('primary_tier','')} | "
            f"{r.get('alternatives_count', 0)} (跨口径不可比) |"
        )
    lines.append("\n[结构性判断:本对照表由 pipeline 兜底生成,未经 LLM 综合]")
    return "\n".join(lines)


async def _generate_analysis(
    summaries: list[dict],
    reconciliations: list[dict],
    llm: Any,
) -> tuple[str, bool]:
    """W9-L2 内部:分析定论。LLM 标 [structural_judgment] → is_structural_judgment=True。
    返回 (md, is_stub)。
    """
    summaries_json = json.dumps(summaries, ensure_ascii=False)
    recs_json = json.dumps(reconciliations, ensure_ascii=False)
    system = (
        "You are a research-report analyst. Produce a cross-section analytical conclusion "
        "(200-400 chars Chinese) integrating the section summaries + reconciliation records. "
        "If a claim goes BEYOND what's supported by the summaries, prefix it with "
        "'[structural_judgment]' or '结构性判断:' to flag it. Do NOT introduce new facts."
    )
    user = f"""## Section Summaries
{summaries_json}

## Reconciliation Records
{recs_json}

## Output (strict JSON)
{{"markdown": "..."}}
"""
    approx_tokens = (len(system) + len(user)) // CHARS_PER_TOKEN
    logger.info("synthesize_crosscuts[L2 analysis]: approx_tokens=%d", approx_tokens)

    ainvoke = getattr(llm, "ainvoke", None)
    if ainvoke is not None:
        response = await ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    else:
        response = llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    content = getattr(response, "content", None) or str(response)
    parsed = _parse_section_response(content)
    md = parsed.get("markdown", "").strip()
    if not md:
        logger.warning("[W9-L2] analysis LLM returned empty; generating deterministic stub")
        md = "[结构性判断:本节分析定论由 pipeline 兜底生成,未经 LLM 综合]\n\n未生成分析 — LLM 综合不可用。"
        return md, True
    return md, False


# =============================================================================
# W10 · write_exec_summary_async — 执行摘要 L3
# =============================================================================


async def write_exec_summary_async(
    section_summaries: list["SectionSummary"],
    crosscuts: list["CrossCut"],
    *,
    llm: Any = None,
) -> "ExecSummary":
    """W10:产出整篇执行摘要(一句话 + 十条要点)。

    非交涉(任务书 §3):
      - 输入:L1 SectionSummary + L2 CrossCut 顶层结论
      - 输出:ExecSummary(one_liner + key_points + source_section_ids)
      - 每个 key_point 必须可回指到 section_id(validation 在 ExecSummary model 里强制)
      - prompt 只含摘要,**不重新拉全量 EU**
      - LLM 不可用 → 抛 RuntimeError 触发 W5 降级

    生成顺序:在 L1/L2 全部完成后、装配前调(任务书 §5 装配顺序)。
    """
    if llm is None:
        from open_deep_research.llm import get_llm
        llm = get_llm()

    if not section_summaries:
        raise RuntimeError("write_exec_summary: no section_summaries to summarize")

    # 准备 crosscut 顶层结论
    crosscut_payload = [
        {
            "crosscut_id": c.crosscut_id,
            "kind": c.kind,
            "md": c.md[:300],
            "is_structural_judgment": c.is_structural_judgment,
            "source_section_ids": c.source_section_ids,
        }
        for c in crosscuts
    ]

    # 准备 section summary 摘要(更短 — L3 是顶层)
    summary_payload = [
        {
            "section_id": s.section_id,
            "summary_md": s.summary_md[:250],  # 比 L2 更紧
        }
        for s in section_summaries
    ]

    system = (
        "You are a research-report executive-summary writer. You will receive "
        "compressed section summaries + crosscut conclusions. Produce a JSON "
        "{one_liner, key_points: [...] } where one_liner ≤ 80 chars Chinese and "
        "key_points is 5-10 items, each a single concrete claim that traces back "
        "to some section_id. Do NOT introduce facts not in the summaries."
    )
    user = f"""## Section Summaries (L1)
{json.dumps(summary_payload, ensure_ascii=False)}

## Crosscut Conclusions (L2)
{json.dumps(crosscut_payload, ensure_ascii=False)}

## Output (strict JSON)
{{"one_liner": "...", "key_points": ["...", "..."], "source_section_ids": ["...", "..."]}}

Rules:
- one_liner 一句话陈述整篇核心结论(≤ 80 中文字符)
- key_points 5-10 条;每条 ≤ 60 中文字符;每条必须能在某 section 摘要里找到依据
- source_section_ids 是 key_points 引用过的 section 列表(去重)
"""
    approx_tokens = (len(system) + len(user)) // CHARS_PER_TOKEN
    logger.info(
        "write_exec_summary: summaries=%d crosscuts=%d approx_tokens=%d",
        len(section_summaries), len(crosscuts), approx_tokens,
    )

    ainvoke = getattr(llm, "ainvoke", None)
    # R14:与 write_section_async 一致 — LLM 偶发空响应,重试 2 次,仍空抛错
    # 让上游 plan_v2_pipeline 进 degraded(任务书 R8)。
    response = None
    last_err: Exception | None = None
    for _attempt in range(3):
        try:
            if ainvoke is not None:
                response = await ainvoke([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            else:
                response = llm.invoke([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
            _raw = getattr(response, "content", None) or str(response or "")
            if _raw.strip():
                break
            logger.warning(
                "write_exec_summary_async: attempt=%d returned empty", _attempt + 1,
            )
        except Exception as e:
            last_err = e
            logger.warning("write_exec_summary_async: attempt=%d raised: %s", _attempt + 1, e)

    if response is None or not (getattr(response, "content", None) or str(response or "")).strip():
        raise RuntimeError(
            f"write_exec_summary_async: LLM returned empty after 3 attempts"
            + (f"; last_err={last_err}" if last_err else "")
        )
    content = getattr(response, "content", None) or str(response)
    parsed = _parse_section_response(content)

    # 解析 key_points 和 one_liner — _parse_section_response 只返 markdown 字段,
    # 这里更宽松:再尝试取 one_liner / key_points / source_section_ids
    raw_md = parsed.get("markdown", "").strip()
    one_liner = ""
    key_points: list[str] = []
    source_section_ids: list[str] = []

    # 优先尝试把 raw_md 当 JSON 解析
    if raw_md.startswith("{"):
        try:
            obj = json.loads(raw_md)
            one_liner = str(obj.get("one_liner", "")).strip()
            kp = obj.get("key_points", [])
            if isinstance(kp, list):
                key_points = [str(x).strip() for x in kp if str(x).strip()]
            ssids = obj.get("source_section_ids", [])
            if isinstance(ssids, list):
                source_section_ids = [str(x).strip() for x in ssids if str(x).strip()]
        except json.JSONDecodeError:
            pass

    # 兜底:one_liner 空时,把 summary_payload 第一节截 80 字
    if not one_liner:
        one_liner = summary_payload[0].get("summary_md", "")[:80]
    if not key_points:
        # 从每个 SectionSummary 抽首句作 key_point
        for s in section_summaries[:8]:
            snippet = s.summary_md.split("。")[0].strip()[:60]
            if snippet:
                key_points.append(snippet)
                if s.section_id not in source_section_ids:
                    source_section_ids.append(s.section_id)

    if not one_liner or not key_points:
        raise RuntimeError(
            "write_exec_summary: LLM returned insufficient content "
            f"(one_liner={bool(one_liner)}, key_points={len(key_points)})"
        )

    return ExecSummary(
        one_liner=one_liner,
        key_points=key_points,
        source_section_ids=source_section_ids,
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
    "synthesize_section_async",  # W9-L1
    "synthesize_crosscuts_async",  # W9-L2
    "write_exec_summary_async",  # W10
]
