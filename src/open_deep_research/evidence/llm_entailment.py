"""Phase 4 (= Runbook v1 阶段 2.4) Entailment 校验 LLM 调用层。

依据: notes/evidence-pipeline-runbook-v1.md 阶段 2.4

提供:
  - verify_entailment_batch(items, llm) -> EntailmentBatchResult
  - 内部调 LLM(用 llm.py.get_llm),批量 20 条一次调用
  - 失败/部分失败由 parse_entailment_response 处理(unverifiable 兜底)
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from open_deep_research.evidence.verify import (
    ENTAILMENT_PROMPT,
    EntailmentBatchResult,
    _render_entailment_items,
    parse_entailment_response,
)

logger = logging.getLogger(__name__)


DEFAULT_BATCH_SIZE = 20


def _build_messages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """组装 prompt messages(单条 system + 单条 human)。"""
    user = ENTAILMENT_PROMPT.format(items=_render_entailment_items(items))
    return [
        {"role": "system", "content": "You are a strict claim-entailment judge."},
        {"role": "user", "content": user},
    ]


async def verify_entailment_batch(
    items: list[dict[str, Any]],
    llm: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    fallback_unverifiable: bool = True,
) -> list[EntailmentBatchResult]:
    """对 items 做 entailment 校验,分批调用 LLM。

    items: list[{claim, span}]
    llm: BaseChatModel 实例(支持 ainvoke)

    返回:每个 item batch 一条 EntailmentBatchResult。
    异常被 try/except 兜底:整批 unverifiable,parse_warnings 记录异常。
    """
    if not items:
        return []

    out: list[EntailmentBatchResult] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        try:
            messages = _build_messages(chunk)
            response = await llm.ainvoke(messages)
            raw = getattr(response, "content", str(response))
            if not isinstance(raw, str):
                raw = str(raw)
        except Exception as e:
            logger.warning(
                "entailment LLM call failed (batch start=%d, size=%d): %s",
                start, len(chunk), e,
            )
            if fallback_unverifiable:
                result = EntailmentBatchResult(
                    results=[
                        # 占位(index 由调用方用 batch 偏移修正)
                        __import__("open_deep_research.evidence.verify", fromlist=["EntailmentResult"]).EntailmentResult(
                            index=i, verdict="unverifiable", score=0.0,
                            reason=f"llm_call_failed: {e}",
                        )
                        for i in range(len(chunk))
                    ],
                    raw_response="",
                    parse_warnings=[f"llm_call_failed: {e}"],
                )
            else:
                raise
        else:
            result = parse_entailment_response(raw, n_items=len(chunk))

        # 修 index:chunk 内 0..N-1 → 实际 item 偏移
        offset = start
        for r in result.results:
            r.index += offset
        out.append(result)

    return out


def verify_entailment_batch_sync(
    items: list[dict[str, Any]],
    llm: Any,
    **kwargs: Any,
) -> list[EntailmentBatchResult]:
    """同步版本(测试 / 离线场景用)。"""
    import asyncio
    return asyncio.run(verify_entailment_batch(items, llm, **kwargs))


__all__ = [
    "verify_entailment_batch",
    "verify_entailment_batch_sync",
    "DEFAULT_BATCH_SIZE",
]