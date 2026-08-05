"""A3b · entailment 门(LLM, fail-fast)

按 plan A3b:第三门是 LLM 蕴含判断。
上次事故:MiniMax key 是 placeholder、401、verifier 静默失败,
fallback 文本冒充成功报告——本卡核心是**杜绝**这个。

接口:
    entail_gate(claim, source_text, *, llm=None) -> Literal["entailed","contradicted","unknown"]

行为:
    - auth 失败 / 超时 / 空响应 → raise LLMGateError(**绝不** fallback)
    - 返回结果 → ential / contradicted / unknown
    - 与 llm.py 解耦(可选传 llm;None 时 get_llm())
    - 与 A3a 两门解耦(本函数只看 entail,不管 span/drift)

错误类:
    LLMGateError(Exception):fail-fast 抛错类型
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal, Optional


logger = logging.getLogger(__name__)


# =============================================================================
# 错误类
# =============================================================================

class LLMGateError(Exception):
    """A3b entailment 门的 fail-fast 异常。

    触发条件:
        - LLM auth 失败(401/403)
        - LLM 超时
        - LLM 返回空响应
        - JSON parse 失败(无法解析为合法 entail 标签)
    """


# =============================================================================
# Prompt
# =============================================================================

_ENTAIL_SYSTEM = (
    "You are a strict claim-entailment judge. "
    "Classify whether the claim is logically entailed, contradicted, "
    "or unknown (cannot determine) by the source text. "
    "Reply with EXACTLY one JSON object: "
    '{"verdict": "entailed" | "contradicted" | "unknown", "score": 0.0-1.0}'
)

_ENTAIL_USER_TEMPLATE = """## Claim
{claim}

## Source Text
{source}

## Output (JSON only)
{{"verdict": "entailed|contradicted|unknown", "score": 0.0-1.0}}
"""


def _build_messages(claim: str, source_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _ENTAIL_SYSTEM},
        {"role": "user", "content": _ENTAIL_USER_TEMPLATE.format(
            claim=claim, source=source_text,
        )},
    ]


# =============================================================================
# Verdict parsing
# =============================================================================

import json
import re


_VERDICT_RE = re.compile(
    r'"verdict"\s*:\s*"(entailed|contradicted|unknown)"', re.IGNORECASE
)


def _parse_verdict(text: str) -> Optional[Literal["entailed", "contradicted", "unknown"]]:
    """从 LLM 响应中解析 verdict。

    支持:
        - 纯 JSON
        - JSON 在 ```json ... ``` 围栏里
        - 任意文本含 "verdict":"..."
    """
    if not text:
        return None
    s = text.strip()
    # 1. 严格 JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            v = obj.get("verdict", "").lower()
            if v in ("entailed", "contradicted", "unknown"):
                return v  # type: ignore[return-value]
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. 围栏 JSON
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                v = obj.get("verdict", "").lower()
                if v in ("entailed", "contradicted", "unknown"):
                    return v  # type: ignore[return-value]
        except (json.JSONDecodeError, ValueError):
            pass
    # 3. 正则匹配
    m = _VERDICT_RE.search(s)
    if m:
        return m.group(1).lower()  # type: ignore[return-value]
    return None


# =============================================================================
# LLM 调用(fail-fast)
# =============================================================================

async def _call_llm(messages: list[dict[str, str]], llm: Any) -> str:
    """调 LLM 并返回原始文本。**任何异常都向上抛**,调用方处理 fail-fast。

    触发 LLMGateError 的场景:
        - llm.ainvoke / llm.invoke 抛 auth/timeout/网络异常
        - 响应为空 / 响应.content 为空
    """
    if llm is None:
        from open_deep_research.llm import get_llm
        llm = get_llm()

    # 优先 async
    ainvoke = getattr(llm, "ainvoke", None)
    try:
        if ainvoke is not None:
            response = await ainvoke(messages)
        else:
            invoke = getattr(llm, "invoke", None)
            if invoke is None:
                raise LLMGateError(
                    f"llm object {type(llm).__name__} has neither ainvoke nor invoke"
                )
            response = invoke(messages)
    except LLMGateError:
        raise
    except Exception as e:
        # auth / timeout / network 全部包装为 LLMGateError
        raise LLMGateError(f"LLM call failed: {type(e).__name__}: {e}") from e

    # 提取 content
    content = getattr(response, "content", None)
    if content is None:
        content = str(response) if response is not None else ""
    if not content or not content.strip():
        raise LLMGateError("LLM returned empty response")
    return content


# =============================================================================
# 公开 API:entail_gate
# =============================================================================

async def entail_gate_async(
    claim: str,
    source_text: str,
    *,
    llm: Any = None,
) -> Literal["entailed", "contradicted", "unknown"]:
    """异步 entailment 门。

    Args:
        claim: 待验证的断言
        source_text: 源文本(原文)
        llm: 可选 BaseChatModel;None 时 llm.get_llm()

    Returns:
        "entailed" / "contradicted" / "unknown"

    Raises:
        LLMGateError: auth fail / timeout / empty response / parse fail
                      **绝不** fallback,绝不被调用方吞掉
    """
    if not claim or not source_text:
        raise LLMGateError("entail_gate requires non-empty claim and source_text")

    messages = _build_messages(claim, source_text)
    raw = await _call_llm(messages, llm)

    verdict = _parse_verdict(raw)
    if verdict is None:
        raise LLMGateError(
            f"could not parse verdict from LLM response: {raw[:200]!r}"
        )
    return verdict


def entail_gate(
    claim: str,
    source_text: str,
    *,
    llm: Any = None,
) -> Literal["entailed", "contradicted", "unknown"]:
    """同步版本(供非 async 调用方)。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 当前已经在 async 上下文 → 返回 coroutine 让调用方 await
            return entail_gate_async(claim, source_text, llm=llm)  # type: ignore[return-value]
        return loop.run_until_complete(entail_gate_async(claim, source_text, llm=llm))
    except RuntimeError:
        # 无 event loop
        return asyncio.run(entail_gate_async(claim, source_text, llm=llm))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LLMGateError",
    "entail_gate",
    "entail_gate_async",
    "_parse_verdict",
]
