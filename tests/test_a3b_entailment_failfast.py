"""A3b · entailment 门 fail-fast 验收测试

按 SPEC §5 A3b:
- **必须注入 401 场景** → 抛 LLMGateError,**不**返回"通过"(回归上次事故)
- 三样本(entail / contradict / unknown)分类正确
- verified 仅当三门全过(本测试只验 entail 这一门)
- 与 llm.py 集成(可选 llm 参数)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from open_deep_research.evidence.gates_failfast import (
    LLMGateError,
    _parse_verdict,
    entail_gate,
    entail_gate_async,
)


# =============================================================================
# Mock LLM factory
# =============================================================================

class _MockLLM:
    """模拟一个会抛异常 / 返回内容的 LLM。"""

    def __init__(self, *, content: str = "", raise_exc: Exception | None = None):
        self._content = content
        self._raise = raise_exc

    async def ainvoke(self, messages: list[dict]) -> Any:
        if self._raise is not None:
            raise self._raise
        return MagicMock(content=self._content)

    def invoke(self, messages: list[dict]) -> Any:
        if self._raise is not None:
            raise self._raise
        return MagicMock(content=self._content)


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro) \
        if not asyncio.get_event_loop().is_running() else asyncio.run(coro)


# -----------------------------------------------------------------------------
# 验收 1: parse_verdict 三样本
# -----------------------------------------------------------------------------

class TestParseVerdict:
    @pytest.mark.parametrize("text,expected", [
        ('{"verdict": "entailed", "score": 0.95}', "entailed"),
        ('{"verdict": "contradicted", "score": 0.8}', "contradicted"),
        ('{"verdict": "unknown", "score": 0.3}', "unknown"),
        ('```json\n{"verdict": "entailed", "score": 0.9}\n```', "entailed"),
        ('Some preamble. "verdict": "contradicted" trailing text', "contradicted"),
        ('', None),
        ('{"no_verdict_key": true}', None),
        ('not json at all', None),
        ('{"verdict": "maybe", "score": 0.5}', None),  # 非合法 verdict
    ])
    def test_parse_various(self, text: str, expected) -> None:
        assert _parse_verdict(text) == expected


# -----------------------------------------------------------------------------
# 验收 2: 三样本分类正确(成功路径)
# -----------------------------------------------------------------------------

class TestThreeSamples:
    def test_entailed_sample(self) -> None:
        llm = _MockLLM(content='{"verdict": "entailed", "score": 0.95}')
        verdict = entail_gate("X happened", "X happened.", llm=llm)
        assert verdict == "entailed"

    def test_contradicted_sample(self) -> None:
        llm = _MockLLM(content='{"verdict": "contradicted", "score": 0.9}')
        verdict = entail_gate("X happened", "Y happened.", llm=llm)
        assert verdict == "contradicted"

    def test_unknown_sample(self) -> None:
        llm = _MockLLM(content='{"verdict": "unknown", "score": 0.2}')
        verdict = entail_gate("X happened", "source has nothing relevant.", llm=llm)
        assert verdict == "unknown"

    def test_fenced_json(self) -> None:
        llm = _MockLLM(content='```json\n{"verdict": "entailed"}\n```')
        verdict = entail_gate("x", "x", llm=llm)
        assert verdict == "entailed"


# -----------------------------------------------------------------------------
# 验收 3: ★ 401 注入场景(plan 硬教训 b)
# -----------------------------------------------------------------------------

class TestFailFast:
    def test_401_raises_LLMGateError_not_pass(self) -> None:
        """核心回归测试:MiniMax 401 时必须抛 LLMGateError,不返回 'unknown' 假装通过。"""
        # 模拟 MiniMax 401 异常
        llm = _MockLLM(raise_exc=Exception("401 Unauthorized: invalid api key"))
        with pytest.raises(LLMGateError) as exc_info:
            entail_gate("some claim", "some source", llm=llm)
        # 异常信息应含 401 关键字
        assert "401" in str(exc_info.value) or "Unauthorized" in str(exc_info.value)

    def test_timeout_raises_LLMGateError(self) -> None:
        llm = _MockLLM(raise_exc=asyncio.TimeoutError("LLM call timed out"))
        with pytest.raises(LLMGateError):
            entail_gate("claim", "source", llm=llm)

    def test_connection_error_raises_LLMGateError(self) -> None:
        llm = _MockLLM(raise_exc=ConnectionError("refused"))
        with pytest.raises(LLMGateError):
            entail_gate("claim", "source", llm=llm)

    def test_empty_response_raises_LLMGateError(self) -> None:
        llm = _MockLLM(content="")
        with pytest.raises(LLMGateError, match="empty"):
            entail_gate("claim", "source", llm=llm)

    def test_whitespace_only_raises_LLMGateError(self) -> None:
        llm = _MockLLM(content="   \n\t  ")
        with pytest.raises(LLMGateError, match="empty"):
            entail_gate("claim", "source", llm=llm)

    def test_unparseable_response_raises(self) -> None:
        # LLM 返回不可解析内容
        llm = _MockLLM(content="I think the claim is true")
        with pytest.raises(LLMGateError, match="parse"):
            entail_gate("claim", "source", llm=llm)

    def test_empty_claim_raises(self) -> None:
        llm = _MockLLM(content='{"verdict":"entailed"}')
        with pytest.raises(LLMGateError):
            entail_gate("", "source", llm=llm)

    def test_empty_source_raises(self) -> None:
        llm = _MockLLM(content='{"verdict":"entailed"}')
        with pytest.raises(LLMGateError):
            entail_gate("claim", "", llm=llm)


# -----------------------------------------------------------------------------
# 验收 4: 与 llm.py 集成(可选参数)
# -----------------------------------------------------------------------------

class TestLLMIntegration:
    def test_passes_llm_directly_no_imports(self) -> None:
        """当 llm 参数非 None 时,不调 get_llm()(避免触发真实网络)。"""
        llm = _MockLLM(content='{"verdict":"entailed"}')
        # 显式传 llm,不应触发任何 get_llm 路径
        verdict = entail_gate("x", "x", llm=llm)
        assert verdict == "entailed"

    def test_async_path(self) -> None:
        llm = _MockLLM(content='{"verdict":"entailed"}')

        async def _call():
            return await entail_gate_async("x", "x", llm=llm)

        result = _run_async(_call())
        assert result == "entailed"


# -----------------------------------------------------------------------------
# 验收 5: 与 A3a 解耦(plan A3a DoD 反向)
# -----------------------------------------------------------------------------

class TestDecoupling:
    def test_no_import_of_deterministic_gates(self) -> None:
        import open_deep_research.evidence.gates_failfast as m
        src = m.__file__
        assert src is not None
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        assert "gates_deterministic" not in content, (
            "gates_failfast must NOT depend on gates_deterministic "
            "(plan A3b / A3a 各自独立)"
        )

    def test_does_not_silently_swallow(self) -> None:
        """验证 _call_llm 不返回 fallback 值。"""
        from open_deep_research.evidence.gates_failfast import _call_llm
        llm = _MockLLM(raise_exc=Exception("network down"))
        with pytest.raises(LLMGateError):
            asyncio.run(_call_llm([{"role": "user", "content": "x"}], llm))


# -----------------------------------------------------------------------------
# 验收 6: 与旧 llm_entailment 的关系(不删旧路径,但新路径默认 fail-fast)
# -----------------------------------------------------------------------------

class TestLegacyPathIntact:
    """plan §4:不删旧 llm_entailment.verify_entailment_batch(保留向后兼容)。"""
    def test_legacy_module_still_importable(self) -> None:
        from open_deep_research.evidence import llm_entailment
        assert hasattr(llm_entailment, "verify_entailment_batch")

    def test_legacy_module_default_silent(self) -> None:
        """旧路径默认行为:异常被吞(unverifiable 兜底)——这是 plan 硬教训 b 的来源。"""
        from open_deep_research.evidence.llm_entailment import verify_entailment_batch
        import inspect
        sig = inspect.signature(verify_entailment_batch)
        # 旧函数签名有 fallback_unverifiable 参数(默认 True 即吞异常)
        assert "fallback_unverifiable" in sig.parameters
        # 默认值 = True(表示"异常时返回 unverifiable 而非抛错")
        assert sig.parameters["fallback_unverifiable"].default is True, (
            "旧路径默认 fallback_unverifiable=True,这就是 plan 硬教训 b 的根。"
            "A3b 改的是新路径 gates_failfast.entail_gate,旧路径保留向后兼容。"
        )
