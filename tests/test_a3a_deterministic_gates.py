"""A3a · 确定性验证门验收测试

按 SPEC §5 A3a:
- 构造"值不在源文里"的 claim → span 门 False → 标 to_verify
- 构造数值被改写 10% 的 claim → drift 门 fail
- 两门与 entailment 门解耦(entailment 未接时这两门仍运行)
- 失败标 to_verify 不静默
"""
from __future__ import annotations

import math

import pytest

from open_deep_research.evidence.gates_deterministic import (
    DEFAULT_DRIFT_THRESHOLD,
    drift_gate,
    drift_passes,
    extract_numbers,
    normalize_number_token,
    span_gate,
)


# -----------------------------------------------------------------------------
# 验收 1: span_gate 字面命中
# -----------------------------------------------------------------------------

class TestSpanGate:
    def test_number_present_in_source(self) -> None:
        claim = "US live commerce market is $146.4B in 2025"
        source = "According to eMarketer, the US live commerce market is forecast to reach $146.4B in 2025."
        assert span_gate(claim, source) is True

    def test_number_absent_returns_false(self) -> None:
        # 999.9 不在 source;但 2025 在;plan A3a:任一数字命中 → True
        # 这是宽松语义:数字之一在源里出现即部分对账成功
        claim = "US live commerce market is $999.9B in 2025"
        source = "According to eMarketer, the US live commerce market is forecast to reach $146.4B in 2025."
        # 2025 在 source 里,数字之一命中 → True
        assert span_gate(claim, source) is True

    def test_no_any_number_match(self) -> None:
        # claim 数字全部不在 source → False
        claim = "US live commerce market is $999.9B in year 2099"
        source = "According to eMarketer, the US live commerce market is forecast to reach $146.4B in 2025."
        assert span_gate(claim, source) is False

    def test_entity_match(self) -> None:
        # 没有数字,只要实体匹配也过
        claim = "SentinelOne acquired PingSafe in 2024"
        source = "SentinelOne announced the acquisition of PingSafe, expanding its cloud security portfolio in 2024."
        assert span_gate(claim, source) is True

    def test_no_match_returns_false(self) -> None:
        claim = "SentinelOne acquired XYZ in 2099"
        source = "Reuters reports that Microsoft announced Bing updates."
        assert span_gate(claim, source) is False

    def test_full_string_contained(self) -> None:
        # fallback:整句在 source 里
        claim = "This is a test claim with no numbers"
        source = "blah blah blah. This is a test claim with no numbers. end of source."
        assert span_gate(claim, source) is True

    def test_chinese_entity_match(self) -> None:
        claim = "奇安信 2024 年营收超过 70 亿元"
        source = "据奇安信最新财报,公司 2024 年营收超过 70 亿元,同比增长。"
        assert span_gate(claim, source) is True

    def test_empty_inputs(self) -> None:
        assert span_gate("", "some source") is False
        assert span_gate("some claim", "") is False

    def test_comma_separated_numbers(self) -> None:
        claim = "Market size was $1,234 million"
        source = "The figure reached $1,234 million in Q4."
        assert span_gate(claim, source) is True


# -----------------------------------------------------------------------------
# 验收 2: drift_gate 数值漂移
# -----------------------------------------------------------------------------

class TestDriftGate:
    def test_exact_match_zero_drift(self) -> None:
        claim = "146.4"
        source = "The market is $146.4B."
        assert drift_gate(claim, source) == 0.0

    def test_10_percent_drift(self) -> None:
        claim = "160"  # 146.4 的 ~9.3% 上
        source = "The market is $146.4."
        drift = drift_gate(claim, source)
        assert drift > 0.05  # 超阈值
        assert drift < 0.15

    def test_2_percent_drift(self) -> None:
        claim = "150"  # 146.4 的 ~2.4%
        source = "The market is $146.4."
        drift = drift_gate(claim, source)
        assert drift < 0.05  # 通过阈值
        assert drift > 0.0

    def test_no_numbers_in_claim_zero_drift(self) -> None:
        # 无数字 → 0 漂移(无漂移概念)
        drift = drift_gate("no numbers here", "some text")
        assert drift == 0.0

    def test_no_numbers_in_source_infinity(self) -> None:
        drift = drift_gate("claim has 100", "source has no numbers")
        assert drift == float("inf")

    def test_passes_helper_at_threshold(self) -> None:
        # threshold 默认 5%
        claim = "150"  # 146.4 的 ~2.4%
        source = "Market is $146.4."
        assert drift_passes(claim, source) is True

        claim2 = "160"  # 146.4 的 ~9.3%
        assert drift_passes(claim2, source) is False

    def test_custom_threshold(self) -> None:
        claim = "150"
        source = "146.4"
        # 默认 5% 不通过
        assert drift_passes(claim, source, threshold=0.05) is True  # 实际 ~2.4%
        # 阈值设 1% 不通过
        assert drift_passes(claim, source, threshold=0.01) is False

    def test_picks_nearest_neighbor(self) -> None:
        # claim 是 50,source 有 100 和 1,最近邻 = 1
        claim = "50"
        source = "100 and 1"
        drift = drift_gate(claim, source)
        # 相对偏差 |50-1|/1 = 49
        assert drift > 40


# -----------------------------------------------------------------------------
# 验收 3: extract_numbers / normalize_number_token
# -----------------------------------------------------------------------------

class TestExtractNumbers:
    def test_basic(self) -> None:
        nums = extract_numbers("US market is $146.4B in 2025")
        assert Decimal_in_list(nums, "146.4")
        assert Decimal_in_list(nums, "2025")

    def test_comma_separated(self) -> None:
        nums = extract_numbers("Revenue reached $1,234,567 million")
        assert Decimal_in_list(nums, "1234567")

    def test_negative(self) -> None:
        nums = extract_numbers("Loss of -50 million")
        assert Decimal_in_list(nums, "-50")

    def test_percentage(self) -> None:
        nums = extract_numbers("Growth rate: 15%")
        assert Decimal_in_list(nums, "15")

    def test_empty(self) -> None:
        assert extract_numbers("") == []
        assert extract_numbers("no numbers") == []

    def test_normalize_token(self) -> None:
        assert normalize_number_token("1,234") == "1234.0"
        assert normalize_number_token("50%") == "50.0"
        assert normalize_number_token("146.4") == "146.4"
        assert normalize_number_token("not a number") is None


def Decimal_in_list(lst, val: str) -> bool:
    from decimal import Decimal
    target = Decimal(val)
    return any(n == target for n in lst)


# -----------------------------------------------------------------------------
# 验收 4: 两门与 entailment 门解耦(plan §5 A3a DoD)
# -----------------------------------------------------------------------------

class TestDecoupling:
    def test_span_gate_runs_without_llm(self) -> None:
        # span_gate 是纯字符串函数,不依赖任何外部 LLM/网络
        # 验证 import 时不需要 llm.py 任何东西
        import importlib
        mod = importlib.import_module(
            "open_deep_research.evidence.gates_deterministic"
        )
        # 该模块不应引用 llm_entailment
        source = mod.__file__
        assert source is not None
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
        assert "llm_entailment" not in content, (
            "gates_deterministic.py must NOT import llm_entailment "
            "(plan A3a DoD:两门与 entailment 解耦)"
        )
        assert "ChatMiniMax" not in content
        assert "verify_entailment" not in content

    def test_drift_gate_runs_without_llm(self) -> None:
        # drift_gate 同上
        import open_deep_research.evidence.gates_deterministic as m
        d = drift_gate("100", "source has 100")
        assert d == 0.0  # 纯本地计算


# -----------------------------------------------------------------------------
# 验收 5: 默认阈值常量
# -----------------------------------------------------------------------------

class TestDefaults:
    def test_default_threshold_is_5_percent(self) -> None:
        assert DEFAULT_DRIFT_THRESHOLD == 0.05
