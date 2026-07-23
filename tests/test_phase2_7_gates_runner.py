"""Phase 2.7 gates_runner 单元测试。

不调 LLM(run_nli=False),只验证:
  - 闸 1: source_span 非空 → span_verified=True
  - 闸 1: source_span 空   → span_verified=False
  - 闸 2: claim 数值 1:1 在 span → numeric_drift=False
  - 闸 2: claim 数值 漂移 > 0.5% → numeric_drift=True
  - 写回 PG:update_verification 被调用,字段值正确
  - run_nli=False 时:不调 LLM,entailment_* 字段为 None
"""
from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

os.environ.setdefault("POSTGRES_HOST", "172.17.0.2")
os.environ.setdefault("POSTGRES_PASSWORD", "odr_v2_pg_pass_change_me")


def _make_v2_eu(
    *,
    claim: str = "GPT-4 达到 86.4% on MMLU",
    span: str = "GPT-4 reached 86.4% accuracy on the MMLU benchmark",
    eu_id: uuid.UUID | None = None,
):
    """构造一个轻量 v2 EU 替身,只暴露 gates_runner 实际用到的字段。"""
    eu = MagicMock()
    eu.eu_id = eu_id or uuid.uuid4()
    eu.claim = claim
    eu.source_span = span
    return eu


@pytest.mark.asyncio
async def test_gate1_empty_span_rejected():
    from open_deep_research.evidence.gates_runner import run_gates_and_persist

    eus = [
        _make_v2_eu(span=""),  # 空 span
        _make_v2_eu(span="real content here"),
    ]
    # 拦截 EuDAO,避免写 PG
    with patch("open_deep_research.evidence.gates_runner.EuDAO") as mock_dao:
        mock_dao.return_value.__enter__ = MagicMock()
        mock_dao.return_value.__exit__ = MagicMock()
        stats = await run_gates_and_persist(eus, run_id="test", run_nli=False)

    assert stats["total"] == 2
    assert stats["span_verified"] == 1
    assert stats["span_rejected"] == 1
    # 写回被调 2 次(每个 EU 一次)
    assert mock_dao.return_value.__enter__.return_value.update_verification.call_count == 2
    # 第一个 EU span_verified=False, 第二个 True
    calls = mock_dao.return_value.__enter__.return_value.update_verification.call_args_list
    first_call_kwargs = calls[0].kwargs
    second_call_kwargs = calls[1].kwargs
    assert first_call_kwargs["span_verified"] is False
    assert second_call_kwargs["span_verified"] is True


@pytest.mark.asyncio
async def test_gate2_no_drift_when_numbers_match():
    from open_deep_research.evidence.gates_runner import run_gates_and_persist

    eus = [
        _make_v2_eu(
            claim="The model reached 86.4% accuracy",
            span="Our model reached 86.4% accuracy on the test set",
        ),
    ]
    with patch("open_deep_research.evidence.gates_runner.EuDAO") as mock_dao:
        mock_dao.return_value.__enter__ = MagicMock()
        mock_dao.return_value.__exit__ = MagicMock()
        stats = await run_gates_and_persist(eus, run_id="test", run_nli=False)

    assert stats["numeric_drift"] == 0
    call_kwargs = mock_dao.return_value.__enter__.return_value.update_verification.call_args.kwargs
    assert call_kwargs["numeric_drift"] is False


@pytest.mark.asyncio
async def test_gate2_detects_drift_when_numbers_differ():
    from open_deep_research.evidence.gates_runner import run_gates_and_persist

    eus = [
        _make_v2_eu(
            claim="The cost was $50 million",
            span="The cost was $30 million in the report",  # 50 vs 30 — 40% drift
        ),
    ]
    with patch("open_deep_research.evidence.gates_runner.EuDAO") as mock_dao:
        mock_dao.return_value.__enter__ = MagicMock()
        mock_dao.return_value.__exit__ = MagicMock()
        stats = await run_gates_and_persist(eus, run_id="test", run_nli=False)

    assert stats["numeric_drift"] == 1
    call_kwargs = mock_dao.return_value.__enter__.return_value.update_verification.call_args.kwargs
    assert call_kwargs["numeric_drift"] is True


@pytest.mark.asyncio
async def test_nli_disabled_writes_none_verdict():
    """run_nli=False 时不调 LLM,entailment_verdict=None(不入 PG)。"""
    from open_deep_research.evidence.gates_runner import run_gates_and_persist

    eus = [_make_v2_eu()]
    with patch("open_deep_research.evidence.gates_runner.EuDAO") as mock_dao:
        mock_dao.return_value.__enter__ = MagicMock()
        mock_dao.return_value.__exit__ = MagicMock()
        with patch(
            "open_deep_research.evidence.gates_runner.verify_entailment_batch"
        ) as mock_nli:
            mock_nli.return_value = []
            stats = await run_gates_and_persist(eus, run_id="test", run_nli=False)

    assert stats["entailed"] == 0
    assert stats["contradicted"] == 0
    assert stats["unverifiable"] == 0
    # run_nli=False 路径不应调 NLI
    assert mock_nli.call_count == 0


@pytest.mark.asyncio
async def test_empty_input_returns_zero_stats():
    from open_deep_research.evidence.gates_runner import run_gates_and_persist

    stats = await run_gates_and_persist([], run_id="test", run_nli=False)
    assert stats == {
        "total": 0,
        "span_verified": 0,
        "span_rejected": 0,
        "numeric_drift": 0,
        "entailed": 0,
        "contradicted": 0,
        "unverifiable": 0,
    }
