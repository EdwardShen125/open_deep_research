"""阶段 6 = Runbook v1 阶段 6: grade retro loop 反馈回路测试。

Mock 化完整离线,跑得快。

验收标准(Runbook 阶段 6):
1. D 占比超阈值 → 触发 retry(reset merge+grade)
2. 重试 N 次后通过 → 状态 ok / fallback_used
3. 重试耗尽仍不达标 → 状态 failed(写 failures)
4. 第一次跑就通过 → 无 retry history
5. 第一次跑 stage 异常 → failed,带具体 stage+error
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest

from open_deep_research.evidence import (
    DEFAULT_D_THRESHOLD,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_STAGES,
    ResearchJob,
    STAGES,
    detect_grade_d_pct,
    reset_run,
    retro_summary,
    run_with_retro_loop,
    should_retry,
)
from open_deep_research.evidence import checkpoint as ckpt_mod
from open_deep_research.evidence import grade_retro
from open_deep_research.evidence.report import Failure, ReportResult


# =============================================================================
# Mock CheckpointDAO (与 test_phase6_checkpoint_job 同步 — 复用)
# =============================================================================


class MockCheckpointDAO:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict] = {}

    def __enter__(self) -> "MockCheckpointDAO":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def upsert(
        self,
        run_id: str | Any,
        stage: str,
        *,
        status: str,
        payload: dict | None = None,
    ) -> None:
        self._store[(str(run_id), stage)] = {
            "run_id": str(run_id),
            "stage": stage,
            "status": status,
            "payload": payload or {},
        }

    def get(self, run_id: str | Any, stage: str) -> dict | None:
        return self._store.get((str(run_id), stage))


@pytest.fixture(autouse=True)
def _inject_mock_dao():
    mock = MockCheckpointDAO()
    prev = ckpt_mod._dao_override
    ckpt_mod.set_dao_override(mock)
    yield mock
    ckpt_mod.set_dao_override(prev)


# =============================================================================
# 1. detect_grade_d_pct + should_retry
# =============================================================================


class _Claim:
    """minimal duck-typed claim with .grade."""

    def __init__(self, grade: str) -> None:
        self.grade = grade


class TestDetectGradeDPct:
    def test_empty_claims_returns_zero(self):
        assert detect_grade_d_pct([]) == 0.0

    def test_all_d(self):
        c = [_Claim("D") for _ in range(5)]
        assert detect_grade_d_pct(c) == 1.0

    def test_no_d(self):
        c = [_Claim("A") for _ in range(3)] + [_Claim("B") for _ in range(2)]
        assert detect_grade_d_pct(c) == 0.0

    def test_mixed(self):
        c = [_Claim("A"), _Claim("D"), _Claim("D"), _Claim("B")]
        assert detect_grade_d_pct(c) == 0.5

    def test_dict_claims_supported(self):
        c = [{"grade": "D"}, {"grade": "A"}]
        assert detect_grade_d_pct(c) == 0.5


class TestShouldRetry:
    def test_default_threshold_is_half(self):
        assert DEFAULT_D_THRESHOLD == 0.5

    def test_retry_when_d_above_threshold(self):
        c = [_Claim("D") for _ in range(4)] + [_Claim("A")]
        assert should_retry(c, threshold=0.5) is True

    def test_no_retry_when_d_at_threshold(self):
        """boundary:D 占比 == threshold 不重试 (strictly >)"""
        c = [_Claim("D"), _Claim("A")]
        # D=0.5, threshold=0.5 → NOT retry
        assert detect_grade_d_pct(c) == 0.5
        assert should_retry(c, threshold=0.5) is False

    def test_no_retry_when_d_below_threshold(self):
        c = [_Claim("A"), _Claim("B")]
        assert should_retry(c, threshold=0.5) is False

    def test_custom_threshold(self):
        c = [_Claim("A"), _Claim("A"), _Claim("A"), _Claim("A")]
        assert detect_grade_d_pct(c) == 0.0
        assert should_retry(c, threshold=0.1) is False

    def test_threshold_zero_means_retro_disabled(self):
        """threshold=0.0 时 D=0 也不触发(因为严格 >),等价于禁用。"""
        c = [_Claim("A"), _Claim("A"), _Claim("B")]
        assert should_retry(c, threshold=0.0) is False


# =============================================================================
# 2. run_with_retro_loop — 端到端 mock 编排
# =============================================================================


def _claim_obj(grade: str) -> Any:
    """构造一个 minimal claim, 接受 .grade 属性 — duck type."""
    return _Claim(grade)


def _make_stages_with_initial_state(initial_claims: list[Any]) -> ResearchJob:
    """5-stage pipeline,merge + grade 每次跑都基于 'claims' state 推导 grade。

    注:为了模拟"重跑 merge+grade 后 grade 变好",我们用 counter 跟踪 retry 次数。
    """

    counter = {"merge_runs": 0, "grade_runs": 0}

    async def extract(state, ctx):
        return {"extracted": 5}

    async def verify(state, ctx):
        return None

    async def merge(state, ctx):
        counter["merge_runs"] += 1
        # 第一次跑给出 D-heavy claims; 后续跑给出 A-heavy
        if counter["merge_runs"] == 1:
            state["claims"] = list(initial_claims)
        else:
            state["claims"] = [_claim_obj("A"), _claim_obj("A"), _claim_obj("B")]
        return state

    async def grade(state, ctx):
        counter["grade_runs"] += 1
        return {"graded": True}

    async def write(state, ctx):
        return {"wrote": True}

    job = ResearchJob(stages=list(zip(STAGES, [extract, verify, merge, grade, write])))
    return job


class TestRunWithRetroLoop:
    @pytest.mark.asyncio
    async def test_first_pass_succeeds_no_retry(self):
        """第一次跑通过(D 占比 < 阈值)— 不应触发 retro loop。"""
        # 全 A
        initial = [_claim_obj("A"), _claim_obj("A"), _claim_obj("B")]
        job = _make_stages_with_initial_state(initial)

        rid = str(uuid4())
        result = await run_with_retro_loop(
            job, rid, {"claims": [], "research_brief": "test"},
            threshold=0.5,
        )

        assert result.status == "ok"
        assert result.ok is True
        assert len(result.warnings) == 0
        assert len(result.failures) == 0

    @pytest.mark.asyncio
    async def test_retro_loop_triggers_and_succeeds(self):
        """核心验收: D 占比过高 → 触发 retro → 重试后通过。"""
        initial = [_claim_obj("D"), _claim_obj("D"), _claim_obj("D"), _claim_obj("A")]
        job = _make_stages_with_initial_state(initial)
        rid = str(uuid4())

        result = await run_with_retro_loop(
            job, rid, {"claims": [], "research_brief": "test"},
            threshold=0.5, max_retries=3,
        )

        assert result.status in ("ok", "fallback_used")
        assert result.ok is True
        # retro 至少跑过一次(attempt #1 succeeded)
        assert any("retro" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_retro_exhausted_returns_failed(self):
        """retro 重试用尽仍不达标 → status=failed。"""
        # 把 merge 写成永远 D-heavy
        async def merge_forever_d(state, ctx):
            state["claims"] = [_claim_obj("D"), _claim_obj("D"), _claim_obj("A")]
            return state

        async def noop(state, ctx):
            return None

        async def extract(state, ctx):
            return {}

        async def verify(state, ctx):
            return None

        async def grade(state, ctx):
            return {}

        async def write(state, ctx):
            return {}

        job = ResearchJob(stages=list(zip(STAGES, [extract, verify, merge_forever_d, grade, write])))
        rid = str(uuid4())

        result = await run_with_retro_loop(
            job, rid, {"claims": []},
            threshold=0.5, max_retries=2,
        )

        assert result.status == "failed"
        assert result.ok is False
        # 应该有 HighGradeDPct failure 记录
        assert any(
            f.error_type == "HighGradeDPct" for f in result.failures
        )

    @pytest.mark.asyncio
    async def test_threshold_zero_means_no_retry_logic(self):
        """threshold=0.5 时 D=0.0 不触发(D <= 阈值); D=0.25 也不触发。

        通过 (不要在测试里再模拟 stage — 只要验证 should_retry 路径)"""
        from open_deep_research.evidence.grade_retro import should_retry as _sr
        # 全 A
        claims_a = [_claim_obj("A"), _claim_obj("A"), _claim_obj("A")]
        assert _sr(claims_a, threshold=0.5) is False
        # D=0.25
        claims_d_low = [_claim_obj("A"), _claim_obj("A"), _claim_obj("A"), _claim_obj("D")]
        assert _sr(claims_d_low, threshold=0.5) is False

    @pytest.mark.asyncio
    async def test_threshold_value_propagation(self):
        """threshold 作为 kwarg 传给 run_with_retro_loop,影响 retro 决策。"""
        # 全 A, 默认 threshold=0.5 → no retro, status=ok
        initial = [_claim_obj("A"), _claim_obj("A"), _claim_obj("B")]
        job = _make_stages_with_initial_state(initial)
        rid = str(uuid4())
        result = await run_with_retro_loop(
            job, rid, {"claims": []},
            threshold=0.5,
        )
        assert result.status == "ok"
        assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_initial_stage_failure_returns_failed(self):
        """第一次跑 stage 抛异常 → 直接 failed(不进入 retro 循环)。"""

        async def bad_extract(state, ctx):
            raise RuntimeError("extract kafka dropped")

        async def verify(state, ctx):
            return None

        async def merge(state, ctx):
            return {}

        async def grade(state, ctx):
            return {}

        async def write(state, ctx):
            return {}

        job = ResearchJob(stages=list(zip(STAGES, [bad_extract, verify, merge, grade, write])))
        rid = str(uuid4())

        result = await run_with_retro_loop(
            job, rid, {"claims": []},
        )
        assert result.status == "failed"
        assert result.ok is False
        assert any(
            "extract kafka dropped" in f.error_message for f in result.failures
        )

    @pytest.mark.asyncio
    async def test_retro_resets_merge_and_grade_only(self):
        """core 验收:retro 只 reset merge + grade,extract/verify 保持 done。"""
        from open_deep_research.evidence import list_completed_stages
        counter = {"extract_runs": 0, "verify_runs": 0, "merge_runs": 0, "grade_runs": 0}

        async def extract(state, ctx):
            counter["extract_runs"] += 1
            return {"e": 1}

        async def verify(state, ctx):
            counter["verify_runs"] += 1
            return None

        async def merge(state, ctx):
            counter["merge_runs"] += 1
            # 第二次跑就改成好 grade
            if counter["merge_runs"] == 1:
                state["claims"] = [_claim_obj("D"), _claim_obj("D"), _claim_obj("A")]
            else:
                state["claims"] = [_claim_obj("A"), _claim_obj("A"), _claim_obj("A")]
            return state

        async def grade(state, ctx):
            counter["grade_runs"] += 1
            return {}

        async def write(state, ctx):
            return {}

        job = ResearchJob(stages=list(zip(STAGES, [extract, verify, merge, grade, write])))
        rid = str(uuid4())
        await run_with_retro_loop(job, rid, {"claims": []}, threshold=0.5, max_retries=3)

        # extract + verify 只跑过一次,merge + grade 跑过 2+ 次
        assert counter["extract_runs"] == 1, "extract should NOT be retried"
        assert counter["verify_runs"] == 1, "verify should NOT be retried"
        assert counter["merge_runs"] >= 2, "merge should be retried"
        assert counter["grade_runs"] >= 2, "grade should be retried"


# =============================================================================
# 3. retro_summary 调试输出
# =============================================================================


class TestRetroSummary:
    def test_summary_ok(self):
        result = ReportResult.from_markdown_and_status(
            "body", status="ok",
        )
        s = retro_summary(result)
        assert "最终 status: **ok**" in s
        assert "ok 硬信号: **True**" in s

    def test_summary_with_warnings(self):
        result = ReportResult.from_markdown_and_status(
            "", status="fallback_used",
            warnings=["retro retry #1: D pct was 0.75"],
            failures=[Failure(stage="retro_attempt_1", error_type="HighGradeDPct", error_message="still > 0.5")],
        )
        s = retro_summary(result)
        assert "status: **fallback_used**" in s
        assert "Warnings" in s
        assert "Failures" in s


# =============================================================================
# 4. 默认常量
# =============================================================================


class TestDefaults:
    def test_threshold(self):
        assert DEFAULT_D_THRESHOLD == 0.5

    def test_max_retries(self):
        assert DEFAULT_MAX_RETRIES == 3

    def test_retry_stages(self):
        assert DEFAULT_RETRY_STAGES == ("merge", "grade")
