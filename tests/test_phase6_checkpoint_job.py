"""阶段 4 = Runbook v1 阶段 4: stage-level checkpoint + observability + job_runner 测试。

设计:MockCheckpointDAO 不依赖 psycopg/Postgres,纯 dict 存储,跑得快。
阶段 4 的真集成测试 (test_eu_dao.py 已有 26 个 + 3 skipped) 等 pgvector 装上再说。

Runbook 验收标准:
1. checkpoint 续跑:同 run_id 跑两次,第二次跳过已完成 stage
2. stage 失败后重启从失败点继续
3. Langfuse 不可用时 stage_trace 退化为 no-op(打日志)
4. stage_trace 不阻塞业务(失败不抛)
5. ResearchJob 串行执行 + state 累加
6. stage 顺序 STAGES 与 Runbook 设计一致
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

import pytest

from open_deep_research.evidence import (
    STAGES,
    ResearchJob,
    StageAlreadyDone,
    flush_observability,
    get_resume_point,
    is_run_complete,
    list_completed_stages,
    list_failed_stages,
    mark_stage_done,
    mark_stage_failed,
    mark_stage_running,
    observability_status,
    reset_run,
    stage_trace,
)
from open_deep_research.evidence import checkpoint as ckpt_mod


# =============================================================================
# Mock CheckpointDAO (完全离线)
# =============================================================================


class MockCheckpointDAO:
    """dict-backed CheckpointDAO 实现。"""

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
        payload: Optional[dict] = None,
    ) -> None:
        self._store[(str(run_id), stage)] = {
            "run_id": str(run_id),
            "stage": stage,
            "status": status,
            "payload": payload or {},
        }

    def get(self, run_id: str | Any, stage: str) -> Optional[dict]:
        return self._store.get((str(run_id), stage))


@pytest.fixture(autouse=True)
def _inject_mock_dao():
    """每个测试自动注入 MockCheckpointDAO,teardown 还原。"""
    mock = MockCheckpointDAO()
    prev = ckpt_mod._dao_override
    ckpt_mod.set_dao_override(mock)
    yield mock
    ckpt_mod.set_dao_override(prev)


# =============================================================================
# 1. STAGES 元组 + checkpoint 基础 API
# =============================================================================


class TestSTAGES:
    def test_stage_order_matches_runbook(self):
        """STAGES 顺序必须与 Runbook 4.3 一致。"""
        assert STAGES == ("extract", "verify", "merge", "grade", "write")

    def test_stages_tuple_immutable(self):
        """防呆:STAGES 是 tuple,不能外部修改。"""
        with pytest.raises((AttributeError, TypeError)):
            STAGES[0] = "wrong"  # type: ignore[index]


class TestCheckpointAPI:
    def test_mark_running_then_done(self):
        rid = str(uuid4())
        mark_stage_running(rid, "extract")
        assert get_resume_point(rid) == "extract"  # running ≠ done
        mark_stage_done(rid, "extract")
        assert get_resume_point(rid) == "verify"

    def test_mark_done_with_payload(self):
        rid = str(uuid4())
        mark_stage_done(rid, "extract", payload={"eu_count": 17})
        row = ckpt_mod.get_dao().get(rid, "extract")
        assert row["status"] == "done"
        assert row["payload"]["eu_count"] == 17
        assert row["payload"]["finished"] is True

    def test_mark_failed_preserves_error(self):
        rid = str(uuid4())
        mark_stage_failed(rid, "verify", error="ValueError: bad span")
        row = ckpt_mod.get_dao().get(rid, "verify")
        assert row["status"] == "failed"
        assert "bad span" in row["payload"]["error"]

    def test_get_resume_point_returns_first_undone(self):
        rid = str(uuid4())
        for s in STAGES:
            mark_stage_done(rid, s)
        assert get_resume_point(rid) is None
        # 删掉 write → write 是 resume point
        ckpt_mod.get_dao()._store.pop((rid, "write"))
        assert get_resume_point(rid) == "write"

    def test_get_resume_point_after_partial_run(self):
        rid = str(uuid4())
        for s in ("extract", "verify"):
            mark_stage_done(rid, s)
        assert get_resume_point(rid) == "merge"

    def test_list_completed_stages_ordered_by_STAGES(self):
        rid = str(uuid4())
        # 反向写 — list 仍按 STAGES 顺序
        for s in reversed(STAGES):
            mark_stage_done(rid, s)
        assert list_completed_stages(rid) == list(STAGES)

    def test_list_failed_stages(self):
        rid = str(uuid4())
        mark_stage_done(rid, "extract")
        mark_stage_failed(rid, "verify", "boom")
        mark_stage_done(rid, "merge")
        assert list_failed_stages(rid) == ["verify"]

    def test_is_run_complete(self):
        rid = str(uuid4())
        assert not is_run_complete(rid)
        for s in STAGES:
            mark_stage_done(rid, s)
        assert is_run_complete(rid)

    def test_reset_run_clears_specified_stages(self):
        rid = str(uuid4())
        for s in STAGES:
            mark_stage_done(rid, s)
        assert is_run_complete(rid)
        reset_run(rid, stages=["merge", "grade"])
        assert not is_run_complete(rid)
        assert get_resume_point(rid) == "merge"

    def test_get_stage_payload(self):
        rid = str(uuid4())
        assert ckpt_mod.get_stage_payload(rid, "extract") is None
        mark_stage_done(rid, "extract", payload={"eu_count": 42})
        assert ckpt_mod.get_stage_payload(rid, "extract") == {
            "stage": "extract", "finished": True, "eu_count": 42,
        }


# =============================================================================
# 2. observability — @stage_trace
# =============================================================================


class TestStageTrace:
    def test_decorator_runs_function_normally(self):
        @stage_trace("extract")
        def f(x):
            return x * 2

        assert f(21) == 42

    def test_decorator_async(self):
        @stage_trace("merge")
        async def g(x):
            return x + 1

        assert asyncio.run(g(41)) == 42

    def test_decorator_does_not_propagate_metadata_failure(self, caplog):
        """即使 extract_counts callable 抛错,业务函数仍正常返回。"""
        @stage_trace("grade", extract_counts=lambda r: 1 / 0)
        def h():
            return "ok"

        with caplog.at_level(logging.WARNING):
            assert h() == "ok"

    def test_observability_status_returns_dict(self):
        s = observability_status()
        assert "enabled" in s

    def test_flush_observability_is_noop_when_disabled(self):
        # Langfuse env vars 没设 → no-op,不抛
        flush_observability()


# =============================================================================
# 3. ResearchJob — 编排
# =============================================================================


async def _extract(state, ctx):
    return {"eu_count": 10}


async def _verify(state, ctx):
    return None  # verify 类 stage 不改 state


async def _merge(state, ctx):
    return {"claim_count": 3}


async def _grade(state, ctx):
    return {"a_count": 1}


async def _write(state, ctx):
    return {"wrote": True}


def _default_stages() -> list:
    return list(zip(STAGES, [_extract, _verify, _merge, _grade, _write]))


class TestResearchJobBasics:
    @pytest.mark.asyncio
    async def test_run_all_stages_in_order(self):
        calls: list[str] = []

        async def s1(state, ctx):
            calls.append("extract")
            return {"e": 1}

        async def s2(state, ctx):
            calls.append("verify")
            return {"v": 2}

        async def s3(state, ctx):
            calls.append("merge")
            return {"m": 3}

        async def s4(state, ctx):
            calls.append("grade")
            return {"g": 4}

        async def s5(state, ctx):
            calls.append("write")
            return {"w": 5}

        job = ResearchJob(stages=list(zip(STAGES, [s1, s2, s3, s4, s5])))
        rid = str(uuid4())
        state = await job.run(rid, {"plan": "demo"})
        assert calls == list(STAGES)
        assert state == {"plan": "demo", "e": 1, "v": 2, "m": 3, "g": 4, "w": 5}

    @pytest.mark.asyncio
    async def test_state_accumulates_across_stages(self):
        """5-stage pipeline,state 在每个 stage 间累加。"""
        rid = str(uuid4())

        async def ext(state, ctx):
            return {"eu_count": 10}

        async def ver(state, ctx):
            assert state["eu_count"] == 10
            return None

        async def mer(state, ctx):
            assert state["eu_count"] == 10
            return {"claim_count": 3}

        async def gra(state, ctx):
            assert state["claim_count"] == 3
            return {"a_count": 1}

        async def wr(state, ctx):
            assert state["a_count"] == 1
            return {"wrote": True}

        job = ResearchJob(stages=list(zip(STAGES, [ext, ver, mer, gra, wr])))
        state = await job.run(rid, {})
        assert state == {"eu_count": 10, "claim_count": 3, "a_count": 1, "wrote": True}

    @pytest.mark.asyncio
    async def test_skip_done_stages_on_resume(self):
        """核心验收 1: 同 run_id 跑两次,第二次跳过已完成 stage。"""
        extract_calls = 0
        write_calls = 0

        async def ext(state, ctx):
            nonlocal extract_calls
            extract_calls += 1
            return {"e": 1}

        async def ver(state, ctx):
            return None

        async def mer(state, ctx):
            return {"m": 1}

        async def gra(state, ctx):
            return None

        async def wr(state, ctx):
            nonlocal write_calls
            write_calls += 1
            return {"w": 1}

        job1 = ResearchJob(stages=list(zip(STAGES, [ext, ver, mer, gra, wr])))
        rid = str(uuid4())
        await job1.run(rid, {})
        assert extract_calls == 1
        assert write_calls == 1

        # 第二次跑 — 全部 stage 已 done,应全部跳过
        await job1.run(rid, {})
        assert extract_calls == 1  # 没重跑
        assert write_calls == 1  # 没重跑

    @pytest.mark.asyncio
    async def test_resume_after_failure(self):
        """核心验收 2: stage 失败后,重跑从失败点继续。"""
        extract_calls = 0
        merge_calls = 0
        write_calls = 0

        async def ext(state, ctx):
            nonlocal extract_calls
            extract_calls += 1
            return {"e": 1}

        async def ver(state, ctx):
            return None

        async def bad_merge(state, ctx):
            nonlocal merge_calls
            merge_calls += 1
            raise RuntimeError("merge failed")

        async def gra(state, ctx):
            return None

        async def wr(state, ctx):
            nonlocal write_calls
            write_calls += 1
            return {"w": 1}

        rid = str(uuid4())
        job1 = ResearchJob(stages=list(zip(STAGES, [ext, ver, bad_merge, gra, wr])))
        with pytest.raises(RuntimeError, match="merge failed"):
            await job1.run(rid, {})
        assert extract_calls == 1
        assert merge_calls == 1
        assert write_calls == 0

        # 把 merge 修好,重跑 — extract 已 done 跳过
        async def good_merge(state, ctx):
            nonlocal merge_calls
            merge_calls += 1
            return {"m": 1}

        job2 = ResearchJob(stages=list(zip(STAGES, [ext, ver, good_merge, gra, wr])))
        await job2.run(rid, {})
        assert extract_calls == 1
        assert merge_calls == 2
        assert write_calls == 1

    @pytest.mark.asyncio
    async def test_strict_mode_raises_on_done_stage(self):
        rid = str(uuid4())

        async def s(state, ctx):
            return {}

        mark_stage_done(rid, "extract")
        job = ResearchJob(stages=[("extract", s)], strict=True)
        with pytest.raises(StageAlreadyDone):
            await job.run(rid, {})

    @pytest.mark.asyncio
    async def test_empty_stages_rejected(self):
        with pytest.raises(ValueError, match="至少需要"):
            ResearchJob(stages=[])

    @pytest.mark.asyncio
    async def test_duplicate_stage_names_rejected(self):
        async def s(state, ctx):
            return {}
        with pytest.raises(ValueError, match="重复"):
            ResearchJob(stages=[("extract", s), ("extract", s)])

    @pytest.mark.asyncio
    async def test_stage_exception_marks_failed(self):
        """核心验收 3: stage 抛错 → mark_stage_failed,run_id 进入 list_failed_stages。"""

        async def bad(state, ctx):
            raise ValueError("kaboom")

        async def after(state, ctx):
            return {}

        rid = str(uuid4())
        # 其他 stage 配 no-op 防止 resume_point=verify 报 false positive
        async def noop(state, ctx):
            return None

        job = ResearchJob(stages=[
            ("extract", bad),
            ("verify", noop),
            ("merge", after),
            ("grade", noop),
            ("write", noop),
        ])
        with pytest.raises(ValueError, match="kaboom"):
            await job.run(rid, {})
        assert list_failed_stages(rid) == ["extract"]
        # merge 没跑,不在 done
        assert "merge" not in list_completed_stages(rid)

    @pytest.mark.asyncio
    async def test_stage_returning_none_does_not_clobber_state(self):
        async def verifier(state, ctx):
            return None

        async def writer(state, ctx):
            return {"written": True}

        rid = str(uuid4())
        async def noop(state, ctx):
            return None
        async def empty(state, ctx):
            return {}
        job = ResearchJob(stages=[
            ("extract", noop), ("verify", verifier),
            ("merge", noop), ("grade", noop),
            ("write", writer),
        ])
        state = await job.run(rid, {"original": "keep"})
        assert state == {"original": "keep", "written": True}


# =============================================================================
# 4. 集成场景:ResearchJob + checkpoint + observability 三件套
# =============================================================================


class TestJobCheckpointIntegration:
    @pytest.mark.asyncio
    async def test_long_running_pipeline_with_observability(self, caplog):
        """核心验收 4: stage metrics 在日志里可见(无 Langfuse 时降级)。"""

        @stage_trace("extract", extract_counts=lambda r: {"eu_count": r.get("eu", 0)})
        async def ext(state, ctx):
            await asyncio.sleep(0.01)
            return {"eu": 5}

        @stage_trace("verify")
        async def ver(state, ctx):
            await asyncio.sleep(0.01)
            return None

        @stage_trace("merge", extract_counts=lambda r: {"claim_count": r.get("claim", 0)})
        async def mer(state, ctx):
            await asyncio.sleep(0.01)
            return {"claim": 3}

        @stage_trace("grade", extract_counts=lambda r: {"a_count": r.get("a", 0)})
        async def gra(state, ctx):
            await asyncio.sleep(0.01)
            return {"a": 1}

        async def wr(state, ctx):
            return None

        rid = str(uuid4())
        job = ResearchJob(stages=[
            ("extract", ext), ("verify", ver), ("merge", mer),
            ("grade", gra), ("write", wr),
        ])
        with caplog.at_level(logging.INFO):
            state = await job.run(rid, {})

        assert state == {"eu": 5, "claim": 3, "a": 1}
        assert is_run_complete(rid)

        # 日志应至少出现 stage-metric 行(Langfuse 不可用时降级到 logger)
        assert "[stage-metric]" in caplog.text

    @pytest.mark.asyncio
    async def test_end_to_end_checkpoint_with_state_propagation(self):
        """完整端到端:5 stage 全跑完,state 累加,checkpoint 全 done。"""

        async def ext(state, ctx):
            return {"eus": [1, 2, 3, 4, 5]}

        async def ver(state, ctx):
            return {"verified_eus": len(state["eus"])}

        async def mer(state, ctx):
            return {"claims": [{"id": i} for i in range(2)]}

        async def gra(state, ctx):
            return {"graded": len(state["claims"])}

        async def wr(state, ctx):
            return {"report": {"claims": state["claims"], "grade_dist": "A:2"}}

        job = ResearchJob(stages=list(zip(STAGES, [ext, ver, mer, gra, wr])))
        rid = str(uuid4())
        state = await job.run(rid, {"plan": "test"})

        assert state["verified_eus"] == 5
        assert state["graded"] == 2
        assert state["report"]["grade_dist"] == "A:2"
        assert is_run_complete(rid)
        assert list_completed_stages(rid) == list(STAGES)
        assert list_failed_stages(rid) == []