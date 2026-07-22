"""阶段 4 = Runbook v1 阶段 4.1 + 4.3: pipeline job runner。

设计依据: notes/evidence-pipeline-runbook-v1.md 4.1 / 4.3 节。

关键设计 (方案 C: 混合 — pipeline 一个 job,内部 stage 走 checkpoint 续跑):
- ResearchJob.run(run_id, plan) 是单一入口,串行跑 5 个 stage
- 每个 stage 开始前调用 get_resume_point(run_id),如果 stage 已 done 则跳过
- 每个 stage 失败时 mark_stage_failed 然后 raise(让上层决定 retry)
- 每 stage 包了 @stage_trace,Langfuse 可见
- 失败后用户可以重新调 run(run_id, plan),自动从失败点继续

为什么选方案 C (而不是细粒度 Redis enqueue):
- 当前架构没有 HTTP 入口,不需要拆 worker 进程边界
- checkpoint 已经能提供"重启跳过已完成 stage"的语义
- 阶段 6 自动降级需要重跑 merge/grade 时,方案 C 只调一次 run() 就够
- 避免 Redis job 队列的额外复杂度(序列化 EU、job timeout、worker 并发)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional
from uuid import UUID

from open_deep_research.evidence.checkpoint import (
    STAGES,
    get_resume_point,
    is_run_complete,
    list_completed_stages,
    list_failed_stages,
    mark_stage_done,
    mark_stage_failed,
    mark_stage_running,
)
from open_deep_research.evidence.observability import stage_trace

logger = logging.getLogger(__name__)


# 一个 stage = (name, function) 的契约
# function 签名: async def fn(state: dict, ctx: dict) -> dict (返回新 state)
# ctx 里至少包含 run_id

StageFn = Callable[[dict, dict], Any]


class StageAlreadyDone(Exception):
    """某个 stage 已经 done,跳过执行。仅在 strict 模式下抛出。"""


class ResearchJob:
    """pipeline 编排器。

    用法:
        job = ResearchJob(stages=[
            ("extract", extract_fn),
            ("verify", verify_fn),
            ("merge", merge_fn),
            ("grade", grade_fn),
            ("write", write_fn),
        ])
        result = await job.run(run_id, initial_state={"plan": plan})

    重启续跑:
        # 同一个 run_id,只跑未完成的 stage
        result = await job.run(run_id, initial_state={"plan": plan})
    """

    def __init__(
        self,
        stages: list[tuple[str, StageFn]],
        *,
        strict: bool = False,
    ) -> None:
        """初始化。

        Args:
            stages: [(stage_name, async_fn), ...] 按顺序
            strict: 如果 True,stage 已 done 时抛 StageAlreadyDone;默认 False 直接跳过

        校验:
        - stages 不能为空
        - stage_name 必须唯一
        - stage_name 顺序应与 STAGES 一致(否则 get_resume_point 可能返回错位)
        """
        if not stages:
            raise ValueError("ResearchJob 至少需要一个 stage")
        names = [s for s, _ in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage name 重复: {names}")
        self.stages = stages
        self.strict = strict

    async def run(self, run_id: str | UUID, initial_state: dict) -> dict:
        """串行跑所有未 done 的 stage。

        Returns:
            最终 state(含各 stage 输出)
        """
        rid = str(run_id)
        ctx = {"run_id": rid, "logger": logger}
        state = dict(initial_state)

        completed = list_completed_stages(rid)
        failed = list_failed_stages(rid)
        if failed:
            logger.warning("[job %s] 失败后重启,跳过已完成 %s,失败 stage: %s",
                           rid, completed, failed)
        elif completed:
            logger.info("[job %s] 续跑模式,跳过已完成 stage: %s", rid, completed)

        for stage_name, stage_fn in self.stages:
            # 续跑逻辑:已 done 的 stage 跳过
            if stage_name in completed and not self.strict:
                logger.info("[job %s] stage '%s' 已 done,跳过", rid, stage_name)
                continue
            if self.strict and stage_name in completed:
                raise StageAlreadyDone(f"stage '{stage_name}' 已 done")

            # 真正的 stage 执行(包 stage_trace)
            await self._execute_stage(rid, stage_name, stage_fn, state, ctx)

        # 最终验证
        if not is_run_complete(rid):
            resume = get_resume_point(rid)
            raise RuntimeError(
                f"[job {rid}] pipeline 跑完后仍不是 complete,resume_point={resume}"
            )
        logger.info("[job %s] 全部 stage 完成", rid)
        return state

    async def _execute_stage(
        self,
        run_id: str,
        stage_name: str,
        stage_fn: StageFn,
        state: dict,
        ctx: dict,
    ) -> None:
        """执行一个 stage,带 checkpoint + observability + 失败处理。"""
        mark_stage_running(run_id, stage_name)
        logger.info("[job %s] stage '%s' 开始", run_id, stage_name)
        # stage_trace 装饰 stage_fn — 但因为 stage_fn 是用户传入的,装饰一次
        wrapped = stage_trace(stage_name)(stage_fn)
        try:
            new_state = await wrapped(state, ctx)
            # stage 返回 None 时当作没改 state(允许 verify 类 stage)
            if new_state is not None:
                state.update(new_state)
            mark_stage_done(run_id, stage_name, payload={"stage": stage_name})
            logger.info("[job %s] stage '%s' done", run_id, stage_name)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            mark_stage_failed(run_id, stage_name, err_msg)
            logger.error("[job %s] stage '%s' failed: %s", run_id, stage_name, err_msg)
            raise


def build_default_stages() -> list[tuple[str, StageFn]]:
    """构造默认的 5-stage pipeline。

    真正的 stage_fn 应该是从现有 deep_researcher 节点抽出来:
    - extract: llm_extractor.extract_eus_from_summary (阶段 2 已实现)
    - verify: verify.verify_eus (阶段 2 已实现)
    - merge: merge.merge_units + pipeline.build_claims_from_eus (阶段 3 已实现)
    - grade: pipeline.build_claims_from_eus 已含 grade (阶段 3)
    - write: ReportResult 结构化输出 (阶段 5 待实现)

    阶段 4 不强行把这些 stage 接进 deep_researcher 的 LangGraph 节点
    (那是阶段 7 planner DAG 的工作)。本模块提供编排原语,
    阶段 5 ReportResult + 阶段 7 planner DAG 时再 wire-up。
    """
    # 占位 — 真正的 stage_fn 在阶段 7 wire-up
    raise NotImplementedError(
        "build_default_stages 需要在阶段 7 planner DAG 中 wire-up "
        "extract → verify → merge → grade → write 的具体函数实现。"
        "阶段 4 只提供编排原语 ResearchJob。"
    )


__all__ = [
    "ResearchJob",
    "StageAlreadyDone",
    "StageFn",
    "STAGES",
]