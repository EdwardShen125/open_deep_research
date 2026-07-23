"""阶段 4 实施:把 plan_v2_pipeline.run_pipeline 包成 ResearchJob 5-stage。

设计依据:notes/evidence-pipeline-runbook-v1.md 4.1-4.3 节。

本模块是 **阶段 4 wire-up** 的最终接缝:
- 把 plan_v2_pipeline.run_pipeline (PlanV2RunResult) 拆成 5 个幂等 stage
- 每个 stage 包了 checkpoint mark_running/mark_done (通过 ResearchJob)
- 续跑自动跳过 done stage
- Stage 失败时 mark_stage_failed + raise
- Langfuse 可见 (通过 stage_trace decorator)

5 个 stage:
- setup    : 生成 run_id + planner(幂等:plan_from_brief deterministic)
- extract  : search + eu_extractor → 落 PG evidence_unit 表(upsert 幂等)
- verify   : 3 闸 + 写回 EU 表 gates 列(幂等:用同一 run_id 覆写)
- merge    : build_claims_from_eus + claim 落 PG + claim_id 回填(幂等)
- write    : cited report + verifier + RDO + Rule 4 audit(无 DB 写入)

为什么 5 stage 而不是 1 stage:
- 1 stage = 整个 run_pipeline,粒度太粗,失败需要从 planner 重跑
- 5 stage = 失败可从 search/verify/merge 任一阶段恢复
- 每个 stage 边界是 PG upsert,自然幂等

幂等性保证:
- stage_setup: plan_from_brief 是 LLM 调用,不是幂等的;但 stage 不写 DB,
  重跑会重新调用 LLM — 可接受(只是成本);stage 失败时 LLM 已调过,
  resume 重新调时 prompt 已带 run_id 信息。
- stage_extract: EuDAO.upsert_many 用 ON CONFLICT DO UPDATE,重复调用 OK
- stage_verify: run_gates_and_persist 也是 ON CONFLICT DO UPDATE
- stage_merge: ClaimDAO.upsert_many 同上,claim_id deterministic
- stage_write: 无 DB 写入,可重复执行

为什么不用细粒度 Redis enqueue (方案 C 选择):
- checkpoint 表已经能提供"重启跳过已完成 stage"语义
- 阶段 6 自动降级重跑 merge 时,只调一次 run() 就够
- 避免 worker 进程序列化 EU 的复杂度
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from open_deep_research.evidence.job_runner import ResearchJob
from open_deep_research.plan_v2_pipeline import (
    PlanV2RunResult,
    default_components,
    run_pipeline,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Stage 函数:把 plan_v2_pipeline.run_pipeline 的阶段拆成 ResearchJob 兼容函数
# =============================================================================
# 签名契约: async def fn(state: dict, ctx: dict) -> dict
# - state: 跨 stage 共享的 dict。包含 plan_v2_run_result 引用
# - ctx: {"run_id": str, "logger": logger, ...}
# - 返回: 更新后的 state (允许 None = no-op)


async def _stage_setup(state: dict, ctx: dict) -> dict:
    """stage 1: planner(纯 deterministic,不调 LLM,不写 DB)。

    失败影响:plan_from_brief 重新调,成本 ~0ms(deterministic 拆分)。
    """
    from open_deep_research.planner_v2 import plan_from_brief

    query = state["query"]
    rid = ctx["run_id"]
    logger.info("[stage_setup] query=%s run_id=%s", query[:60], rid)
    plan = plan_from_brief(query, max_subtopics=state.get("max_subtopics", 4))
    state["plan"] = plan
    state["stages_completed"] = state.get("stages_completed", []) + ["setup"]
    return state


async def _stage_extract(state: dict, ctx: dict) -> dict:
    """stage 2: search + eu_extractor + 落 PG evidence_unit + 3 闸 + 写 claim。

    设计妥协:plan_v2_pipeline.run_pipeline 内部 phase 没拆成 phase_fn,
    本 stage 调 run_pipeline 跑完全流程(phase 2-7)。续跑粒度 = 整 pipeline,
    不是 5 sub-stage。后续 commit 把 plan_v2_pipeline 拆 phase_fn 后,本 stage
    会改成只跑 phase 2-3.5;stage_verify 跑 3.55;stage_merge 跑 3.6;
    stage_write 跑 4-7。这样能实现真正的细粒度断点续跑。

    幂等保证(从 PG 端验证,不是从函数调用):
    - EuDAO.upsert_many:ON CONFLICT (eu_id) DO UPDATE
    - ClaimDAO.upsert_many:ON CONFLICT (claim_id) DO UPDATE
    - run_gates_and_persist:同 ON CONFLICT 写回 EU 表的 gates 列
    - 即使 stage_extract 重跑 100 次,PG 端 EU/claim 数量不变(upsert 覆盖)
    """
    rid = ctx["run_id"]
    logger.info(
        "[stage_extract] running full plan_v2 pipeline (phase 2-7) "
        "under checkpoint; run_id=%s",
        rid,
    )
    result: PlanV2RunResult = await run_pipeline(
        state["query"],
        run_id=rid,
        primary=state.get("primary"),
        fallback=state.get("fallback"),
        sources_dao=state.get("sources_dao"),
        cache=state.get("cache"),
        crawler=state.get("crawler"),
        writer_response=state.get("writer_response"),
        title=state.get("title", "Plan v2 Report"),
        max_subtopics=state.get("max_subtopics", 4),
    )
    state["plan_v2_result"] = result
    state["stages_completed"] = state.get("stages_completed", []) + ["extract"]
    if result.error:
        logger.warning(
            "[stage_extract] pipeline returned error=%s "
            "(likely 'no evidence units extracted' under evidence-only mode; "
            "后续 stage 仍 mark_done,但 final passed=False)",
            result.error,
        )
    return state


async def _stage_verify(state: dict, ctx: dict) -> dict:
    """stage 3: 3 闸 + 写回 EU gates 列(由 run_pipeline 已跑过,本 stage marker)。

    续跑时若 state 没有 plan_v2_result(极端:stage_setup 中断),直接返回 None。
    否则仅作为 checkpoint marker,因为 phase 3.55 已由 stage_extract 跑完。

    后续 commit:把 run_gates_and_persist 从 plan_v2_pipeline 拆出,
    本 stage 调 run_gates_and_persist(state["plan_v2_result"].evidence_units)。
    """
    result = state.get("plan_v2_result")
    if not isinstance(result, PlanV2RunResult):
        logger.warning(
            "[stage_verify] no plan_v2_result in state; 跳过 (run_id=%s)",
            ctx["run_id"],
        )
        return state
    logger.info(
        "[stage_verify] gate_stats=%s (本 stage marker; 3 闸 已由 stage_extract 跑)",
        result.gate_stats,
    )
    state["stages_completed"] = state.get("stages_completed", []) + ["verify"]
    return state


async def _stage_merge(state: dict, ctx: dict) -> dict:
    """stage 4: build_claims_from_eus + claim 落 PG + claim_id 回填(marker)。

    同 _stage_verify — 已由 stage_extract 跑完,本 stage 仅作为 checkpoint marker。
    """
    result = state.get("plan_v2_result")
    if not isinstance(result, PlanV2RunResult):
        logger.warning(
            "[stage_merge] no plan_v2_result in state; 跳过 (run_id=%s)",
            ctx["run_id"],
        )
        return state
    logger.info(
        "[stage_merge] %d claims (dist=%s) (本 stage marker)",
        len(result.claims), result.claim_grade_dist,
    )
    state["stages_completed"] = state.get("stages_completed", []) + ["merge"]
    return state


async def _stage_write(state: dict, ctx: dict) -> dict:
    """stage 5: cited report + verifier + RDO + Rule 4 audit(marker)。

    同 _stage_verify / _stage_merge — 已由 stage_extract 跑完。
    """
    result = state.get("plan_v2_result")
    if not isinstance(result, PlanV2RunResult):
        logger.warning(
            "[stage_write] no plan_v2_result in state; 跳过 (run_id=%s)",
            ctx["run_id"],
        )
        return state
    logger.info(
        "[stage_write] verifier.passes=%s url_compliance.high=%d passed=%s (marker)",
        result.verification.passes if result.verification else None,
        sum(1 for u in result.url_compliance if u.severity == "high"),
        result.passed,
    )
    state["stages_completed"] = state.get("stages_completed", []) + ["write"]
    return state


# =============================================================================
# 公开 API
# =============================================================================


def build_plan_v2_stages() -> list[tuple[str, Any]]:
    """构造 plan_v2 pipeline 的 5-stage 编排。

    Returns:
        [(stage_name, async_fn), ...] 给 ResearchJob
    """
    return [
        ("setup", _stage_setup),
        ("extract", _stage_extract),
        ("verify", _stage_verify),
        ("merge", _stage_merge),
        ("write", _stage_write),
    ]


def build_default_research_job(*, strict: bool = False) -> ResearchJob:
    """构造默认的 ResearchJob(plan_v2 5-stage)。

    Args:
        strict: True 时已 done stage 抛 StageAlreadyDone;默认 False 自动跳过

    Returns:
        ResearchJob 实例
    """
    return ResearchJob(build_plan_v2_stages(), strict=strict)


async def run_pipeline_resumable(
    query: str,
    *,
    run_id: str,
    primary: Optional[Any] = None,
    fallback: Optional[Any] = None,
    sources_dao: Optional[Any] = None,
    cache: Optional[Any] = None,
    crawler: Optional[Any] = None,
    writer_response: Optional[str] = None,
    title: str = "Plan v2 Report",
    max_subtopics: int = 4,
) -> PlanV2RunResult:
    """以 ResearchJob 5-stage 模式跑 plan_v2 pipeline。

    与 plan_v2_pipeline.run_pipeline 的区别:
    - 5 个 stage,每个 stage 落 checkpoint(可续跑)
    - Langfuse 可见 (stage_trace decorator)
    - 失败时 mark_stage_failed,resume 跳过已完成 stage
    - 续跑:同一个 run_id 重调,自动从第一个未 done 的 stage 开始

    限制:见 _stage_extract docstring — 由于 plan_v2_pipeline 内部未拆 phase,
    续跑粒度 = 整 pipeline,不是 sub-stage。后续 commit 拆 phase 后能细化。

    续跑返回:若所有 stage 已 done 且 state 中 plan_v2_result 不存在(因
    stage_extract 跳过),自动从 PG 读回 EU + claim 重新组装 PlanV2RunResult。
    """
    job = build_default_research_job()
    initial_state = {
        "query": query,
        "primary": primary,
        "fallback": fallback,
        "sources_dao": sources_dao,
        "cache": cache,
        "crawler": crawler,
        "writer_response": writer_response,
        "title": title,
        "max_subtopics": max_subtopics,
    }
    final_state = await job.run(run_id, initial_state=initial_state)
    result = final_state.get("plan_v2_result")

    if result is not None:
        # 第一次跑 — stage_extract 实际执行,plan_v2_result 在 state 里
        return result

    # 续跑场景:全部 stage skip,state 中无 plan_v2_result。
    # 从 PG 读回 EU + claim 重新组装 PlanV2RunResult。
    logger.info(
        "[run_pipeline_resumable] 续跑命中:all stages skipped, "
        "从 PG 重新读 EU + claim 装配 PlanV2RunResult (run_id=%s)",
        run_id,
    )
    return _rehydrate_from_pg(run_id, query, title)


def _rehydrate_from_pg(run_id: str, query: str, title: str) -> PlanV2RunResult:
    """续跑命中时,从 PG 读 EU + claim 重新组装 PlanV2RunResult。

    当前实现:
    - 读 EU (EuDAO.list_by_run)
    - 读 claim (ClaimDAO.list_by_run)
    - 算 claim_grade_dist
    - 其他字段(cited_report/verification/report_data/url_compliance)为 None/空
      — 因为 stage_write 是 marker,没存到 PG

    这是"伪 run result" — 用于 caller 不抛 KeyError。完整数据应调
    GET /runs/{id}/report 走 PG 聚合路径。
    """
    from open_deep_research.evidence import EuDAO, ClaimDAO

    out = PlanV2RunResult(query=query, run_id=run_id)
    try:
        with EuDAO() as edao:
            eus = edao.list_by_run(run_id)
        with ClaimDAO() as cdao:
            claims = cdao.list_by_run(run_id)
    except Exception as e:
        logger.warning("PG rehydrate failed for run_id=%s: %s", run_id, e)
        out.error = f"rehydrate failed: {e}"
        return out

    # EU 列表是 EvidenceUnitV2(从 EuDAO.list_by_run)而非 EvidenceUnit。
    # plan_v2_pipeline.evidence_units 字段名义上是 list[EvidenceUnit] 但
    # 实际跑的时候也是 V1 对象 — 这里赋 V2 也算行为差异。已知 caveat。
    # type: ignore[arg-type]
    out.evidence_units = eus
    out.claims = claims
    out.claim_grade_dist = {
        g: sum(1 for c in claims if getattr(c, "grade", None) == g)
        for g in "ABCD"
    }
    logger.info(
        "[rehydrate] run_id=%s -> %d EU + %d claims (dist=%s)",
        run_id, len(eus), len(claims), out.claim_grade_dist,
    )
    # passed / verification / cited_report / report_data / url_compliance / gate_stats
    # 留 None/空 — caller 应走 /runs/{id}/report 拿完整聚合
    out.passed = False  # 续跑不重跑 verifier,passed 状态未知
    return out


__all__ = [
    "build_plan_v2_stages",
    "build_default_research_job",
    "run_pipeline_resumable",
]