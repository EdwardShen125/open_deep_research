"""阶段 6 = Runbook v1 阶段 6: 基于 grade 分布的反馈回路。

设计依据: notes/evidence-pipeline-runbook-v1.md 阶段 6 节。

问题: 当 EU 抽取 / 验证 阶段产出质量过低时,grade D 占比会飙升
(>50% 的 claim 都没有任何 EU 通过 entailment)。
此时归并 + grade 阶段重跑也没有意义 — 根因是上游(EU 抽取)。
但**当前 stage 里最经济的动作是:重跑 merge + grade**,让它们重新
consume 已抽取的 EU 列表(可能在重试阶段重新用更严的阈值)。

机制:
- detect_grade_d_pct(claims) -> 推导 D 占比
- should_retry(claims, threshold=0.5) -> True if D 占比 > threshold
- run_with_retro_loop(...) -> 包装 ResearchJob.run,
    失败时 reset_run(['merge','grade']) 重跑,最多 max_retries 次
- 仍失败 → 写 failure + status=failed,告知上层

## 接口边界

本模块**不**修改 ResearchJob / stage_fn 本身 — 它是编排的上层包装。
阶段 7 planner DAG 时也可以套这个 retro loop(per-dimension)。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional
from uuid import UUID

from open_deep_research.evidence.checkpoint import (
    list_failed_stages,
    mark_stage_failed,
    reset_run,
)
from open_deep_research.evidence.job_runner import ResearchJob, StageFn
from open_deep_research.evidence.report import Failure, ReportResult

logger = logging.getLogger(__name__)


# 默认 D 占比阈值。Runbook 验收第 1 条: P3 "无自动降级" → 整改后 D>50% 触发重试
DEFAULT_D_THRESHOLD = 0.5

# 默认最大重试次数。Runbook 第 3 条验收: 3 次仍不达标 → 标 failed
DEFAULT_MAX_RETRIES = 3

# 哪些 stage 在重试时会 reset (Runbook: 仅 merge / grade 阶段)
DEFAULT_RETRY_STAGES = ("merge", "grade")


def detect_grade_d_pct(claims: list[Any]) -> float:
    """推导 D 占比 (grade='D' 数 / 总数)。

    兼容 ClaimV2 / dict / 任意带 .grade 属性的对象。
    Returns 0.0 if claims 为空。
    """
    if not claims:
        return 0.0
    d_count = 0
    for c in claims:
        grade = getattr(c, "grade", None)
        if grade is None and isinstance(c, dict):
            grade = c.get("grade")
        if grade == "D":
            d_count += 1
    return d_count / len(claims)


def should_retry(claims: list[Any], threshold: float = DEFAULT_D_THRESHOLD) -> bool:
    """是否应触发降级重试:D 占比 > threshold。

    threshold=0.5 表示 D 占 >50% 触发。
    threshold=0.0 表示永不触发。
    threshold=1.0 表示 100% D 才触发(几乎不可能)。
    """
    return detect_grade_d_pct(claims) > threshold


async def run_with_retro_loop(
    job: ResearchJob,
    run_id: str | UUID,
    initial_state: dict,
    *,
    threshold: float = DEFAULT_D_THRESHOLD,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_stages: tuple[str, ...] = DEFAULT_RETRY_STAGES,
    extract_claims: Optional[Callable[[dict], list[Any]]] = None,
) -> ReportResult:
    """带 retro loop 的 ResearchJob.run 包装。

    Args:
        job: 已构造的 ResearchJob(stages)
        run_id: 本次调研 run ID
        initial_state: 初始 state dict
        threshold: D 占比阈值,超过触发重试
        max_retries: 重试 merge+grade 的最大次数
        retry_stages: 重试时 reset 哪些 stage(默认 merge+grade)
        extract_claims: 从 state 取出 claims 列表的回调。
            默认 = lambda s: s.get('claims', [])

    Returns:
        ReportResult:
        - status=ok          if 无需重试 或 重试后通过
        - status=fallback_used if 重试后通过但有 retry history
        - status=failed      if 重试用尽仍不达标
    """
    rid = str(run_id)
    extract_claims_fn = extract_claims or (lambda s: s.get("claims", []))
    warnings: list[str] = []
    failures: list[Failure] = []

    # 第一次跑(含完整 5 stage)
    try:
        state = await job.run(rid, initial_state)
    except Exception as e:
        # 第一次就 stage 失败 → 走与后续重试一致的路径
        failures.append(Failure(
            stage="(initial)",
            error_type=type(e).__name__,
            error_message=str(e)[:400],
        ))
        return ReportResult.from_markdown_and_status(
            "",
            status="failed",
            failures=failures,
            warnings=warnings,
            run_id=rid,
            research_brief=initial_state.get("research_brief"),
        )

    # 检查 grade 分布
    claims = extract_claims_fn(state)
    d_pct = detect_grade_d_pct(claims)
    logger.info("[retro] initial run done, %d claims, D pct=%.3f (threshold=%.3f)",
                len(claims), d_pct, threshold)

    if not should_retry(claims, threshold):
        # 通过,无重试
        return ReportResult.from_markdown_and_status(
            initial_state.get("__body__", ""),
            status="ok",
            failures=failures,
            warnings=warnings,
            run_id=rid,
            research_brief=initial_state.get("research_brief"),
        )

    # 需要降级 — 重试 merge+grade
    logger.warning("[retro] D 占比 %.3f > 阈值 %.3f,触发降级重试(最多 %d 次)",
                   d_pct, threshold, max_retries)

    success_after_retry = False
    for attempt in range(1, max_retries + 1):
        warnings.append(f"retro retry #{attempt}: D pct was {d_pct:.3f} > threshold {threshold:.3f}")

        # reset merge + grade (extract / verify 保留 — 它们是真上游,重跑也没用)
        reset_run(rid, stages=list(retry_stages))

        try:
            state = await job.run(rid, state)  # 从 merge 重新跑
        except Exception as e:
            failures.append(Failure(
                stage=f"retro_attempt_{attempt}",
                error_type=type(e).__name__,
                error_message=str(e)[:400],
            ))
            logger.error("[retro] attempt %d failed: %s", attempt, e)
            continue

        # 检查是否已达标
        claims = extract_claims_fn(state)
        d_pct = detect_grade_d_pct(claims)
        logger.info("[retro] attempt %d done, D pct=%.3f", attempt, d_pct)

        if not should_retry(claims, threshold):
            success_after_retry = True
            warnings.append(f"retro retry #{attempt} succeeded: D pct now {d_pct:.3f}")
            break
        else:
            failures.append(Failure(
                stage=f"retro_attempt_{attempt}",
                error_type="HighGradeDPct",
                error_message=f"D pct={d_pct:.3f} still > threshold {threshold:.3f}",
            ))

    if success_after_retry:
        return ReportResult.from_markdown_and_status(
            initial_state.get("__body__", ""),
            status="fallback_used" if warnings else "ok",
            failures=failures,
            warnings=warnings,
            run_id=rid,
            research_brief=initial_state.get("research_brief"),
        )

    # 重试耗尽仍不达标 → failed
    mark_stage_failed(rid, "retro_loop",
                      error=f"D pct={d_pct:.3f} 超过阈值 {threshold:.3f} 经 {max_retries} 次重试仍不达标")
    return ReportResult.from_markdown_and_status(
        "",
        status="failed",
        failures=failures,
        warnings=warnings,
        run_id=rid,
        research_brief=initial_state.get("research_brief"),
    )


def retro_summary(result: ReportResult) -> str:
    """调试:把 retro 决策历史打成一段 markdown(给 operator log / Langfuse span metadata 用)。"""
    parts: list[str] = []
    parts.append(f"- 最终 status: **{result.status}**")
    parts.append(f"- ok 硬信号: **{result.ok}**")
    if result.warnings:
        parts.append(f"- Warnings ({len(result.warnings)}):")
        for w in result.warnings:
            parts.append(f"  - {w}")
    if result.failures:
        parts.append(f"- Failures ({len(result.failures)}):")
        for f in result.failures:
            parts.append(f"  - **{f.stage}** [{f.error_type}]: {f.error_message}")
    return "\n".join(parts)


__all__ = [
    "DEFAULT_D_THRESHOLD",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_STAGES",
    "detect_grade_d_pct",
    "retro_summary",
    "run_with_retro_loop",
    "should_retry",
]
