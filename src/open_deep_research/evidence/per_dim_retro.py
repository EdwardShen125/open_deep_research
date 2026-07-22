"""阶段 7 = Runbook v1 阶段 7.2: per-dimension retro loop。

设计依据: notes/evidence-pipeline-runbook-v1.md 阶段 7 节。

Runbook 设计意图:retro 应该 per-dimension 触发,而不是 per-run。
- run = 多个 dimension 的并行研究
- 每个 dimension 可能 grade 分布差异巨大
  (例: A 维度主源丰富 → A 占比 60%;B 维度冷门 → A 占比 10%)
- 全局 retro 会因为单维度差就重试整个 run → 浪费 token
- per-dimension retro 让 A 维度 OK 时不重试,只重试 B 维度

设计:
- per_dim_retro(job, run_id, state, dimensions, extract_per_dim_claims)
- extract_per_dim_claims(state, dim) -> claims[dim]
- 对每个 dim:if d_pct > threshold → reset run + rerun ResearchJob
- 只要一个 dim 不达标 → 继续重试(run-level)
- 优点: 整 run 失败前能精确诊断是哪个 dim 失败
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional
from uuid import UUID

from open_deep_research.evidence.grade_retro import (
    DEFAULT_D_THRESHOLD,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_STAGES,
    detect_grade_d_pct,
    should_retry,
)
from open_deep_research.evidence.job_runner import ResearchJob
from open_deep_research.evidence.report import Failure, ReportResult

logger = logging.getLogger(__name__)


async def run_with_per_dim_retro(
    job: ResearchJob,
    run_id: str | UUID,
    initial_state: dict,
    dimensions: list[str],
    *,
    threshold: float = DEFAULT_D_THRESHOLD,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_stages: tuple[str, ...] = DEFAULT_RETRY_STAGES,
    extract_per_dim_claims: Optional[Callable[[dict, str], list[Any]]] = None,
) -> ReportResult:
    """per-dimension retro loop 编排。

    Args:
        job: ResearchJob(stages per-dim 已展开)
        run_id: 本次调研 run ID
        initial_state: 初始 state(merged + write 节点都已 wired)
        dimensions: 本 run 的 dimension 列表
        threshold / max_retries / retry_stages: 与 run_with_retro_loop 同
        extract_per_dim_claims: 从 state 取出指定 dim 的 claims 列表的回调
            默认 = lambda s, dim: s.get(f'claims__{dim}', [])

    Returns:
        ReportResult (与 run_with_retro_loop 同)
    """
    rid = str(run_id)
    extract_fn = extract_per_dim_claims or (lambda s, dim: s.get(f"claims__{dim}", []))
    warnings: list[str] = []
    failures: list[Failure] = []

    # 第一次跑
    try:
        state = await job.run(rid, initial_state)
    except Exception as e:
        failures.append(Failure(
            stage="(initial)",
            error_type=type(e).__name__,
            error_message=str(e)[:400],
        ))
        return ReportResult.from_markdown_and_status(
            "", status="failed",
            failures=failures, warnings=warnings,
            run_id=rid,
            research_brief=initial_state.get("research_brief"),
        )

    # per-dim 检查
    dim_d_pct = _per_dim_d_pct(state, dimensions, extract_fn)
    bad_dims = [d for d, p in dim_d_pct.items() if should_retry(
        extract_fn(state, d), threshold=threshold
    )]

    logger.info("[per-dim retro] initial d_pct: %s, bad dims: %s", dim_d_pct, bad_dims)

    if not bad_dims:
        return ReportResult.from_markdown_and_status(
            initial_state.get("__body__", ""),
            status="ok", failures=failures, warnings=warnings,
            run_id=rid, research_brief=initial_state.get("research_brief"),
        )

    # 触发 retro(per-dim 决策,但 run-level 重试)
    for attempt in range(1, max_retries + 1):
        warning_lines = [f"{d}={dim_d_pct[d]:.2f}" for d in bad_dims]
        warnings.append(
            f"per-dim retro #{attempt}: bad dims ({{{', '.join(warning_lines)}}}) > threshold {threshold}"
        )

        # 我们不知道哪些 stage 要 reset — 在 batch_dag_for_dimensions 后,
        # 节点的 name 都是 {stage}__{dim}。retry_stages 是 ('merge', 'grade'),
        # 需要展开成 [f'merge__{d}' for d in bad_dims] + [f'grade__{d}' for d in bad_dims]
        from open_deep_research.evidence.checkpoint import reset_run
        stages_to_reset = []
        for stage in retry_stages:
            for d in bad_dims:
                stages_to_reset.append(f"{stage}__{d}")
        reset_run(rid, stages=stages_to_reset)

        try:
            state = await job.run(rid, state)
        except Exception as e:
            failures.append(Failure(
                stage=f"per_dim_attempt_{attempt}",
                error_type=type(e).__name__,
                error_message=str(e)[:400],
            ))
            continue

        # 重检
        dim_d_pct = _per_dim_d_pct(state, dimensions, extract_fn)
        bad_dims = [d for d, p in dim_d_pct.items() if should_retry(
            extract_fn(state, d), threshold=threshold
        )]

        if not bad_dims:
            warnings.append(
                f"per-dim retro #{attempt} succeeded; all dims <= {threshold}"
            )
            break
        failures.append(Failure(
            stage=f"per_dim_attempt_{attempt}",
            error_type="HighGradeDPct",
            error_message=(
                f"dims still bad: {[(d, dim_d_pct[d]) for d in bad_dims]}"
            ),
        ))

    if not bad_dims:
        return ReportResult.from_markdown_and_status(
            initial_state.get("__body__", ""),
            status="fallback_used" if warnings else "ok",
            failures=failures, warnings=warnings,
            run_id=rid, research_brief=initial_state.get("research_brief"),
        )
    return ReportResult.from_markdown_and_status(
        "", status="failed",
        failures=failures, warnings=warnings,
        run_id=rid, research_brief=initial_state.get("research_brief"),
    )


def _per_dim_d_pct(
    state: dict,
    dimensions: list[str],
    extract_fn: Callable,
) -> dict[str, float]:
    """返回 {dim: d_pct}。"""
    out: dict[str, float] = {}
    for dim in dimensions:
        claims = extract_fn(state, dim)
        out[dim] = detect_grade_d_pct(claims)
    return out


__all__ = [
    "run_with_per_dim_retro",
]
