"""阶段 4 = Runbook v1 阶段 4.4: Langfuse + stage metrics 观测。

设计依据: notes/evidence-pipeline-runbook-v1.md 4.4 节。

关键设计:
- @stage_trace 装饰器把整个 stage 函数包成一个 Langfuse span
- span metadata 带 stage_name / run_id / duration_ms / eu_count / claim_count
- Langfuse 不可用时自动降级为 no-op decorator
- 阶段 7 planner DAG 的 stage 命名要与 STAGES 对齐

与 llm.py._get_langfuse 的关系:
- _get_langfuse 返回 Langfuse 客户端(可能为 None)
- 本模块不直接依赖 Langfuse SDK,而是通过 _get_langfuse 拿客户端 + OTel tracer
- 这样万一 Langfuse 升级 SDK 破坏 API,只影响 llm.py 一个文件
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

from open_deep_research.llm import _get_langfuse

logger = logging.getLogger(__name__)


def stage_trace(
    stage_name: str,
    *,
    run_id: Optional[str] = None,
    extract_counts: Optional[Callable[[Any], dict[str, int]]] = None,
) -> Callable:
    """装饰器:包一个 stage 函数为 Langfuse span。

    Args:
        stage_name: stage 名(extract / verify / merge / grade / write)
        run_id: 可选,如果装饰的函数接受 run_id kwarg,会自动注入
        extract_counts: 可选 callable,接收函数返回值,返回 {"eu_count": N, "claim_count": M}
                        用于把 stage 输出统计打到 span metadata

    用法:
        @stage_trace("extract")
        async def extract_stage(state):
            ...

        @stage_trace("merge", run_id="my_run", extract_counts=lambda r: {"claim_count": len(r)})
        def merge_stage(eus):
            return claims

    Langfuse 不可用时(no env vars / SDK 失败):decorator 是 no-op,函数照常执行。
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            lf = _get_langfuse()
            start = time.monotonic()
            if lf is None:
                result = await fn(*args, **kwargs)
                _log_local_metric(stage_name, time.monotonic() - start, result, extract_counts)
                return result
            try:
                tracer = lf._otel_tracer
                with tracer.start_as_current_span(stage_name) as span:
                    metadata: dict[str, Any] = {"stage": stage_name}
                    if run_id:
                        metadata["run_id"] = run_id
                    result = await fn(*args, **kwargs)
                    duration_ms = (time.monotonic() - start) * 1000
                    metadata["duration_ms"] = round(duration_ms, 1)
                    if extract_counts is not None:
                        try:
                            metadata.update(extract_counts(result))
                        except Exception as e:
                            logger.warning("stage_trace extract_counts failed: %s", e)
                    lf.update_current_span(metadata=metadata)
                    return result
            except Exception as e:
                logger.warning("stage_trace Langfuse span failed (degrading): %s", e)
                return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            lf = _get_langfuse()
            start = time.monotonic()
            if lf is None:
                result = fn(*args, **kwargs)
                _log_local_metric(stage_name, time.monotonic() - start, result, extract_counts)
                return result
            try:
                tracer = lf._otel_tracer
                with tracer.start_as_current_span(stage_name) as span:
                    metadata: dict[str, Any] = {"stage": stage_name}
                    if run_id:
                        metadata["run_id"] = run_id
                    result = fn(*args, **kwargs)
                    duration_ms = (time.monotonic() - start) * 1000
                    metadata["duration_ms"] = round(duration_ms, 1)
                    if extract_counts is not None:
                        try:
                            metadata.update(extract_counts(result))
                        except Exception as e:
                            logger.warning("stage_trace extract_counts failed: %s", e)
                    lf.update_current_span(metadata=metadata)
                    return result
            except Exception as e:
                logger.warning("stage_trace Langfuse span failed (degrading): %s", e)
                return fn(*args, **kwargs)

        import inspect
        if inspect.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator


def _log_local_metric(
    stage_name: str,
    duration_s: float,
    result: Any,
    extract_counts: Optional[Callable[[Any], dict[str, int]]],
) -> None:
    """Langfuse 不可用时,把 metric 打到本地 logger (INFO 级,阶段 4 验收用)。

    阶段 4 验收第 3 条"stage metrics 在 LangSmith/Langfuse 可见":没接 Langfuse 的环境
    也至少能在日志里看到 duration + count。
    """
    extras: dict[str, Any] = {"duration_ms": round(duration_s * 1000, 1)}
    if extract_counts is not None:
        try:
            extras.update(extract_counts(result))
        except Exception:
            pass
    logger.info("[stage-metric] %s %s", stage_name, extras)


def flush_observability() -> None:
    """在 run 结束 / 进程退出前调用,flush Langfuse 事件。"""
    from open_deep_research.llm import flush_langfuse
    flush_langfuse()


def observability_status() -> dict[str, Any]:
    """诊断:Langfuse 是否配置 + 上次 flush 是否成功。"""
    from open_deep_research.llm import langfuse_status
    return langfuse_status()