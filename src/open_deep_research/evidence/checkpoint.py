"""阶段 4 = Runbook v1 阶段 4.3: stage-level checkpoint 续跑。

设计依据: notes/evidence-pipeline-runbook-v1.md 4.3 节。

与 RunCheckpointDAO 的关系:
- RunCheckpointDAO 提供单条 (run_id, stage) 的 upsert/get 原子操作
- 本模块提供"stage progress"语义:
    * STAGES 元组 = pipeline 全 stage 顺序 (extract/verify/merge/grade/write)
    * get_resume_point(run_id) → 第一个未完成 stage(从前往后)
    * mark_stage_done / mark_stage_failed / mark_stage_running
    * list_completed_stages / list_failed_stages

设计约束:
- 不引入新的 DB 表,只用 evidence.run_checkpoint (migrations/002)
- status ∈ {"running", "done", "failed"}
- stage 顺序由 STAGES 决定,而不是 DB 查出来排序(避免 stage 改名后顺序错乱)
- 内部 DAO 调用走 get_dao() 工厂函数 — 测试时可注入 mock
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol
from uuid import UUID


# Pipeline stage 顺序。重启时按此顺序跳过已完成。
STAGES: tuple[str, ...] = (
    "extract",    # LLM 抽取 EU → evidence_unit
    "verify",     # span/numeric/entailment 三闸 → 更新 EU 字段
    "merge",      # 归并 EU → 生成 Claim
    "grade",      # 计算 grade → 更新 Claim
    "write",      # ReportResult 结构化输出
)


# =============================================================================
# DAO 接口(测试时可 mock)
# =============================================================================


class CheckpointDAOProtocol(Protocol):
    """RunCheckpointDAO 必须实现的最小契约。"""

    def __enter__(self) -> "CheckpointDAOProtocol": ...
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool: ...
    def upsert(self, run_id: str | UUID, stage: str, *, status: str, payload: Optional[dict] = None) -> None: ...
    def get(self, run_id: str | UUID, stage: str) -> Optional[dict]: ...


_dao_override: Optional[CheckpointDAOProtocol] = None


def set_dao_override(dao: Optional[CheckpointDAOProtocol]) -> None:
    """注入 mock DAO(测试用)。传 None 恢复默认 RunCheckpointDAO。"""
    global _dao_override
    _dao_override = dao


def get_dao() -> CheckpointDAOProtocol:
    """返回当前 DAO(测试 override 或默认 RunCheckpointDAO)。

    Lazy import:避免模块级 import 数据库依赖。
    """
    if _dao_override is not None:
        return _dao_override
    from open_deep_research.evidence.eu_dao import RunCheckpointDAO
    return RunCheckpointDAO()


def _run(run_id: str | UUID, stage: str, *, status: str, payload: Optional[dict] = None) -> None:
    """通用 upsert:running 用空 payload,done/failed 才打 finished_at。"""
    with _ctx() as dao:
        dao.upsert(run_id, stage, status=status, payload=payload or {})


def _ctx():
    """以 context manager 形式打开 DAO。"""
    dao = get_dao()
    if hasattr(dao, "__enter__"):
        return dao
    # 裸对象(测试 mock)直接返回 — 调用方负责
    class _NoopCtx:
        def __enter__(self_inner):
            return dao
        def __exit__(self_inner, *a):
            return False
    return _NoopCtx()


# =============================================================================
# Public API
# =============================================================================


def mark_stage_running(run_id: str | UUID, stage: str) -> None:
    """记录 stage 开始(覆盖之前的 status)。"""
    _run(run_id, stage, status="running", payload={"stage": stage})


def mark_stage_done(run_id: str | UUID, stage: str, payload: Optional[dict] = None) -> None:
    """记录 stage 完成。"""
    merged: dict[str, Any] = {"stage": stage, "finished": True}
    if payload:
        merged.update(payload)
    _run(run_id, stage, status="done", payload=merged)


def mark_stage_failed(run_id: str | UUID, stage: str, error: str) -> None:
    """记录 stage 失败(保留 payload 里的 error 供排错)。"""
    _run(
        run_id, stage, status="failed",
        payload={"stage": stage, "error": error},
    )


def get_stage_status(run_id: str | UUID, stage: str) -> Optional[str]:
    """返回某 stage 的 status (None / 'running' / 'done' / 'failed')。"""
    with _ctx() as dao:
        row = dao.get(run_id, stage)
        return row["status"] if row else None


def _stages_with_status(
    run_id: str | UUID,
    status: str,
    *,
    stage_names: Optional[Iterable[str]] = None,
) -> set[str]:
    """返回所有 status=<status> 的 stage 名集合。

    stage_names: 可选 — 指定枚举哪些 stage 名。
        None = 用全局 STAGES 元组(向后兼容)
        Iterable = 指定列表(例如 DAG 节点名)

    内部辅助:避免 list_completed / list_failed 重复 SQL。
    Mock DAO 实现只需提供 .get(),不需要 list 接口。
    """
    targets = list(stage_names) if stage_names is not None else list(STAGES)
    found = set()
    for s in targets:
        with _ctx() as dao:
            row = dao.get(run_id, s)
        if row and row.get("status") == status:
            found.add(s)
    return found


def list_completed_stages(
    run_id: str | UUID,
    *,
    stage_names: Optional[Iterable[str]] = None,
) -> list[str]:
    """列出所有 status='done' 的 stage,按 STAGES 顺序(若不指定)或 stage_names 顺序。

    Args:
        run_id: 调研 run ID
        stage_names: 可选 — 限制查哪些 stage(None = STAGES 元组)
    """
    if stage_names is None:
        # 默认枚举全部 STAGES
        done = _stages_with_status(run_id, "done")
        return [s for s in STAGES if s in done]
    # 指定了 stage_names — 按 stage_names 顺序返回
    targets = list(stage_names)
    done = _stages_with_status(run_id, "done", stage_names=targets)
    return [s for s in targets if s in done]


def list_failed_stages(
    run_id: str | UUID,
    *,
    stage_names: Optional[Iterable[str]] = None,
) -> list[str]:
    """列出所有 status='failed' 的 stage,按 STAGES 顺序(若不指定)或 stage_names 顺序。"""
    if stage_names is None:
        failed = _stages_with_status(run_id, "failed")
        return [s for s in STAGES if s in failed]
    targets = list(stage_names)
    failed = _stages_with_status(run_id, "failed", stage_names=targets)
    return [s for s in targets if s in failed]


def get_resume_point(
    run_id: str | UUID,
    *,
    stage_names: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """返回第一个未 'done' 的 stage (按 STAGES 顺序或 stage_names 顺序)。

    Returns:
        - None: 全部 stage 已 done
        - 第一个 status != 'done' 的 stage 名
    """
    if stage_names is None:
        done = set(list_completed_stages(run_id))
        targets = list(STAGES)
    else:
        targets = list(stage_names)
        done = set(list_completed_stages(run_id, stage_names=targets))
    for s in targets:
        if s not in done:
            return s
    return None


def is_run_complete(
    run_id: str | UUID,
    *,
    stage_names: Optional[Iterable[str]] = None,
) -> bool:
    """判断整个 run 的全部 stage 是否都已 done。

    stage_names: None = 用 STAGES 元组;Iterable = 自定义集合(DAG 节点名)
    """
    return get_resume_point(run_id, stage_names=stage_names) is None


def reset_run(run_id: str | UUID, *, stages: Optional[Iterable[str]] = None) -> None:
    """清空指定 stage 的 checkpoint 记录。

    实现:把 status 写成 'reset' 标记,然后从 done 列表里剔除即可。
    这样 mock DAO (没有 _cur) 也能 work。

    用例:
    - 阶段 4 验收测试中需要"从干净状态跑"
    - 阶段 6 自动降级:grade D 过多时清空 merge + grade 重新跑
    """
    target = list(stages) if stages is not None else list(STAGES)
    with _ctx() as dao:
        for s in target:
            # mock DAO: 删 _store 条目;真 DAO: 走 SQL
            store = getattr(dao, "_store", None)
            if store is not None and (str(run_id), s) in store:
                del store[(str(run_id), s)]
                continue
            cur = getattr(dao, "_cur", None)
            if cur is None:
                # 既不是 mock 也不是有 _cur 的真 DAO — 跳过(测试场景不该走到这里)
                continue
            cur().execute(
                "DELETE FROM evidence.run_checkpoint WHERE run_id = %s AND stage = %s",
                (str(run_id), s),
            )
        if hasattr(dao, "_commit"):
            try:
                dao._commit()
            except Exception:
                pass


def get_stage_payload(run_id: str | UUID, stage: str) -> Optional[dict]:
    """取出某 stage 的 payload (用于跨 stage 传递元信息)。"""
    with _ctx() as dao:
        row = dao.get(run_id, stage)
        return row["payload"] if row else None