"""HTTP 入口 — 让客户端能异步启动 / 监控调研 run。

设计依据: 路线图 阶段 2 — "HTTP 入口"
依据 P0 后路线: 改最小、向后兼容、可独立部署。

Endpoints:
  POST /runs               → 启动新 run, 返回 run_id + 202
  GET  /runs/{id}          → 查 run 状态 + stage 进度 + claim_stats
  GET  /runs/{id}/report   → 查 ReportResult(从 PG evidence.claim 聚合)
  GET  /healthz            → 健康检查
  GET  /                   → server info

启动方式:
    POSTGRES_HOST=... POSTGRES_PASSWORD=... .venv/bin/python -m open_deep_research.api.server
    # 或
    POSTGRES_HOST=... POSTGRES_PASSWORD=... .venv/bin/uvicorn open_deep_research.api.server:app --host 0.0.0.0 --port 8000

设计选择:
  - 后台任务用 FastAPI BackgroundTasks(进程内);不接 Redis / ARQ。
    真生产环境可换 celery / arq;接口形态保持一致。
  - 状态查询走 PG: evidence.run_checkpoint + evidence.claim 聚合。
  - evidence-only 模式(无 Tavily key): 跑 pipeline 全程,但无网络搜索,
    EU 数为 0。HTTP 入口仍跑通,只为客户端可用性。
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# 让 server 可作 module 跑(`python -m open_deep_research.api.server`)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from open_deep_research.evidence.eu_dao import ClaimDAO, EuDAO, RunCheckpointDAO
from open_deep_research.evidence.report import (
    ClaimStats,
    Failure,
    ReportResult,
    ReportSection,
)

logger = logging.getLogger("api.server")


# =============================================================================
# In-process run registry(run_id → metadata)
# 进程内: 记录 started/finished_at/mode/error/result
# 持久化(用于跨重启): evidence.run_checkpoint + evidence.claim
# =============================================================================

_RUN_REGISTRY: dict[str, dict[str, Any]] = {}


def _register_run(run_id: str, *, query: str, mode: str) -> None:
    _RUN_REGISTRY[run_id] = {
        "run_id": run_id,
        "query": query,
        "mode": mode,
        "status": "queued",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "error": None,
        "result_summary": None,
    }


def _update_run(run_id: str, **fields: Any) -> None:
    if run_id in _RUN_REGISTRY:
        _RUN_REGISTRY[run_id].update(fields)


# =============================================================================
# Lifespan: 启动时连接 PG, 关闭时清理
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api.server starting (PG host=%s, port=%s)",
                os.environ.get("POSTGRES_HOST", "127.0.0.1"),
                os.environ.get("POSTGRES_PORT", "5432"))
    # Smoke check
    try:
        with EuDAO() as dao:
            dao._cur().execute("SELECT 1")
        logger.info("PG connection OK")
    except Exception as e:
        logger.error("PG connection failed at startup: %s", e)
        # 不 raise — 让 server 起来,健康检查会暴露问题
    yield
    logger.info("api.server shutting down")


app = FastAPI(
    title="Open Deep Research API",
    version="0.1.0",
    description="异步启动 / 监控调研 run 的 HTTP 入口。依据 P0 后路线图阶段 2 设计。",
    lifespan=lifespan,
)


# =============================================================================
# Pydantic models — request / response shapes
# =============================================================================

class StartRunRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="研究简报")
    mode: str = Field(
        default="evidence-only",
        description="evidence-only = 不调网络;full = 调 Tavily (需 TAVILY_API_KEY)",
    )
    run_id: Optional[str] = Field(default=None, description="可选,自定义 run_id")
    max_subtopics: int = Field(default=4, ge=1, le=10)


class StartRunResponse(BaseModel):
    run_id: str
    status: str
    query: str
    mode: str
    started_at: str


class StageProgress(BaseModel):
    name: str
    status: str  # pending / running / done / failed


class RunStatusResponse(BaseModel):
    run_id: str
    query: str
    mode: str
    status: str
    started_at: Optional[str]
    finished_at: Optional[str]
    duration_ms: Optional[float]
    error: Optional[str]
    stages: list[StageProgress]
    claim_stats: Optional[dict[str, Any]] = None
    eu_stats: Optional[dict[str, Any]] = None  # {total, by_dimension, top_source_domains, by_source_tier}


class ReportResponse(BaseModel):
    run_id: str
    ok: bool
    status: str
    body_markdown: str
    sections: list[dict[str, Any]]
    claim_stats: Optional[dict[str, Any]] = None
    eu_stats: Optional[dict[str, Any]] = None
    failures: list[dict[str, Any]]
    warnings: list[str]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    pg_ok: bool
    run_registry_size: int


# =============================================================================
# Background task — 跑 plan_v2_pipeline
# =============================================================================

async def _run_pipeline_background(run_id: str, query: str, mode: str, max_subtopics: int) -> None:
    """后台任务:跑 pipeline,更新 registry + checkpoint。

    设计:
      - 用 plan_v2_pipeline.run_pipeline,evidence-only 模式(无 Tavily key 也行)
      - 失败不 raise — 写 registry + Failure record
      - 不阻塞主进程(BackgroundTasks 在请求返回后跑)
    """
    started = datetime.now(timezone.utc)
    _update_run(run_id, status="running")

    def _ckpt(stage: str, status: str, payload: Optional[dict] = None) -> None:
        try:
            with RunCheckpointDAO() as rdao:
                rdao.upsert(run_id=run_id, stage=stage, status=status, payload=payload or {})
        except Exception as e:
            logger.warning("checkpoint upsert failed (%s/%s): %s", stage, status, e)

    _ckpt("api_received", "done", {"query": query, "mode": mode})

    from open_deep_research.plan_v2_pipeline import run_pipeline
    from open_deep_research.search_providers import TavilyProvider, SearXNGProvider

    primary = None
    if mode == "full":
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            primary = TavilyProvider(api_key=tavily_key)
    searxng_url = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8080")
    fallback = SearXNGProvider(base_url=searxng_url, timeout=30.0)

    _ckpt("pipeline", "running")
    try:
        result = await run_pipeline(
            query=query,
            run_id=run_id,
            primary=primary,
            fallback=fallback,
            max_subtopics=max_subtopics,
        )

        _ckpt(
            "pipeline", "done",
            {
                "passed": getattr(result, "passed", None),
                "n_eus": len(result.evidence_units or []),
                "has_report": result.cited_report is not None,
                "warnings": list(result.cited_report_warnings or []),
            },
        )

        _update_run(
            run_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result_summary={
                "passed": getattr(result, "passed", None),
                "n_eus": len(result.evidence_units or []),
                "has_report": result.cited_report is not None,
            },
        )
        logger.info("run %s completed in %.1fs (passed=%s, n_eus=%d)",
                    run_id,
                    (datetime.now(timezone.utc) - started).total_seconds(),
                    getattr(result, "passed", None),
                    len(result.evidence_units or []))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error("run %s failed: %s\n%s", run_id, err, traceback.format_exc())
        _ckpt("pipeline", "failed", {"error": err})
        _update_run(
            run_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=err,
        )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=dict)
async def root():
    return {
        "service": "open_deep_research_api",
        "version": "0.1.0",
        "endpoints": ["/runs", "/runs/{id}", "/runs/{id}/report", "/healthz", "/docs"],
    }


@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    pg_ok = True
    try:
        with EuDAO() as dao:
            dao._cur().execute("SELECT 1")
    except Exception as e:
        logger.warning("healthz PG check failed: %s", e)
        pg_ok = False
    return HealthResponse(
        status="ok" if pg_ok else "degraded",
        pg_ok=pg_ok,
        run_registry_size=len(_RUN_REGISTRY),
    )


@app.post(
    "/runs",
    response_model=StartRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_run(req: StartRunRequest, background_tasks: BackgroundTasks):
    """启动一个新 run(异步)。"""
    if req.mode not in ("evidence-only", "full"):
        raise HTTPException(400, f"mode must be 'evidence-only' or 'full', got {req.mode!r}")
    # mode='full' 允许跑当至少一种 search provider 可用:
    #   - TAVILY_API_KEY 存在(走 Tavily primary)
    #   - SEARXNG_URL 存在(走 SearXNG fallback — 即使 Tavily 配额耗尽也能跑)
    # evidence-only 不调网络
    if req.mode == "full":
        if not os.environ.get("TAVILY_API_KEY") and not os.environ.get("SEARXNG_URL"):
            raise HTTPException(
                400,
                "mode='full' requires TAVILY_API_KEY or SEARXNG_URL env var; "
                "use mode='evidence-only' to run without network calls",
            )

    rid = req.run_id or str(uuid4())
    try:
        UUID(rid)  # 校验
    except ValueError:
        raise HTTPException(400, f"run_id must be a valid UUID, got {rid!r}")

    if rid in _RUN_REGISTRY:
        raise HTTPException(409, f"run_id {rid!r} already exists in registry")

    _register_run(rid, query=req.query, mode=req.mode)
    background_tasks.add_task(_run_pipeline_background, rid, req.query, req.mode, req.max_subtopics)

    return StartRunResponse(
        run_id=rid,
        status="queued",
        query=req.query,
        mode=req.mode,
        started_at=_RUN_REGISTRY[rid]["started_at"],
    )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """查 run 状态 + stage 进度 + claim_stats + eu_stats。"""
    try:
        UUID(run_id)
    except ValueError:
        raise HTTPException(400, f"run_id must be a valid UUID, got {run_id!r}")

    meta = _RUN_REGISTRY.get(run_id)

    # 读 checkpoint stages
    stages: list[StageProgress] = []
    try:
        with RunCheckpointDAO() as ckdao:
            cur = ckdao._cur()
            cur.execute(
                "SELECT stage, status, started_at, finished_at FROM evidence.run_checkpoint "
                "WHERE run_id = %s ORDER BY started_at",
                (run_id,),
            )
            for r in cur.fetchall():
                stages.append(StageProgress(name=r[0], status=r[1]))
    except Exception as e:
        logger.warning("checkpoint read for %s failed: %s", run_id, e)

    # claim_stats(从 PG 聚合 — phase 3 闸 2 才有 claim)
    claim_stats: Optional[dict[str, Any]] = None
    try:
        with ClaimDAO() as cdao:
            claims = cdao.list_by_run(run_id)
        if claims:
            stats = ClaimStats.from_claim_list(claims)
            claim_stats = stats.model_dump()
    except Exception as e:
        logger.warning("claim_stats for %s failed: %s", run_id, e)

    # eu_stats(从 PG 聚合 — claim_stats 还没接时,也能立刻看见 EU 分布)
    eu_stats: Optional[dict[str, Any]] = None
    try:
        eu_stats = _build_eu_stats(run_id)
    except Exception as e:
        logger.warning("eu_stats for %s failed: %s", run_id, e)

    duration_ms: Optional[float] = None
    started_at = meta["started_at"] if meta else None
    finished_at = meta.get("finished_at") if meta else None
    if started_at and finished_at:
        try:
            t0 = datetime.fromisoformat(started_at)
            t1 = datetime.fromisoformat(finished_at)
            duration_ms = (t1 - t0).total_seconds() * 1000
        except Exception:
            pass

    return RunStatusResponse(
        run_id=run_id,
        query=meta["query"] if meta else "",
        mode=meta["mode"] if meta else "unknown",
        status=meta["status"] if meta else "not_found",
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        error=meta.get("error") if meta else "run not in registry (maybe restarted)",
        stages=stages,
        claim_stats=claim_stats,
        eu_stats=eu_stats,
    )


def _build_eu_stats(run_id: str) -> dict[str, Any]:
    """从 PG 聚合 EU 统计:{total, by_dimension, top_source_domains, by_source_tier}。

    claim_stats 是 phase 3 闸 2 才填的;eu_stats 立刻可观测 — 跑完一次
    pipeline 就能在 /runs/{id} 看到 EU 分布,不需要等 claim pipeline。
    by_source_tier (P0) 让数据准确性导向可量化 (primary 占比)。
    """
    with EuDAO() as edao:
        total = edao.count_by_run(run_id)
        by_dim = edao.count_by_dimension(run_id)
        top_doms = edao.count_by_source_domain(run_id, limit=10)
        by_tier = edao.count_by_source_tier(run_id)
    return {
        "total": total,
        "by_dimension": dict(by_dim),  # {market_size: 48, adoption: 17, ...}
        "top_source_domains": [{"domain": d, "count": c} for d, c in top_doms],
        "source_domain_count": len(top_doms),
        "by_source_tier": dict(by_tier),  # {primary: 137, secondary: 0, ...}
    }


@app.get("/runs/{run_id}/report", response_model=ReportResponse)
async def get_run_report(run_id: str):
    """从 PG 聚合 ReportResult(结构化,含 status / claim_stats / sections)。"""
    try:
        UUID(run_id)
    except ValueError:
        raise HTTPException(400, f"run_id must be a valid UUID, got {run_id!r}")

    # 从 PG 读 EU + Claim
    try:
        with EuDAO() as edao:
            eus = edao.list_by_run(run_id)
        with ClaimDAO() as cdao:
            claims = cdao.list_by_run(run_id)
    except Exception as e:
        raise HTTPException(500, f"PG read failed: {e}")

    if not claims and not eus:
        raise HTTPException(404, f"no data for run_id {run_id!r}")

    stats = ClaimStats.from_claim_list(claims, eus=eus)
    sections: list[dict[str, Any]] = []
    for c in claims[:50]:
        sections.append({
            "section_id": f"s_{c.dimension_id}_{c.claim_type}",
            "title": f"[{c.grade}] {c.dimension_id}: {c.canonical_claim[:80]}",
            "body_markdown": c.canonical_claim,
            "claim_ids": [str(c.claim_id)],
            "grade": c.grade,
        })

    # 装配 body_markdown
    body_lines = [
        f"# Report for run {run_id}",
        "",
        f"Total claims: {stats.total_claims}",
        f"Grade distribution: {stats.grade_dist_pct}",
        f"Total EUs: {stats.total_eus} (usable: {stats.usable_eus})",
        "",
    ]
    for s in sections:
        body_lines.append(f"## {s['title']}")
        body_lines.append(s["body_markdown"])
        body_lines.append("")

    # status 判定
    if not claims:
        status_str = "failed"
        ok = False
    elif stats.unverified_claims == stats.total_claims and stats.total_claims > 0:
        status_str = "fallback_used"
        ok = True
    elif claims:
        status_str = "ok"
        ok = True
    else:
        status_str = "failed"
        ok = False

    return ReportResponse(
        run_id=run_id,
        ok=ok,
        status=status_str,
        body_markdown="\n".join(body_lines),
        sections=sections,
        claim_stats=stats.model_dump(),
        eu_stats=_build_eu_stats(run_id),
        failures=[],
        warnings=[],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


__all__ = ["app", "start_run", "get_run_status", "get_run_report"]


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("open_deep_research.api.server:app", host=host, port=port, reload=False)