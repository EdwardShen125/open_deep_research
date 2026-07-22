# Phase 11 — HTTP 入口 (server.py) 验收文档

> 依据 P0 后路线图阶段 2: HTTP / GraphQL 入口。改动最小、向后兼容、客户端可用。

## 验收清单

| # | 验收项 | 状态 | 验证 |
| --- | --- | :---: | --- |
| 1 | FastAPI server.py 装在 `open_deep_research/api/` | ✅ | `from open_deep_research.api.server import app` |
| 2 | 4 个 endpoints + 文档 | ✅ | `GET /`, `GET /healthz`, `POST /runs`, `GET /runs/{id}`, `GET /runs/{id}/report`, `GET /docs` |
| 3 | `POST /runs` 异步启动 + 202 + run_id | ✅ | curl `POST /runs {"query":...,"mode":"evidence-only"}` → `202` + JSON `{run_id, status:queued}` |
| 4 | `GET /runs/{id}` 状态 + checkpoint stages | ✅ | 进程内 registry + `evidence.run_checkpoint` 聚合 |
| 5 | `GET /runs/{id}/report` ReportResult 聚合 | ✅ | 从 `evidence.claim` + `evidence.evidence_unit` 聚合 → ClaimStats / sections |
| 6 | `mode=evidence-only` 不需要 Tavily key 跑通 | ✅ | 实测 55ms 完成(空 EU pipeline,checkpoint 写 PG) |
| 7 | `mode=full` 没 TAVILY_API_KEY 时拒绝 (400) | ✅ | test_start_full_mode_without_tavily_key |
| 8 | 自定义 run_id / UUID 校验 / 409 重号 | ✅ | 3 个 test |
| 9 | Background task 失败不挂 server | ✅ | exception → registry status=failed,error 记录 |
| 10 | PG 连接检查(lifespan + healthz) | ✅ | `/healthz` 返 `{pg_ok: true/false}` |
| 11 | 现有 474 测试零破坏 | ✅ | **490 passed, 6 skipped**(从 474 → 490, +16) |

## 启动方式

```bash
# 方式 1: uvicorn 直跑
POSTGRES_HOST=172.17.0.2 POSTGRES_PASSWORD=odr_v2_pg_pass_change_me \
  .venv/bin/uvicorn open_deep_research.api.server:app --host 0.0.0.0 --port 8000

# 方式 2: module entry
POSTGRES_HOST=172.17.0.2 POSTGRES_PASSWORD=odr_v2_pg_pass_change_me \
  .venv/bin/python -m open_deep_research.api.server

# 方式 3: 测试
.venv/bin/python -m pytest tests/test_phase11_api.py -v
```

## Smoke test 实际结果

```bash
$ curl http://127.0.0.1:8765/healthz
{"status":"ok","pg_ok":true,"run_registry_size":0}

$ curl -X POST http://127.0.0.1:8765/runs \
    -H 'Content-Type: application/json' \
    -d '{"query":"State of AI 2024","mode":"evidence-only"}'
{"run_id":"130baf0c-cacc-4307-b85f-a8ca5ff05814","status":"queued",...}

$ curl http://127.0.0.1:8765/runs/130baf0c-cacc-4307-b85f-a8ca5ff05814
{"run_id":"...","status":"completed","duration_ms":55.382, ...}
```

## 设计决策(以及为什么)

| 选择 | 为什么 |
| --- | --- |
| FastAPI BackgroundTasks(进程内) | 不引外部队列,改最小;生产可换 arq/celery,接口不变 |
| evidence-only + full 两种 mode | 不强制 TAVILY_API_KEY;sandbox 可跑通 |
| RunCheckpointDAO 走 PG(不用 singleton) | singleton `get_dao()` 是无 ctx mgr,需每个调用显式 with |
| Registry 进程内(非持久) | 状态可从 PG `evidence.run_checkpoint` 聚合恢复;restart 不丢真状态 |
| ReportResult 直接用阶段 5 类 | 不重写状态机;复用 ok / status / claim_stats 字段 |
| 不接 OAuth / Streaming | 路线图后续阶段(分布式 worker / SSE) |
| 不重写 supervisor | run_pipeline 调用证据-only 模式,向后兼容 |

## 文件清单

```
src/open_deep_research/api/
  __init__.py      (新增) subpackage marker
  server.py        (新增) FastAPI app, 4 endpoints, BackgroundTask

tests/
  test_phase11_api.py  (新增) 16 测试 (1 集成 skipped by env)

deploy/
  PHASE_11_ACCEPTANCE.md (新增) 本文件
```

## 已知限制

- **Registry 不持久化**:server restart 后,process 内 _RUN_REGISTRY 丢;状态仍可从 PG `run_checkpoint` 聚合,但"status 字段"会显示 "not_found"。这是设计选择(后续阶段会接 Redis 或类似)
- **Background task 单进程**:单机跑可,生产多 worker 需换 arq/celery。当前接口形态不变
- **没 streaming/SSE**:GET /runs/{id}/report 是 polling。后续阶段加
- **没 OAuth / 多租户**:路线图后续

## 后续阶段(路线图)

1. **Metrics + Langfuse 补全** (1 天)— Prometheus `/metrics` 端点 + Langfuse 全局 trace
2. **supervisor DAG 重写** (3-4 天)— 1526 行黑盒调度改 DAG-driven
3. **分布式 worker + 多租户** (2-3 天)— arq + Redis + tenant_id 隔离
4. **真 Tavily key 接 full mode** (1 小时)— 环境变量改一下即用

要不要继续走 Metrics + Langfuse?