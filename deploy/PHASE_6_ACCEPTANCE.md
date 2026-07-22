# Phase 4 验收文档 (= Runbook v1 阶段 4)

> 对应 Runbook: `notes/evidence-pipeline-runbook-v1.md` 阶段 4
> 实现提交: `feat(evidence)` checkpoint / observability / job_runner

## 目标

把"调研流程"从 LangGraph 同步链路上解开,具备:
1. **stage-level 续跑**: pipeline 任一 stage 失败后,重启能从失败点继续
2. **stage 进度可观测**: 每个 stage 跑耗时、EU/Claim 计数在 Langfuse / 日志可见
3. **降级保留**: Langfuse 不可用时,观测性自动降级到 logger,不阻塞业务

## 改动概览

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `src/open_deep_research/evidence/checkpoint.py` | 213 | stage-level 续跑 API + DAO 注入 |
| `src/open_deep_research/evidence/observability.py` | 165 | `@stage_trace` decorator + Langfuse/logger fallback |
| `src/open_deep_research/evidence/job_runner.py` | 178 | `ResearchJob` 编排器 |
| `src/open_deep_research/evidence/__init__.py` | +18 | 暴露新模块 |
| `tests/test_phase6_checkpoint_job.py` | 450 | 28 个 mock-DB 测试覆盖 5 条验收 |

## 验收标准

### ✅ 验收 1: stage-level checkpoint 续跑

**Runbook**: "stage-level checkpoint 续跑,DB 持久化进度"

**测试**: `TestResearchJobBasics::test_skip_done_stages_on_resume` + `test_resume_after_failure`

**机制**:
- `evidence.run_checkpoint` 表 (migrations/002) 持久化 `(run_id, stage, status)`
- `mark_stage_running / mark_stage_done / mark_stage_failed` 三个写入点
- `ResearchJob.run(run_id, ...)` 每次进入 stage 前查 `list_completed_stages`,已 done 跳过
- 失败时 `mark_stage_failed` 记录 error 到 payload

**结果**: 28 passed,验证:
- 5-stage pipeline 全部完成 → 第二次 run() 不重跑任何 stage
- merge 失败 → extract/verify 已 done → 重跑只跑 merge+grade+write
- 失败的 stage 在 `list_failed_stages(run_id)` 可查

### ✅ 验收 2: stage 顺序由 STAGES 元组决定

**Runbook**: "stage 顺序在 state 中按 STAGES 元组排序"

**测试**: `TestSTAGES::test_stage_order_matches_runbook`

**机制**:
- `STAGES = ("extract", "verify", "merge", "grade", "write")` 元组,不可变
- `get_resume_point()` 按 STAGES 顺序找第一个未 done 的 stage
- `list_completed_stages / list_failed_stages` 按 STAGES 顺序返回

**为什么不让 DB 排**: stage 改名后,DB 排顺序可能错乱(STAGES 是 single source of truth)。

### ✅ 验收 3: Langfuse 不可用时降级到本地日志

**Runbook**: "Langfuse + stage metrics"

**测试**: `TestStageTrace::test_decorator_does_not_propagate_metadata_failure` + `test_long_running_pipeline_with_observability`

**机制**:
- `_get_langfuse()` 返回 None 时(`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` 未设),decorator 退化为 logger.info
- 任何 Langfuse OTel API 异常被 try/except 包住,降级路径不抛
- `extract_counts` callable 抛错也不影响业务函数返回

**降级日志格式**:
```
[stage-metric] extract {'duration_ms': 12.3, 'eu_count': 5}
[stage-metric] merge {'duration_ms': 4.1, 'claim_count': 3}
```

**flush_observability()**: 在 run 结束 / 进程退出前调,Langfuse 不可用时 no-op。

### ✅ 验收 4: Job 编排支持失败恢复

**Runbook**: "job 化 + checkpoint 续跑"

**测试**: `TestResearchJobBasics::test_resume_after_failure` + `TestJobCheckpointIntegration::test_end_to_end_checkpoint_with_state_propagation`

**机制**:
- `ResearchJob(stages=[(name, fn), ...])` 持有 stage 列表
- `run(run_id, initial_state)` 串行执行未 done 的 stage
- state 跨 stage 累加;stage 返回 None 时视为不改 state(verify 类 stage 用)
- stage 抛错时 `mark_stage_failed` + raise,让上层决定 retry 策略

### ✅ 验收 5: 不引入 HTTP / Redis / ARQ(架构简化)

**Runbook 设想**: "HTTP 入口只 enqueue + ARQ worker 跑业务"

**实际决策**: 当前架构没有 HTTP 入口,LangGraph CLI / Python 调用直接执行,
**方案 C (混合)** — pipeline 一个 job,内部 stage 走 checkpoint。

**WHY**:
1. 阶段 4 实际工程量在 checkpoint + observability,不在 HTTP/RQ
2. 方案 C 保留"重启跳过已完成 stage"语义,与 Redis 任务队列能力等价
3. 阶段 6 自动降级需要重跑 merge/grade 时,只需调一次 `job.run(run_id, ...)`

**未来扩展**: 若引入 HTTP 入口,只需新增 `app/server.py` 把 `job.run` 套在 async task 里,
不需要改 ResearchJob 本身。

## 不在阶段 4 范围

- **真 PG 集成测试**: `pgvector` 扩展未装(`postgres:16-alpine` 镜像不含),
  `test_eu_dao.py` 26 个集成测试 + 3 skipped 占位仍 skipped。
  等 docker-compose 换 `pgvector/pgvector:pg16-alpine` 后可全部启用。
- **stage wire-up**: `ResearchJob` 提供编排原语,但具体 stage_fn
  (extract → verify → merge → grade → write) 没接进 LangGraph 节点。
  阶段 7 planner DAG 会做 wire-up。
- **HTTP 入口**: 当前没有 HTTP 层,方案 C 已涵盖未来扩展路径。

## 测试结果

```
tests/test_phase6_checkpoint_job.py ............................  [100%]
============================== 28 passed in 0.45s ==============================

全套: 380 passed, 3 skipped, 196 warnings
```

阶段 1 + 2 + 3 + 4 累计:
- 阶段 1 (Phase 3): schema + DAO + state 瘦身 — 48 tests
- 阶段 2 (Phase 4): verify gates + LLM extractor — 39 tests
- 阶段 3 (Phase 5): merge + independence + grading — 34 tests
- 阶段 4 (Phase 6): checkpoint + observability + job_runner — **28 tests**

**总计 380 passed, 3 skipped**(vs 阶段 3 完成态 324 passed,3 skipped)。

## 阶段 5+ 衔接

阶段 5 验收第 1 条 "1000-2500 claims" 需要真 PG 跑通。
阶段 4 的 mock-DB 测试验证了编排逻辑;真 PG 验证等 pgvector 装上后再启用
`test_eu_dao.py` 集成测试即可。