# Runbook v1 — 整改项归档

> 立项日期:2026-07-22 | 最近复核:2026-07-23

## 目的

把 Runbook v1 中**已被架构迭代消解**的整改项归档,避免后人按 Runbook 对照清单再"找一遍"。

## 归档项

### #2 "搬进 ARQ worker + 降级保留"(P1 整改)

**Runbook 原文**:
> P1 summarize 60s 超时 ×301 | 同步请求内长跑 | 搬进 ARQ worker + 降级保留 | 4

**当前状态**:**只做了"降级保留",没做"ARQ worker"**,且**不是漏掉的整改**。

**为什么**:
- `evidence/job_runner.py:12-16` 显式拒绝用 Redis 队列:
  > "避免 Redis job 队列的额外复杂度(序列化 EU、job timeout、worker 并发)"
- evidence-only 模式 pipeline 跑完时间 <1s,BackgroundTasks 完全够用
- ARQ worker 真正需要的场景是 Phase 14(分布式 + 多租户)

**当前部署状态**:
- `docker-compose.yml` 有 `redis` service 但**无 worker service**
- `api/server.py` 用 FastAPI BackgroundTasks(进程内)
- 状态从 `evidence.run_checkpoint` 聚合,跨重启不丢

**何时重启 ARQ worker 整改**:
- Phase 14 多 worker 部署
- 长跑任务 >30s 上限触发 BackgroundTasks 卡死

### #4 "RSS 3.5GB / asyncio 死锁"(P2 整改)

**Runbook 原文**:
> P2 RSS 3.5GB / asyncio 死锁 | state 里驮着 60K EU | EU 出 state(预期连带消失) | 1, 4

**当前状态**:**不存在 RSS 驱动可改**。

**为什么**:
- `git log -- '*rss*' '*feedparser*'` 返回空
- 当前 search 链路是 Tavily API (`AsyncTavilyClient`) + SearXNG(自部署搜索引擎)
- 没有任何代码引用 `feedparser` / RSS / atom_feed
- RSS 是 plan_v1 早期方案,plan_v2 架构切换时已删

**当前部署状态**:
- `search_providers.py` 完整实现 Tavily + SearXNG,**全部异步**
- `crawler.py` 同步 fetcher 走 `loop.run_in_executor`(非阻塞)
- state 已删 `raw_notes / notes`,EU 不在 state 里堆积

**何时重启 RSS 整改**:
- 永远不会,除非 plan_v2 重新引入 RSS 驱动(目前无此计划)

## 仍然有效的整改项(11 条)

| # | Runbook 整改 | 阶段 | 状态 |
| --- | --- | --- | --- |
| 1 | EU 持久化 + 分节检索 | 1, 5 | ✅ Phase 1 + 5 |
| 2 | span 校验 | 2 | ✅ Phase 2 |
| 3 | 语义归并 | 3 | ✅ Phase 3 |
| 4 | ReportResult 结构化 | 5 | ✅ Phase 7 |
| 5 | 降级保留(summarize 180s timeout) | 4 | ✅ Phase 6 |
| 6 | job 化 + checkpoint 续跑 | 4 | ✅ Phase 6 |
| 7 | claim_stats 入 ReportResult | 5 | ✅ Phase 7 |
| 8 | EU 出 state | 1 | ✅ Phase 1 |
| 9 | 基于 grade 分布的反馈回路 | 6 | ✅ Phase 8 |
| 10 | planner 显式 DAG | 7 | ✅ Phase 9 |
| 11 | RunConfig 显式配额 | 7 | ✅ Phase 9 |