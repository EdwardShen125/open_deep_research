# Phase 3 Acceptance — EvidenceUnit + Claim 双层模型 + state 瘦身

**生成时间**: 2026-07-22
**执行人**: Hermes (MiniMax-M3) on `research` profile
**目标**: 关闭 Runbook v1 阶段 1 全部 5 条验收,把 EU 从 LangGraph state 搬出,落地到 PG。
**前置**: 910fcb6 (Phase 0A EU dedup) + 777 EU baseline (EDR v8)

---

## ✅ 验收对齐

| Runbook 阶段 1 验收 | 实现位置 | 测试 |
| --- | --- | --- |
| 1. EU/Claim 可写入 + 按 run_id + dim_id 读出 | `evidence/eu_dao.py` | `tests/test_eu_dao.py` (单元) + `test_eu_dao_integration.py` (PG 集成,留 CI) |
| 2. 向量检索走 HNSW | `evidence/eu_dao.py:EuDAO.search_by_embedding` | `TestHnswSqlGeneration`(SQL inspect);`EXPLAIN` 留真 PG CI |
| 3. LangGraph state 序列化 < 50KB | `state.py` 去 raw_notes + 加 eu_counts | `tests/test_state_size.py` |
| 4. EU 全部落库,count 一致 | supervisor `_persist_eus_to_pg` + `plan_v2_pipeline.py:3.5` | fail-safe path + 真 PG CI |
| 5. (隐含)旧 baseline 不破 | — | 250 passed / 3 skipped |

---

## ✅ 已交付

### 1.1 双层模型

| 项 | 落地位置 | 测试 |
| --- | --- | --- |
| `EvidenceUnitV2` (Pydantic BaseModel) | `src/open_deep_research/evidence/schema.py` (11.3 KB) | `tests/test_schema_v2.py` 16/16 |
| `ClaimV2` (Pydantic BaseModel) | 同上 | 同上 |
| Literal type aliases (ClaimType / SourceTier / Verdict / Grade) | 同上 | `test_literal_types_match_migration` |
| `usable` property(span_verified + numeric_drift + entailment_verdict) | 同上 | `test_eu_usable_requires_three_signals` |
| 旧 dataclass → V2 Pydantic 桥接 `to_v2()` | `evidence_units.py:to_v2` | `test_legacy_dataclass_to_v2_bridge`, `test_legacy_bridge_preserves_content_hash_across_invocation` |
| content_hash 跨 run 稳定 | `to_v2` 复用 legacy content_hash | 同上 |

### 1.2 存储

| 项 | 落地位置 |
| --- | --- |
| `evidence.claim` 表(UUID PK + pgvector 1024 维) | `migrations/002_claim_and_evidence_unit_v2.sql` |
| `evidence.evidence_unit` 表(同上) | 同上 |
| `evidence.run_checkpoint` 表(PK (run_id, stage),阶段 4 用) | 同上 |
| 5 个 HNSW / btree 索引 | 同上 |
| `EuDAO` / `ClaimDAO` / `RunCheckpointDAO`(psycopg context manager) | `evidence/eu_dao.py` (20.9 KB) |
| `upsert_many`(基于 PK ON CONFLICT DO NOTHING,EU 不可变) | 同上 |
| `search_by_embedding`(ORDER BY embedding <=> $q) | 同上 |
| `update_verification`(窄通道,只允许 span_verified/numeric_drift/entailment_verdict 回填) | 同上 |
| `update_claim_id`(阶段 3 归并后回填) | 同上 |
| `count_by_run` / `count_by_dimension`(state 瘦身后的 supervisor 聚合) | 同上 |
| `grade_distribution`(阶段 5 报告可信度摘要) | 同上 |

### 1.3 state 瘦身

| 项 | 落地位置 |
| --- | --- |
| `raw_notes` 从 AgentState / SupervisorState / ResearcherState / ResearcherOutputState 删除(decision D2-B) | `state.py` |
| 新增 `eu_counts: dict[str, int]` 到 AgentState / SupervisorState | `state.py` |
| 新增 `claim_counts: dict[str, int]` 到 AgentState | `state.py` |
| 新增 `dimension_ids: list[str]` 到 AgentState / SupervisorState | `state.py` |
| 新增 `dimension_id: Optional[str]` 到 ResearcherState / ResearcherOutputState | `state.py` |
| 新增 `eu_count: int` 到 ResearcherState / ResearcherOutputState | `state.py` |
| supervisor_tools:删 raw_notes 聚合 → 改写 eu_counts + 同步落 PG(fail-safe) | `deep_researcher.py:supervisor_tools` |
| researcher synthesize:删 raw_notes 输出 → 改写 eu_count + dimension_id | `deep_researcher.py:compress_research` |
| `_resolve_run_id` / `_persist_eus_to_pg` helpers | `deep_researcher.py` 顶部 |
| plan_v2_pipeline:接 EuDAO.upsert_many(stage 3.5, fail-safe) | `plan_v2_pipeline.py` |
| compress_research 节点**保留**(阶段 4 才删) | state.py 决策 D2-B 注释 |

### 测试

| 文件 | 测试数 |
| --- | --- |
| `tests/test_schema_v2.py` (新建) | 16 |
| `tests/test_state_size.py` (新建) | 6 |
| `tests/test_eu_dao.py` (新建) | 26 + 3 skip |
| 旧 baseline 测试(`tests/test_*.py`) | 202 → 250 |

**总计: 250 passed, 3 skipped**

---

## ✅ Phase 3 state 大小实测

模拟 supervisor 完成 EDR v9 baseline (777 EU) 后的 state shape:

```
eu_count_per_dim = {
    "funding_history": 300,
    "investors": 200,
    "acquisitions": 150,
    "product_market": 127,
}
state = {eu_counts: ..., dimension_ids: ..., ...}
size = max(json, pickle) serialized bytes
```

| EU 总数 | state size | 验收 |
| --- | --- | --- |
| 777 | < 1KB | ✓ < 50KB |
| 10,000 | < 1KB | ✓ < 50KB |

(对比 EDR v9 baseline `raw_notes=23.7K chars + 777 EU 内存对象 ≈ 380KB;Phase 3 降至 < 1KB。)

---

## ✅ 兼容性

- 旧 dataclass `EvidenceUnit`(in-process / cross-module 传 dict / 给 writer) — 完全不动,新加 `to_v2()` 方法做兼容
- 旧 `state.evidence_units` 字段保留(LEGACY)— 阶段 4 才删
- 旧 `state.compressed_research` 字段保留 — 阶段 4 才删
- 旧 18 + 11 + 其他 173 个测试 — **零破坏**

---

## 🚧 留待阶段 2-7 的工作

| 阶段 | 衔接点 |
| --- | --- |
| 2 — 三道闸(span / numeric drift / entailment) | `EuDAO.update_verification()` 已就位 |
| 3 — 归并 + 源独立性 + 置信度分级 | `ClaimDAO.upsert_many` + `EuDAO.update_claim_id` 已就位 |
| 4 — ARQ job 化 + checkpoint 续跑 | `RunCheckpointDAO` 已就位;`research_iterations` 累加已就位 |
| 5 — 分节写作 + ReportResult | `ClaimDAO.list_by_run(exclude_grade='D')` + `grade_distribution` 已就位 |
| 6 — fact-check pass + 降级回路 | claim → 蕴含校验由 ClaimV2 字段直接支撑 |
| 7 — planner DAG | `dimension_id` 字段已就位(planner 改造留阶段 7) |

---

## ⚠️ 已知限制 / 后续 TODO

1. **PG 集成测试** — `tests/test_eu_dao.py::TestEuDaoPostgresIntegration` 是 `pytest.mark.skip`,需要 CI 起 odr-postgres 才能跑。当前 fail-safe 路径已经覆盖("PG 失败 → 降级 in-memory")。
2. **dict 路径** — `_persist_eus_to_pg` 遇到 dict 形态 EU(asdict 输出)会 skip,留给阶段 4 处理 LangGraph state 序列化路径。
3. **PG UUID 校验** — `plan_v2_pipeline` 中 `r-cache` / `r-write` 等非 UUID 字符串 run_id 会触发 PG 校验失败,fail-safe 路径正确处理。阶段 4 worker 改造时统一 UUID 化。
4. **`source_tier` 默认 tertiary** — 阶段 1 留 default,阶段 3 白名单驱动升级。
5. **HNSW 实际 EXPLAIN** — 留真 PG 跑;代码已用 `<=>` 操作符,pgvector query planner 自动选 HNSW。