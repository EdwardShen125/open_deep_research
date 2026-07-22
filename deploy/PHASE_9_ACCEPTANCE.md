# Phase 7 验收文档 (= Runbook v1 阶段 7)

> 对应 Runbook: `notes/evidence-pipeline-runbook-v1.md` 阶段 7
> 实现提交: `feat(evidence)` planner 显式 DAG + RunConfig 显式配额

## 目标

把"调研流程"的调度从 supervisor 黑盒升级为**显式 DAG**:
- 用户可写 `DAG.add(DAGNode(name=..., fn=..., depends_on=[...]))`
- `dag_to_stages()` 把 DAG 转 `ResearchJob.stages`,自动拓扑排序
- `NodeQuota` 给每个节点独立配额(token / time / retry / cosine threshold)
- `batch_dag_for_dimensions()` 把 merge/grade 节点按 dimension 展开
- `run_with_per_dim_retro` 是**阶段 6 retro 的 per-dim 升级**

## 改动概览

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `src/open_deep_research/evidence/dag.py` | 439 | `DAG / DAGNode / NodeQuota / validate / topo_sort / dag_to_stages / batch / default_pipeline_dag` |
| `src/open_deep_research/evidence/per_dim_retro.py` | 170 | `run_with_per_dim_retro` per-dim 降级 |
| `src/open_deep_research/evidence/job_runner.py` | 改 ~10 | `ResearchJob.stage_names` + 用 stage_names 查 checkpoint |
| `src/open_deep_research/evidence/checkpoint.py` | 改 ~50 | `list_completed / failed / get_resume_point / is_run_complete` 加 `stage_names` kwarg |
| `src/open_deep_research/evidence/__init__.py` | 暴露 9 个新符号 | |
| `tests/test_phase7_dag_planner.py` | 555 | 34 个测试覆盖 |

## 验收标准

### ✅ 验收 1: DAG 校验 — 环 / 缺失依赖 / 重复名 (Runbook 7.1)

**测试**: `TestValidateDAG` + `TestDAGBasics`
- `test_self_dependency_cycle`: a 依赖自己 → DAGValidationError
- `test_three_node_cycle`: a→b→c→a → DAGValidationError
- `test_missing_dependency`: 依赖不存在的节点 → DAGValidationError

### ✅ 验收 2: 拓扑排序确定性 (Runbook 7.1)

**测试**: `TestTopoSort`
- `test_linear_chain`: a→b→c → [a, b, c]
- `test_diamond_dependency`: a→{b, c}→d → [a, ..., d] (b/c 顺序稳定)
- `test_two_roots_kept_in_insertion_order`: 同入度节点按添加顺序

**机制**: Kahn's algorithm + 稳定插入。dag_to_stages 自动 topological sort + @stage_trace 装饰 fn。

### ✅ 验收 3: RunConfig 显式配额 (Runbook 7.1)

**测试**: `TestNodeQuota`
```python
q = NodeQuota(
    token_budget=10_000,         # 各 stage token 上限
    time_budget_s=60.0,          # 时间预算
    retry_on_transient=True,     # transient 错误是否重试
    max_retries=3,
    cosine_threshold=0.92,       # 仅 merge 节点用
    entailment_strict=False,     # 仅 verify 节点用
)
dag.add(DAGNode(name="merge", fn=merge_fn, quota=q))
```

`NodeQuota.to_metadata()` → 传给 `@stage_trace` → Langfuse span metadata 可见。

### ✅ 验收 4: default_pipeline_dag 工厂 (Runbook 7.1)

**测试**: `TestDefaultPipelineDAG`
- 5 个节点: extract → verify → merge → grade → write
- `merge_per_dimension=True` → merge/grade 节点带 per_dimension 标记
- `merge_per_dimension=False` → 经典 linear chain

### ✅ 验收 5: per-dimension batching

**测试**: `TestBatchDAGForDimensions`
- `per_dimension=False` 节点不动
- `per_dimension=True` 节点被复制成 N 个 (name + `__dim`)
- write 节点的 dep 自动 fan-in 展开到所有 per-dim grade

**机制**:
```
原始 DAG:
    extract → verify → merge(per_dim) → grade(per_dim)
                       ↓
                       write(depends_on=[grade])

batch 后 (dimensions=[d1, d2]):
    extract → verify → merge__d1, merge__d2
                       ↓
                       grade__d1, grade__d2
                       ↓
                       write(depends_on=[grade__d1, grade__d2])
```

### ✅ 验收 6: per-dim retro 反馈回路 (Runbook 阶段 7 衔接阶段 6)

**测试**: `TestPerDimRetro`
- `test_no_dim_retro_when_all_good`: 全 A → status=ok,0 warnings
- `test_one_dim_bad_triggers_retry`: d1 重试,d2 不动(独立决策)
- `test_failed_when_no_dim_improves`: 重试用尽 → status=failed

**核心价值**: 单维度差不会再拖累整个 run。

```python
result = await run_with_per_dim_retro(
    job, run_id, state,
    dimensions=["pricing", "reviews", "features"],
    threshold=0.5, max_retries=3,
)
# 仅对 D 占比 > 0.5 的 dim 触发 retro(独立决策)
```

## 设计权衡 / NOT-范围

| 不做 | WHY |
| --- | --- |
| 重写 supervisor (1526 行 deep_researcher.py) | 风险巨大;现 use_explicit_dag 切换向后兼容 |
| 引入 networkx / 第三方图库 | Runbook 期望小改动;Kahn 算法 30 行内可实现 |
| 接 LangGraph conditional_edges | 是更深的重写,留待后续 |
| pgvector 集成 + BGE-M3 真接 | pgvector 仍缺失,阶段 7 路由验收仍 mock;DAG 接口设计不变,装扩展后直接 wire 真 EU |
| 真 PG 跑 777 EU 基线 | 同上 |

## 测试结果

```
tests/test_phase7_dag_planner.py ..................................      [100%]
============================== 34 passed in 0.37s ==============================

全套: 461 passed, 3 skipped, 194 warnings
```

**总计 461 passed, 3 skipped**(vs 阶段 6 完成态 427 passed,+34)。

## Runbook 阶段 7 全部完成

按 Runbook 整改 7/7 完成:

| 阶段 | Runbook 主题 | 状态 |
| --- | --- | --- |
| 1 | EU 一等公民 (schema / state 瘦身) | ✅ |
| 2 | span 校验 + 语义归并 (三道闸) | ✅ |
| 3 | 归并 + 源独立性 + 置信度分级 | ✅ |
| 4 | ARQ worker + checkpoint 续跑 + Langfuse | ✅ |
| 5 | ReportResult 结构化输出 + fallback 信号硬化 | ✅ |
| 6 | 基于 grade 分布的反馈回路 | ✅ |
| 7 | planner 显式 DAG + RunConfig 显式配额 | ✅ |

## Phase 7+ 路线图

1. **真 PG 集成验证**: docker-compose 换 `pgvector/pgvector:pg16-alpine` + 重跑 19,955 EU baseline
2. **BGE-M3 真接**: 在 `extract` 阶段算 embedding,落库 `VECTOR(1024)`,merge 阶段用 cosine 归并
3. **supervisor 重构**: 用 DAG 取代 supervisor 黑盒(本阶段架构已为它铺路)
4. **HTTP 入口**: 加 GraphQL / REST endpoint 套 `ResearchJob` + `run_with_per_dim_retro`
