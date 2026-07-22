# Phase 6 验收文档 (= Runbook v1 阶段 6)

> 对应 Runbook: `notes/evidence-pipeline-runbook-v1.md` 阶段 6
> 实现提交: `feat(evidence)` grade retro loop

## 目标

解决 **P3 无自动降级**:当上游(extract / verify)产出质量过低、归并 + grade
后 D 占比飙升时,系统应**自动 retry** reset merge+grade,而不是直接摆烂。

## 改动概览

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `src/open_deep_research/evidence/grade_retro.py` | 195 | `run_with_retro_loop` 编排 + `detect_grade_d_pct` + `should_retry` + `retro_summary` |
| `src/open_deep_research/evidence/__init__.py` | +12 | 暴露新模块 + 6 个新公共符号 |
| `tests/test_phase6_grade_retro.py` | 405 | 23 个测试覆盖 |

## 验收标准

### ✅ 验收 1: D 占比超阈值自动 retro (Runbook P3)

**机制**:
- `DEFAULT_D_THRESHOLD = 0.5`(D 占 >50% 触发)
- `DEFAULT_MAX_RETRIES = 3`
- `DEFAULT_RETRY_STAGES = ("merge", "grade")`(仅 retry 这两个,extract/verify 保留)

**测试**: `TestRunWithRetroLoop::test_retro_loop_triggers_and_succeeds`

**流程**:
1. 第一次跑 `ResearchJob.run(rid, init_state)`,5 stage 全跑
2. 算 `d_pct = D_count / total_count`,> 阈值 → 进入 retro
3. `reset_run(rid, stages=["merge","grade"])`(extract/verify 保留 done)
4. 再调 `ResearchJob.run(rid, state)`(从 merge 重新跑)
5. 重检 d_pct → 通过则 break;否则循环
6. N 次仍不达标 → `status=failed`,带 `HighGradeDPct` failure

### ✅ 验收 2: retro 只 reset merge+grade,不动 extract/verify

**测试**: `TestRunWithRetroLoop::test_retro_resets_merge_and_grade_only`

**机制**: `ResearchJob.run()` 看到 merge/grade 没在 `list_completed_stages()` 中,所以跳过 extract/verify(已 done),直接重跑 merge/grade。

```python
async def test_retro_resets_merge_and_grade_only():
    counter = {"extract_runs": 0, "verify_runs": 0, "merge_runs": 0, "grade_runs": 0}
    # ... 用 counter 跟踪 ...
    await run_with_retro_loop(job, rid, state, threshold=0.5, max_retries=3)
    assert counter["extract_runs"] == 1   # NOT retried
    assert counter["verify_runs"] == 1    # NOT retried
    assert counter["merge_runs"] >= 2     # retried
    assert counter["grade_runs"] >= 2     # retried
```

### ✅ 验收 3: retro 失败有硬信号 (复用阶段 5 ReportResult)

**测试**: `TestRunWithRetroLoop::test_retro_exhausted_returns_failed`

**机制**:
- `run_with_retro_loop()` 返回 `ReportResult`
- 失败时 `status='failed'` + `ok=False` + `failures=[Failure(stage="retro_loop", ...)]`
- 调用方用 `is_report_success(result) is False` 拦截伪成功

### ✅ 验收 4: 第一次跑通过 → 无 retro history

**测试**: `TestRunWithRetroLoop::test_first_pass_succeeds_no_retry`

**机制**: `d_pct <= threshold` → 直接 `status='ok',warnings=[],failures=[]`,零 retro 开销。

### ✅ 验收 5: 第一次 stage 异常 → failed 不进入 retro

**测试**: `TestRunWithRetroLoop::test_initial_stage_failure_returns_failed`

**机制**: `ResearchJob.run(rid, state)` 抛异常 → catch 后写 `Failure(stage="(initial)")`,直接返回 `status='failed'`。**不进入 retro**(retro 是为 grade 分布差设计,不是为 stage 抛错设计)。

## API 概述

```python
from open_deep_research.evidence import (
    run_with_retro_loop, detect_grade_d_pct, should_retry,
    DEFAULT_D_THRESHOLD, DEFAULT_MAX_RETRIES, DEFAULT_RETRY_STAGES,
    retro_summary, ReportResult, Failure,
)

# 1. 派生 D 占比
pct = detect_grade_d_pct(claims)  # 0.0 - 1.0

# 2. 决策
should_retry(claims, threshold=0.5)  # True if pct > 0.5

# 3. 编排
result: ReportResult = await run_with_retro_loop(
    job, run_id, initial_state,
    threshold=0.5, max_retries=3, retry_stages=("merge", "grade"),
    extract_claims=lambda s: s.get("claims", []),
)

# 4. 调试输出
print(retro_summary(result))
# - 最终 status: **failed**
# - ok 硬信号: **False**
# - Warnings (3): ...
# - Failures (3): ...
```

## 默认值 vs 调参

| 参数 | 默认 | 含义 | Runbook 设计意图 |
| --- | --- | --- | --- |
| `threshold` | 0.5 | D 占比阈值 | 验收第 1 条: D>50% 触发 |
| `max_retries` | 3 | 重试次数 | 验收第 3 条: 3 次仍不达标 → failed |
| `retry_stages` | ("merge", "grade") | 重试哪些 stage | 验收第 2 条: 仅归并+grade,extract/verify 不动 |

调参场景:
- **生产系统保守**: `threshold=0.3`(更早触发),`max_retries=2`(更早放弃)
- **实验跑激进**: `threshold=0.7`(高容忍),`max_retries=10`(多试)
- **阶段 7 per-dimension retro**: 把 `extract_claims` 换成"按 dimension 切片",逐个 dimension 检 d_pct

## 测试结果

```
tests/test_phase6_grade_retro.py .......................                 [100%]
============================== 23 passed in 0.34s ==============================

全套: 427 passed, 3 skipped, 194 warnings
```

阶段 1+2+3+4+5+6 累计:
- 阶段 1 (Phase 3): 48 tests
- 阶段 2 (Phase 4): 39 tests
- 阶段 3 (Phase 5): 34 tests
- 阶段 4 (Phase 6): 28 tests
- 阶段 5 (Phase 7): 24 tests
- 阶段 6 (Phase 8): **23 tests**

**总计 427 passed, 3 skipped**(vs 阶段 5 完成态 404 passed)。

## 阶段 7+ 衔接

- **阶段 7 planner DAG**: `run_with_retro_loop` 的 `extract_claims` 改成"按 dimension 切片",
  retro 自然 per-dimension 触发,与 Runbook 设计一致。
- **HTTP 入口 + Retry-After header**: `ReportResult.status=failed` + `pipeline_duration_ms` 可
  直接喂给上层(202 接受 + retry_after,401-like 错误码),阶段 7+ 路线图但非 Runbook 必选。
