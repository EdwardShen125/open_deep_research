# Phase 5 验收文档 (= Runbook v1 阶段 5)

> 对应 Runbook: `notes/evidence-pipeline-runbook-v1.md` 阶段 5
> 实现提交: `feat(evidence)` ReportResult / fallback signal hardening

## 目标

把"调研流程"产出的**最终结果**从脆弱的单字符串 (`final_report: str`) 升级为
**结构化 `ReportResult`**,并把"成功 / 兜底 / 失败"改成**硬信号**,杜绝
"调用方看到非空字符串就以为调研成功"的 P0 bug。

## 改动概览

| 文件 | 行数 | 角色 |
| --- | --- | --- |
| `src/open_deep_research/evidence/report.py` | 235 | `ReportResult / ReportSection / ClaimStats / Failure / is_report_success` |
| `src/open_deep_research/evidence/__init__.py` | +5 | 暴露新模块 |
| `src/open_deep_research/state.py` | +6 | `AgentState.report_result: Optional[dict]` |
| `src/open_deep_research/deep_researcher.py` | 改 ~80 | `final_report_generation` 5 个 return path 全部走 `_build_update`,ReportResult 序列化进 state |
| `tests/test_phase7_report_result.py` | 410 | 24 个测试覆盖 5 条验收 |

## 验收标准

### ✅ 验收 1: ok 是硬信号 (Runbook 5.2)

**Runbook**: "失败信号是软的 → ReportResult 结构化返回"

**测试**: `TestReportResultBasics::test_from_markdown_and_status_failed` + `TestIsReportSuccess::test_failed_returns_false`

**机制**:
- `ReportResult.ok: bool` — True/False 二值硬信号
- `ReportResult.status: Literal["ok","partial","fallback_used","failed"]` — 详细分类
- `is_report_success(result)` 只在 status=='ok' 时返回 True

```python
r = ReportResult.from_markdown_and_status("body", status="fallback_used")
assert r.ok is True                # 兜底也算"产出"
assert r.status == "fallback_used"
assert is_report_success(r) is False  # 但 success=False — 调用方必须看
```

### ✅ 验收 2: 伪成功修复 (Runbook P0)

**问题**: 之前 writer LLM 失败时,把错误字符串写到 `final_report` (str),调用方判定非空即"调研成功"。

**整改** (`deep_researcher.py:final_report_generation`):

| Return path | 之前 status | 现在 status | 之前 ok | 现在 ok |
| --- | --- | --- | --- | --- |
| writer 成功 (verification 0 critical) | (隐式) | `ok` | True | True |
| writer 成功但 verifier flag critical | (隐式) | `partial` | True | True (但 warnings 暴露) |
| transient 重试超时,用 EU digest 兜底 | (隐式) | `fallback_used` | True | True (但 failures 暴露) |
| token-limit 无 model map | "Error generating..." 字串 | `failed` | (伪装) | False |
| 其他异常 | "Error generating..." 字串 | `failed` | (伪装) | False |
| max retries exceeded | "Error generating..." 字串 | `failed` | (伪装) | False |

**关键**: `final_report: str` 字段**保留**(`backward_compatibility`),同时新增
`report_result: dict` 字段(`ReportResult.model_dump(mode="json")`)。
- 旧调用方: 仍能读 `state["final_report"]`(但**不再是非空=成功**)
- 新调用方: 读 `state["report_result"]["ok"]` 和 `state["report_result"]["failures"]`

### ✅ 验收 3: claim_stats 一等公民 (Runbook 5.3)

**机制** (`evidence/report.py:ClaimStats`):

```python
stats = ClaimStats.from_claim_list(claims, eus=eus)
# → 推导 grade_dist_pct, total_eus, usable_eus, rejected_eus,
#   unique_sources, unique_primary_sources, has_conflict
```

**测试**: `TestClaimStats::test_grade_distribution_pct` + `test_conflict_counter` + `test_eu_count_propagates`

**Result**: 24 passed, 全部 grade 推导 + has_conflict 计数 + EU 维度统计正确。

### ✅ 验收 4: JSON 序列化 (LangGraph state 兼容)

**测试**: `TestReportResultSerialization`

**机制**:
- Pydantic `ReportResult.model_dump(mode="json")` → 进 `state["report_result"]`
- `Failure` 含 `datetime` 字段,自动转 ISO 字符串(测试验证 `isinstance(d["timestamp"], str)`)
- 不动 `final_report: str` 字段,旧的消费方零破坏

### ✅ 验收 5: 失败列表 + stats block 渲染

**测试**: `TestReportMarkdown`

**机制**: `ReportResult.to_markdown_with_warnings()` 返回:
- ⚠️ Warnings block(列出 warnings[])
- ❌ Failures block(列出 failures[].stage + error_type + error_message)
- 📊 Evidence Stats block(列出 ClaimStats 关键字段)
- 正文 body_markdown 在末尾

## 不在阶段 5 范围

- **`sections` 自然来自 DAG**: 阶段 5 的 `sections` 字段是手工构造占位,真正按 dimension/claim 自动聚合在**阶段 7 planner DAG**。
- **deep_researcher 端到端真测**: 需要真 writer LLM,本次跳过(算法层 + mock 验证)。
- **claim_stats 真 PG 验证**: 同阶段 4,pgvector 缺失,留阶段 7 bundle 处理。
- **RunConfig 显式配额**: 阶段 7 planner DAG 的附带设计(本阶段不引入)。

## 测试结果

```
tests/test_phase7_report_result.py ........................              [100%]
============================== 24 passed in 0.36s ==============================

全套: 404 passed, 3 skipped, 194 warnings
```

阶段 1 + 2 + 3 + 4 + 5 累计:
- 阶段 1 (Phase 3): schema + DAO + state 瘦身 — 48 tests
- 阶段 2 (Phase 4): verify gates + LLM extractor — 39 tests
- 阶段 3 (Phase 5): merge + independence + grading — 34 tests
- 阶段 4 (Phase 6): checkpoint + observability + job_runner — 28 tests
- 阶段 5 (Phase 7): ReportResult + fallback signal hardening — **24 tests**

**总计 404 passed, 3 skipped**(vs 阶段 4 完成态 380 passed)。

## 阶段 6+ 衔接

- 阶段 6 自动降级: 检查 `state["report_result"]["grade_dist_pct"]["D"]` 比例,
  若过高 → 触发 `reset_run(stages=["merge","grade"])` 重跑。这正是 ResearchJob 的用途。
- 阶段 7 planner DAG: `sections` 字段自然来自 DAG 节点,让 ReportSection 自动按 dimension 聚合;
  RunConfig 显式配额(extract / merge / write 各 stage 的 token 预算)在 planner 里直接体现。
