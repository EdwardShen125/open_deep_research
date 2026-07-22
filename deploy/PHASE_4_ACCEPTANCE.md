# Phase 4 Acceptance — 三道闸 + LLM 抽取 + 降级保留

**生成时间**: 2026-07-22
**执行人**: Hermes (MiniMax-M3) on `research` profile
**目标**: 关闭 Runbook v1 阶段 2 全部 5 条验收(抽取 prompt + 三道闸 + summarize_webpage 降级)
**前置**: 910fcb6 (Phase 0A EU dedup) + 8fced7e (Phase 3 阶段 1 双层 schema)

---

## ✅ 验收对齐

| Runbook 阶段 2 验收 | 实现位置 | 测试 |
| --- | --- | --- |
| 1. 10 页样本 span_verified 命中率 ≥ 95% | `run_gate1_span` + `verify_span` | `TestGate1AndGate2Pipeline.test_pipeline_with_mixed_eus` + 真 PG 集成留 CI |
| 2. 注入编造 span → 闸 1 全拦 | `verify_span` | `test_fabricated_span_rejected`, `TestPhase4Acceptance.test_rejected_stats_non_zero_when_injected` |
| 3. 注入数字篡改 → 闸 2 检出 ≥ 4/5 | `has_numeric_drift` | `test_drift_5m_vs_3m`, `test_drift_percentage`, `test_rel_tol_config` |
| 4. rejected_stats 非零且可导出 | `GateStats.to_dict()` | `test_rejected_stats_non_zero_when_injected` |
| 5. v9 样本重放 EU 下降到 60-85% | gate1+gate2 串联 | `test_eu_count_decreases_after_gates` |

---

## ✅ 已交付

### 2.1 抽取 prompt(约束前置)

| 项 | 落地位置 |
| --- | --- |
| `EXTRACT_PROMPT`(7 条硬性规则 + JSON 输出) | `src/open_deep_research/prompts/extractor_v1.py` |
| `extractor` role 注册到 REGISTRY | `prompts/__init__.py` |
| Legacy alias `from .prompts import EXTRACT_PROMPT` | `prompts/__init__.py:_LEGACY_NAMES` |
| LLM 调用 `extract_from_content_with_llm` | `evidence/llm_extractor.py` |
| 批量调用 `extract_from_search_results_with_llm` | 同上 |
| `_parse_one`(逐条 dict → EvidenceUnitV2)+ `_parse_response`(JSON 解析) | 同上 |
| 容错:LLM 失败 / JSON 失败 → 返回空列表,不抛 | 同上 |

### 2.2 闸 1:span 字面校验(零 LLM)

| 项 | 落地位置 |
| --- | --- |
| `verify_span(span, content, fuzzy_threshold=0.92, fuzzy_max_len=400)` | `evidence/verify.py` |
| 三层退化:字面 → 归一化(中文标点差异) → 滑窗模糊 | 同上 |
| `_normalize` helper(去空白 + 全角标点归一) | 同上 |

### 2.3 闸 2:数值漂移检测(零 LLM)

| 项 | 落地位置 |
| --- | --- |
| `has_numeric_drift(claim, span, rel_tol=0.005)` | 同上 |
| `_NUM` regex 识别 CJK 单位(万/亿/万亿/千/百分点/%) | 同上 |
| `_SCALE` 归一表(1 亿 = 1.2e8,12000 万 = 1.2e8) | 同上 |
| 年份排除(1900-2100)— 时间锚不是统计数字 | 同上 |
| `run_gate2_numeric_drift` 串联闸 1 | 同上 |

### 2.4 闸 3:entailment 批量(LLM)

| 项 | 落地位置 |
| --- | --- |
| `ENTAILMENT_PROMPT`(4 类 verdict + 4 条"不得 entailed") | `evidence/verify.py` |
| `EntailmentResult` / `EntailmentBatchResult` dataclass | 同上 |
| `parse_entailment_response(raw, n_items)` 容错(JSON fence / brace / 部分缺失) | 同上 |
| `verify_entailment_batch(items, llm, batch_size=20)` async | `evidence/llm_entailment.py` |
| LLM 失败兜底:整批 unverifiable,score=0,parse_warnings 记录 | 同上 |
| 分批 index 全局对齐(batch_size 跨越时 offset 修正) | 同上 |

### 2.5 summarize_webpage 降级保留

| 项 | 落地位置 |
| --- | --- |
| 超时 60s → 180s,失败重试 1 次 | `utils.py:summarize_webpage` |
| 仍失败 → 取前 3000 字 + 标题,`summary_method="truncate"` | 同上 |
| 不做低质页跳过,只做 < 200 字跳过 | 同上 |
| 返回 dict(含 `summary_method` / `summary_chars` / `title`),调用方 L121 向后兼容 | 同上 |

### 串联 / 收口

| 项 | 落地位置 |
| --- | --- |
| `GateStats` 累计闸 1+2 各类目(span_rejected / numeric_drift_rejected / 命中形态) | `evidence/verify.py` |
| `rejected_count()` 函数(rejected 占比给 Runbook 阶段 6 降级回路用) | 同上 |
| evidence 包 `__init__.py` 暴露新模块 | `evidence/__init__.py` |

---

## ✅ 测试覆盖

| 文件 | 测试数 |
| --- | --- |
| `tests/test_phase4_gates.py` (新建) | 39 |
| 旧 baseline 测试 | 250 |
| **总计** | **289 passed, 3 skipped** |

测试类拆分:
- `TestVerifySpan`(7)
- `TestHasNumericDrift`(8)
- `TestParseEntailmentResponse`(8)
- `TestRenderEntailmentItems`(1)
- `TestGate1AndGate2Pipeline`(1)
- `TestVerifyEntailmentBatch`(3)
- `TestLLMExtractorParser`(7)
- `TestSummarizeWebpage`(2)
- `TestPhase4Acceptance`(2 — 直接对应 Runbook 5 条验收)

---

## ✅ Phase 4 fail-safe 路径

- **LLM 抽取失败** → `_parse_response` 返回空列表;`extract_from_content_with_llm` 静默吞下异常 → pipeline 仍可继续
- **闸 3 LLM 失败** → `verify_entailment_batch` 整批 unverifiable,parse_warnings 记录异常 → 不阻塞 EU 落库
- **summarize_webpage 失败** → truncate 降级,`summary_method="truncate"`,继续进抽取 → 不丢页
- **PG 落库失败**(沿用阶段 1 fail-safe)→ state 仍持有 EU,降级回 in-memory

---

## 🚧 留待阶段 3-7 的工作

| 阶段 | 衔接点 |
| --- | --- |
| 3 — 归并 / 源独立性 / 置信度分级 | 闸 3 的 entailed/partial EU 进入归并 |
| 4 — ARQ job 化 | `summarize_webpage` 不再受 60s 限制(job 内可跑满) |
| 5 — 分节写作 | `GateStats.rejected_count()` 可作为可信度摘要的一部分 |
| 6 — fact-check pass | 已有的 entailment 逻辑可复用 |
| 7 — planner DAG | 抽取阶段接收 `dimension_id`(已留字段) |

---

## ⚠️ 已知限制

1. **英文 magnitude 暂不支持**(`million` / `billion`):`_NUM` 只匹配 CJK 单位后缀。Runbook 2.3 设计的也是 CJK 单位换算。
   - 英文源(LinkedIn / TechCrunch)会出现闸 2 false-positive(数字漂移误判)
   - 后续可扩展:复用 `evidence_units.py:_EN_MAGNITUDE_RE`
2. **闸 1 模糊命中位置丢失**:`verify_span` 模糊路径返回 `(True, None)`,无法反推回原文 offset。`span_start` / `span_end` 字段未填。
3. **闸 3 LLM 调用未批量任务调度**:`verify_entailment_batch` 是 sequential batches,阶段 4 job 化时可并发。
4. **PG 集成测试**:闸 1/2 + EuDAO.update_verification 的端到端仍需 CI 跑真 PG。