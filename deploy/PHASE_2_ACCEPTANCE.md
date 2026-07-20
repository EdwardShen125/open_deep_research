# Phase 2 Acceptance — EvidenceUnit 数据流改造

**生成时间**: 2026-07-19
**执行人**: Hermes (MiniMax-M3)
**目标**: 关闭差距 #1 (raw_notes=1 → EU≥N) + #3 部分 (writer chain-of-citation)

---

## ✅ 已交付

### 2.1 EvidenceUnit Pydantic-style model

| 项 | 落地位置 | 测试 |
|---|---|---|
| EvidenceUnit dataclass | `src/open_deep_research/evidence_units.py` (12 KB) | `tests/test_evidence_units.py` 18/18 |
| NumberBinding + 中文+英文 numeric 解析 | 同上 | covered |
| EntityRef | 同上 | covered |
| content_hash 跨运行 dedup | `content_hash` property (sha256 of normalized payload) | `test_eu_content_hash_dedup` |

**核心不变量(单元测试断言)**:
- `claim` 非空,>500 字符自动 truncate
- `confidence ∈ [0,1]`
- `source_url` 非空
- `content_hash` 跨 `extracted_at` / `run_id` 稳定(确保重跑同一查询同一来源 → 同一 EU ID)

### 2.2 Deterministic EU extractor

| 项 | 落地位置 | 测试 |
|---|---|---|
| `extract_from_search_result(result)` | `src/open_deep_research/eu_extractor.py` (10.5 KB) | `tests/test_eu_extractor.py` 11/11 |
| `split_sentences()` 中英文 support | 同上 | covered |
| `mine_entities()` CI vendor lexicon | 同上 | covered |
| `_confidence_for()` heuristic | 同上 | covered |
| 集成 SourcesDAO (SQLite parity path) | 同上 | `test_extractor_integration_with_sources_dao_sqlite` ✅ |

**关键设计**(避免 v1 的 LLM 双重 hop 信息丢失):
- Tavily summary → 句子切分 → 每句一个 EU (deterministic)
- 不再二次 LLM summarize,verbatim 句子保留在 EU.quote 字段
- 数字 / 实体锚点由 deterministic miner 抽出

### 2.3 chain-of-citation schema + prompt + validator

| 项 | 落地位置 | 测试 |
|---|---|---|
| `CitedClaim / CitedSection / CitedReport` dataclass | `src/open_deep_research/cited_report.py` (13.5 KB) | `tests/test_cited_report.py` 14/14 |
| `CITED_REPORT_PROMPT` 模板(JSON-only) | 同上 | covered |
| `parse_cited_report(raw_response)` parser | 同上 | covered (含 fenced JSON + 自由 prose 抽取) |
| `validate_cited_report(report, eu_pool)` | 同上 | covered |

**Validator 关闭的差距**(对应 v1 baseline anchors):
- 无 eu_ids claim → `[WARN] ... NO eu_ids` (gap-C:unsourced) — 7 个锚点
- Ownership/relation 单源引用 → `[WARN] ... ownership but cites only 1` (gap-A: A1 Kompyte / A2 Algorithmia) — 4 个锚点
- 引用数字但 EU 不含该 NumberBinding → `[WARN] ... no matching NumberBinding` (gap-A: A3 valuation / A4 TAM chain) — 4 个锚点
- 未知 EU ID → `unresolved_eu_ids` 列表

---

## 📂 已新增 / 改动文件

```
src/open_deep_research/evidence_units.py           12 KB  ✅  EU 模型 + NumberBinding + EntityRef
src/open_deep_research/eu_extractor.py             10.5 KB ✅  句子切分 + 抽取 + DAO 集成
src/open_deep_research/cited_report.py             13.5 KB ✅  chain-of-citation schema + prompt + 验证器
tests/test_evidence_units.py                        9 KB  ✅ 18/18
tests/test_eu_extractor.py                          8 KB  ✅ 11/11
tests/test_cited_report.py                          9.7 KB ✅ 14/14
deploy/PHASE_2_ACCEPTANCE.md                        (this) ✅
```

**累计单元测试**: 9 (Phase 1.1) + 7 (Phase 1.2) + 13 (Phase 1.5) + 18 (Phase 2.1) + 11 (Phase 2.2) + 14 (Phase 2.3) = **72 tests, all PASS**

---

## 🎯 Phase 2 验收标准对照

| v1 baseline 锚点 | Phase 2 闭环路径 | 状态 |
|---|---|---|
| **A1** Kompyte ownership | validator 标 [WARN] ownership + EU < 2 | ✅ 验证规则到位 |
| **A2** Klue/Algorithmia 虚构建关系 | 同上 + EU 内 `acquired_by` EntityRef 记录 | ✅ 抽取规则到位 |
| **A3** Klue/Crayon 估值 | NumberBinding 强约束 validator 检查 | ✅ 验证规则到位 |
| **A4** TAM 估算链 | 同上 + NumberBinding confidence 标记 | ✅ 验证规则到位 |
| **C1-C6** 中文数字无 binding | NumberBinding chi regex + EU quote 强制 | ✅ 解析规则到位 |
| **粗腰 gap**: raw_notes=1 → 0 EUs | extractor 把 Tavily summary 折成 EU 列表(≥3 per result) | ✅ 数据流改造到位 |

---

## 🚦 e2e 验收(待 docker 回来)

### Phase 2.4 — 给 docker / LangGraph 回来的清单

| e2e 测试 | 命令草案(假设 server 已起) |
|---|---|
| 用 baseline 同样的负样本 `baselines/v1/question.txt` 重跑 | `uvx langgraph dev --allow-blocking` 然后 `cd /root/open_deep_research && python3 tests/run_negative_sample.py` |
| 验证 `state.notes:` 不再只剩 "raw_notes=1" | inspect trace `final_state.json` → 查 `notes: ≥N` 或新字段 `evidence_units: ≥5×N` |
| 验证 cited_report.orphan_claim_text=[] (no unsourced prose) | inspect trace `final_state.json['cited_report']['orphan_claim_text']` |
| 验证 validator 命中 A1-A4, C1-C6 中至少 80% | `grep -c '\[WARN\]' final_state.json` |

**目前的 proceed 路径**:代码 + 72 unit tests + 静态 validators 全到位,只要 docker 一活,可以一小时内跑完 e2e 并收到 acceptance 信号。

---

## 下一步候选

- **Phase 3a** Verifier engine:用 rule 1/2/3 自动化跑 `validate_cited_report + 自有 heu rules + 中文扫描器`,作为独立的 verifier 服务
- **回到 Phase 1.3/1.4**(docked 阻塞,代码就位就行)
- **Phase 3b**:ReportDataObject 单源 + 规则四(URL 二次解析)
- **Phase 4**:Planner 增强(独立 module)

按"成本/杠杆"我推荐 **Phase 3a verifier**——它能让 Phase 2 的 chain-of-citation 在不联网的情况下自洽运行,产出最终 acceptance。
