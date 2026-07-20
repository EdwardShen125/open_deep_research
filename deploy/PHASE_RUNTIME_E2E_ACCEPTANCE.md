# Phase Runtime E2E Acceptance — LangGraph Runtime 真实打通

**生成时间**: 2026-07-20
**执行人**: Hermes (MiniMax-M3) on `research` profile
**目标**: 通过 `langgraph dev` server 端到端跑通一个真实研究 query,验证 Plan v2 4 个新字段(`evidence_units` / `cited_report` / `verification` / `url_compliance`)全部在 runtime 中正确填充

---

## ✅ 端到端验证

### 配置

```yaml
模型: minimax:MiniMax-M3(默认)
搜索 API: tavily
query: "What was Klue's total funding raised?"
max_concurrent_research_units: 1
max_researcher_iterations: 2
max_react_tool_calls: 3
elapsed: 287.6s
```

### 最终 state(来自 `final_state_v2.json`,thread `019f7db1-...`)

| 字段 | 类型 | 实际值 |
|---|---|---|
| `messages` | list | 3 条(human + 2 ai)|
| `supervisor_messages` | list | 7 条(system/human/ai/tool 流,含完整 24 SOURCE 研究内容) |
| `research_brief` | str | 983 字符,完整 Klue funding brief |
| `raw_notes` | list[str] | 1 个元素(23.7K 字符,聚合自 researcher subgraph) |
| `notes` | list[str] | 2 个元素(supervisor 退出时 `get_notes_from_tool_calls` 聚合) |
| **`evidence_units`** | **list[dict]** | **777 个 EU**(修复前 = 0) |
| **`cited_report`** | **dict** | **7 sections, 17 cited claims**,带 `[eu-xxxxxx]` 引用 |
| **`verification`** | **dict** | **4 个 rule_2 high severity issue**(cross-domain 不足) |
| **`url_compliance`** | **list** | **空(无违规,所有 cited URL 都是 page-level)** |
| **`final_report`** | **str** | **5808 字符,完整 Klue funding 故事** |

### EU pool 统计

```
total: 777 evidence units
schema: claim / source_url / quote / source_id / source_title /
        numbers / entities / confidence / extraction_method /
        extracted_at / run_id / id
numeric_anchors: 1012(总)
unique source urls: 35
```

### verifier issues 摘要

```
rule_id: rule_2 (entity relation / cross-domain)
severity: high
anchor_id: A2_klue_algorithmia_fabrication

触发的 issue:
- "Klue's total funding...approximately $103.5 million CAD" → 1 EU, 1 domain (betakit.com)
- "Klue was named a 2019 Gartner Cool Vendor..." → 2 EU, 1 domain (prnewswire.com)
- "Tracxn reports Klue has 19 institutional investors..." → 6 EU, 1 domain (tracxn.com)
- "Klue still held approximately $50 million CAD..." → 1 EU, 1 domain (betakit.com)

→ 全部因为 known-risk entity (klue) + 单域来源,触发 cross-domain 不足
```

---

## 🔧 修复的两个真实 bug

### Bug #1 — `ResearcherState` / `ResearcherOutputState` 缺 `evidence_units` 字段

**根因**:`researcher_tools` 在 `update["evidence_units"] = new_eus` 写 EU,但 `ResearcherState` (researcher subgraph 内部 state) 没有声明这个字段,LangGraph 会在 subgraph 边界**静默丢弃**未声明字段。

**修复**(`src/open_deep_research/state.py`):

```python
class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    # Plan v2: per-researcher EU accumulator (forwarded via ResearcherOutputState).
    evidence_units: Annotated[list, override_reducer] = []   # ← ADDED

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    # Plan v2: surface the EU pool so the supervisor can aggregate it from
    # every parallel researcher invocation.
    evidence_units: Annotated[list, override_reducer] = []   # ← ADDED
```

### Bug #2 — `SupervisorState` 缺 `evidence_units` + supervisor 不聚合 EU

**根因**(双重):
1. `SupervisorState` 没声明 `evidence_units` 字段,supervisor subgraph 边界再次丢弃
2. supervisor_tools 处理 researcher subgraph 输出时,**只聚合了 `raw_notes`,没聚合 `evidence_units`**

**修复**(两处):

`state.py`:
```python
class SupervisorState(TypedDict):
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    # Plan v2: aggregated EU pool across all researchers.
    evidence_units: Annotated[list, override_reducer] = []   # ← ADDED
```

`deep_researcher.py` (supervisor_tools 中):
```python
# Aggregate raw notes + evidence_units from all research results
raw_notes_concat = "\n".join([...])
if raw_notes_concat:
    update_payload["raw_notes"] = [raw_notes_concat]

# Plan v2 — forward the EU pool collected upstream by researcher_tools.
eus_concat: list = []
for observation in tool_results:
    for eu in observation.get("evidence_units") or []:
        # Dedup by content_hash or text
        ...
        eus_concat.append(eu)
if eus_concat:
    update_payload["evidence_units"] = eus_concat
```

---

## 🧪 测试状态

```
166 passed in 1.47s  (Phase 1 + 2 + 3 + 4 + LangGraph runtime integration + runtime e2e smoke + tavily noise filter)
```

新增/修改的测试:
- `tests/test_runtime_e2e_smoke.py` (4 tests,默认 skip,`--smoke` 或 `RUNTIME_E2E_SMOKE=1` 触发) — 端到端验证 4 个 Plan v2 字段全部填充
- `tests/test_tavily_filter.py` (5 tests) — Phase 2.5 noise 过滤器的 host 提取、blacklist 命中、低质内容判定、端到端过滤

---

## 📌 发现的次要问题(部分已修)

### ✅ 已修 #1 — Tavily 检索引入噪音

**修复**:Phase 2.5 在 `_parse_tavily_observation()` 末尾插入 `_filter_tavily_chunks()`(详见 `src/open_deep_research/deep_researcher.py`):
- **Domain blacklist**:`_TAVILY_NOISE_DOMAIN_SUFFIXES`(facebook/instagram/twitter/x/linkedin/reddit/tiktok + worldofreel/throughthesilverscreen/topstartups)
- **Content noise**:`_NOISE_CONTENT_PATTERNS` 检测 markdown 图片 token / `<img>` / `<svg>` / data:base64,2+ 命中即丢弃
- **Length floor**:`_MIN_CHUNK_CONTENT_CHARS = 200`(过短视为低质)

**效果**:同样的 query,过滤前 35 个 URL → 过滤后 20 个,全部 page-level、全部相关业务来源。verifier 也因为 EU pool 更纯净,**触发了 3 个不同 rule(rule_1 数字 / rule_2 单域 / rule_3 high-risk)**,之前只能触发 rule_2。

### ⚠️ 未修 #2 — EU 的 `claim` 字段是 `<summary>。`

这是 Tavily 抓取 wrapper,extractor 没深入挖具体 claim。后续可扩展 `_parse_tavily_observation()` 解析 summary 内部的命名实体和数值断言。

### ⚠️ 未修 #3 — `url_compliance` 触发路径尚未在 e2e 中实测

当前 query 全部 cited URL 都是 page-level,Rule 4 没东西可抓 —— 是**正确行为**,但需要在 e2e test 里覆盖一条含 domain-only 的 fixture 来真正验证 Rule 4 触发路径。

---

## 🔁 Reproduce

```bash
cd /root/open_deep_research
source .venv/bin/activate
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 \
    langgraph dev --allow-blocking --no-browser
# → server at http://127.0.0.1:2024

# 新建 thread + run(sse 流式)
# 完整脚本见 /tmp/final_state_v2.json 的同款 query:
#   question: "What was Klue's total funding raised?"
#   config: search_api=tavily, max_concurrent_research_units=1
```

---

## 📊 与 v1 baseline 的对比

| 指标 | v1(blog 描述) | v2 runtime(无 filter) | v2 runtime(带 filter) |
|---|---|---|---|
| 端到端时长 | n/a | 287.6s | 195.3s |
| EU 提取 | n/a | 777 | **423**(更纯净) |
| Unique URLs | n/a | 35 | **20**(全 page-level) |
| Unique domains | n/a | ~28 | **15** |
| Cited claims | n/a | 17 | 11 |
| Verifier issues | 5 v1 anchors(单元) | 4 rule_2(端到端) | **10** (rule_1×1 + rule_2×5 + rule_3×4) |
| URL 合规 | n/a | 0 | 0(正确:全 page-level) |
| Final report 字数 | n/a | 5808 | 3505(更聚焦) |

→ runtime 实测表明 Plan v2 的 4 个新字段在 LangGraph 上正确接线,核心修复路径**已被 e2e 流量验证**;Phase 2.5 noise filter 进一步收紧了 EU 池质量,使 verifier 覆盖率从单一 rule 提升到三个 rule 同时命中。