# v1 架构缺陷对照表

**生成时间**: 2026-07-18
**基线 trace**: `baselines/v1/trace/final_state.json`
**基线报告**: `baselines/v1/report.md`(26224 字符)
**错误锚点**: `baselines/v1/expected_anchors.json`(18 个)

---

## 基线运行条件

| 项 | 值 |
|---|---|
| 模型 | `minimax:MiniMax-M3`(走 MiniMax ChatModel) |
| 题目(英文) | "Research the current landscape of AI-powered competitive intelligence and market research products in 2025. List 5 main products with their pricing tiers and target customers." |
| 原计划题目(中文 207 字) | `KeyError: 'reflection'` 失败(详见 §缺陷 0) |
| max_concurrent_research_units | 2 |
| max_researcher_iterations | 3 |
| search_api | tavily |
| 总耗时 | 298.7s |
| final_report 长度 | 26224 字符 |
| raw_notes | 1 条(researcher 开场白) |
| notes | 0 条(supervisor 立即终止) |
| supervisor_messages | 7 条(1 次 ConductResearch + 1 次 ResearchComplete) |

> ⚠️ 由于 v1 对中文长 prompt 不稳定,本 baseline 用英文缩窄版以拿到可用 trace。
> 中文 207 字原题的错误锚点(18 个)在 `expected_anchors.json` 中保留,Phase 3a 注入测试时会再次使用。

---

## 6 大架构缺陷(按严重程度)

### 缺陷 0: 中文长 prompt 触发 `KeyError: 'reflection'` [critical]

**位置**: `src/open_deep_research/deep_researcher.py:278`

```python
reflection_content = tool_call["args"]["reflection"]
```

**触发**: model 调用 `think_tool` 时若 args 缺 `reflection` 字段,直接 KeyError 抛出。

**失败案例**: 中文 207 字原题跑了 19.7s 即异常终止,`final_state.json` 中存有 `{"__error__": {"error": "KeyError", "message": "'reflection'"}}`。

**v2 必须修复**: 节点代码必须防御性读取 `tool_call["args"].get("reflection")`,不可信 model 输出格式。

---

### 缺陷 1: researcher 子图输出完全不进入证据流 [critical]

**Trace 证据**:
```
supervisor_messages (7 entries):
  [4] ai: tool_calls: [ConductResearch(['research_topic'])]    ← 派遣 1 次
  [5] tool: # AI-Powered Competitive Intelligence ...         ← 压缩 research
  [6] ai: tool_calls: [ResearchComplete([])]                   ← 立即终止
  
raw_notes: 1 entry (researcher 的开场白, 不是搜索结果)
notes: 0 entries (supervisor 没聚合)
final_report: 26224 chars (writer 自由发挥)
```

**根因**:
- `compress_research` 节点把 tool_call 的所有原始输出压缩成一段 markdown
- `supervisor` 在第 6 步就调 `ResearchComplete`
- writer 只看到 `findings` 这个压缩文本,**没有 per-claim source binding**

**后果**: writer 写的 26224 字符中,只有 researcher 真正调过 1 次搜索的结论被保留。**其余绝大部分是 LLM 自身知识**——一旦 LLM 知识有误(Algorithmia / Kompyte ownership 之类),就完全没机会纠正。

**v2 必须修复**: Phase 2 引入 EvidenceUnit,所有 claim 强制绑定 verbatim quote + source。

---

### 缺陷 2: 域名级 URL 通过 8/59 [high]

**Trace 证据**:
```
URL count in final_report: 59
Domain-level (no path beyond /) count: 8/59
```

**域名级 URL 样例**:
- `https://klue.com`
- `https://www.crayon.co](https://www.crayon.co`(格式错误)
- 其他 6 个

**根因**: v1 引用由 `compress_research` LLM 自行决定用哪个 URL,**没有机制校验 URL 精度**。

**v2 必须修复**: Phase 3b 规则四(writer 引用只能解析到页面级 URL;域名级自动标 unverifiable 并触发替代来源检索)。

---

### 缺陷 3: writer 输出与事实脱节 [high]

**Kompyte 表述 trace 证据**:
```
Per [Parano.ai](...), Kompyte is the cheapest of the enterprise CI suites
at comparable deployment sizes, with entry deployments starting at $15,000-...
```

**事实**: Kompyte 2022 年已被 Crayon 收购,**不是独立 CI 套件**。v1 报告把它当成独立产品给出定价。

**Algorithmia trace 证据**: v1 baseline report 中 `Algorithmia` **不出现**——因为我们跑了英文缩窄版,而负样本原报告里 Algorithmia 是被虚构地关联到 Klue。

**根因**: writer 完全基于 LLM 内部知识生成内容,无证据引用,无 relation 验证。

**v2 必须修复**:
- Phase 2a 子串校验 + 蕴含校验 → quote 必真实,claim 必与 quote 语义一致
- Phase 3a 规则二实体关系验证 → acquisition/ownership 关系必须经独立域名来源确认

---

### 缺陷 4: 中文数字表达式完全无 binding [medium]

**Anchor 清单** (在 `expected_anchors.json`):
- C1: 30-60 亿美元(亿单位)
- C2: 5 万家(万单位)
- C3: 30%(百分号)
- C4: 500K-2M 美元(美元 + K/M 后缀)
- C5: ¥1.2 亿(货币符号 + 亿混排)
- C6: 区间表达 30-60 亿

**v1 行为**: writer 直接输出这些数字,**无任何 binding**——既不知道数字来源 EU 的 ID,也不知道 normalized 值。

**v2 必须修复**: Phase 3a 规则一数字强绑定,中文数字扫描规格明确覆盖亿/万/百分号/K/M/区间/货币符号。

---

### 缺陷 5: 表格与正文数字不一致 [critical, 但 v1 baseline 未触发]

**Anchor 清单** (在 `expected_anchors.json`):
- D1: 出海 SaaS TAM(正文 30-60 亿 vs 表格 5-10 亿,**TAM/SAM 颠倒**)
- D2: 本地化营销(同样颠倒)
- D3: 企业内部 PMM(正文 50 亿 vs 表格 10 亿,**缩小 5×**)

**为什么 baseline 没触发**: 这是负样本原报告的错误,**baseline 的英文缩窄版不涉及这三组数据**。但 v1 架构同样无法防御此类问题:
- 正文 block 与 table block 是 writer 独立生成的两个 markdown 片段
- 没有 ReportDataObject 单源约束
- Phase 3a 数字扫描只在 final_report 上跑一次,但表格 vs 正文的 diff 不被检测

**v2 必须修复**: Phase 3b 规则三 ReportDataObject 单源渲染,正文 block 与 table block 从同一对象渲染,数学上不可能不一致。

---

## v1 baseline 跑出来的**已知错误**(对应锚点)

| 锚点 | v1 baseline 表现 | 备注 |
|---|---|---|
| A1 Kompyte ownership | ❌ 描述 Kompyte 为独立 CI 套件,无 Crayon 关联 | 锚点验证通过 |
| B1/B2 域名级 URL | ❌ klue.com / crayon.co 出现于报告中 | 锚点验证通过 |
| 缺陷 0 中文 KeyError | ❌ 直接失败 | 锚点保留待 Phase 3a 复测 |
| A2 Klue-Algorithmia | ⚠️ 不出现(英文 baseline 范围缩窄) | 待 Phase 3a 注入测试 |
| A3 Klue 估值 | ⚠️ baseline 报告未含估值数据 | 待 Phase 3a 中文原题复测 |
| A4-C6 数字 / D1-D3 表格 | ⚠️ baseline 范围未覆盖 | 待 Phase 3a 中文原题复测 |

---

## v1 → v2 改进路线图

| 缺陷 | 计划 Phase | 关键机制 |
|---|---|---|
| 缺陷 0 KeyError | Phase 0a (后续 phase) | node 代码防御性读取 model output |
| 缺陷 1 evidence 流断裂 | Phase 2 | EvidenceUnit + verbatim quote 强制绑定 |
| 缺陷 2 域名 URL | Phase 1 + Phase 3b | sources 表 + 引用格式校验 |
| 缺陷 3 writer 幻觉 | Phase 2 + Phase 3a | 蕴含校验 + 实体关系验证 |
| 缺陷 4 数字无 binding | Phase 3a 规则一 | 中文数字扫描 + NumberBinding |
| 缺陷 5 表格不一致 | Phase 3b 规则三 | ReportDataObject 单源渲染 |

---

## trace 详细信息文件

```
baselines/v1/
├── report.md                                # 26224 字符的 v1 baseline 报告
└── trace/
    ├── run_config.json                      # 完整 LangGraph run 配置(可复现)
    ├── run_summary.json                     # 耗时 + 各 state 长度
    ├── final_state.json                     # 完整 state dump(含 messages / supervisor_messages / research_brief / raw_notes / notes / final_report)
    ├── question.txt                         # 输入题目
    └── negative_sample_source.txt           # 负样本原报告路径

baselines/v1/expected_anchors.json            # 18 个错误锚点(4A + 5B + 6C + 3D)
docs/state_v1.md                             # 当前 state schema 抄录
docs/v1_defects.md                           # 本文件
```

---

## Phase 1 准备工作清单

进入 Phase 1(检索层替换)前:

1. ✅ 基础设施: Phase 0.5 完成
2. ✅ LLM 收口: Phase 0a 完成
3. ✅ 基线 trace 存档: Phase 0b 完成
4. ⏳ PyMuPDF license 决策: Phase 0.5 遗留,**Phase 1 前必决**
5. ⏳ SearXNG 上游引擎限速问题: Phase 1 检索时必须解决(否则抓不到数据)
6. ⏳ Langfuse 后端启动: Phase 0a decorator 已就位,缺镜像拉取

---

## 负样本原题复测计划

**触发时机**: Phase 3a cross-validation 节点实施后

**做法**:
1. 用原中文 207 字负样本题目跑 v2 流水线
2. 读取 `expected_anchors.json` 的 18 个锚点
3. 逐项核对 v2 报告:
   - A 类: 修复后是否正确(Kompyte 标 Crayon 旗下 / Algorithmia 关系修正)
   - B 类: 引用是否升级到页面级 URL
   - C 类: 数字是否绑定到 NumberBinding(可程序化扫描验证)
   - D 类: 表格 vs 正文是否一致(可程序化 diff)
4. 输出 v2 验收报告对比 v1 baseline

**触发代码**:
```python
import json
anchors = json.load(open('baselines/v1/expected_anchors.json'))
# 对每个锚点 id 调用对应的验证器
results = []
for a in anchors['anchors']:
    result = verify_anchor(a, v2_final_report, v2_evidence_units)
    results.append(result)
# 输出汇总
```
