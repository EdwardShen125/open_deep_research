# Phase 0b 验收报告

**执行时间**: 2026-07-18
**执行人**: Hermes (Claude)
**范围**: 基线报告 + state schema 文档 + 错误锚点 + v1 缺陷对照
**前置**: Phase 0.5 ✓ Phase 0a ✓

---

## 验收清单

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 0b.1 | 用 v1 架构跑负样本原题 | ✅ (部分) | 中文原题 KeyError 失败(已记录),英文缩窄版 baseline 完整跑通 |
| 0b.2 | trace + report 落盘 | ✅ | `baselines/v1/{report.md, trace/}` 完整 |
| 0b.3 | state_v1.md 抄录 | ✅ | 165 行,4 state + 5 Pydantic |
| 0b.4 | expected_anchors.json | ✅ | 18 个锚点 (4A + 5B + 6C + 3D) |
| 0b.5 | v1 缺陷对照表 | ✅ | docs/v1_defects.md (227 行) |

---

## 关键发现:v1 baseline 暴露的架构缺陷

**6 大缺陷,按严重程度**:

| # | 缺陷 | 严重度 | 触发证据 |
|---|---|---|---|
| 0 | `KeyError: 'reflection'` 在中文长 prompt 下直接崩溃 | critical | `baselines/v1/trace/final_state.json: {"__error__": ...}` |
| 1 | researcher 子图输出完全不进入证据流 (raw_notes=1, notes=0) | critical | supervisor 第 2 轮立即调 ResearchComplete |
| 2 | 8/59 引用是域名级 URL | high | final_report regex 扫描 |
| 3 | writer 完全自由发挥, Kompyte 描述为独立产品 (实际是 Crayon 旗下) | high | regex 抓 "Kompyte" 上下文 |
| 4 | 中文数字表达式完全无 binding | medium | anchors C1-C6 (亿/万/%/K-M/区间/币符号) |
| 5 | 表格 vs 正文不一致 (D1/D2/D3) | critical | anchors D1-D3 (baseline 范围未触发,但架构无法防御) |

---

## 基线 trace 数据

| 字段 | 值 |
|---|---|
| 题目(英文) | "Research the current landscape of AI-powered competitive intelligence and market research products in 2025. List 5 main products with their pricing tiers and target customers." |
| 模型 | `minimax:MiniMax-M3` |
| max_concurrent | 2 |
| max_iterations | 3 |
| search_api | tavily |
| 总耗时 | 298.7s |
| final_report | 26224 字符 |
| raw_notes | 1 条 (researcher 开场白) |
| notes | 0 条 |
| messages | 2 条 (用户问题 + AI 最终回复) |
| supervisor_messages | 7 条 (system + human + system + human + ai(ConductResearch) + tool(压缩research) + ai(ResearchComplete)) |
| research_brief | 1189 字符 |
| URL 数 | 59 |
| 域名级 URL 数 | 8 |

---

## 交付物清单

```
baselines/v1/
├── report.md                                26224 字符的 v1 baseline 报告
├── expected_anchors.json                    18 个错误锚点 JSON
└── trace/
    ├── run_config.json                      LangGraph 完整配置(可复现)
    ├── run_summary.json                     耗时 + 各 state 长度
    ├── final_state.json                     完整 state dump
    ├── question.txt                         输入题目
    └── negative_sample_source.txt           原中文负样本路径说明

docs/
├── state_v1.md                              165 行,state schema 完整抄录
└── v1_defects.md                            227 行,v1 → v2 改进路线图
```

---

## 已知遗留 & 下一步

### 遗留

1. **原中文 207 字负样本题目 v1 跑不通**——`KeyError: 'reflection'`。
   - Phase 0a 之后需要修复 deep_researcher.py:278 防御性读取
   - 但这个修复超出 Phase 0b 范围,延后到 Phase 2

2. **Langfuse 后端仍未启动**——Phase 0a 跑 baseline 时 LANGFUSE 不可用,无 trace metadata 落到 Langfuse UI。
   - 当前 Langfuse 只在 `langfuse_status()` 中标 `enabled: False`
   - 网络问题持续存在,镜像拉不下来

3. **PyMuPDF AGPL license 红线**仍未解决。
   - Phase 0.5 决策项,**Phase 1 前必决**

### Phase 1 进入条件

✅ 基础设施 (Phase 0.5)
✅ LLM 收口 (Phase 0a)
✅ Baseline + 锚点 (Phase 0b)
⏳ **PyMuPDF license 决策** (Phase 1 前必决)
⏳ **SearXNG 上游引擎限速解决** (Phase 1 检索需要数据)

### 推荐下一步

进 **Phase 1**(检索层替换)前,需要先解决两个 blocker:
1. PyMuPDF license 决策(3 选 1: fork / 商业 / 换 pdfplumber)
2. SearXNG 上游引擎问题(Tavily 付费 / SearXNG 多实例 / 接受当前限速)

你选哪个?
- **A**: 先解决这两个 blocker(预计 1-2 天)
- **B**: 直接进 Phase 1,blocker 在 Phase 1 内解决
- **C**: 跳到其他主题(比如先做 Langfuse 后端启动 / 修 deep_researcher.py 防御性读取)
