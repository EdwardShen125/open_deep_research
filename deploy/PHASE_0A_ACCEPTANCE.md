# Phase 0a 验收报告

**执行时间**: 2026-07-18
**执行人**: Hermes (Claude)
**范围**: LLM 收口 + Prompt 版本化 + Langfuse 接入
**前置**: Phase 0.5 基础设施 ✓

---

## 验收清单

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 0a.1 | llm.py 收口 + MiniMax 路由 | ✅ | `get_llm('minimax:MiniMax-M3')` → `ChatMiniMax` |
| 0a.2 | prompts/ 拆分 + 版本化 + registry | ✅ | 7 个 role,每个含 `PROMPT_VERSION` 常量 |
| 0a.3 | Langfuse 接入(prompt_version metadata) | ✅ | `trace_llm` decorator 同步/异步双实现 |
| 0a.4 | MiniMax provider 与 llm.py 适配 | ✅ | 走原有 `__init__._resolve_chat_model` 路由 |
| 0a.5 | prompts.py 冻结 + tag | ✅ | `git tag pre-prompts-migration` |
| 0a.6 | grep 无绕过 llm.py 的直接调用 | ✅ | `init_chat_model` 0 命中 active code |

**附加**: 端到端 smoke run 通过(`15.5s / 3911 字符报告`),LangGraph server 自动 reload 无破坏。

---

## 文件变更

```
新增:
  src/open_deep_research/llm.py            (177 行,LiteLLM 路由 + Langfuse)
  src/open_deep_research/prompts/__init__.py    (registry + backward-compat shim)
  src/open_deep_research/prompts/clarify_v1.py
  src/open_deep_research/prompts/research_brief_v1.py
  src/open_deep_research/prompts/supervisor_v1.py
  src/open_deep_research/prompts/researcher_v1.py
  src/open_deep_research/prompts/compressor_v1.py
  src/open_deep_research/prompts/writer_v1.py
  src/open_deep_research/prompts/webpage_summarizer_v1.py
  src/open_deep_research/prompts_legacy.py      (deprecated shim, 仍 import-ok)

修改:
  src/open_deep_research/__init__.py        (+3 行:tags 透传)
  src/open_deep_research/utils.py           (15 行:summarization_model 改走 get_llm)
  src/open_deep_research/deep_researcher.py (-1 行:删 unused import + 1 行注释更新)

删除:
  src/open_deep_research/prompts.py         (被 prompts/ 取代,已重命名为 prompts_legacy.py 保留)
```

git tag: `pre-prompts-migration` (commit `ff3b70b`)

---

## 核心 API

### `llm.py` 公共接口

```python
from open_deep_research.llm import (
    get_llm,             # 统一 LLM 入口(config → BaseChatModel)
    get_prompt,          # 按 role 加载 prompt 文本
    get_prompt_version,  # 返回版本字符串(用于 Langfuse metadata)
    trace_llm,           # @decorator,自动写 span + prompt_version metadata
    flush_langfuse,      # 收尾时 flush
    langfuse_status,     # 诊断
)
```

### `prompts/` Registry

```python
from open_deep_research.prompts import REGISTRY, load_prompt
# 7 roles: clarify, research_brief, supervisor, researcher, compressor, writer, webpage_summarizer

text, version = load_prompt("supervisor")
# version = "supervisor_v1"  # 写进 Langfuse metadata
```

### 向后兼容

```python
# 旧调用方式仍可用(__init__.py 的 __getattr__ 自动转发)
from open_deep_research.prompts import lead_researcher_prompt  # → supervisor_v1.lead_researcher_prompt
```

---

## Langfuse 集成设计

### 当前状态

| 项 | 状态 |
|---|---|
| SDK | `langfuse==4.14.0`(已装) |
| 客户端初始化 | 条件初始化(LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY 存在才启) |
| 后端服务 | **未启动**(Phase 0.5 网络问题,镜像未拉) |
| 降级行为 | 无 LANGFUSE creds → decorator 透传,无错误 |

### Span API(Langfuse 4.x OTel-native)

```python
tracer = client._otel_tracer
with tracer.start_as_current_span("node_name"):
    client.update_current_span(metadata={
        "prompt_role": "researcher",
        "prompt_version": "researcher_v1",
        "node": "researcher_node",
    })
```

> Phase 0b 跑基线报告时,LANGFUSE creds 仍可能不在(网络问题)。decorator 设计保证运行无破坏,只是 trace 落空。

---

## 已知遗留(进入 Phase 0b 前)

1. **Langfuse 后端未起**: LANGFUSE_HOST 镜像未拉,Phase 0a 在无 Langfuse 下完成验证。Phase 0b 前需决策:
   - (A) 等网络恢复拉镜像
   - (B) 用 local log + LangSmith 替代(LANGSMITH_API_KEY 空,当前不可用)
   - (C) 跳过 Langfuse 验证,延后到 Phase 1

2. **PyMuPDF license 红线仍未解决**: Phase 0.5 标记为 Phase 1 前必决。当前 pip 装的是 1.26.0 AGPL,不在 Phase 0a 改动范围,但**任何新增 Python 依赖**已被 `deploy/check_license.py` 监控。

3. **deep_researcher.py 节点未加 `@trace_llm`**: 当前架构 `configurable_model` 是 lazy-routing,无法静态装饰。Phase 0b 实施时考虑:
   - (A) 在每个 node 函数内手动调用 `trace_llm(role=..., node_name=...)` 包裹 LLM 调用
   - (B) 改造 `configurable_model` 让其支持自动 trace hook
   - (C) 推迟到 Phase 2(配合 EvidenceUnit 重写一起做)

4. **prompts 没版本化压缩后的细分子角色**: v1 只有 7 个 role,Phase 2 引入 extractor/validator/entity_resolver 后扩展到 10+ 个 role。现有 `prompts/{role}_v{N}.py` 命名模式已支持。

---

## 回滚路径

```bash
git checkout pre-prompts-migration
# 或:
git revert <phase-0a-commit-sha>
```

`prompts_legacy.py` 仍存在,旧 import 路径在回滚后立刻可用。
`llm.py` 可单独删除(其他模块不依赖),回滚风险低。

---

## Phase 0b 准备工作清单

进入 Phase 0b(基线报告 + state_v1.md + 错误锚点 JSON)需要:

1. **决定 Langfuse 状态**(见遗留 1)
2. **基线题目已确认**: `LLM竞品-市场调研分析产品-20260716-2245.md`(已读)
3. **跑 v1 基线 trace**: `curl POST /threads/{tid}/runs/wait` 跑负样本原题
4. **写 state_v1.md**: 抄录 AgentState / SupervisorState / ResearcherState + 5 个 Pydantic 结构
5. **固化 4 类错误锚点**:
   - A 类(虚构 relation / 所有权缺失 / 估值断言)
   - B 类(13 个域名级 URL)
   - C 类(中文数字表达)
   - D 类(表格 vs 正文数字不一致 3 处)
