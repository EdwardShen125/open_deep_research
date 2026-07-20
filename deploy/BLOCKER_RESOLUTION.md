# Phase 0 Blocker 决策报告

**生成时间**: 2026-07-18
**执行人**: Hermes (Claude)
**目的**: 解决 Phase 0.5 遗留的两个 Blocker,清零 Phase 1 入口

---

## Blocker 1: PyMuPDF AGPL-3.0 License 红线

### 现状摘要

- 包:`pymupdf 1.26.0`(自 1.24 版本起改 AGPL-3.0 dual-licensed)
- 来源:`Required-by: open-deep-research` —— 上游 ODR v0.10.0 pyproject.toml 直接依赖
- 我们源码 `src/`:**未 import**(验证过 `Loaded pymupdf/fitz modules: []`)
- Plan v2 红线:**禁止 AGPL**——网络服务跑 AGPL 等于强制开源

### 决策

**采用方案(c):从 pyproject.toml 剔除 PyMuPDF,完全用 pdfplumber 替代**

理由:
1. 零代码引用——移除无功能损失
2. `pdfplumber 0.11.10` (MIT) 已装好作为 PDF 替代
3. Plan v2 早已对齐(§"license 红线")
4. 商业 license 需付费 + 续费维护
5. 等上游剔除 PR 不可控(网络拉取 github 也受 sandbox 限制)

### 执行

```bash
# pyproject.toml 去掉 pymupdf>=1.25.3 行
uv pip uninstall pymupdf     # 实际包卸载
uv lock                       # 重新锁定
```

**已确认**:
- ✅ `uv pip show pymupdf` → "Package(s) not found"
- ✅ `deploy/check_license.py` 输出 `STATUS: PASS (no actionable license violations)`(首次)
- ✅ LangGraph server 自动 reload 未破坏
- ✅ smoke run 177.7s 出 16471 字符报告,v1 流水线正常

### 回滚路径

```bash
git checkout pyproject.toml
uv pip install "pymupdf>=1.25.3"
```

---

## Blocker 2: SearXNG 上游引擎限速 → Tavily 切换决策

### 现状摘要

| 层级 | 状态 |
|---|---|
| SearXNG 容器 (odr-searxng, healthy) | ✅ |
| SearXNG HTTP API 容器内 | ✅ 200,但 `unresponsive_engines: []` |
| SearXNG 所有上游引擎 (brave/duckduckgo/wikipedia/google) | ❌ 全部限速或 CAPTCHA |
| SearXNG 容器外端口 8888 | ❌ Connection refused(只接 outside-bridge 网络,设计上不暴露) |
| Tavily 直接 HTTP 调用 | ✅ 返回 5 个高相关结果(本文档测试) |

### 决策

**Phase 1 默认走 Tavily,SearXNG 保留为 fallback**

理由:
1. Tavily 已集成到 src/:`utils.py:42` `@tool(description=TAVILY_SEARCH_DESCRIPTION)` + `configuration.py` 默认 `SearchAPI.TAVILY`
2. 检索质量验证:5 个高相关结果 + **页面级 URL**(满足 Phase 3b B 类规则四)
3. Phase 1 plan v2 §"Phase 1 任务" 的"低成本自托管检索"目标:host 网络环境对 SearXNG 上游限速不可控,但 Tavily 是 metered pay-per-call,**完全绕开限速问题**
4. SearXNG 容器仍保留作为:
   - 当 host 网络恢复后的 fallback(`search_api: searxng` switch)
   - LLM 失败时的 metadata 兜底
   - Phase 4 multi-source cross-validation 的对照源

### 待补(Phase 1 实施时)

- `.env` 配置 `TAVILY_API_KEY` 的运行时已经存在(`tvly-dev-***`),但 **Phase 1 应替换为 production key**(`tvly-prod-...`)
- Tavily 配额监控(避免检索爆发 cost spike)
- 来源 URL 解析到 page-level 的二次校验(我们目前的 Tavily 命中已经全是 page-level,但需 SDK 抓 `raw_content` 时校验)

### 决策之外的次要观察

Tavily 第一命中结果(`kompyte.com/...comparison`)把 Kompyte 描述为**独立 CI 套件**(实际 2022 年已被 Crayon 收购)——这是评测站立场而非事实真相。

→ Phase 3a 规则二(实体关系验证)就为解决这种**来源立场混淆**而设:独立来源交叉验证时,Kompyte ownership 应该有 ≥2 个独立域名证据(比如 Crayon 官方新闻 + 第三方报道)确认"Kompyte 是 Crayon 旗下",而不是单一来源的厂商页。

---

## Phase 1 入口状态

| 项 | 状态 |
|---|---|
| 基础设施 (Phase 0.5) | ✅ |
| LLM 收口 (Phase 0a) | ✅ |
| Baseline + 锚点 (Phase 0b) | ✅ |
| PyMuPDF license | ✅ 已剔除 |
| SearXNG 上游引擎 | ⏸️ → Tavily 绕过决策 |
| Langfuse 后端 | ⏳ 后端镜像拉不下来,decorator 已就位(Phase 0a 完成),降级为本地 log |
| 防御性 model output | ⏳ Phase 2 顺手处理 `KeyError: 'reflection'` |

**Phase 1 入口**:所有 P0 项已就位,P1 项(Langfuse 后端)有降级路径,可以开始。

---

## 下一步

进 **Phase 1**(检索层替换),按 plan v2:
1. Crawl4AI 集成取代 summarize_webpage LLM 抽取
2. sources 表设计 + Postgres 落库
3. TTL 缓存层
4. SEARXNG_FALLBACK_URL 配置(Tavily 主路径,SearXNG 备份)
