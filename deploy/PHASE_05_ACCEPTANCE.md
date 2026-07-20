# Phase 0.5 验收报告

**执行时间**: 2026-07-18
**执行人**: Hermes (Claude)
**范围**: 仅基础设施,不动业务代码

---

## 验收清单

| # | 验收项 | 结果 | 证据 |
|---|---|---|---|
| 0.5.1 | compose + .env + searxng/settings.yml | ✅ | `docker compose config --quiet` 通过 |
| 0.5.2 | 核心服务启动 + 健康检查 | ✅ | odr-postgres / odr-redis / odr-searxng 全部 `healthy` |
| 0.5.3 | 网络隔离验证 | ✅ | `RESULT: ALL_PASS` 6/6 |
| 0.5.4 | 数据库 + schemas | ✅ | `evidence` / `langfuse` / `public` |
| 0.5.5 | Python 依赖 + license | ⚠️ 部分 | 4/5 已装,1 license 告警需 Phase 1 前决策 |

---

## 启动的容器

```
NAME                STATUS                    PORTS
odr-postgres        Up X minutes (healthy)    5432/tcp
odr-redis           Up X minutes (healthy)    6379/tcp
odr-searxng         Up X minutes (healthy)    8080/tcp
odr-pipeline-shell  Up X minutes              (隔离验证用)
```

`Langfuse` 和 `TEI (BGE-M3)` **未启动**——Docker Hub 网络拉镜像超时,见 §已知问题。

---

## 网络隔离验证结果

测试脚本: `deploy/test_isolation.py`
测试方法: 在 `odr-pipeline-shell`(仅接 `pipeline-internal` 内部网)内测试 6 项:

| 目标 | 期望 | 实际 | 结果 |
|---|---|---|---|
| bing.com:443 | BLOCKED | BLOCKED (gaierror DNS 失败) | ✅ |
| google.com:443 | BLOCKED | BLOCKED | ✅ |
| github.com:443 | BLOCKED | BLOCKED | ✅ |
| odr-searxng:8080 | OK HTTP | OK HTTP 200 | ✅ |
| odr-postgres:5432 | OK TCP | OK TCP open | ✅ |
| odr-redis:6379 | OK TCP | OK TCP open | ✅ |

**关键拓扑**:
- `outside-bridge` (普通 bridge): postgres / searxng / TEI 需出外网的服务
- `data` (`internal: true`): 服务间互访
- `pipeline-internal` (`internal: true`): pipeline-shell 专用,不可外联

容器对网络 iptables 规则验证通过。

---

## 数据库

```
postgres 16-alpine
  db: odr_v2
  user: postgres
  schemas: public, evidence, langfuse
```

`migrations/` 目录已建,占位文件 `000_phase05_init.sql`。完整 schema(sources/evidence_units/entities/eu_entities/claim_clusters/report_claims)按计划 v2 在 Phase 1/2 实施时落地。

---

## Python 依赖

```
psycopg     3.3.4   (LGPL-3.0)     Phase 1+ 数据库
langfuse    4.14.0  (MIT)          Phase 0a 可观测性
arq         0.28.0  (MIT)          队列(预留)
crawl4ai    0.9.2   (Apache-2.0)   Phase 1 抓取
pdfplumber  0.11.10 (MIT)          Phase 1+ 文档解析(替代 PyMuPDF)
pymupdf     1.26.0  (AGPL-3.0 ⚠️)  上游依赖,见下文
```

**docling 延后到 Phase 2**: 拖拽 torch + nvidia CUDA 全套(数 GB),PyPI 镜像超时无法一次装完。

---

## ⚠️ 关键发现:License 红线

`deploy/check_license.py` 检测到上游 `open_deep_research` 项目 `pyproject.toml` 直接依赖 **`pymupdf 1.26.0`**,该版本采用 **AGPL-3.0 dual-licensed**(或购买 Artifex 商业 license)。

**计划 v2 红线明文**: "PyMuPDF、MinerU、marker 禁止(AGPL)。文档解析仅用 Docling / pdfplumber。"

**但 PyMuPDF 在当前架构下是 transitive dependency**——上游 ODR 框架本身(可能在 PDF 解析某个节点)用了它,我们没有 import 但间接依赖树带上了。

### 决策项(Phase 1 前必须拍板)

| 选项 | 含义 | 风险 |
|---|---|---|
| (a) 上游 fork / 改 pyproject.toml 剔除 pymupdf | 工作量大,需 fork + PR upstream | 维护负担 |
| (b) 购买 Artifex 商业 license | 付费,简单 | 成本 |
| (c) 全部改用 pdfplumber + pypdfium2(MIT/Apache) | 替换上游 PDF 解析 | 兼容测试 |

**本 Phase 0.5 不解决此问题**,但已建立 `check_license.py` 作为持续监控——任何新增依赖都会被检查。Phase 1 实施时第一件事是解决它。

---

## 已知问题

### 1. Langfuse / TEI 镜像未拉取

- `langfuse/langfuse:2` + `langfuse/langfuse-clickhouse:23` + `ghcr.io/huggingface/text-embeddings-inference:1.5`
- 镜像过大(各 1-3 GB),Docker Hub 当前网络条件下拉取超时
- **影响**: Phase 0a 的 Langfuse 埋点暂不可用;Phase 2 的 BGE-M3 聚类暂不可用
- **缓解**:
  - Phase 0a 可先代码层集成 langfuse SDK,运行时连接失败降级为本地 log
  - BGE-M3 聚类可暂时用 sentence-transformers 本地跑(BGE-M3 模型 ~2GB)
  - 待网络恢复后再补镜像

### 2. SearXNG 上游搜索引擎受网络限速

- 隔离验证 HTTP 200,但实际搜索结果为 0(`brave: too many requests`, `duckduckgo: CAPTCHA`, `google: suspended`)
- **这是上游引擎的限速,不是 SearXNG 配置问题**
- Phase 1 实施时,需要解决:
  - 选项 A: 启用 Tavily / SerpAPI 等付费引擎
  - 选项 B: 接多个 SearXNG 实例做负载分散
  - 选项 C: 在 `.env` 里只配当前可达的引擎

### 3. SearXNG 2026.7.3 配置变更

新版 SearXNG 移除 DOI resolver 功能,settings.yml 需省略 `default_doi_resolver` 字段。
已在 `deploy/searxng/settings.yml` 中处理,留注释说明。

---

## Phase 0.5 交付物清单

```
deploy/
├── docker-compose.yml           容器编排(网络隔离拓扑)
├── .env                         凭证与配置
├── searxng/settings.yml         SearXNG 网络隔离版
├── test_isolation.py            网络隔离验证脚本
├── init_db.py                   DB init 工具脚本(供后续 phase 复用)
├── check_license.py             License 审计(AGPL 红线持续监控)
└── README.md                    (待补)

migrations/
└── 000_phase05_init.sql         Phase 0.5 占位,完整 schema 在后续 phase 落地
```

---

## Phase 0a 准备工作清单

进入 Phase 0a 还需要做的事:
1. **PyMuPDF license 决策**——见上文三选一
2. **Langfuse 镜像拉取**——网络恢复后补
3. **TEI 镜像拉取**——同上,或先用 sentence-transformers 本地跑 BGE-M3
4. **SearXNG 上游引擎限速**——Phase 0a 不影响(只读 trace),Phase 1 检索时必须解决
5. **freeze prompts.py** (git tag `pre-prompts-migration`)

---

## 回滚指引

`docker compose down -v` 即可清空所有容器与卷,无破坏性改动到业务代码。
