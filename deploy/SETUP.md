# Open Deep Research v2 — 一键部署 / 迁移指南

本指南面向**空机器 / 新 CI runner / 想从单机迁移到全栈容器**的场景。每一步都对应 `Makefile` 的目标,不用记 `docker compose` 长命令。

> TL;DR(假设有 docker + uv):
>
> ```bash
> git clone <this-repo> odr && cd odr
> cp .env.example .env        # 填入 API keys
> make install                # 装 .venv + 依赖
> make docker-up PROFILE=full # 拉起全套
> curl http://127.0.0.1:2024/healthz
> ```

---

## 1. 必备前置

| 工具 | 版本 | 用途 |
|------|------|------|
| Python | 3.13 | 本地 / 单测 |
| [`uv`](https://astral.sh/uv) | ≥0.4 | 依赖管理(本仓库 `uv.lock` 锁版)|
| Docker Engine | ≥24 | `make docker-up` 用 |
| `docker compose` v2 | 随 docker | 同上 |
| (可选)`.github_token` | — | git push 用,本机放 `/root/.github_token` |

如果是升级到 `pgvector`,Postgres 镜像必须支持扩展。`postgres:16-alpine` **不含** pgvector,见 §5。

---

## 2. 三种启动方式

### 2.1 仅本机裸跑(最快,适合改代码)

```bash
make install                         # uv 同步依赖
cp .env.example .env && vim .env     # 填 key
make migrate                         # 在 (另起的) PG 上应用 migrations/*.sql
make run                             # = uvicorn open_deep_research.api.server:app
```

调用:
```bash
curl -s -X POST http://127.0.0.1:2024/runs \
  -H 'content-type: application/json' \
  -d '{"query":"open source LLM framework", "mode":"evidence-only"}'
```

### 2.2 docker compose 拉基础设施 + 裸跑 API(推荐开发)

```bash
make docker-up-infra                  # 只起 postgres/redis/searxng
make migrate                         # 把 schema 刷进容器化 PG
POSTGRES_HOST=127.0.0.1 make run    # API 仍裸跑,但连到容器 PG
```

### 2.3 docker compose 全栈(推荐部署 / 给别人用)

```bash
make docker-build                    # 构建 odr/api 镜像
make docker-up PROFILE=full          # 一起起 postgres/redis/searxng/api/crawler
make docker-logs                     # 看 api 日志
make docker-down                     # 停
```

容器内 api 默认监听 `0.0.0.0:2024`,通过 `${API_BIND:-127.0.0.1}:${API_PORT:-2024}:2024` 暴露到 host。

---

## 3. profiles

`docker-compose.yml` 用 `profiles:` 切部署形态:

| Profile | 起哪些 |
|---------|--------|
| `infra` (default) | postgres + redis + searxng |
| `api` | infra + api(开发最小集)|
| `full` | infra + api + crawler-sidecar |
| `test-isolation` | 单独的 `pipeline-shell` 网络隔离测试容器(见 `deploy/test_isolation.py`)|

调用 `make docker-up PROFILE=api` / `=full` / `=test-isolation`。

---

## 4. 环境变量

`.env` 顶层文件会被 `langgraph.json` + entrypoint + docker-compose 同时消费。
所有变量参考 `.env.example`。**绝密值** (`TAVILY_API_KEY`, `MINIMAX_API_KEY`, `LANGSMITH_API_KEY`, `SUPABASE_KEY`, `POSTGRES_PASSWORD`) 不要 commit。`.gitignore` 已屏蔽顶层和 `deploy/.env`。

最小可用集(evidence-only 模式,无需外网 API):

```ini
POSTGRES_HOST=127.0.0.1
POSTGRES_USER=postgres
POSTGRES_PASSWORD=*****
POSTGRES_DB=odr_v2
SEARXNG_URL=http://127.0.0.1:8080
```

跑 `mode=full` 时再多一条 `TAVILY_API_KEY=tvly-*****`。

---

## 5. PG / pgvector

迁移由 `deploy/init_db.py` 顺序应用 `migrations/*.sql`,含:

- `000_phase05_init.sql` — `evidence` / `langfuse` schema
- `001_phase1_sources.sql` — sources 表
- `002_claim_and_evidence_unit_v2.sql` — evidence_unit / claim v2
- `003_pgvector.sql` — pgvector 扩展 + `embedding vector(384)` 列 + HNSW 索引

**重要**:`postgres:16-alpine` 默认不含 pgvector。要启用 RAG 向量库,改镜像为:

```yaml
image: pgvector/pgvector:pg16
```

或在本机 PG 上: `apt install postgresql-16-pgvector`。

`make migrate` 会自动应用 003;若扩展不可用,会**报错**,不会 silent skip(确保不被遗忘)。

---

## 6. SearXNG

`deploy/searxng/settings.yml` 是本地配置文件,跟上游默认略有差异。
如果发现 `arxiv / github` 引擎返回 "Suspended":

1. 进入容器:`docker compose exec searxng sh`
2. 看 secret / limiter:`cat /etc/searxng/settings.yml | grep -E 'secret|limiter'`
3. 改高 limiter 或换上游:`engines:replacements`

我们发现本机 SearXNG 默认 arxiv 被 rate-limit 杀。需要:

```yaml
engines:
  - name: arxiv
    disabled: false
    # 适度降低并发
    engine_kwargs:
      hl: en-US
```

---

## 7. CI / 部署到第三方平台

CI 部分目前只跑 review(.github/workflows/claude*.yml)。生产部署流水线留作后续 work。

如要用 **Fly.io / Render / Railway**:

- 复制 `Dockerfile.api` 作为平台 build 源
- 暴露 `PORT=2024`
- 把 §4 列出的 env 在平台 dashboard 注入

---

## 8. 验证清单

部署完后,逐条确认:

```bash
# 1. 健康
curl -sf http://127.0.0.1:2024/healthz | jq .pg_ok   # 应为 true

# 2. 起一次证据-only brief
RID=$(curl -s -X POST http://127.0.0.1:2024/runs -H 'content-type: application/json' \
  -d '{"query":"pgvector hybrid search","mode":"evidence-only"}' | jq -r .run_id)
# 轮询直到 completed/failed

# 3. 单测
make test                                              # 600+ passed in 60s
```

---

## 9. 出问题怎么办

| 现象 | 看哪里 |
|------|--------|
| `pg_ok=false` | `make docker-logs postgres` / `POSTGRES_HOST` 是否对得上 |
| `healthz` 返 200 但 `runs` 0 EU | SearXNG 引擎被限流(§6)|
| uvicorn 启动报 `tavily_python not installed` | `pip install tavily-python` 或跑 evidence-only 模式 |
| Docker 镜像构建失败 | `Dockerfile.api` 安装 `uv` 步骤,可能需要换下载源 |
| `migrate` 报 pgvector extension 找不到 | §5 |

---

## 10. 备份 / 还原 / systemd 常驻

### 备份(每日 crontab 推荐)

```bash
make backup
# 默认输出到 ./backups/<db>_<ts>.sql.gz
# 自动保留 7 天
```

或者手工:`POSTGRES_PASSWORD=*** ./scripts/backup_pg.sh`。

要挂 crontab:

```cron
# 每天凌晨 3 点备份
0 3 * * * cd /opt/open_deep_research && ./scripts/backup_pg.sh >> /var/log/odr-backup.log 2>&1
```

### 还原

```bash
make restore                                  # 用 ./backups/ 最新文件
# 或者
./scripts/restore_pg.sh backups/odr_v2_20260723_120000.sql.gz
```

脚本会提示 `confirm DROP+RECREATE`(安全保护)。

### systemd 常驻部署(host 跑 uvicorn,不用 docker)

适合把 ODR 跑在没 docker 的旧服务器上:

```bash
sudo cp deploy/systemd/open_deep_research.env.example \
    /etc/open_deep_research/open_deep_research.env
sudo chmod 0600 /etc/open_deep_research/open_deep_research.env
sudo vim /etc/open_deep_research/open_deep_research.env     # 填 keys
sudo cp deploy/systemd/open_deep_research.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now open_deep_research
sudo systemctl status open_deep_research
sudo journalctl -u open_deep_research -f
```

或者走 `make install-systemd` 看步骤。

服务监听 `0.0.0.0:2024`,再用 nginx / caddy 反向代理到 443(HTTPS)。


