# =============================================================================
# Open Deep Research v2 — operator harness.
#
# 目标拆 4 组:
#   install     python / docker 一键装
#   dev         本地裸跑 (无 docker)
#   test        单测 / 端到端
#   deploy      docker compose 一键起 / 停 / 进
#   clean       拆台
#
# 自变量 (env override):
#   PORT=2024  API_BIND=0.0.0.0  PROFILE=full|api|infra|test-isolation
# =============================================================================

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

PROJECT  := open-deep-research
PY       := .venv/bin/python
PIP      := .venv/bin/pip
UV       := uv
COMPOSE  := docker compose -f deploy/docker-compose.yml

# ---- install / setup ------------------------------------------------------------

.PHONY: install
install:                    ## 一键本地安装 (uv 同步依赖)
	uv sync
	cp -n .env.example .env 2>/dev/null || true
	@echo ">> edit .env if you need real keys"

.PHONY: install-dev
install-dev:                ## 安装测试/lint 依赖
	uv sync --dev

# ---- dev ------------------------------------------------------------------------

.PHONY: run
run:                        ## 裸跑 uvicorn (用 shell env)
	@if [ ! -x .venv/bin/uvicorn ]; then echo "run 'make install' first"; exit 1; fi
	.venv/bin/uvicorn open_deep_research.api.server:app \
	    --host 127.0.0.1 --port $${PORT:-2024} --log-level $${LOG_LEVEL:-info}

.PHONY: langgraph
langgraph:                  ## 起 LangGraph dev (Studio)
	.venv/bin/langgraph dev --port $${PORT:-2024}

.PHONY: migrate
migrate:                    ## 应用 migrations/*.sql 到 PG (幂等)
	@POSTGRES_HOST=$${POSTGRES_HOST:-127.0.0.1} \
	 POSTGRES_USER=$${POSTGRES_USER:-postgres} \
	 POSTGRES_PASSWORD=$${POSTGRES_PASSWORD:-odr_v2_pg_pass_change_me} \
	 POSTGRES_DB=$${POSTGRES_DB:-odr_v2} \
	 .venv/bin/python deploy/init_db.py

# ---- test -----------------------------------------------------------------------

.PHONY: test
test:                       ## 全套单测
	POSTGRES_HOST=$${POSTGRES_HOST:-172.17.0.2} \
	POSTGRES_PASSWORD=$${POSTGRES_PASSWORD:-odr_v2_pg_pass_change_me} \
	.venv/bin/pytest tests/ -x -q

.PHONY: test-phase
test-phase:                 ## 跑单个 phase (PHASE=10)
	POSTGRES_HOST=$${POSTGRES_HOST:-172.17.0.2} \
	POSTGRES_PASSWORD=$${POSTGRES_PASSWORD:-odr_v2_pg_pass_change_me} \
	.venv/bin/pytest tests/ -x -q -k "phase$(PHASE)"

.PHONY: e2e
e2e:                        ## 端到端 smoke (baseline_e2e.py)
	@echo ">> make sure api is up on $${PORT:-2024}, or run \`make docker-up\` first"
	.venv/bin/python scripts/baseline_e2e.py

# ---- deploy (docker compose) ----------------------------------------------------

.PHONY: docker-build
docker-build:              ## 构建 api 镜像
	$(COMPOSE) --profile $${PROFILE:-full} build api

.PHONY: docker-up
docker-up:                  ## 起全套 (默认 profile=full: postgres+redis+searxng+api+crawler)
	$(COMPOSE) --profile $${PROFILE:-full} up -d
	@echo ">> waiting for api..."
	@for i in $$(seq 1 30); do \
	  if curl -fsS http://127.0.0.1:$${API_PORT:-2024}/healthz >/dev/null 2>&1; then echo "api ready"; exit 0; fi; \
	  sleep 2; \
	done; echo "api did not become ready in 60s"; $(COMPOSE) logs api

.PHONY: docker-up-infra
docker-up-infra:            ## 只起基础设施 (不带 api 容器)
	$(COMPOSE) --profile infra up -d postgres redis searxng

.PHONY: docker-down
docker-down:                ## 停全套并清网络 (保留 volumes)
	$(COMPOSE) --profile $${PROFILE:-full} down

.PHONY: docker-logs
docker-logs:                ## 跟踪 api 日志
	$(COMPOSE) logs -f --tail=50 api

.PHONY: docker-shell
docker-shell:               ## 进 api 容器 shell
	$(COMPOSE) --profile $${PROFILE:-full} exec api /bin/sh

.PHONY: docker-isolation
docker-isolation:           ## 进隔离验证容器
	$(COMPOSE) --profile test-isolation run --rm pipeline-shell

# ---- clean ----------------------------------------------------------------------

.PHONY: clean
clean:                      ## 清 venv + pycache + .pytest_cache
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean docker-down ## 全清 (含 docker 卷)
	$(COMPOSE) down -v

# ---- help -----------------------------------------------------------------------

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
