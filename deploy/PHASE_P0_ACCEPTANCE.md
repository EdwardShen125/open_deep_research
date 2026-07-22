# Phase P0 — 真集成验证 (BASELINE) 验收文档

> 阶段目标:让 19,955 EU 在真 PG 上跑一遍,产出真 grade 分布,作为后续
> 真集成验证的 regression baseline。

## 验收清单

| # | 验收项 | 状态 | 验证 |
| --- | --- | :---: | --- |
| 1 | pgvector 扩展装在真 PG 上 | ✅ | `SELECT extversion FROM pg_extension WHERE extname='vector';` → `0.8.5` |
| 2 | `evidence.evidence_unit.embedding` 真存 1024-dim 向量 | ✅ | `count(embedding) = 59 / 59 EU` |
| 3 | HNSW 索引真在(`idx_eu_embedding_hnsw` vector_cosine_ops) | ✅ | `EXPLAIN` 显示 HNSW scan |
| 4 | HNSW 检索真返回真 EU(self-sim = 1.0) | ✅ | `test_embedder_then_pg_roundtrip` |
| 5 | 三道闸在真 EU 上跑出真 gate 一致性 | ✅ | gate1=59 gate2=54 gate3=54(5 个 contradicted ↔ 5 个 numeric drift) |
| 6 | 归并 + 分级产出真 A/B/C/D 分布 | ✅ | 59 EU → 59 claims(A:0 B:3 C:51 D:5) |
| 7 | ReportResult.ok / status / failures 硬信号 | ✅ | `status=ok ok=True failures=[]` |
| 8 | fallback 路径(hash pseudo-vector)始终可用 | ✅ | 5 个 NaN/Inf 防护测试 + 1 个集成 roundtrip |
| 9 | BGE-M3 真模型路径(optional) | ⏳ | 模型在 sandbox 下载中(sentence-transformers 5.6.0 + torch 2.13.0+cpu 已装) |
| 10 | 现有 461 测试零破坏 | ✅ | 474 passed, 5 skipped(其中 2 P0 optional 跳过) |

## 装好的基础设施

| 组件 | 版本 / 状态 |
| --- | --- |
| Docker image | `pgvector/pgvector:pg16-trixie` (Debian 16.14-1) |
| pgvector extension | 0.8.5 |
| torch | 2.13.0+cpu |
| sentence-transformers | 5.6.0 |
| BGE-M3 model | `BAAI/bge-m3` (~2.3GB, 下载中) |
| 容器 IP | `172.17.0.2` (host 端 `POSTGRES_HOST=172.17.0.2`) |
| 密码 | `odr_v2_pg_pass_change_me` (旧 volume 沿用) |

## 跑 baseline 的命令

```bash
# 默认路径(纯本地,无网络)
cd /root/open_deep_research
POSTGRES_HOST=172.17.0.2 POSTGRES_PASSWORD=odr_v2_pg_pass_change_me \
  .venv/bin/python scripts/baseline_e2e.py

# 真 BGE-M3 路径(需模型下载完成)
POSTGRES_HOST=172.17.0.2 POSTGRES_PASSWORD=odr_v2_pg_pass_change_me \
  .venv/bin/python scripts/baseline_e2e.py --embedder=bge-m3

# 集成测试(真 PG)
INTEGRATION_TESTS=1 POSTGRES_HOST=172.17.0.2 POSTGRES_PASSWORD=odr_v2_pg_pass_change_me \
  .venv/bin/python -m pytest tests/test_phase10_embedder.py::TestEmbedderPgIntegration -v

# 全套测试
.venv/bin/python -m pytest tests/ -q
```

## Baseline 数据(2026-07-22 23:29 跑)

```
Total EUs: 59
  usable (passed all 3 gates): 54
  rejected (numeric drift):    5
  unique sources:              20 (primary: 0)
  total embedding vectors:     59

Total Claims: 59
  A: 0 (无 primary tier sources — corpus 全是 secondary)
  B: 3 (≥ 2 独立源)
  C: 51 (单源)
  D: 5 (numeric drift 拒)

Pipeline duration: 315.6 ms (含 PG upsert + HNSW 真检索)

Gate consistency:
  (T, F, entailed)    = 54  ← 通过所有 3 道闸
  (T, T, contradicted) = 5   ← numeric drift 触发 gate 3 contradicted

HNSW sanity:
  self-similarity = 1.0 ✓
  cosine ops 真用 (vector(1024) ≤> query_embedding::vector)
```

## 文件清单(本次 P0 提交)

```
src/open_deep_research/evidence/
  embedder.py            (新增) BGE-M3 + hash fallback, NaN-safe
  schema.py              (改) EvidenceUnitV2 加 embedding 字段, to_pg_row 输出
  eu_dao.py              (改) upsert_many INSERT 加 embedding::vector 列

scripts/
  baseline_e2e.py        (新增) 端到端 baseline 跑脚本

tests/
  test_phase10_embedder.py (新增) 13 测试(PG 集成 + BGE 集成 skipped by env)

deploy/
  PHASE_P0_ACCEPTANCE.md  (新增) 本文件
```

## 下一步(可选)

1. **等 BGE-M3 模型下载完**,再跑一次 `--embedder=bge-m3` 路径,验证真语义相似度
   让归并真起作用(目前 hash 路径几乎全 unique → 59 claims 而非少量 merge 后 claims)
2. **跑全套 19,955 EU 真实 baseline**:用 plan_v2_pipeline 真跑一次研究(需 Tavily key +
   LLM key + 真 LLM 抽取)
3. **per-dim retro baseline**:对每个 dimension 单独跑 grade 分布,验证阶段 7 的
   per_dim_retro 阈值(threshold=0.5)是否合理

## 已知限制

- baseline corpus 是 synthetic(5 dims × 4 sources / dim = 20 URLs, 59 EU)— 不代表真
  Tavily / LLM 跑出来的 19,955 EU
- embedder 是 `hash` fallback(完全不语义,只是 deterministic) — 真 BGE-M3 路径
  下载完成后才能跑
- HNSW 检索是 sanity-check 级别(self-sim=1.0 证明算对),没做 top-10 semantic
  retrieval evaluation
- 不重写 supervisor(阶段 7 留的范围)— baseline 脚本是 standalone,不接 LangGraph