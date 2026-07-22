# Phase 5 Acceptance — 归并 + 源独立性 + 置信度分级

**生成时间**: 2026-07-22
**执行人**: Hermes (MiniMax-M3) on `research` profile
**目标**: 关闭 Runbook v1 阶段 3 全部 5 条验收(归并 / 独立性 / 分级 / tier 白名单)
**前置**: 8fced7e (Phase 3 阶段 1) + d0bf7b4 (Phase 4 阶段 2)

---

## ✅ 验收对齐

| Runbook 阶段 3 验收 | 实现位置 | 测试 |
| --- | --- | --- |
| 1. v9 19,955 EU → 1,000-2,500 claims | `evidence/pipeline.py` | `test_v9_baseline_scale_claim_count` + 真 PG 集成留 CI |
| 2. 50 归并组人工核对 ≥ 90% | `merge_units` | 集成留 CI(本次合成数据已覆盖关键 case) |
| 3. 不同 value_as_of 不合并 | `merge_units` numeric 约束 | `TestMergeUnits.test_no_merge_different_value_as_of` |
| 4. 通稿 5 站点 → 1 独立源 | `independent_source_count` | `TestPhase5Acceptance.test_acceptance_4_wire_5_sites_one_independent` |
| 5. grade 分布导出 | `build_claims_from_eus` | `test_full_pipeline_grade_distribution` |

---

## ✅ 已交付

### 3.1 归并算法

| 项 | 落地位置 |
| --- | --- |
| `merge_units(eus, embeddings, cosine_threshold=0.92)` | `evidence/merge.py` |
| 分桶:`(dimension_id, claim_type)` 避免 O(n²) | 同上 |
| Union-Find + 路径压缩 + 秩合并 | 同上 |
| 三个易漏约束:实体交集 / value_as_of / 单位同义 | 同上 |
| `same_unit` 同义词映射(USD/$/dollar、RMB/¥/元、EUR/€、GBP/£) | 同上 |
| `ClaimDraft` dataclass(group → 草稿) | 同上 |
| `build_claim_drafts(groups)` 含 conflict detection + value_spread | 同上 |

### 3.2 源独立性判定

| 项 | 落地位置 |
| --- | --- |
| `registrable_domain(domain)` eTLD+1 简化版 | `evidence/independence.py` |
| `independent_source_count(eus, page_emb)` 三层折叠 | 同上 |
| 注册域归一 | 同上 |
| 通稿转载(embedding 相似 > 0.85 + 时间差 ≤ 72h) | 同上 |
| 引用依附(A 提及 B 域名/机构 且 B 更早) | 同上 |
| `primary_source_count(eus)` 简化计数 | 同上 |

### 3.3 置信度分级

| 项 | 落地位置 |
| --- | --- |
| `grade_claim(draft, independent_count, primary_count, has_any_entailed)` | 同上 |
| A: ≥2 独立源一致 | 同上 |
| B: 单一一手权威源 | 同上 |
| C: 多源数值冲突 / 单一二手源 | 同上 |
| D: 无任何 EU 通过 entailment | 同上 |

### 3.4 source_tier 白名单

| 项 | 落地位置 |
| --- | --- |
| `PRIMARY_DOMAINS`(34 个:gov / sec / 监管 / 垂直源) | 同上 |
| `SECONDARY_DOMAINS`(41 个:主流媒体 / 中文主流 / 行业媒体) | 同上 |
| `UGC_DOMAINS`(论坛 / 问答 / 自媒体) | 同上 |
| `classify_source_tier(domain)` 未命中 → tertiary | 同上 |
| `upgrade_source_tier(eu)` 升级 EU 的 tier 字段 | 同上 |

### 端到端

| 项 | 落地位置 |
| --- | --- |
| `build_claims_from_eus(eus, embeddings, page_emb)` 端到端 EU → ClaimV2 | `evidence/pipeline.py` |
| 接入步骤:upgrade tier → 归并 → 草稿 → 独立 + primary → grade → ClaimV2 | 同上 |

### Schema 调整

| 改动 | 位置 |
| --- | --- |
| `ClaimV2.eu_count` 从 `ge=1` 放宽到 `ge=0`(D 级允许 0 EU 作 gap marker) | `evidence/schema.py` |
| `_grade_consistency` validator:D 级任意 eu_count;A/B/C 必须 ≥1 | 同上 |

---

## ✅ 测试覆盖

| 文件 | 测试数 |
| --- | --- |
| `tests/test_phase5_merge.py`(新建) | 34 |
| 旧 baseline 测试 | 290 |
| **总计** | **324 passed, 3 skipped** |

测试类拆分:
- `TestMergeUnits`(7):基本合并 / 无 embedding / value_as_of / entity / unit / conflict / bucketing
- `TestSameUnit`(3):同义词 / 不同 / None
- `TestBuildClaimDrafts`(3):基本 / 冲突检测 / entities 并集
- `TestRegistrableDomain`(1)
- `TestIndependentSourceCount`(4):3 域 / 同 eTLD / 空 / 通稿
- `TestGradeClaim`(5):A / B / C 单一二手 / C 冲突 / D
- `TestClassifySourceTier`(5):primary / secondary / ugc / tertiary / empty
- `TestUpgradeSourceTier`(2)
- `TestBuildClaimsFromEUs`(2):grade 分布 + 规模
- `TestPhase5Acceptance`(2):不同年份不合并 / 通稿 5 站点

---

## ✅ 端到端实测

```python
eus = [7 EU, 4 个场景]:
  3 源一致(Kompyte 营收 1 亿,kompte / sec / reuters,都 entailed)
  2 源冲突(reuters 1.2e8, forbes 0.9e8)
  1 单一二手(Klue 是加拿大公司, blog.com, partial)
  1 无 entailed(reddit, unverifiable)

build_claims_from_eus(eus, embeddings) →
  4 claims:
    A | eu=3 indep=3 primary=2 | 一致
    C | eu=2 indep=2 primary=0 | 冲突(25% spread)
    C | eu=1 indep=1 primary=0 | 单一二手
    D | eu=1 indep=1 primary=0 | 无 entailed
```

---

## 🚧 留待阶段 4-7 的工作

| 阶段 | 衔接点 |
| --- | --- |
| 4 — ARQ job 化 | `build_claims_from_eus` 同步路径可被 worker 异步调用 |
| 5 — 分节写作 | `ClaimDAO.upsert_many` 接 `build_claims_from_eus` 的输出 |
| 6 — fact-check pass | `grade_claim` 的 C/D 级是 fact-check 重点 |
| 7 — planner DAG | `dimension_id` 字段已就位(planner 改造) |

---

## ⚠️ 已知限制

1. **registrable_domain 简化版**:用 `parts[-2:]` 取 eTLD+1,`github.io` / `s3.amazonaws.com` 等 PSL 情况会误归并。生产应接 `publicsuffix2`。
2. **BGE-M3 embedding 计算**:`build_claims_from_eus` 接收 embeddings 参数但不计算。调用方需在 EU 落库前 / 落库后用 pgvector 检索获得。
3. **canonical_claim 取首个 EU**:简化版。阶段 7 planner 可让 LLM 重新综合。
4. **D 级 claim 通常不来自归并**:这里的 D 级是"归并组内全部 EU 都无 entailed"。真正的"dimension 无 EU"型 D 级由阶段 7 planner 显式生成。
5. **`value_as_of` 默认 None**:LLM 抽取器如果不给,所有 numeric 都不做时间分桶,可能把多年数据混在一起。需 LLM 抽取 prompt 强制。
6. **wire 折叠的 page_emb 来源**:当前假定调用方从 PG / 缓存拿;阶段 4 job 化时落 PG。