-- ============================================================
-- Phase 3 (= Runbook v1 阶段 1): Claim + EvidenceUnit v2 双层模型
-- ============================================================
-- 依据: notes/evidence-pipeline-runbook-v1.md 阶段 1.1–1.3 + 验收
-- 命名沿用 evidence.<table>,因为 Phase 1 已经创建 schema。
--
-- 双层模型:
--   claim         : 跨源归并后的"可解释结论"。报告只消费 Claim。
--   evidence_unit : 单源原子观察。不可变、只增。闸 1/2/3 (阶段 2) 的入口。
--   run_checkpoint: 长跑续跑 (阶段 4 用,本阶段先建表)。
--
-- 与已有 evidence.sources 表的关系:
--   sources: 每个 URL 一行。Phase 1 页面级校验的入口。
--   evidence_unit.source_url 引用 sources.url (软引用,不强制 FK,因为 EU
--   可来自 cached / pre-Crawl4AI summary 而 source 未登记)。
--
-- 设计:
--   1. eu_id / claim_id 是 UUID,前端友好且利于跨 run 引用。
--   2. embedding 1024 维 = BGE-M3,与 langchain embeddings 一致。
--   3. run_checkpoint 的 (run_id, stage) PK 让 worker 重启时幂等续跑。
--   4. span / span_start / span_end 是 EU 相对 source_url 正文文本的
--      字符偏移,阶段 2 闸 1 用它们做字面/归一/模糊匹配。
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------- Claim ----------
CREATE TABLE IF NOT EXISTS evidence.claim (
    claim_id                 UUID PRIMARY KEY,
    run_id                   UUID NOT NULL,
    dimension_id             TEXT NOT NULL,
    canonical_claim          TEXT NOT NULL,
    claim_type               TEXT NOT NULL,        -- numeric / event / attribute / relation / opinion
    entities                 TEXT[] NOT NULL DEFAULT '{}',
    norm_value               NUMERIC,
    unit                     TEXT,
    value_as_of              DATE,
    value_spread             REAL,                 -- 各源数值最大相对偏差
    eu_count                 INT  NOT NULL,
    independent_source_count INT  NOT NULL,
    primary_source_count     INT  NOT NULL,
    earliest_published_at    TIMESTAMPTZ,
    has_conflict             BOOLEAN NOT NULL DEFAULT FALSE,
    conflicting_values       JSONB NOT NULL DEFAULT '[]'::jsonb,
    grade                    TEXT NOT NULL,        -- A / B / C / D
    grade_reason             TEXT NOT NULL,
    embedding                VECTOR(1024),         -- BGE-M3
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_claim_run_dimension_grade
    ON evidence.claim (run_id, dimension_id, grade);
CREATE INDEX IF NOT EXISTS idx_claim_embedding_hnsw
    ON evidence.claim USING hnsw (embedding vector_cosine_ops);

-- ---------- EvidenceUnit v2 ----------
CREATE TABLE IF NOT EXISTS evidence.evidence_unit (
    eu_id              UUID PRIMARY KEY,
    run_id             UUID NOT NULL,
    dimension_id       TEXT,                     -- Nullable in 阶段 1;阶段 3 强制
    claim              TEXT NOT NULL,
    claim_type         TEXT NOT NULL,             -- numeric / event / attribute / relation / opinion
    entities           TEXT[] NOT NULL DEFAULT '{}',
    norm_value         NUMERIC,
    unit               TEXT,
    value_as_of        DATE,
    source_url         TEXT NOT NULL,
    source_domain      TEXT NOT NULL,
    source_title       TEXT,
    published_at       TIMESTAMPTZ,
    source_tier        TEXT NOT NULL,             -- primary / secondary / tertiary / ugc
    source_span        TEXT NOT NULL,             -- 逐字片段,≥10 字符 (阶段 2 闸 1 校验)
    span_start         INT,
    span_end           INT,
    extractor_model    TEXT NOT NULL,
    extracted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    span_verified      BOOLEAN NOT NULL DEFAULT FALSE,
    numeric_drift      BOOLEAN NOT NULL DEFAULT FALSE,
    entailment_verdict TEXT,                     -- entailed / partial / contradicted / unverifiable
    entailment_score   REAL,
    claim_id           UUID REFERENCES evidence.claim(claim_id) ON DELETE SET NULL,
    embedding          VECTOR(1024),
    -- 兼容字段:旧 EU dataclass 用 content_hash 做 dedup,新表继续保留 hash 索引
    content_hash       CHAR(64)
);

CREATE INDEX IF NOT EXISTS idx_eu_run_dimension
    ON evidence.evidence_unit (run_id, dimension_id);
CREATE INDEX IF NOT EXISTS idx_eu_claim_id
    ON evidence.evidence_unit (claim_id);
CREATE INDEX IF NOT EXISTS idx_eu_embedding_hnsw
    ON evidence.evidence_unit USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_eu_content_hash
    ON evidence.evidence_unit (content_hash);
CREATE INDEX IF NOT EXISTS idx_eu_span_unverified
    ON evidence.evidence_unit (run_id) WHERE span_verified = FALSE;

-- ---------- run_checkpoint (阶段 4 用,阶段 1 先建) ----------
CREATE TABLE IF NOT EXISTS evidence.run_checkpoint (
    run_id       UUID NOT NULL,
    stage        TEXT NOT NULL,
    status       TEXT NOT NULL,                  -- running | done | failed
    payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at  TIMESTAMPTZ,
    PRIMARY KEY (run_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_status
    ON evidence.run_checkpoint (status) WHERE status IN ('running', 'failed');