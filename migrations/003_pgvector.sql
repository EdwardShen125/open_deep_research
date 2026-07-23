-- 003_pgvector.sql
-- pgvector 扩展 + evidence_unit embedding 列 (Runbook v2 §1 RAG vector store)。
-- 幂等:IF NOT EXISTS。可以反复执行。
--
-- 依赖:debian apt: postgresql-16-pgvector, 或 docker image `pgvector/pgvector:pg16`。
-- alpine 镜像需要手动 `apk add pgvector` 或换 base。

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 仅在 v2 schema 存在时执行 (init_db.py 先建)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'evidence') THEN
    -- embedding 列
    ALTER TABLE evidence.evidence_unit
      ADD COLUMN IF NOT EXISTS embedding vector(384);  -- sentence-transformers/all-MiniLM-L6-v2

    -- HNSW 索引 (pgvector ≥0.5 推荐,比 ivfflat 维护成本低)
    CREATE INDEX IF NOT EXISTS evidence_unit_embedding_hnsw
      ON evidence.evidence_unit USING hnsw (embedding vector_cosine_ops);

    -- claim 也开 embedding (Runbook v2 §3)
    ALTER TABLE evidence.claim
      ADD COLUMN IF NOT EXISTS embedding vector(384);
    CREATE INDEX IF NOT EXISTS claim_embedding_hnsw
      ON evidence.claim USING hnsw (embedding vector_cosine_ops);
  ELSE
    RAISE NOTICE 'evidence schema not yet present — skipping column add (run init_db.py first)';
  END IF;
END
$$;
