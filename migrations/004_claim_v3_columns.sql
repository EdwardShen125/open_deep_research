-- ============================================================
-- Migration 004: ClaimV3 fields on evidence.evidence_unit
-- ============================================================
-- plan A1:扩展 Pydantic + 写 PG migration。
-- 加 6 列(全部 nullable / default 满足回填,12,598 已有 EU 不丢):
--   tier                TEXT                      A/B/C/D,A2 填
--   caliber_id          TEXT                      F2 注册表 id,C2 填
--   verification_status TEXT NOT NULL DEFAULT 'to_verify'
--   gate_results        JSONB NOT NULL DEFAULT '{}'::jsonb
--   origin_chain        TEXT[] NOT NULL DEFAULT '{}'
--   embedding_model     TEXT                      版本列
--
-- 索引:
--   idx_eu_tier             (E1 来源页 + C1 路由按 tier 过滤)
--   idx_eu_verification     (E1 主结论门 + 待核实附录)
--   idx_eu_caliber          (D1 聚簇按 caliber 过滤)
--
-- Down:全部 DROP IF EXISTS,幂等可重放。
-- ============================================================

-- ---------- Up ----------

ALTER TABLE evidence.evidence_unit
    ADD COLUMN IF NOT EXISTS tier TEXT,
    ADD COLUMN IF NOT EXISTS caliber_id TEXT,
    ADD COLUMN IF NOT EXISTS verification_status TEXT NOT NULL DEFAULT 'to_verify',
    ADD COLUMN IF NOT EXISTS gate_results JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS origin_chain TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;

CREATE INDEX IF NOT EXISTS idx_eu_tier
    ON evidence.evidence_unit (tier);
CREATE INDEX IF NOT EXISTS idx_eu_verification
    ON evidence.evidence_unit (verification_status);
CREATE INDEX IF NOT EXISTS idx_eu_caliber
    ON evidence.evidence_unit (caliber_id);

-- ---------- Down(可上可下,plan DoD) ----------

-- DROP 操作单独写一段,便于人工 rollback:
-- DROP INDEX IF EXISTS evidence.idx_eu_tier;
-- DROP INDEX IF EXISTS evidence.idx_eu_verification;
-- DROP INDEX IF EXISTS evidence.idx_eu_caliber;
-- ALTER TABLE evidence.evidence_unit
--     DROP COLUMN IF EXISTS embedding_model,
--     DROP COLUMN IF EXISTS origin_chain,
--     DROP COLUMN IF EXISTS gate_results,
--     DROP COLUMN IF EXISTS verification_status,
--     DROP COLUMN IF EXISTS caliber_id,
--     DROP COLUMN IF EXISTS tier;
