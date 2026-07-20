-- ============================================================
-- Phase 1.1: sources table
-- ============================================================
-- 用途:
--   - 持久化每一次 search 调用获得的所有 URL
--   - 提供 page-level URL 校验(B 类锚点要求"必须页面级,非域名级")
--   - 支持 TTL 缓存(下一条迁移)
--   - 作为 Phase 1.4 Crawl4AI 抓取的入参
--
-- 表关系:
--   sources        — 每个 fetch 命中一个 URL → page
--   source_cache   — Phase 1.2 TTL cache 上挂的 share layer
--   evidence_units — Phase 2 才落,本迁移不涉及
--
-- 设计:
--   1. URL 为唯一性约束的关键 — sha256(normalized_url) 防止 http/https 重复
--   2. page_level 字段标记 URL 是否页面级(路径有意义) — B 类锚点校验点
--   3. fetch_status 走 small enum 替代 bool flag(便于后续扩展 'crawled'/'error'/'pending')
--   4. JSONB 保留原始 provider payload(便于溯源和回放)
-- ============================================================

CREATE SCHEMA IF NOT EXISTS evidence;

-- ---------- ENUM ----------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'source_fetch_status') THEN
        CREATE TYPE evidence.source_fetch_status AS ENUM (
            'fetched',   -- 命中并返回 content
            'cached',    -- 命中缓存
            'pending',   -- 已登记 URL,等后续抓取(Crawl4AI 队列)
            'failed',    -- 抓取/抓取失败
            'skipped'    -- 主动跳过(域名级、robots.txt disallow 等)
        );
    END IF;
END$$;

-- ---------- TABLE ----------
CREATE TABLE IF NOT EXISTS evidence.sources (
    id                  BIGSERIAL PRIMARY KEY,

    -- URL 及其 normalized 形式
    url                 TEXT NOT NULL,
    url_normalized      TEXT NOT NULL,                  -- 小写 host, 去 trailing slash, 去 utm_*
    url_hash            CHAR(64) NOT NULL UNIQUE,       -- sha256(url_normalized)
    domain              TEXT NOT NULL,                  -- 仅 host(eTLD+1 已简化)

    -- 元数据(由 search provider 提供)
    title               TEXT,
    provider            TEXT NOT NULL,                 -- 'tavily' / 'searxng' / 'crawl4ai'
    provider_query      TEXT,                          -- 触发该命中的搜索词
    provider_score      DOUBLE PRECISION,               -- Tavily score 等

    -- 页面级校验(B 类锚点必查)
    page_level          BOOLEAN NOT NULL DEFAULT FALSE,
    page_level_reason   TEXT,                           -- 为什么不是 page_level(便于 Phase 3b 规则四)

    -- 抓取状态
    fetch_status        evidence.source_fetch_status NOT NULL DEFAULT 'fetched',
    http_status         INTEGER,                       -- 远端返回码
    content_type        TEXT,

    -- 原始 payload 留痕(回放/调试用)
    provider_payload    JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Crawl4AI / 网页正文 (Phase 1.4 才填充,先留 NULL)
    raw_content         TEXT,
    raw_content_hash    CHAR(64),                      -- sha256(raw_content),便于 dedup
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMPTZ,                  -- Phase 1.2 TTL 用

    -- 用户级 / 提示链(可空)
    research_topic      TEXT,                          -- 哪条 supervisor topic 拉到这个 URL
    run_id              TEXT,                          -- LangGraph run_id,溯源

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT url_hash_format CHECK (url_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT raw_content_hash_format CHECK (
        raw_content_hash IS NULL OR raw_content_hash ~ '^[0-9a-f]{64}$'
    )
);

-- ---------- INDEX ----------
CREATE INDEX IF NOT EXISTS idx_sources_url_normalized ON evidence.sources (url_normalized);
CREATE INDEX IF NOT EXISTS idx_sources_domain         ON evidence.sources (domain);
CREATE INDEX IF NOT EXISTS idx_sources_run_id         ON evidence.sources (run_id);
CREATE INDEX IF NOT EXISTS idx_sources_topic          ON evidence.sources (research_topic);
CREATE INDEX IF NOT EXISTS idx_sources_fetched_at     ON evidence.sources (fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_sources_expires_at     ON evidence.sources (expires_at)
    WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sources_page_level     ON evidence.sources (page_level)
    WHERE page_level = FALSE;  -- B 类锚点扫描优化

-- 派生的 not-page-level 视图(B 类锚点过滤器)
CREATE OR REPLACE VIEW evidence.v_domain_only_sources AS
SELECT id, url, domain, title, run_id, page_level_reason
FROM evidence.sources
WHERE page_level = FALSE;

-- ---------- TRIGGER ----------
-- updated_at 自动维护
CREATE OR REPLACE FUNCTION evidence.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sources_updated_at ON evidence.sources;
CREATE TRIGGER trg_sources_updated_at
    BEFORE UPDATE ON evidence.sources
    FOR EACH ROW
    EXECUTE FUNCTION evidence.set_updated_at();

-- ---------- HELPER: 判页面级 ----------
-- "页面级"定义: URL path 段数 ≥ 1,且不含 raw query string only
-- 域名级例子: https://klue.com/, https://klue.com
-- 页面级例子: https://klue.com/product/battlecards, https://crayon.co/crayon-vs-klue
CREATE OR REPLACE FUNCTION evidence.is_page_level(url_in TEXT)
RETURNS BOOLEAN AS $$
DECLARE
    u    TEXT;
    path TEXT;
BEGIN
    -- 去掉 scheme
    u := regexp_replace(url_in, '^https?://', '', 'i');
    -- 去掉 trailing slash
    u := regexp_replace(u, '/$', '');
    -- 取 path 部分(去掉 query/fragment)
    u := split_part(u, '?', 1);
    u := split_part(u, '#', 1);
    -- 重新拆 host vs path
    path := substring(u from position('/' in u) for length(u));

    -- 无 path 或 path 只有 '/' → 域名级
    IF path IS NULL OR path = '' OR path = '/' THEN
        RETURN FALSE;
    END IF;
    -- 仅 query(以 ? 或 # 起) → 域名级
    IF path ~ '^[\?#]' THEN
        RETURN FALSE;
    END IF;
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMIT;

-- ============================================================
-- Bootstrap ack
-- ============================================================
SELECT 'Phase 1.1 sources table ready' AS status;
