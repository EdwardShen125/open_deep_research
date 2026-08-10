"""ODR end-to-end CLI entry point.

Note: ODR = Open Deep Research (the project's exact name). We deliberately
do NOT use "EDR" here because that abbreviation is taken by
Endpoint Detection and Response in cybersecurity.

Runs `run_pipeline_resumable(query: str, ...)` from `plan_v2_pipeline.py`
against either:
  --mode fixture  : in-process deterministic fixture provider (no network)
  --mode live     : SearXNGProvider (172.18.0.2:8080, falls back to fixture)

Usage:
  python run_odr.py --brief "What is the size of the US live commerce market?"
  python run_odr.py --brief "..." --mode live --output report.md

This is the missing CLI wrapper that connects:
  planner_v2 → query_constructor → UnifiedSearch → EU extractor
    → claim build → verifier → RDO render → markdown output
Per SPEC §2 (plan_v2_writer_v2.py placeholder) — implemented as
`run_odr.py` to avoid clashing with the existing `run_pipeline` import.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.staged_runner import run_pipeline_resumable
from open_deep_research.search_providers import (
    SearchProvider, SearchQuery, SearchResult,
)


# =============================================================================
# LLM key detection
# =============================================================================

def detect_llm_keys() -> dict:
    """Probe which LLM provider keys are available. Used for honest reporting.

    Returns dict like {'MiniMax': False, 'ANTHROPIC_API_KEY': False, 'OPENAI_API_KEY': False}
    Returns False for all if the LLM gate fail-fasts on any provider.
    """
    keys = {
        "MINIMAX_API_KEY": bool(os.environ.get("MINIMAX_API_KEY", "").strip()),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
    }
    return keys


import os  # noqa: E402  (after sys.path manipulation above)


# =============================================================================
# Fixture provider — deterministic, no network
# =============================================================================

class FixtureProvider(SearchProvider):
    """Return canned results so end-to-end runs without network."""

    name = "fixture"

    def __init__(self):
        # Three EUs that match the Golden Eval brief: US live commerce market.
        self._docs = [
            {
                "url": "https://www.emarketer.com/content/us-live-commerce",
                "title": "US Live Commerce Market 2024",
                "content": "US live commerce GMV reached $146.4B in 2024, projected to "
                           "hit $678B by 2027 (eMarketer live retail definition).",
                "domain": "emarketer.com",
                "score": 0.95,
            },
            {
                "url": "https://www.coresight.com/research/livestream-shopping",
                "title": "Livestream Shopping Global Outlook",
                "content": "Coresight Research estimates US livestream shopping at "
                           "$550B by 2025 using broader livestream retail scope.",
                "domain": "coresight.com",
                "score": 0.88,
            },
            {
                "url": "https://www.mckinsey.com/industry/retail/live-commerce",
                "title": "Live Commerce: The Next Retail Channel",
                "content": "McKinsey: US live commerce could reach $25-35B by 2026 "
                           "in a narrow livestream-only definition.",
                "domain": "mckinsey.com",
                "score": 0.91,
            },
        ]

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        # Always return the same 3 docs regardless of query (fixture).
        return [
            SearchResult(
                url=d["url"],
                title=d["title"],
                content=d["content"],
                score=d["score"],
                provider=self.name,
                provider_query="us-live-commerce",
                raw_payload={"domain": d["domain"]},
            )
            for d in self._docs
        ]


# =============================================================================
# Live provider — SearXNG with fallback to fixture
# =============================================================================

class SearXNGOrFixtureProvider(SearchProvider):
    """Try SearXNG first; fall back to fixture if unreachable."""

    name = "searxng-or-fixture"

    def __init__(self, searxng_url: str = "http://172.18.0.2:8080"):
        self._searxng_url = searxng_url
        self._fixture = FixtureProvider()
        self._searxng_reachable: bool | None = None

    async def _check_searxng(self) -> bool:
        if self._searxng_reachable is not None:
            return self._searxng_reachable
        import urllib.request, urllib.error
        try:
            urllib.request.urlopen(
                f"{self._searxng_url}/healthz", timeout=2
            ).read()
            self._searxng_reachable = True
        except (urllib.error.URLError, OSError):
            self._searxng_reachable = False
        return self._searxng_reachable

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        if await self._check_searxng():
            # Real SearXNG path. U0 ★ (2026-08-10 验证 v38): SearXNGProvider
            # default timeout=5.0 is too short — SearXNG meta-search needs to
            # fan out to 15 engines, takes ~10s+ under load. Without explicit
            # timeout, provider raises ReadTimeout, fixture fallback silently
            # returns emarketer/mckinsey US live commerce data (pollutes
            # CN EDR research pipeline).
            try:
                from open_deep_research.search_providers import SearXNGProvider
                p = SearXNGProvider(self._searxng_url, timeout=30.0)
                return await p.search(query)
            except Exception:
                import traceback
                traceback.print_exc()
        # Fallback to fixture.
        return await self._fixture.search(query)


# =============================================================================
# Markdown rendering
# =============================================================================

def render_markdown(result, brief: str, pg_summary: dict | None = None) -> str:
    """Render PlanV2RunResult → human-readable markdown."""
    lines = [
        f"# ODR Report: {brief}",
        "",
        f"_Run ID: `{result.run_id}` · Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Status",
        "",
        f"- **Error**: `{result.error or 'none'}`",
        f"- **Planner sub-topics**: {len(result.planner.sub_topics) if result.planner else 0}",
        f"- **Evidence units collected**: {len(result.evidence_units)}",
        f"- **Claims built**: {len(result.claims)}",
        f"- **Cited report warnings**: {len(result.cited_report_warnings)}",
        f"- **Passed**: {result.passed}",
        "",
    ]

    if pg_summary:
        lines.append("## PG Persistence")
        lines.append("")
        if pg_summary.get("error"):
            lines.append(f"- **ERROR**: `{pg_summary['error']}`")
        lines.append(f"- **Sources inserted**: {pg_summary['sources']}")
        lines.append(f"- **Evidence units inserted**: {pg_summary['evidence_units']}")
        lines.append(f"- **Claims inserted**: {pg_summary['claims']}")
        lines.append("")

    if result.planner:
        lines.append("## Planner Output")
        lines.append("")
        for i, st in enumerate(result.planner.sub_topics, 1):
            lines.append(f"### {i}. {st.title}")
            lines.append(f"- dimension: `{st.dimension_id}`")
            lines.append(f"- question: {st.question}")
            lines.append(f"- expected_keywords: {st.expected_keywords}")
            lines.append("")

    if result.evidence_units:
        lines.append(f"## Evidence Units ({len(result.evidence_units)})")
        lines.append("")
        for i, eu in enumerate(result.evidence_units[:20], 1):
            nums = ", ".join(
                f"{n.text} ({n.unit or ''})" for n in eu.numbers[:3]
            ) or "<no number>"
            lines.append(
                f"{i}. **{eu.claim[:80]}** — {nums}"
            )
            lines.append(f"   - source: {eu.source_url[:60]} · tier: {eu.source_tier}")
        if len(result.evidence_units) > 20:
            lines.append(f"\n_…and {len(result.evidence_units) - 20} more._")
        lines.append("")

    if result.cited_report:
        lines.append("## Cited Report")
        lines.append("")
        lines.append("```json")
        try:
            lines.append(json.dumps(result.cited_report.to_dict()
                                    if hasattr(result.cited_report, "to_dict")
                                    else result.cited_report.__dict__,
                                    indent=2, default=str)[:5000])
        except Exception as e:
            lines.append(f"<serialization error: {e}>")
        lines.append("```")
        lines.append("")

    if result.verification:
        lines.append("## Verifier")
        lines.append("")
        lines.append(f"- **Passed**: {result.verification.passes}")
        lines.append(f"- **Issue count**: {len(result.verification.issues)}")
        lines.append(f"- **By rule**: {result.verification.by_rule}")
        lines.append(f"- **By severity**: {result.verification.by_severity}")
        lines.append("")

    if result.cited_report_warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in result.cited_report_warnings:
            lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

async def _run(args) -> int:
    if args.mode == "fixture":
        provider = FixtureProvider()
    elif args.mode == "live":
        provider = SearXNGOrFixtureProvider(args.searxng_url)
    else:
        print(f"unknown mode: {args.mode}", file=sys.stderr)
        return 2

    print(f"[run_odr] brief: {args.brief!r}", file=sys.stderr)
    print(f"[run_odr] mode: {args.mode}", file=sys.stderr)
    print(f"[run_odr] max_subtopics: {args.max_subtopics}", file=sys.stderr)
    print(f"[run_odr] pg_persist: {args.persist_pg}", file=sys.stderr)
    print(f"[run_odr] 4-layer: archetypes={args.archetype} ontology={args.ontology} "
          f"registry={args.registry} instance.market={args.market} "
          f"instance.category={args.category} instance.year={args.year}",
          file=sys.stderr)

    t0 = time.time()
    # R6/R7:为 W3 llm_extractor + W9/W10 综合层注入真 LLM 客户端;
    # 失败/缺 key → 退化为 None,trigger plan_v2_pipeline 的显式 degraded。
    try:
        from open_deep_research.llm import get_llm
        _llm = get_llm()
        if _llm is None:
            print("[run_odr] WARNING: get_llm() returned None (R6/R7 degraded path)", file=sys.stderr)
    except Exception as _e:
        print(f"[run_odr] WARNING: get_llm() raised: {_e} (R6/R7 degraded path)", file=sys.stderr)
        _llm = None

    # R16 ★:optional crawler wiring. Returns None when --crawler-url is
    # unset (pipeline then uses MockCrawlProvider, snippet-only behavior).
    def _build_crawler(args):
        url = args.crawler_url
        if not url:
            if args.use_real_crawler:
                print("[run_odr] WARNING: --use-real-crawler set but --crawler-url is empty; "
                      "pipeline will fall back to MockCrawlProvider.", file=sys.stderr)
            return None
        try:
            from open_deep_research.crawler import Crawl4AIHttpProvider
            return Crawl4AIHttpProvider(
                base_url=url,
                timeout=30.0,
                api_token=args.crawler_token,  # R19b:bearer auth
            )
        except Exception as _ce:
            print(f"[run_odr] WARNING: Crawl4AIHttpProvider init failed: {_ce}", file=sys.stderr)
            return None

    try:
        result = await run_pipeline_resumable(
            query=args.brief,
            run_id=str(uuid.uuid4()),
            primary=provider,
            fallback=provider if args.mode == "live" else None,
            # R16 ★:wire up Crawl4AIHttpProvider when --crawler-url is set.
            # Without it, pipeline falls back to MockCrawlProvider → snippet-only.
            crawler=_build_crawler(args),
            max_subtopics=args.max_subtopics,
            title=f"ODR: {args.brief[:60]}",
            # 4-layer (post-architecture-v2)
            archetypes=args.archetype.split(",") if args.archetype else None,
            ontology=args.ontology,
            registry_vertical=args.registry,
            instance_brand=args.brand,
            instance_market=args.market,
            instance_category=args.category,
            instance_year=args.year,
            # R6/R7
            llm=_llm,
        )
    except Exception as e:
        print(f"[run_odr] PIPELINE FAILED: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - t0
    print(f"[run_odr] pipeline took {elapsed:.1f}s", file=sys.stderr)

    # -------------------------------------------------------------------------
    # PG persistence (staged_runner already wrote checkpoints + EU + claim)
    # -------------------------------------------------------------------------
    pg_summary = {"sources": 0, "evidence_units": 0, "claims": 0, "error": None}
    if args.persist_pg:
        try:
            import psycopg
            with psycopg.connect(
                host=os.environ.get("POSTGRES_HOST", "172.17.0.2"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                user=os.environ.get("POSTGRES_USER", "postgres"),
                password=os.environ.get("POSTGRES_PASSWORD", "odr_v2_pg_pass_change_me"),
                dbname=os.environ.get("POSTGRES_DB", "odr_v2"),
            ) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM evidence.evidence_unit WHERE run_id::text = %s",
                    (result.run_id,),
                )
                pg_summary["evidence_units"] = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM evidence.claim WHERE run_id::text = %s",
                    (result.run_id,),
                )
                pg_summary["claims"] = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(*) FROM evidence.sources WHERE run_id = %s",
                    (result.run_id,),
                )
                pg_summary["sources"] = cur.fetchone()[0]
                conn.commit()
        except Exception as e:
            pg_summary["error"] = repr(e)[:300]
            print(f"[run_odr] PG read-back failed: {e}", file=sys.stderr)

    print(
        f"[run_odr] PG: {pg_summary['sources']} sources, "
        f"{pg_summary['evidence_units']} EUs, {pg_summary['claims']} claims"
        + (f" | ERROR: {pg_summary['error']}" if pg_summary['error'] else ""),
        file=sys.stderr,
    )

    # 任务书 §5:W9/W10 reduce-tree 产出 assembled_markdown;若存在则优先用
    # (render_markdown 是 W7 旧产物,我们不替换它,只是新增 W9/W10 长报告优先路径)
    md_assembled = getattr(result, "assembled_markdown", None)
    if md_assembled:
        md = md_assembled
        # 附加一段 Status 元信息(原 render_markdown 干的活)
        status_block = (
            f"<!-- status: run_id={result.run_id} | "
            f"EU={len(result.evidence_units)} claims={len(result.claims)} "
            f"summaries={len(getattr(result, 'section_summaries', []))} "
            f"crosscuts={len(getattr(result, 'crosscuts', []))} "
            f"exec={'yes' if getattr(result, 'exec_summary', None) else 'no'} "
            f"passed={result.passed} degraded={result.degraded} "
            f"reconciliations={len(getattr(result, 'reconciliations', []))} -->\n\n"
        )
        md = status_block + md
    else:
        md = render_markdown(result, args.brief, pg_summary)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[run_odr] wrote {args.output} ({len(md)} chars)", file=sys.stderr)
    else:
        print(md)
    return 0


def main():
    # R1-class regression fix: run_odr 启动时主动加载 .env,不再依赖外部 shell
    # 必须先 source .env — W9/W10 综合层 LLM 调用才拿得到 MINIMAX_API_KEY。
    # 若 .env 不存在则 fallback(不影响 fixture mode)。
    try:
        from dotenv import load_dotenv
        from pathlib import Path as _P
        _env_path = _P(__file__).resolve().parent / ".env"
        if _env_path.exists():
            load_dotenv(_env_path, override=False)
    except Exception as _env_e:
        print(f"[run_odr] .env load skipped: {_env_e}", file=sys.stderr)

    p = argparse.ArgumentParser(
        description="ODR end-to-end runner (planner → search → EU → verifier → RDO)"
    )
    p.add_argument("--brief", required=True, help="Research brief / question")
    p.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    p.add_argument("--searxng-url", default="http://172.18.0.2:8080")
    # R16 ★:optional Crawl4AI sidecar (HTTP). When set, the pipeline fetches
    # full-text markdown for each search result before extraction — which is
    # the difference between 'extract from snippet' and 'extract from real
    # body'. Mirrors the api/server.py behavior.
    p.add_argument("--crawler-url", default=os.environ.get("CRAWLER_URL"),
                   help="Crawl4AI sidecar URL (e.g. http://127.0.0.1:11235). "
                        "When unset, pipeline uses MockCrawlProvider (snippet-only).")
    # R19b ★:crawl4ai 0.9.x requires bearer auth on every endpoint. Sidecar
    # refuses all requests with 401 unless the matching token is provided.
    # Must match `CRAWL4AI_API_TOKEN` env on the crawler-sidecar container.
    p.add_argument("--crawler-token", default=os.environ.get("CRAWL4AI_API_TOKEN"),
                   help="Crawl4AI API token (env CRAWL4AI_API_TOKEN). Required "
                        "for sidecar auth — without it, sidecar returns 401.")
    p.add_argument("--use-real-crawler", dest="use_real_crawler",
                   action="store_true", default=False,
                   help="Force-enable Crawl4AIHttpProvider even when --crawler-url unset "
                        "(will warn + still degrade to MockCrawlProvider)")
    p.add_argument("--output", "-o", help="Write markdown to file (default: stdout)")
    p.add_argument("--max-subtopics", type=int, default=4)
    p.add_argument("--persist-pg", action="store_true", default=True,
                   help="Persist EUs/claims/sources/checkpoints to production PG (default true)")
    p.add_argument("--no-persist-pg", dest="persist_pg", action="store_false")
    # ---- 4-layer architecture (post-architecture-v2) ----
    p.add_argument("--archetype", default=None,
                   help="Comma-separated archetype list (e.g. 'market_size,competitive,regulatory')")
    p.add_argument("--ontology", default=None,
                   help="Ontology id (e.g. 'cn_cybersec', 'us_livecommerce')")
    p.add_argument("--registry", default=None,
                   help="Registry vertical id (defaults to --ontology)")
    p.add_argument("--brand", default=None, help="Instance brand param")
    p.add_argument("--market", default=None, help="Instance market param (e.g. 'CN', 'US')")
    p.add_argument("--category", default=None, help="Instance category param")
    p.add_argument("--year", type=int, default=None, help="Instance year param")
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()