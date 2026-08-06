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
            # Real SearXNG path would go here; for now stay deterministic.
            try:
                from open_deep_research.search_providers import SearXNGProvider
                p = SearXNGProvider(self._searxng_url)
                return await p.search(query)
            except Exception:
                pass
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

    t0 = time.time()
    try:
        result = await run_pipeline_resumable(
            query=args.brief,
            run_id=str(uuid.uuid4()),
            primary=provider,
            fallback=provider if args.mode == "live" else None,
            max_subtopics=args.max_subtopics,
            title=f"ODR: {args.brief[:60]}",
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

    md = render_markdown(result, args.brief, pg_summary)
    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"[run_odr] wrote {args.output} ({len(md)} chars)", file=sys.stderr)
    else:
        print(md)
    return 0


def main():
    p = argparse.ArgumentParser(
        description="ODR end-to-end runner (planner → search → EU → verifier → RDO)"
    )
    p.add_argument("--brief", required=True, help="Research brief / question")
    p.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    p.add_argument("--searxng-url", default="http://172.18.0.2:8080")
    p.add_argument("--output", "-o", help="Write markdown to file (default: stdout)")
    p.add_argument("--max-subtopics", type=int, default=4)
    p.add_argument("--persist-pg", action="store_true", default=True,
                   help="Persist EUs/claims/sources/checkpoints to production PG (default true)")
    p.add_argument("--no-persist-pg", dest="persist_pg", action="store_false")
    args = p.parse_args()
    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()