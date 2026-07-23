"""Plan v2 end-to-end pipeline.

This is the **integration glue** that connects the Phase 1-4 modules into
a single deterministic research run:

    planner.plan_from_brief(query)
       ↓
    UnifiedSearch.search(SearchQuery)
       ↓
    eu_extractor.extract_from_search_results(results)
       ↓
    cited_report.parse_cited_report(...)
       ↓
    verifier.verify(report, eu_pool)
       ↓
    report_data.ReportDataObject
       ↓
    enforce_page_level(rdo, resolver=MockCrawlProvider)

The pipeline is **not** coupled to LangGraph: it can be invoked from a
test, a CLI, or an HTTP handler. When LangGraph is wired in (Phase 2.4
e2e), the supervisor calls `run_plan_v2_pipeline()` and writes the
structured output back to the state dict.

This module also serves as the single end-to-end smoke test for the
entire plan_v2 stack.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

from open_deep_research.planner_v2 import (
    PlannerPlan, plan_from_brief, validate_plan,
)
from open_deep_research.search_providers import (
    UnifiedSearch, SearchQuery, SearchResult, TavilyProvider, SearXNGProvider,
)
from open_deep_research.search_cache import SearchCache
from open_deep_research.sources_dao import SourcesDAO
from open_deep_research.eu_extractor import (
    extract_from_search_results,
)
from open_deep_research.cited_report import (
    CitedReport, parse_cited_report, validate_cited_report, render_eu_pool,
    CITED_REPORT_PROMPT,
)
from open_deep_research.evidence_units import (
    EvidenceUnit, eus_as_dicts, dedup_eus,
)
from open_deep_research.verifier import (
    verify, VerificationResult,
)
from open_deep_research.report_data import (
    DataRow, ReportDataObject, ReportSection, enforce_page_level,
    UrlComplianceIssue,
)
from open_deep_research.crawler import (
    MockCrawlProvider, CrawlResolver, CrawlResponse,
)
from open_deep_research.evidence import EuDAO, ClaimDAO
from open_deep_research.evidence.pipeline import build_claims_from_eus


# =============================================================================
# Result type for the full pipeline
# =============================================================================

@dataclass
class PlanV2RunResult:
    """Outcome of running the Plan v2 pipeline end-to-end.

    All stages emit a typed output so the pipeline can be partially
    inspected even when an upstream stage returns no results.
    """
    query: str
    run_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    planner: Optional[PlannerPlan] = None
    search_responses: list[dict[str, Any]] = field(default_factory=list)  # per sub-topic
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    claims: list[Any] = field(default_factory=list)  # ClaimV2 (跨源归并)
    claim_grade_dist: dict[str, int] = field(default_factory=dict)  # {A: N, B: N, C: N, D: N}
    cited_report: Optional[CitedReport] = None
    cited_report_warnings: list[str] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    report_data: Optional[ReportDataObject] = None
    url_compliance: list[UrlComplianceIssue] = field(default_factory=list)

    passed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "planner": self.planner.to_dict() if self.planner else None,
            "search_responses": self.search_responses,
            "evidence_units": eus_as_dicts(self.evidence_units),
            "claims": [c.model_dump() if hasattr(c, "model_dump") else c for c in self.claims],
            "claim_grade_dist": self.claim_grade_dist,
            "cited_report": self.cited_report.to_dict() if self.cited_report else None,
            "cited_report_warnings": self.cited_report_warnings,
            "verification": self.verification.to_dict() if self.verification else None,
            "report_data": self.report_data.to_dict() if self.report_data else None,
            "url_compliance": [u.to_dict() for u in self.url_compliance],
            "passed": self.passed,
            "error": self.error,
        }


# =============================================================================
# Pipeline
# =============================================================================

async def run_pipeline(
    query: str,
    *,
    run_id: Optional[str] = None,
    primary: Any = None,
    fallback: Any = None,
    sources_dao: Optional[Any] = None,
    cache: Optional[SearchCache] = None,
    crawler: Any = None,
    writer_response: Optional[str] = None,
    title: str = "Plan v2 Report",
    max_subtopics: int = 4,
) -> PlanV2RunResult:
    """Run the full plan_v2 stack and return a typed result.

    Args
    ----
    query           : the research brief (string).
    run_id          : optional, will be auto-generated if absent.
    primary         : a SearchProvider (e.g. TavilyProvider). Optional — if
                      None, the pipeline runs in "evidence-only" mode and
                      no network calls are made.
    fallback        : a SearchProvider for fallback (SearXNGProvider).
    sources_dao     : an optional SourcesDAO for source persistence.
    cache           : an optional SearchCache (Phase 1.2).
    crawler         : an optional CrawlProvider for the Rule 4 audit.
    writer_response : the writer LLM response (JSON string). When omitted,
                      the pipeline builds a *placeholder* CitedReport from
                      the EUs themselves so downstream verifier still runs.
    title           : report title (used in RDO header).
    max_subtopics   : cap the planner output.
    """
    rid = run_id or f"r-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    out = PlanV2RunResult(query=query, run_id=rid)

    try:
        # ----- 1. Planner -----
        plan = plan_from_brief(query, max_subtopics=max_subtopics)
        planner_issues = validate_plan(plan)
        if planner_issues:
            # Non-fatal — log; planner issues are about topology, not data.
            out.cited_report_warnings.extend(
                f"planner issue: {i.detail}" for i in planner_issues
                if i.severity in ("critical", "high")
            )
        out.planner = plan

        # ----- 2. Search + EU extraction per sub-topic -----
        us = UnifiedSearch(
            primary=primary,
            fallback=fallback,
            cache=cache,
            sources_dao=sources_dao,
        )
        all_eus: list[EvidenceUnit] = []
        for st in plan.sub_topics:
            try:
                resp = await us.search(SearchQuery(
                    queries=[st.question],
                    topic="general",
                    max_results=5,
                    run_id=rid,
                    research_topic=st.title,
                ))
            except Exception as e:
                logger.warning("search failed for sub_topic=%s: %s", st.title, e)
                out.search_responses.append({
                    "sub_topic": st.title,
                    "error": str(e),
                })
                continue
            out.search_responses.append({
                "sub_topic": st.title,
                "source": resp.source,
                "results_count": len(resp.results),
                "latency_ms": resp.latency_ms,
                "primary_used": resp.primary_used,
                "fallback_used": resp.fallback_used,
            })
            # Convert SearchResult → dict for extractor.
            raws = [
                {
                    "url": r.url,
                    "title": r.title,
                    "content": r.content,
                    "raw_content": r.raw_content,
                    "score": r.score,
                    "provider": r.provider,
                    "query": r.provider_query,
                }
                for r in resp.results
            ]
            eus = extract_from_search_results(
                raws,
                run_id=rid,
                sources_dao=sources_dao,
                research_topic=st.title,
                dimension_id=st.dimension_id,
            )
            all_eus.extend(eus)
        out.evidence_units = dedup_eus(all_eus)
        logger.info("extracted %d unique EU across %d sub-topics (with %d dimensioned)",
                    len(out.evidence_units),
                    len(plan.sub_topics),
                    sum(1 for s in plan.sub_topics if s.dimension_id))

        if not out.evidence_units:
            out.error = "no evidence units extracted"
            return out

        # ----- 3. Cited report (writer stage) -----
        if writer_response is None:
            # Build a *placeholder* report from the EUs so downstream verify
            # still runs without an LLM in the loop.
            writer_response = _placeholder_cited_response(
                title=title, eus=out.evidence_units,
            )
        cited, parse_warns = parse_cited_report(writer_response)
        out.cited_report = cited
        out.cited_report_warnings.extend(parse_warns)
        out.cited_report_warnings.extend(
            validate_cited_report(cited, out.evidence_units)
        )

        # ----- 3.5 Phase 3 (= Runbook v1 阶段 1.3): 同步落 PG -----
        # 把确定性抽取的 EU 写 PG evidence.evidence_unit 表,作为"一等公民"。
        # 失败则 fail-safe(pipeline 仍返回 in-memory 结果)。
        v2_eus: list = []  # 同时给下面的 merge phase 用
        try:
            v2_eus = [eu.to_v2(run_id=rid) for eu in out.evidence_units]
            if v2_eus:
                with EuDAO() as dao:
                    dao.upsert_many(v2_eus)
        except Exception as pg_e:
            import warnings
            warnings.warn(
                f"Phase 3 EuDAO.upsert_many failed (run_id={rid}): {pg_e}; "
                "falling back to in-memory EU only",
                RuntimeWarning,
                stacklevel=2,
            )

        # ----- 3.6 Phase 5 (= Runbook v1 阶段 3.1-3.4): EU → ClaimV2 -----
        # 数据准确性导向 (Runbook v1 §3.3):
        #   1. upgrade_source_tier — 基于白名单再次校验 source_tier
        #   2. merge_units — cosine similarity > 0.92 的 EU 归并
        #   3. build_claim_drafts — 每个 group 生成 canonical claim
        #   4. grade_claim — A/B/C/D 评级 (基于 independent + primary count)
        #   5. claim 落 PG (evidence.claim 表) — 让 /runs/{id} 的 claim_stats 立刻可观测
        try:
            if v2_eus:
                claims = build_claims_from_eus(v2_eus)
                out.claims = claims
                out.claim_grade_dist = {
                    g: sum(1 for c in claims if c.grade == g)
                    for g in "ABCD"
                }
                logger.info(
                    "build_claims_from_eus: %d EU -> %d claims (grade dist: %s)",
                    len(v2_eus), len(claims), out.claim_grade_dist,
                )
                if claims:
                    with ClaimDAO() as cdao:
                        cdao.upsert_many(claims)
                    # Phase 3.4: 回填 EU.claim_id — 用 claim 中所含 entities
                    # 反查 EU.content_hash 不可靠,留 P1 单独做 (Runbook §3.4 完整版)。
        except Exception as merge_e:
            import warnings
            warnings.warn(
                f"Phase 5 build_claims_from_eus failed (run_id={rid}): {merge_e}; "
                "falling back to in-memory claims only",
                RuntimeWarning,
                stacklevel=2,
            )

        # ----- 4. Verifier -----
        verification = verify(cited, out.evidence_units)
        out.verification = verification

        # ----- 5. ReportDataObject -----
        rdo = ReportDataObject(title=cited.title or title)
        for sec in cited.sections:
            rsec = rdo.add_section(heading=sec.heading, prose_lead="")
            for c in sec.claims:
                # Build a single DataRow per claim — keeps prose + table
                # aligned even at this synthesis step.
                prose_template = c.text
                # Lift at most one source_url from the cited EUs so Rule 4
                # has something to audit.
                source_url = ""
                for eu in out.evidence_units:
                    if eu.id in c.eu_ids:
                        source_url = eu.source_url
                        break
                rsec.add_row(DataRow(
                    key=c.text[:32],
                    label=c.text[:40],
                    category="claim",
                    values={"claim": c.text, "confidence": c.confidence},
                    provenance="; ".join(c.eu_ids[:5]),
                    source_url=source_url,
                    eu_ids=list(c.eu_ids),
                    confidence=c.confidence,
                    prose_template=prose_template,
                    table_columns=["claim", "confidence"],
                ))
        out.report_data = rdo

        # ----- 6. Rule 4 audit -----
        if crawler is not None:
            resolver = CrawlResolver(crawler)
            # Use a synchronous call via asyncio.run if needed; here we
            # provide the async-version-aware resolver and pass it via
            # call_sync. If `crawler` is async-friendly, swap caller.
            out.url_compliance = enforce_page_level(
                rdo, resolver=_sync_adapter(resolver)
            )
        else:
            out.url_compliance = enforce_page_level(rdo)

        # ----- 7. Pass/fail -----
        out.passed = (
            verification.passes
            and not any(uc.severity == "high" for uc in out.url_compliance)
        )
    except Exception as e:
        out.error = f"{type(e).__name__}: {e}"
    finally:
        out.finished_at = datetime.now(timezone.utc)
    return out


# =============================================================================
# Helpers
# =============================================================================

def _placeholder_cited_response(*, title: str, eus: list[EvidenceUnit]) -> str:
    """Build a JSON response that satisfies the parser, sourced from EUs."""
    import json

    # Group by source host; build at most 6 sections.
    sections = []
    grouped: dict[str, list[EvidenceUnit]] = {}
    for eu in eus:
        from urllib.parse import urlsplit
        host = (urlsplit(eu.source_url).hostname or "unknown").lower()
        grouped.setdefault(host, []).append(eu)

    # Build sections in hostname order for determinism.
    for host, host_eus in list(grouped.items())[:6]:
        claims = []
        for eu in host_eus[:5]:
            claims.append({
                "text": eu.claim,
                "eu_ids": [eu.id or ""],
                "numbers": [
                    {"text": n.text, "value_min": n.value_min,
                     "value_max": n.value_max, "unit": n.unit,
                     "is_estimated": n.is_estimated}
                    for n in eu.numbers
                ],
                "confidence": eu.confidence,
                "rationale": f"grounded in {host}",
            })
        sections.append({"heading": host, "claims": claims})

    if not sections:
        sections.append({
            "heading": "Findings",
            "claims": [{
                "text": eus[0].claim,
                "eu_ids": [eus[0].id or ""],
                "numbers": [],
                "confidence": eus[0].confidence,
                "rationale": "single EU grounding",
            }],
        })

    return json.dumps({"title": title, "sections": sections}, ensure_ascii=False)


def _sync_adapter(resolver: CrawlResolver):
    """Wrap CrawlResolver.call_sync so enforce_page_level sees a sync fn."""
    def _sync(url: str) -> str:
        try:
            out = resolver.call_sync(url)
        except Exception:
            return ""
        return out or ""
    return _sync


# =============================================================================
# Convenience: build a minimal stack for local / e2e testing
# =============================================================================

def default_components(
    *,
    tavily_api_key: Optional[str] = None,
    searxng_url: Optional[str] = None,
    sources_dao: Optional[Any] = None,
    cache: Optional[SearchCache] = None,
    use_real_search: bool = True,
    use_real_crawler: bool = False,
) -> dict[str, Any]:
    """Return a dict with the default components for the pipeline.

    `use_real_search=False` returns None for primary/fallback so callers
    can exercise the planning/extract/verify path without network.
    `use_real_crawler=False` returns a MockCrawlProvider.
    """
    primary = fallback = None
    if use_real_search:
        primary = TavilyProvider(api_key=tavily_api_key or os.environ.get("TAVILY_API_KEY"))
        if searxng_url or os.environ.get("SEARXNG_URL"):
            fallback = SearXNGProvider(base_url=searxng_url)
    crawler: Any = MockCrawlProvider()
    if use_real_crawler:
        # Caller is responsible for installing crawl4ai; we still expose
        # the protocol but don't construct it eagerly.
        crawler = None  # type: ignore
    return {
        "primary": primary,
        "fallback": fallback,
        "sources_dao": sources_dao,
        "cache": cache,
        "crawler": crawler,
    }
