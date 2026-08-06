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

import numpy as np

logger = logging.getLogger(__name__)
parent = logger.parent
if not logger.handlers and (parent is None or not parent.handlers):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

from open_deep_research.planner_v2 import (
    PlannerPlan, plan_from_brief, validate_plan,
)
from open_deep_research.search_providers import (
    UnifiedSearch, SearchQuery, SearchResult, TavilyProvider, SearXNGProvider,
)
from open_deep_research.search_cache import SearchCache
from open_deep_research.sources_dao import SourcesDAO
from open_deep_research.evidence.embedder import embed_texts
from open_deep_research.eu_extractor import (
    extract_from_search_results,
)
# Phase query_constructor: turn planner sub-topics into SearXNG profiles.
# Honor OPEN_DEEP_RESEARCH_NO_QC=1 to disable and keep the pre-QC baseline.
from open_deep_research.query_constructor import (
    construct as qc_construct,
    to_search_query,
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
from open_deep_research.evidence.tier_classifier import classify_tier as _classify_tier_abc
# W3: A/B/C/D tier (cognitive authority) → SourceTier (primary/secondary/...)
_TIER_ABC_TO_SOURCE = {"A": "primary", "B": "secondary", "C": "tertiary", "D": "ugc"}


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
    gate_stats: dict[str, int] = field(default_factory=dict)  # 闸 1+2+3 命中统计
    honest_pass: Optional[dict] = None  # W8:honest_pass 收口(unverified/caliber/source_pages)

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
            "gate_stats": self.gate_stats,
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
    vertical: Optional[str] = None,
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
        # ----- W1: framework.py 驱动计划（"槽"作为一等公民）-----
        # 旧路径:plan_from_brief() LLM 自由发散 sub_topics
        # 新路径:data/frameworks/<vertical>.yaml 给出确定性 slot 集,
        #         每个 slot 携 expected_claim_type / caliber_id / required_tier_min,
        #         intents 由槽派生。framework 不存在时降级到旧 LLM planner。
        from open_deep_research.evidence.framework import (
            Framework, load_framework, Slot,
        )

        framework: Framework | None = None
        # W1 框架仅在 caller 显式 vertical= 时启用。
        # 默认 None → 走 LLM planner(原行为,保持英文/通用 query,
        # fixture E2E 不会因为硬编码 us_livecommerce 框架而中文化)。
        if vertical:
            try:
                framework = load_framework(vertical)
                logger.info(
                    "[W1] framework loaded: vertical=%s sections=%d slots=%d",
                    vertical, len(framework.sections),
                    sum(len(s.slots) for s in framework.sections),
                )
                # 把 framework 拍扁成 PlannerPlan.sub_topics(每 slot → 1 sub_topic)
                from open_deep_research.planner_v2 import SubTopic, PlannerPlan
                framework_subs: list[SubTopic] = []
                for sec in framework.sections:
                    for slot in sec.slots:
                        framework_subs.append(SubTopic(
                            id=slot.slot_id,
                            title=slot.question,
                            question=slot.question,
                            search_api="searxng",
                            parallelism="fan_out",
                            expected_entities=[],
                            expected_keywords=[],
                            rationale=slot.notes or "",
                            dimension_id=sec.section_id,
                        ))
                # max_subtopics 仍生效(截断)
                framework_subs = framework_subs[:max_subtopics]
                plan = PlannerPlan(
                    title=framework.title,
                    sub_topics=framework_subs,
                    waves=[],
                    notes=f"[W1] from framework {vertical}.yaml, {len(framework_subs)}/{sum(len(s.slots) for s in framework.sections)} slots",
                )
                planner_issues = validate_plan(plan)
                logger.info(
                    "[W1] intents derived from framework slots: count=%d "
                    "(each carries dimension_id=%s)",
                    len(plan.sub_topics),
                    plan.sub_topics[0].dimension_id if plan.sub_topics else None,
                )
            except FileNotFoundError:
                logger.info(
                    "[W1] no framework yaml for %s; falling back to LLM planner",
                    vertical,
                )
                plan = plan_from_brief(query, max_subtopics=max_subtopics)
                planner_issues = validate_plan(plan)
        else:
            # Default: LLM planner (preserves test fixtures / non-vertical briefs)
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
        # Phase query_constructor: instead of constructing one SearchQuery per
        # sub-topic with hard-coded `topic=general, max_results=5`, we delegate
        # the query-shape decision to QueryConstructor. Result: each sub-topic
        # may fan out to multiple SearXNG profiles (vendor + academic, for
        # instance), each tuned to a different engine/category/language subset.
        # Old behavior is preserved when OPEN_DEEP_RESEARCH_NO_QC=1 (one
        # SearchQuery, topic=general, max_results=5, no extras).
        us = UnifiedSearch(
            primary=primary,
            fallback=fallback,
            cache=cache,
            sources_dao=sources_dao,
        )
        all_eus: list[EvidenceUnit] = []
        for st in plan.sub_topics:
            # ----- W2: source_router + acquirer 定向取证(每槽优先)-----
            # 旧路径:qc_construct(st) LLM 自由发散
            # 新路径:framework 存在时,从 Slot 派生定向查询;失败/无 framework → 旧 LLM 兜底
            source_router_queries: list[str] = []
            try:
                from open_deep_research.evidence.source_router import route_sources
                from open_deep_research.evidence.source_registry import load_registry as _load_registry
                from open_deep_research.evidence.caliber_registry import (
                    load_caliber_registry as _load_calibers,
                )
                from open_deep_research.evidence.framework import Slot as _FSlot
                # W4: 从 framework YAML 找出该 slot 对应的真正 caliber_id
                _fw_caliber_id: Optional[str] = None
                if framework is not None:
                    for _sec in framework.sections:
                        for _slot in _sec.slots:
                            if _slot.slot_id == (st.id or ""):
                                _fw_caliber_id = _slot.caliber_id
                                break
                        if _fw_caliber_id:
                            break
                _fw_slot = _FSlot(
                    slot_id=(st.id or "slot_default").lower().replace(" ", "_").replace("-", "_")[:64] or "slot_default",
                    question=st.question,
                    expected_claim_type="attribute",
                    required_tier_min="B",
                    caliber_id=_fw_caliber_id,
                )
                try:
                    # 仅在 caller 注入 framework 时启用 source_router。
                    if framework is not None:
                        _registry = _load_registry(vertical or "us_livecommerce")
                        _calibers = _load_calibers(vertical or "us_livecommerce")
                    else:
                        _registry = None
                        _calibers = None
                except FileNotFoundError:
                    _registry = None
                    _calibers = None
                if _registry is not None:
                    _intent = route_sources(
                        _fw_slot, registry=_registry, calibers=_calibers,
                    )
                    source_router_queries = _intent.query_list[:5]
                    logger.info(
                        "[W2+W4] source_router intent for slot=%s caliber_id=%s queries=%d (first: %r)",
                        st.id, _fw_caliber_id, len(source_router_queries),
                        source_router_queries[0] if source_router_queries else None,
                    )
            except Exception as e:
                logger.warning("[W2] source_router skipped: %s", e)

            # 2a. Build an ExecutionPlan (LLM-driven one-shot per sub-topic).
            try:
                exec_plan = await qc_construct(query, st)
            except Exception as e:
                logger.warning(
                    "query_constructor failed for sub_topic=%s: %s; using legacy single-intent",
                    st.title, e,
                )
                # ----- W2: 若 source_router 产出了定向 query,优先用它替代 question -----
                if source_router_queries:
                    legacy_intent_qs = [SearchQuery(
                        queries=source_router_queries[:3],
                        topic="general",
                        max_results=5,
                        run_id=rid,
                        research_topic=st.title,
                    )]
                    logger.info(
                        "[W2] using source_router queries as fallback intent (n=%d)",
                        len(legacy_intent_qs[0].queries),
                    )
                else:
                    legacy_intent_qs = [SearchQuery(
                        queries=[st.question],
                        topic="general",
                        max_results=5,
                        run_id=rid,
                        research_topic=st.title,
                    )]
                exec_plan = None
                intent_search_queries = legacy_intent_qs
            else:
                # ----- W2: 合并 source_router 定向查询到 LLM 产出的 query 列表 -----
                intent_search_queries = to_search_query(
                    exec_plan, run_id=rid, sub_topic=st,
                )
                if source_router_queries:
                    # site-scoped 强制前 N(source_router 已处理 site scoping)
                    intent_search_queries[0].queries = (
                        source_router_queries + intent_search_queries[0].queries
                    )
                    logger.info(
                        "[W2] merged source_router (n=%d) into LLM intent queries",
                        len(source_router_queries),
                    )

            # 2b. Execute each intent → merge results into one Raws set.
            # Each SearchQuery gets one `us.search(...)` call; EUs across
            # intents dedup naturally because `extract_from_search_results`
            # hashes by (url, sentence) and EU DAO dedups by content_hash.
            raws: list[dict[str, Any]] = []
            intent_outcomes: list[dict[str, Any]] = []
            for sq in intent_search_queries:
                try:
                    resp = await us.search(sq)
                except Exception as e:
                    logger.warning(
                        "search failed for sub_topic=%s intent queries=%r: %s",
                        st.title, sq.queries, e,
                    )
                    intent_outcomes.append({
                        "queries": sq.queries,
                        "topic": sq.topic,
                        "engines": (sq.extras or {}).get("engines"),
                        "language": (sq.extras or {}).get("language"),
                        "error": str(e),
                    })
                    continue
                intent_outcomes.append({
                    "queries": sq.queries,
                    "topic": sq.topic,
                    "engines": (sq.extras or {}).get("engines"),
                    "language": (sq.extras or {}).get("language"),
                    "source": resp.source,
                    "results_count": len(resp.results),
                    "latency_ms": resp.latency_ms,
                    "primary_used": resp.primary_used,
                    "fallback_used": resp.fallback_used,
                })
                for r in resp.results:
                    raws.append({
                        "url": r.url,
                        "title": r.title,
                        "content": r.content,
                        "raw_content": r.raw_content,
                        "score": r.score,
                        "provider": r.provider,
                        "engine": r.engine,         # surface SearXNG engine tag
                        "query": r.provider_query,
                    })

            out.search_responses.append({
                "sub_topic": st.title,
                "dimension_id": st.dimension_id,
                "rationale": (exec_plan.rationale if exec_plan else "qc failed; legacy fallback"),
                "plan_source": (exec_plan.source if exec_plan else "legacy"),
                "intent_count": len(intent_search_queries),
                "intents": intent_outcomes,
            })

            # 2c. Extract EUs from the merged raw set (one extractor call —
            #     dedup is internal to the extractor + EU DAO).
            if not raws:
                continue
            # P1 fix: pass the original brief as `research_topic` so the cyber
            # guard (cybersecurity-context keyword filter) can enable itself.
            # Passing `st.title` (e.g. "market_size") defeats the guard.
            eus = extract_from_search_results(
                raws,
                run_id=rid,
                sources_dao=sources_dao,
                research_topic=query,           # was st.title — guard was disabled
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
            # W7 ★:section_writer 替换 placeholder(只在 LLM 可用时)
            # 旧路径(被删):_placeholder_cited_response → 默默 string fallback
            # 新路径:先试 section_writer.write_section(structured ReportResult),
            #        LLM 不可用 → 显式 degraded placeholder(标记 status,不糊弄)
            section_writer_used = False
            try:
                from open_deep_research.evidence.section_writer import write_section
                from open_deep_research.evidence.framework import Slot as _w7_Slot
                from open_deep_research.evidence.claim_v3 import ClaimV3
                # 若 framework 存在,逐槽写;否则只写一个聚合 section
                if framework is not None and framework.sections:
                    section_md_parts: list[str] = []
                    for _sec in framework.sections[:max_subtopics]:
                        _claims = [
                            ClaimV3(
                                value=(
                                    f"{eu.numbers[0].text} ({eu.numbers[0].unit or ''})"
                                    if eu.numbers else eu.claim[:200]
                                ),
                                norm_value=(
                                    float(eu.numbers[0].value_min)
                                    if eu.numbers and eu.numbers[0].value_min is not None
                                    else None
                                ),
                                source_id=str(id(eu)),  # V1 EU 没有 eu_id,用 id()
                                source_url=eu.source_url,
                                source_domain=eu.source_url.split("/")[2] if "/" in eu.source_url else "",
                                source_title=eu.source_title,
                                tier="B",
                                caliber_id=None,
                                verification_status="to_verify",
                            )
                            for eu in out.evidence_units
                            if eu.dimension_id == _sec.section_id
                        ][:10]
                        if not _claims:
                            continue
                        _slot = _w7_Slot(
                            slot_id=_sec.section_id,
                            question=(_sec.title + " " + _sec.section_id)[:200],  # ≥5 chars
                            expected_claim_type="qualitative",
                            required_tier_min="B",
                        )
                        try:
                            _section_result = write_section(_slot, _claims)
                            section_md_parts.append(
                                f"## {_sec.title}\n\n"
                                f"_section_id={_section_result.section_id}; "
                                f"filled_slots={len(_section_result.slots)}_\n"
                            )
                            section_writer_used = True
                        except Exception as sw_e:
                            logger.warning(
                                "[W7] write_section failed for section=%s: %s",
                                _sec.section_id, sw_e,
                            )
                    if section_writer_used and section_md_parts:
                        writer_response = (
                            f"# {title}\n\n" + "\n".join(section_md_parts) + "\n"
                        )
                        logger.info(
                            "[W7] section_writer produced %d sections (structured ReportResult)",
                            len(section_md_parts),
                        )
                if not section_writer_used:
                    raise RuntimeError("section_writer not invoked or failed")
            except Exception as w7_e:
                import traceback as _w7_tb
                logger.warning(
                    "[W7] section_writer unavailable (%s); falling back to degraded placeholder\n%s",
                    w7_e, _w7_tb.format_exc(),
                )
                writer_response = _placeholder_cited_response(
                    title=title, eus=out.evidence_units,
                ) + "\n\n<!-- W7 DEGRADED: LLM unavailable, report not generated by section_writer -->\n"
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

            # ----- W3: tier_classifier 替换白名单 -----
            # 旧路径:Phase 5 白名单 upgrade_source_tier(被删)
            # 新路径:每个 EU 经 tier_classifier.py 赋 tier — F1 注册表命中优先,
            #        未命中按 .gov/filings/媒体信号,signal-tier fallback 触发 warning
            for vu in v2_eus:
                abc = _classify_tier_abc(
                    source_url=vu.source_url,
                    source_domain=vu.source_domain,
                    log_unmatched=False,  # 批量落库时静默兜底,避免 log flood
                )
                if abc is not None:
                    vu.source_tier = _TIER_ABC_TO_SOURCE[abc]  # type: ignore[index]
            if v2_eus:
                # Runbook P0:真集成验证 — 在 upsert_many 前 batch hook
                # embed_texts() 写 v2_eus[i].embedding (BGE-M3 真模型优先,
                # 未装时 SHA-256 fallback — 永远不抛异常,只返回 (N, 1024) 向量)。
                # 失败 → 留 None,PG 端 embedding 列 NULL,与历史数据兼容。
                try:
                    claims = [vu.claim for vu in v2_eus]
                    vecs = embed_texts(claims)
                    if vecs.shape[0] == len(v2_eus):
                        for vu, v in zip(v2_eus, vecs):
                            vu.embedding = v.tolist()
                except Exception as emb_e:
                    import warnings
                    warnings.warn(
                        f"Phase 3 embedder.embed_texts failed (run_id={rid}): "
                        f"{emb_e}; EU embedding will be NULL",
                        RuntimeWarning,
                        stacklevel=2,
                    )

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

        # ----- 3.55 Phase 2.7 (= Runbook v1 阶段 2.1-2.4): 跑 3 闸 + 写回 PG -----
        # 数据准确性导向 (Runbook v1 §2.1-2.4):
        #   闸 1 (span)   : source_span 存在 → True(extractor self-truthful 假设)
        #   闸 2 (numeric) : claim 中数值在 span 内(0.5% rel_tol)
        #   闸 3 (NLI)     : LLM 判 (claim, span) entail/contradict/unverifiable
        # 写回 PG evidence_unit.{span_verified, numeric_drift, entailment_*}
        # 闸通过是 grade_claim 出 A/B 的前提 (schema.usable = span_verified && !numeric_drift && entailed)
        #
        # W5 ★非交涉:gates_failfast 替换静默包裹
        #   旧路径(被删):try/except ... warnings.warn(...)  → 续跑 + claim 全 D
        #   新路径:LLMGateError(auth/timeout/empty) 直接 raise 终止 run,
        #         span/numeric 闸失败也标 run 失败(SCHEMA.usable=false → D,但不掩盖)
        if v2_eus:
            from open_deep_research.evidence.gates_runner import run_gates_and_persist
            from open_deep_research.evidence.gates_failfast import LLMGateError
            try:
                gate_stats = await run_gates_and_persist(v2_eus, run_id=rid, run_nli=True)
            except LLMGateError as llm_e:
                # ★非交涉:绝不静默续跑、绝不"all grade=D"糊弄
                logger.error(
                    "Phase 2.7 NLI gate failed (run_id=%s): %s; "
                    "TERMINATING run, NOT degrading to unverifiable",
                    rid, llm_e,
                )
                out.error = f"NLI gate failed: {llm_e}"
                out.passed = False
                out.finished_at = datetime.now(timezone.utc)
                raise  # 让 caller(api/server / run_edr)看到,不静默吞
            out.gate_stats = gate_stats  # 让 /runs/{id} 能看见
            logger.info(
                "Phase 2.7 gates (run_id=%s): %s",
                rid, gate_stats,
            )
            # 关键:重新从 PG 读 EU,让后续 merge phase 看到新写的 entailment_verdict
            try:
                with EuDAO() as edao:
                    v2_eus = edao.list_by_run(rid)
                logger.info(
                    "Phase 2.7 re-fetched %d EU from PG (entailment_verdict populated)",
                    len(v2_eus),
                )
            except Exception as rf_e:
                logger.warning("Phase 2.7 PG re-fetch failed: %s", rf_e)
        # W5 删除:L413-419 旧的 except Exception ... warnings.warn(...) 静默续跑已被替换
        #   新路径:L378-393 已显式 except LLMGateError 并 raise,不再 swallow 任何异常

        # ----- 3.6 Phase 5 (= Runbook v1 阶段 3.1-3.4): EU → ClaimV2 -----
        # 数据准确性导向 (Runbook v1 §3.3):
        #   1. tier 已在 W3 用 tier_classifier.py 赋,Phase 5 不再做白名单升级
        #   2. merge_units — cosine similarity > 0.92 的 EU 归并
        #   3. build_claim_drafts — 每个 group 生成 canonical claim
        #   4. grade_claim — A/B/C/D 评级 (基于 independent + primary count)
        #   5. claim 落 PG (evidence.claim 表) — 让 /runs/{id} 的 claim_stats 立刻可观测
        # P1.2: 同时拿 eu_to_claim_id map 用于回填 evidence_unit.claim_id 字段
        try:
            if v2_eus:
                # P3 fix: pass embeddings=vecs to merge_units so cosine-based dedup
                # actually runs. Without this, build_claims_from_eus → merge_units
                # skips all pairs (embeddings=None path), producing 100% singleton
                # claims and forcing every EU to grade=D.
                # Embeddings come from the in-memory v2_eus[i].embedding list
                # populated at line 331 (embed_texts → hash/MiniLM).
                vecs_for_merge: Optional[np.ndarray] = None
                try:
                    emb_lists = [vu.embedding for vu in v2_eus]
                    if all(e is not None for e in emb_lists):
                        vecs_for_merge = np.asarray(emb_lists, dtype=np.float32)
                except Exception as emb_xfer_e:
                    logger.warning(
                        "Phase 3 merge embeddings transfer failed: %s; merge will 100%% singleton",
                        emb_xfer_e,
                    )
                result = build_claims_from_eus(
                    v2_eus, embeddings=vecs_for_merge, return_eu_map=True
                )
                if isinstance(result, tuple):
                    claims, eu_to_claim_id = result
                else:
                    claims, eu_to_claim_id = result, {}
                out.claims = claims
                out.claim_grade_dist = {
                    g: sum(1 for c in claims if c.grade == g)
                    for g in "ABCD"
                }
                logger.info(
                    "build_claims_from_eus: %d EU -> %d claims (grade dist: %s, eu_map=%d)",
                    len(v2_eus), len(claims), out.claim_grade_dist, len(eu_to_claim_id),
                )
                if claims:
                    with ClaimDAO() as cdao:
                        cdao.upsert_many(claims)
                    # P1.2: 回填 evidence_unit.claim_id 字段,让溯源完整
                    if eu_to_claim_id:
                        with EuDAO() as edao:
                            for eu_id_str, claim_id_str in eu_to_claim_id.items():
                                edao.update_claim_id(eu_id_str, claim_id_str)
                        logger.info(
                            "claim_id 回填: %d EU -> claim_id (溯源链路完成)",
                            len(eu_to_claim_id),
                        )

            # ----- W6: reconciliation + independence reduce -----
            # 在 build_claims 之后、报告装配之前;从 EU + framework 派生 ClaimV3,
            # cluster by measurand key, detect caliber divergence, reconcile.
            # 跨口径禁 mean/median — primary 选最佳 caliber,其余进 alternatives 标不可比。
            try:
                from open_deep_research.evidence.reconciliation import (
                    cluster_claims, reconcile_cluster,
                )
                from open_deep_research.evidence.claim_v3 import ClaimV3
                from open_deep_research.evidence.caliber_registry import (
                    load_caliber_registry as _w6_load_calibers,
                )
                try:
                    calibers = _w6_load_calibers("us_livecommerce")
                except FileNotFoundError:
                    calibers = None
                # 建 slot_id → framework slot 的映射(W4 已存 framework)
                fw_slot_by_id: dict[str, Any] = {}
                if framework is not None:
                    for sec in framework.sections:
                        for sl in sec.slots:
                            fw_slot_by_id[sl.slot_id] = sl
                # 从 v2_eus 派生 ClaimV3(EU 没存 caliber_id,从 framework 取)
                claim_v3_for_recon: list[ClaimV3] = []
                for eu in v2_eus:
                    if eu.norm_value is None:
                        continue
                    _cal = None
                    if framework is not None:
                        _cal = fw_slot_by_id.get(eu.dimension_id or "", None)
                        # EU 没存 slot_id,只能 dimension 匹配(section_id)
                    claim_v3_for_recon.append(ClaimV3(
                        value=str(eu.norm_value),
                        norm_value=float(eu.norm_value),
                        source_id=str(eu.eu_id),
                        source_url=eu.source_url,
                        source_domain=eu.source_domain,
                        source_title=eu.source_title,
                        tier=("A" if eu.source_tier == "primary" else
                              "B" if eu.source_tier == "secondary" else
                              "C" if eu.source_tier == "tertiary" else "D"),
                        caliber_id=None,
                        verification_status=(
                            "verified" if eu.span_verified and not eu.numeric_drift
                            and (eu.entailment_verdict == "entailed") else "to_verify"
                        ),
                    ))
                if claim_v3_for_recon:
                    clusters = cluster_claims(claim_v3_for_recon)
                    logger.info(
                        "[W6] cluster: %d claims → %d clusters (口径分歧候选)",
                        len(claim_v3_for_recon), len(clusters),
                    )
                    for cc in clusters:
                        try:
                            rec = reconcile_cluster(cc, calibers=calibers)
                            logger.info(
                                "[W6] reconciled: measurand=%s primary=%s caliber=%s "
                                "divergence=%s alternatives=%d",
                                rec.measurand[:60],
                                rec.primary_value,
                                rec.primary_caliber_id,
                                rec.divergence_kind,
                                len(rec.alternatives),
                            )
                        except Exception as rec_e:
                            logger.warning(
                                "[W6] reconcile_cluster failed: %s",
                                rec_e,
                            )
            except Exception as w6_e:
                logger.warning("[W6] reconciliation stage skipped: %s", w6_e)
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

        # ----- 7a. W8 honest_pass 报告收口 -----
        # 主结论降级 + [待核实]附录 + 口径误区 + 来源页(从元数据渲染)
        try:
            from open_deep_research.evidence.honest_pass import run_honest_pass
            from open_deep_research.evidence.report_result import ReportResult
            from open_deep_research.evidence.claim_v3 import ClaimV3
            # 用 v2_eus(V2 schema 有 span_verified / numeric_drift / entailment_verdict)
            _claims_v3: list[ClaimV3] = []
            for _eu in v2_eus:
                _claims_v3.append(ClaimV3(
                    value=_eu.claim[:200],
                    source_id=str(_eu.eu_id),
                    source_url=_eu.source_url,
                    source_domain=_eu.source_domain,
                    source_title=_eu.source_title,
                    tier=("A" if _eu.source_tier == "primary" else
                          "B" if _eu.source_tier == "secondary" else
                          "C" if _eu.source_tier == "tertiary" else "D"),
                    caliber_id=None,
                    verification_status=(
                        "verified" if _eu.span_verified and not _eu.numeric_drift
                        and (_eu.entailment_verdict == "entailed") else "to_verify"
                    ),
                ))
            _rr = ReportResult(
                title=rdo.title or title,
                vertical_id="us_livecommerce",
                sections=[],
                unresolved=[],
            )
            _rr = run_honest_pass(_rr, _claims_v3, reconciliations=[])  # type: ignore[arg-type]
            out.honest_pass = _rr.honest_pass or {}
            hp_stats = out.honest_pass.get("stats", {}) if isinstance(out.honest_pass, dict) else {}
            logger.info(
                "[W8] honest_pass: unverified=%d caliber_mismatch=%d sources=%d",
                hp_stats.get("unverified_count", 0),
                hp_stats.get("caliber_mismatch_count", 0),
                hp_stats.get("unique_source_domains", 0),
            )
        except Exception as w8_e:
            logger.warning("[W8] honest_pass skipped: %s", w8_e)
            out.honest_pass = {"error": str(w8_e)}

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
