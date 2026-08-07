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
    EvidenceUnit, NumberBinding, eus_as_dicts, dedup_eus,
)
from open_deep_research.evidence.schema import EvidenceUnitV2
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
# R6 helpers: V2 → V1 转换,保留 metric_type 信息给 reconciliation
# =============================================================================


def _v2_to_v1(eu_v2: EvidenceUnitV2) -> EvidenceUnit:
    """把 LLM 抽取器产出的 EvidenceUnitV2 转成下游消费的 V1 dataclass。

    设计妥协:downstream(cited_report / verifier / W6 reconciliation 旧路径)期望
    V1 字段集(.numbers / .claim / .quote / .source_url / .dimension_id)。
    V2 的 metric_type / entities / source_span / source_tier 通过 V1 字段透传:
    - metric_type → extraction_method 后缀(给 reconciler 解析用)
    - entities → V1.entities(每个转为 EntityRef)
    - source_span → V1.quote
    - source_tier → V1.source_tier
    """
    from open_deep_research.evidence_units import EntityRef

    # claim 截断 ≤500 chars(V1 __post_init__ 强校验)
    claim = (eu_v2.claim or "").strip()
    if len(claim) > 500:
        claim = claim[:497] + "..."

    numbers: list[NumberBinding] = []
    if eu_v2.norm_value is not None and eu_v2.unit:
        try:
            numbers.append(NumberBinding(
                text=str(eu_v2.norm_value) + " " + str(eu_v2.unit),
                value_min=float(eu_v2.norm_value),
                value_max=float(eu_v2.norm_value),
                unit=str(eu_v2.unit),
                is_estimated=False,
            ))
        except Exception:
            pass

    # R6:把 metric_type 编码到 extraction_method,让 reconciler 能取
    method_suffix = f"|metric={eu_v2.metric_type or 'unknown'}"
    extraction_method = (eu_v2.extractor_model or "llm_extractor_v1") + method_suffix

    # R6:entity_type heuristic — 公司类(V2.entities 里含 "公司/Co./Inc./集团/科技" 等关键词)
    return EvidenceUnit(
        claim=claim,
        source_url=eu_v2.source_url,
        quote=eu_v2.source_span,
        source_title=eu_v2.source_title,
        numbers=numbers,
        entities=[
            EntityRef(name=e, entity_type=_infer_entity_type(e))
            for e in (eu_v2.entities or [])
        ],
        confidence=0.7,
        extraction_method=extraction_method,
        run_id=str(eu_v2.run_id) if eu_v2.run_id else None,
        dimension_id=eu_v2.dimension_id,
        source_tier=eu_v2.source_tier,
    )


_COMPANY_HINTS = ("公司", "集团", "科技", "Co.", "Inc.", "Corp", "厂商", "厂商")


def _infer_entity_type(name: str) -> str:
    """启发式:V2.entities 的 name → V1.entity_type。

    公司类:含公司/集团/科技/Co./Inc./Corp
    产品类:含 EDR/产品/平台/系统
    metric:含 规模/份额/CAGR/率
    其余:unknown
    """
    if any(h in name for h in _COMPANY_HINTS):
        return "company"
    if any(h in name for h in ("EDR", "产品", "平台", "系统", "aES")):
        return "product"
    if any(h in name for h in ("规模", "份额", "CAGR", "率", "USD", "亿元", "%")):
        return "metric"
    return "unknown"


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
    # R6:保留 V2 列表(带 metric_type / entities / norm_value)给 reconciliation
    evidence_units_v2: list[Any] = field(default_factory=list)
    claims: list[Any] = field(default_factory=list)  # ClaimV2 (跨源归并)
    claim_grade_dist: dict[str, int] = field(default_factory=dict)  # {A: N, B: N, C: N, D: N}
    cited_report: Optional[CitedReport] = None
    cited_report_warnings: list[str] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    report_data: Optional[ReportDataObject] = None
    url_compliance: list[UrlComplianceIssue] = field(default_factory=list)
    gate_stats: dict[str, int] = field(default_factory=dict)  # 闸 1+2+3 命中统计
    honest_pass: Optional[dict] = None  # W8:honest_pass 收口(unverified/caliber/source_pages)
    # W9-L2 输入:W6 产出的 ReconciliationRecord 列表(跨切综合层读它)
    reconciliations: list[Any] = field(default_factory=list)

    # W9/W10 reduce-tree 产物(任务书 §5 装配)
    section_summaries: list[Any] = field(default_factory=list)
    crosscuts: list[Any] = field(default_factory=list)
    exec_summary: Optional[Any] = None
    assembled_markdown: Optional[str] = None  # 长报告最终 markdown

    passed: bool = False
    error: Optional[str] = None
    # R5 整改: 显式区分 "passed=True 但部分 degraded" 与 "passed=True 且全部 OK"。
    # section_writer 不可用时 degrade=True + passed=False,确保 golden-eval main
    # gate 不把降级 placeholder 当成 pass。
    degraded: bool = False

    # R10:未解析的 slot(section) — 报告装配时显式标"数据不足",不留无声空标题。
    unresolved_sections: list[dict[str, str]] = field(default_factory=list)

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
            "degraded": self.degraded,
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
    archetypes: Optional[list[str]] = None,
    ontology: Optional[str] = None,
    registry_vertical: Optional[str] = None,
    instance_brand: Optional[str] = None,
    instance_market: Optional[str] = None,
    instance_category: Optional[str] = None,
    instance_year: Optional[int] = None,
    # R6/R7:LLM client 透传给 W3 llm_extractor 与 W9/W10 综合层;
    # 若 None → run_odr.py 调 get_llm() 兜底 → 仍 None 则 fallback 规则抽取器 + degraded。
    llm: Optional[Any] = None,
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
        # W1: 单一事实源 = evidence/plan_builder.build_deterministic_plan
        # (R1 整改:原 staged_runner 硬编码 load_framework("us_livecommerce"),
        #  4 层架构在生产路径上实际未生效。改后 W1 hook 只有一处实现,
        #  staged_runner / run_pipeline / CLI 全部经由 plan_builder 调。)
        from open_deep_research.evidence.plan_builder import build_deterministic_plan

        plan, framework, onto = build_deterministic_plan(
            query,
            vertical=vertical,
            archetypes=archetypes,
            ontology=ontology,
            registry_vertical=registry_vertical,
            instance_brand=instance_brand,
            instance_market=instance_market,
            instance_category=instance_category,
            instance_year=instance_year,
            max_subtopics=max_subtopics,
        )
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
        # R6:本批 EU(V2 + 已配 slot.section_id)的临时累加器
        _batch_v2: list[EvidenceUnitV2] = []
        _batch_v2_with_dim: list[tuple[str, EvidenceUnitV2]] = []
        # R15:每 sub_topic 尝试记录(unresolved 诊断用)
        _sub_topic_attempts: list[dict[str, Any]] = []
        for st in plan.sub_topics:
            _slot_dim: str = st.id or st.dimension_id or "unknown"
            # R11:在 source_router try 块之前预声明,防 unbound 报错。
            _fw_expected_claim_type: Optional[str] = None
            # R12:抽 ontology 的主题实体集,作为 relevance gate 兜底(任务书 R12)。
            # 优先用 positive_cn_vendors / positive_en_vendors(EDR 厂商列表),
            # 其次 entity_spaces.vendors.items。空集则跳过校验(放过)。
            _topic_entities: set[str] = set()
            if onto is not None:
                try:
                    _ih = getattr(onto, "indicator_hollows", {}) or {}
                    for _vendors_key in ("positive_cn_vendors", "positive_en_vendors"):
                        for _v in (_ih.get("edr_disambiguation", {}) or {}).get(
                            _vendors_key, []
                        ):
                            if isinstance(_v, str) and _v.strip():
                                _topic_entities.add(_v.strip().lower())
                    for _v in (
                        (onto.entity_spaces or {}).get("vendors", {}).get("items", [])
                    ):
                        _nm = _v.get("name") if isinstance(_v, dict) else None
                        if isinstance(_nm, str) and _nm.strip():
                            _topic_entities.add(_nm.strip().lower())
                except Exception as _e:
                    logger.debug("[R12] topic_entities extraction skipped: %s", _e)
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
                from typing import cast as _cast
                from open_deep_research.evidence.framework import ClaimType as _CT
                # R11 ★:从 framework YAML 读 expected_claim_type 和 caliber_id。
                # 不再用 dimension_id 关键词猜 — 路由与抽取必须用同一个
                # claim_type(任务书 R11:框架脱节是 R1 式漂移的复发)。
                _fw_caliber_id: Optional[str] = None
                _fw_expected_claim_type: Optional[str] = None
                if framework is not None:
                    for _sec in framework.sections:
                        for _slot in _sec.slots:
                            if _slot.slot_id == (st.id or ""):
                                _fw_caliber_id = _slot.caliber_id
                                # framework ClaimType: quantitative / qualitative /
                                # comparative / event / attribute
                                _fw_expected_claim_type = (
                                    str(_slot.expected_claim_type)
                                    if _slot.expected_claim_type
                                    else None
                                )
                                break
                        if _fw_caliber_id:
                            break
                _fw_slot = _FSlot(
                    slot_id=(st.id or "slot_default").lower().replace(" ", "_").replace("-", "_")[:64] or "slot_default",
                    question=st.question,
                    expected_claim_type=_cast(_CT, _fw_expected_claim_type or "attribute"),
                    required_tier_min="B",
                    caliber_id=_fw_caliber_id,
                )
                try:
                    # 仅在 caller 注入 framework 时启用 source_router。
                    # (R1 整改:不再静默回退到 "us_livecommerce"。
                    #  4-layer → ontology 同时也是 registry_vertical;legacy
                    #  → vertical;都没有(None)→ 跳过 source_router。)
                    _reg_v: str | None = None
                    if onto is not None:
                        _reg_v = registry_vertical or ontology
                    elif vertical:
                        _reg_v = vertical
                    if _reg_v:
                        _registry = _load_registry(_reg_v)
                        _calibers = _load_calibers(_reg_v)
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
                    # ----- W2b: indicator_hollows disambiguation -----
                    # When 4-layer plan ran, `onto` is in scope; apply
                    # ontology.indicator_hollows to drop EDR-polysemy noise.
                    if onto is not None and source_router_queries:
                        from open_deep_research.query_constructor import (
                            _apply_indicator_hollows,
                        )
                        source_router_queries = _apply_indicator_hollows(
                            source_router_queries, ontology=onto,
                        )
                        logger.info(
                            "[W2b] indicator_hollows applied: %d queries disambiguated",
                            len(source_router_queries),
                        )
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

            # 2c. Extract EUs from the merged raw set.
            # R6 ★:真 LLM 抽取器。旧路径 extract_from_search_results 是规则抽取
            # (直接拿 search snippet 切,没有 LLM 综合判断,导语/SEO 全进 EU)。
            # 新路径:对每页调 LLM 抽 EU + metric_type/norm_value/entities 全套 schema,
            # quantitative 槽强校验 norm_value+unit,缺失直接丢弃(避免定义句
            # 灌进 TAM/SAM 槽)。
            if not raws:
                continue
            # R11 ★:expected_claim_type 优先用 framework slot 声明的;
            # 只有 framework 没声明时才用 dimension_id 关键词启发式。
            # framework 声明是规范化(quantitative/qualitative/...),
            # heuristic 是临时兜底(numeric / None)。
            # 注意:_fw_expected_claim_type 在 source_router try 块里可能没赋值,
            # 这里预声明 None 防 unbound。
            sub_dim = (st.dimension_id or "").lower()
            _fw_ect: Optional[str] = _fw_expected_claim_type  # type: ignore[name-defined]
            if _fw_ect:
                # R11:把 framework 声明的 claim_type 翻译成 llm_extractor 期望的
                # 词汇表值(framework 用 quantitative/numeric/qualitative 等,
                # llm_extractor schema 用 numeric/attribute/event/relation/opinion)。
                # 翻译规则:quantitative → numeric(其他保持原样)。
                # v15 教训:ect=None 时 LLM 倾向返空;v11 用 numeric 时抽到 9 EU。
                _llm_ect_map = {
                    "quantitative": "numeric",
                    "qualitative": "attribute",
                    "comparative": "attribute",
                    "event": "event",
                    "attribute": "attribute",
                }
                expected_claim_type: Optional[str] = _llm_ect_map.get(
                    _fw_ect, _fw_ect,
                )
            elif any(k in sub_dim for k in (
                "market_size", "sam", "tam", "share", "growth",
                "revenue", "gmv", "sku_rank", "penetration",
            )):
                expected_claim_type = "numeric"
            else:
                expected_claim_type = None
            try:
                from open_deep_research.evidence.llm_extractor import (
                    extract_from_search_results_with_llm,
                )
                if llm is not None:
                    # R13:预算可配置 — quantitative 槽给更高预算(40 页),
                    # 其他槽给默认 10 页。quantitative 槽被 cap 掉会大幅降低召回,
                    # 是 EU 46→9 的主嫌之一(任务书 R13)。
                    # 经验值:quantitative 10 页 → 0~2 EU;40 页 → 8~15 EU。
                    from open_deep_research.evidence.llm_extractor import (
                        extract_from_search_results_with_llm,
                    )
                    # 先算 _is_quantitative(R12 / R13 都用)
                    _is_quantitative = _fw_ect in (
                        "quantitative",
                    ) or expected_claim_type in ("quantitative", "numeric")
                    # R13:可配置预算 — quantitative 槽 40 页,其他槽 10 页
                    # (任务书 R13)。v19 经验:10 页 → 0~2 EU;40 页 → 8~15 EU。
                    _cap_pages = 40 if _is_quantitative else 10
                    _raws_capped = raws[:_cap_pages]
                    _capped_count = max(0, len(raws) - _cap_pages)
                    # R12 ★:relevance gate · 引擎层确定性兜底。
                    # market_size/quantitative 槽不查 arxiv — 学术论文对市场
                    # 规模是纯噪声源(任务书 R12)。同时过滤掉 arxiv.org URL
                    # (有些引擎不标 engine 字段)。
                    if _is_quantitative:
                        _before_filter = len(_raws_capped)
                        _raws_capped = [
                            r for r in _raws_capped
                            if (r.get("engine") or "").lower() != "arxiv"
                            and "arxiv.org" not in (r.get("url") or "").lower()
                        ]
                        _arxiv_filtered = _before_filter - len(_raws_capped)
                        if _arxiv_filtered:
                            logger.info(
                                "[W3 R12] arxiv filtered for quantitative slot=%s: %d pages dropped",
                                st.id, _arxiv_filtered,
                            )
                    eus_v2 = await extract_from_search_results_with_llm(
                        _raws_capped,
                        run_id=rid,
                        llm=llm,
                        sub_query=st.question,
                        extractor_model="extractor_v1",
                        expected_claim_type=expected_claim_type,
                    )
                    # R6 ★:EU.dimension_id → framework.sections[i].section_id(宽标签,
                    # 如 market_size__tam),与 W7 (plan_v2_pipeline:478) 配对规则一致。
                    # 之前用 st.id(如 market_size__market_size__tam__tam_total)不匹配,
                    # W9 L0 装配时 _eu_per_section[sec.section_id] 找不到任何 EU,全部标 unresolved。
                    # st.dimension_id 是 planner 写入的宽标签,framework section_id 同源。
                    for _eu in eus_v2:
                        _eu.dimension_id = st.dimension_id or _slot_dim
                    # R12 ★:relevance gate · 主题实体校验(确定性兜底)。
                    # 即便 LLM 漏检了"本页无关"的语义,我们也用主题实体集合
                    # 做一次硬校验:claim.entities 与 ontology 主题实体集无交集
                    # → 丢弃(任务书 R12)。计 filtered_irrelevant 供诊断。
                    _filtered_irrelevant = 0
                    if _topic_entities and _is_quantitative:
                        _filtered_v2: list[EvidenceUnitV2] = []
                        for _eu in eus_v2:
                            _eu_entities = {
                                (e or "").strip().lower()
                                for e in (getattr(_eu, "entities", []) or [])
                                if isinstance(e, str)
                            }
                            if _eu_entities & _topic_entities:
                                _filtered_v2.append(_eu)
                            else:
                                _filtered_irrelevant += 1
                                logger.info(
                                    "[W3 R12] filtered_irrelevant eu.url=%s "
                                    "entities=%s not in topic set",
                                    getattr(_eu, "source_url", ""),
                                    _eu_entities,
                                )
                        eus_v2 = _filtered_v2
                        if _filtered_irrelevant:
                            logger.info(
                                "[W3 R12] slot=%s filtered_irrelevant=%d (remaining v2_eus=%d)",
                                st.id, _filtered_irrelevant, len(eus_v2),
                            )
                    # V2 → V1 转换:downstream 仍消费 V1 dataclass
                    # (写者/对账都期望 .numbers/.claim/.source_url/.dimension_id)
                    eus = [_v2_to_v1(_e) for _e in eus_v2]
                    # 把 V2 列表暂存到 ctx-style 临时容器,本 sub_topic 结束后并入 out
                    _batch_v2.extend(eus_v2)
                    logger.info(
                        "[W3 R13] llm_extractor sub_topic=%s ect=%s raw_results=%d cap_pages=%d capped_off=%d v2_eus=%d v1_eus=%d",
                        st.id, expected_claim_type, len(raws), _cap_pages, _capped_count,
                        len(eus_v2), len(eus),
                    )
                else:
                    logger.warning(
                        "[W3 R6] llm=None, falling back to deterministic extractor (degraded)"
                    )
                    out.degraded = True
                    eus = extract_from_search_results(
                        raws,
                        run_id=rid,
                        sources_dao=sources_dao,
                        research_topic=query,
                        dimension_id=st.id or st.dimension_id,
                    )
            except Exception as e:
                logger.warning(
                    "[W3 R6] llm_extractor failed (sub_topic=%s): %s; falling back to rule extractor (degraded)",
                    st.id, e,
                )
                out.degraded = True
                eus = extract_from_search_results(
                    raws,
                    run_id=rid,
                    sources_dao=sources_dao,
                    research_topic=query,
                    dimension_id=st.id or st.dimension_id,
                )
            all_eus.extend(eus)
            # R6:本批 V2 累加
            _batch_v2_with_dim.extend(
                [(_slot_dim, _e) for _e in _batch_v2]
            )
            _batch_v2.clear()
            # R15 ★:确定性重试阶梯 · 记录已尝试的查询集。
            # 三档(任务书 R15):
            #   1) 主查询(source_router + LLM intent)
            #   2) 换 query_templates 下一个模板
            #   3) 放宽 tier:允许 C 级(原本仅 A/B)
            # 这里只先做"记录已尝试查询数"的诊断,显式 unresolved 时附带。
            _tried_queries = [
                q for _sq in intent_search_queries for q in (_sq.queries or [])
            ]
            _sub_topic_attempts.append({
                "sub_topic_id": st.id,
                "queries_tried": _tried_queries,
                "raw_pages": len(raws),
                "v2_eus": len(eus_v2) if "eus_v2" in dir() else 0,
                "v1_eus": len(eus),
            })

        # R6:写 out.evidence_units_v2(dimension_id 已配好 slot.section_id)
        if _batch_v2_with_dim:
            out.evidence_units_v2 = [_e for _, _e in _batch_v2_with_dim]
        out.evidence_units = dedup_eus(all_eus)
        logger.info(
            "extracted %d unique EU across %d sub-topics (with %d dimensioned)",
            len(out.evidence_units),
            len(plan.sub_topics),
            sum(1 for s in plan.sub_topics if s.dimension_id),
        )

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
                # R5 整改: section_writer 不可用时显式标记 degraded + passed=False,
                # 防止 golden-eval main gate 把降级 placeholder 当成 pass。
                out.degraded = True
                out.passed = False
                out.cited_report_warnings.append(
                    "W7 section_writer unavailable; placeholder rendered (degraded)"
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
                raise  # 让 caller(api/server / run_odr)看到,不静默吞
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
                # R9:calibers 按 ontology/registry_vertical 加载(原写死 us_livecommerce
                # 导致 cn_cybersec run 永远 calibers=None → primary_caliber_id='unknown')。
                _caliber_vertical = (
                    registry_vertical or ontology or vertical or "us_livecommerce"
                )
                try:
                    calibers = _w6_load_calibers(_caliber_vertical)
                except FileNotFoundError:
                    calibers = None
                # R9:CaliberRegistry.by_metric(metric_type) 找最匹配的 caliber_id
                # 优先 metric_type 全等,次选 metric_type 任意子串;找不到 → None
                def _pick_caliber_id(
                    metric_type: Optional[str], entity: Optional[str],
                ) -> Optional[str]:
                    if calibers is None or not metric_type:
                        return None
                    # 1) 全等 + entity 子串
                    for c in calibers.calibers:
                        if c.metric_type == metric_type:
                            if not entity or entity.lower() in c.entity.lower():
                                return c.id
                    # 2) 全等(忽略 entity)
                    for c in calibers.calibers:
                        if c.metric_type == metric_type:
                            return c.id
                    return None

                # 建 slot_id → framework slot 的映射(W4 已存 framework)
                fw_slot_by_id: dict[str, Any] = {}
                if framework is not None:
                    for sec in framework.sections:
                        for sl in sec.slots:
                            fw_slot_by_id[sl.slot_id] = sl
                # 从 v2_eus 派生 ClaimV3 — R9:metric_type / entity 透传,
                # caliber_id 从 CaliberRegistry by_metric 派生(替代之前写 None)。
                claim_v3_for_recon: list[ClaimV3] = []
                for eu in v2_eus:
                    if eu.norm_value is None:
                        continue
                    # R9 强约束:无 metric_type / 无 entity 的 EU 不进对账(unknown 不可比对)。
                    if not eu.metric_type or eu.metric_type == "other":
                        continue
                    if not eu.entities:
                        continue
                    _cal = _pick_caliber_id(eu.metric_type, eu.entities[0])
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
                        # R9:caliber_id 真匹配 metric_type + entity(替代写死 None)
                        caliber_id=_cal,
                        metric_type=eu.metric_type,
                        entity=eu.entities[0] if eu.entities else None,
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
                            # W9-L2 输入:把 rec 收集到 out.reconciliations
                            out.reconciliations.append(rec)
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

        # ----- 6b. W9 / W10 · 写作 reduce 树(任务书 §5)-----
        # 顺序:L0 (已在 W7 写过) → L1 synthesize_section → L2 synthesize_crosscuts → L3 write_exec_summary → 装配
        # 任一层 LLM 不可用 → degraded=True / passed=False(R5 保留)
        # W7 旧逐槽 path 继续产出 cited_report(不替换);此处是新长报告路径。
        try:
            from open_deep_research.evidence.report_result import (
                CrossCut, ExecSummary, SectionResult, SectionSummary, FilledSlot,
            )
            from open_deep_research.evidence.section_writer import (
                synthesize_section_async,
                synthesize_crosscuts_async,
                write_exec_summary_async,
            )
            from open_deep_research.evidence.claim_v3 import ClaimV3 as _ClaimV3_W9

            # 把 EU 池转成 ClaimV3,按 dimension_id 分组给每节
            _eu_per_section: dict[str, list[Any]] = {}
            for _eu in v2_eus:
                _key = _eu.dimension_id or "_unassigned"
                _eu_per_section.setdefault(_key, []).append(_eu)

            # L0 → SectionResult(按 framework.sections 顺序)
            # 每个 framework.section 一个 SectionResult;
            # 该 section 内所有 EU 合并为一个 FilledSlot(代表本节综合证据池)。
            # 限制:framework.slot_id 是细粒度(如 dossier__qax__overview),
            #       但 EU.dimension_id 仅 section 级(W7 既定配对规则),所以这里
            #       也按 section 级合并 — 与 W7 (plan_v2_pipeline:478) 一致。
            _section_results = []
            # R7 ★:用 run_odr.py 注入的 llm;不再默认 None 触发 stub 静默兜底。
            # 若 llm is None → synthesize_section_async 内部 get_llm() 仍可用(MiniMax 有 key);
            # 真没 key → section_writer 已抛 → plan_v2_pipeline 进入 degraded 路径(已配)。
            _llm_w9 = llm  # type: ignore[assignment]
            if framework is not None and framework.sections:
                for _sec in framework.sections:
                    _sec_eus = _eu_per_section.get(_sec.section_id, [])
                    _filled_slots: list[FilledSlot] = []
                    if _sec_eus:
                        # 同 dim EU 收成一槽(L0 单槽映射)
                        _v3_multi = [
                            _ClaimV3_W9(
                                value=e.claim[:200],
                                source_id=str(e.eu_id),
                                source_url=e.source_url,
                                source_domain=e.source_domain,
                                source_title=e.source_title,
                                tier=("A" if e.source_tier == "primary" else
                                      "B" if e.source_tier == "secondary" else
                                      "C" if e.source_tier == "tertiary" else "D"),
                                caliber_id=None,
                                verification_status=("verified" if e.span_verified
                                                     and not e.numeric_drift else "to_verify"),
                            )
                            for e in _sec_eus[:10]  # bounded k=10(任务书 §0.2)
                        ]
                        from open_deep_research.evidence.report_result import (
                            derive_confidence as _derive_conf,
                        )
                        _filled_slots.append(FilledSlot(
                            slot_id=_sec.section_id,
                            claims=_v3_multi,
                            confidence=_derive_conf(_v3_multi),
                        ))
                    else:
                        # R10 ★ + R15:空 section 显式记入 unresolved(不留无声空标题)。
                        # 报告装配时 render 标题 + "(数据不足)" 标注。
                        # R15:附带"已尝试 N 个查询 + raw_pages"具体诊断信息,
                        # 让空槽可归因到查询问题 vs 数据问题。
                        _matched_attempt = next(
                            (a for a in _sub_topic_attempts if a.get(
                                "sub_topic_id",
                            ) == _sec.section_id),
                            None,
                        )
                        _attempt_detail = ""
                        if _matched_attempt:
                            _tried = _matched_attempt.get("queries_tried") or []
                            _raw_pages = _matched_attempt.get("raw_pages", 0)
                            _attempt_detail = (
                                f" — 已尝试 {len(_tried)} 个查询 / "
                                f"抓取 {_raw_pages} 页"
                            )
                        out.unresolved_sections.append({
                            "section_id": _sec.section_id,
                            "title": _sec.title,
                            "reason": "no EU flowed into this slot — "
                                      "source_router 未命中/ontology 缺覆盖/预期 numeric 但 LLM 未抽到数字"
                                      + _attempt_detail,
                        })
                    _section_results.append(SectionResult(
                        section_id=_sec.section_id,
                        title=_sec.title,  # framework.title 在 build_plan 已渲染过
                        slots=_filled_slots,
                    ))

            # L1 · synthesize_section(每节小结)
            _summaries: list[SectionSummary] = []
            for _sr in _section_results:
                if not _sr.slots:
                    continue
                try:
                    _sm = await synthesize_section_async(_sr, llm=_llm_w9)
                    _summaries.append(_sm)
                except Exception as w9l1_e:
                    logger.warning("[W9-L1] %s failed: %s — degraded (continuing)", _sr.section_id, w9l1_e)
                    out.degraded = True
                    out.passed = False
                    out.cited_report_warnings.append(
                        f"W9-L1 degrade: {_sr.section_id}: {w9l1_e}"
                    )
                    # 改 break → continue:任一节失败不影响后续节
                    # 任务书 §0.2:综合层 LLM 不可用 → degraded=True / passed=False;
                    # 失败节留空,其他节仍可综合。
            out.section_summaries = _summaries

            # L2 · synthesize_crosscuts(对照表 + 分析)
            # 任务书 §7:framework.crosscuts 优先;默认 2 条兜底
            _fw_crosscuts = (
                framework.crosscuts if (framework is not None and framework.crosscuts) else None
            )
            _crosscuts: list[CrossCut] = []
            if _summaries:
                try:
                    _crosscuts = await synthesize_crosscuts_async(
                        _summaries,
                        out.reconciliations,
                        crosscut_specs=_fw_crosscuts,  # 任务书 §7
                        llm=_llm_w9,
                    )
                except Exception as w9l2_e:
                    logger.warning("[W9-L2] crosscuts failed: %s — degraded", w9l2_e)
                    out.degraded = True
                    out.passed = False
                    out.cited_report_warnings.append(
                        f"W9-L2 degrade: {w9l2_e}"
                    )
            out.crosscuts = _crosscuts

            # §6 · 任一综合层 is_stub=True → degraded=True / passed=False
            # (任务书 §0.2 不糊弄:综合层 LLM 不可用必须如实标记)
            if any(getattr(s, "is_stub", False) for s in _summaries):
                out.degraded = True
                out.passed = False
                out.cited_report_warnings.append(
                    f"W9-L1 stub fallback: {sum(1 for s in _summaries if getattr(s, 'is_stub', False))}/{len(_summaries)} sections"
                )
            if any(getattr(c, "is_stub", False) for c in _crosscuts):
                out.degraded = True
                out.passed = False
                out.cited_report_warnings.append(
                    f"W9-L2 stub fallback: {sum(1 for c in _crosscuts if getattr(c, 'is_stub', False))}/{len(_crosscuts)} crosscuts"
                )

            # L3 · write_exec_summary
            _exec: Optional[ExecSummary] = None
            if _summaries:
                try:
                    _exec = await write_exec_summary_async(
                        _summaries, _crosscuts, llm=_llm_w9,
                    )
                except Exception as w10_e:
                    logger.warning("[W10] exec_summary failed: %s — degraded", w10_e)
                    out.degraded = True
                    out.passed = False
                    out.cited_report_warnings.append(
                        f"W10 degrade: {w10_e}"
                    )
            out.exec_summary = _exec

            # 装配:按任务书 §5 顺序输出 long markdown
            try:
                from open_deep_research.evidence.report_result import (
                    ReportResult as _ReportResult_W9,
                )
                _rr_long = _ReportResult_W9(
                    title=rdo.title or title,
                    vertical_id="cn_cybersec" if (ontology or "").startswith("cn") else
                               (ontology or vertical or "unknown"),
                    sections=_section_results,
                    section_summaries=_summaries,
                    crosscuts=_crosscuts,
                    exec_summary=_exec,
                    unresolved=[],
                    degraded=out.degraded,
                )
                # 装配 markdown:exec 在最前、各节 L0+L1、crosscuts 在中、honest_pass 附录由 W8 加
                _md_parts: list[str] = []
                _md_parts.append(f"# {_rr_long.title}\n")
                if _exec is not None:
                    _md_parts.append("## 执行摘要\n")
                    _md_parts.append(f"**{_exec.one_liner}**\n\n")
                    for _kp in _exec.key_points:
                        _md_parts.append(f"- {_kp}\n")
                    _md_parts.append("\n")
                # 各节:L0 槽 + L1 小结
                for _sr in _section_results:
                    _md_parts.append(f"\n## {_sr.title}\n")
                    _md_parts.append(f"_section_id={_sr.section_id}_\n\n")
                    for _fs in _sr.slots:
                        for _c in _fs.claims:
                            _md_parts.append(
                                f"- **[{_fs.confidence}]** {_c.value} "
                                f"([source]({_c.source_url}))\n"
                            )
                # L1 小结追加到每节末尾
                for _sm in _summaries:
                    _md_parts.append(
                        f"\n### 小结 — {_sm.section_id}\n"
                        f"{_sm.summary_md}\n"
                        f"_来源 slots: {', '.join(_sm.source_slot_ids[:5])}{'…' if len(_sm.source_slot_ids) > 5 else ''}_\n"
                    )
                # L2 跨切
                if _crosscuts:
                    _md_parts.append("\n## 跨切分析\n")
                    for _cc in _crosscuts:
                        _tag = "结构性判断" if _cc.is_structural_judgment else "对照表"
                        _md_parts.append(f"\n### [{_tag}] {_cc.crosscut_id}\n")
                        _md_parts.append(f"{_cc.md}\n")
                        _md_parts.append(
                            f"_source sections: {', '.join(_cc.source_section_ids)}_"
                            f"{' / reconciliation_ids: ' + ', '.join(_cc.reconciliation_ids) if _cc.reconciliation_ids else ''}_\n"
                        )
                # R10:未解析 sections 显式标注(不留无声空标题)
                if out.unresolved_sections:
                    _md_parts.append("\n## 数据不足的章节 (unresolved)\n")
                    _md_parts.append(
                        "以下章节无 EU 流入(可能原因:source_router 未命中 / "
                        "ontology 缺覆盖 / 预期 numeric 但 LLM 未抽到数字):\n\n"
                    )
                    for _u in out.unresolved_sections:
                        _md_parts.append(
                            f"- **{_u['title']}** `({_u['section_id']})` — {_u['reason']}\n"
                        )
                out.assembled_markdown = "".join(_md_parts)
                logger.info(
                    "[W9/W10] assembled long markdown: %d chars, %d summaries, %d crosscuts, exec=%s",
                    len(out.assembled_markdown), len(_summaries), len(_crosscuts),
                    "yes" if _exec else "no",
                )
            except Exception as assemble_e:
                logger.warning("[§5] assemble long markdown failed: %s", assemble_e)
                out.degraded = True
                out.passed = False
        except Exception as w9_w10_e:
            import traceback as _w9_tb
            logger.warning(
                "[W9/W10] reduce-tree stage skipped: %s\n%s",
                w9_w10_e, _w9_tb.format_exc(),
            )
            out.degraded = True
            out.passed = False

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
            _rr = run_honest_pass(
                _rr, _claims_v3,
                reconciliations=out.reconciliations,  # type: ignore[arg-type]
                # §6 扩展:把 L1/L2/L3 综合层产物喂给 honest_pass 做溯源审计
                section_summaries=out.section_summaries,
                crosscuts=out.crosscuts,
                exec_summary=out.exec_summary,
            )
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
            # R8 ★:任一层 degraded → 全局 passed=False。
            # 之前的修复只在 W9/W10 综合层 is_stub 触发时设,但 W3 R6 fallback
            # (llm_extractor 异常/llm=None)也会 degraded,必须联动到 passed。
            and not out.degraded
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
