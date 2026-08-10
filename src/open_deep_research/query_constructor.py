"""QueryConstructor — 一站式把 research brief + planner sub-topic 转成精确 SearchQuery。

## Why a single module (instead of separated Intent + Capability + Rewriter layers)

Earlier analysis proposed 5 layered modules (IntentClassifier → CapabilityResolver →
Rewriter → ExecutionPlan). That works for brownfield where each layer can be A/B
tested. In our pipeline each SearchQuery is consumed by exactly one gate (SearXNG
provider → eu_extractor → verifier) before its effect is observable; partial state
between layers risks dirty partial ExecutionPlan leaking to the next run.

This module collapses the chain into ONE function: `construct(...)`. It performs
intent classification + capability resolution + query rewriting in a single LLM
call. Failures fall back to a single deterministic intent, surfaced loudly.

## Public surface

  - `QueryIntent`        Pydantic schema — one SearXNG profile
  - `ExecutionPlan`      Pydantic schema — collection of intents + rationale
  - `construct(brief, sub_topic, *, primary_provider) → ExecutionPlan`
  - `to_search_query(execution_plan, *, sub_topic, run_id) → list[SearchQuery]`
  - `apply_to_search_query(intent, *, base) → SearchQuery` — single-intent projection

## Cache

Per-sub-topic hash → ExecutionPlan cache, TTL 1h. Cache invalidates implicitly
when research_brief OR sub_topic changes. Cache key is `(brief_hash, sub_topic_hash)`.

## Behavior contract

1. The `primary_provider` argument must implement `name` attribute (str).
   We accept SearXNGProvider (only backend configured today) and TavilyProvider
   (kept for future); the `extras` map is universal across both but only
   SearXNGProvider currently honors it — TavilyProvider reads topic/max_results.

2. On LLM failure (timeout / parse error) we log loudly and fall back to
   a single-intent ExecutionPlan that is comparable to the legacy
   `SearchQuery(queries=[st.question], topic="general", max_results=5)`
   but with `engines=[bing, wikipedia, arxiv]` and `language=auto` so we still
   improve over the all-arxiv-dominant baseline.

3. No silent degradation to empty extras — if LLM fails for one sub_topic we
   fall back to deterministic defaults that are *strictly better than the
   pre-QC baseline*.

## Forward compatibility

The `extras` dict on SearchQuery was added in P1 already (search_providers.py).
SearXNGProvider now reads four keys (`engines`, `categories`, `language`,
`time_range`) and forwards them as SearXNG URL parameters. Adding a fifth
means one line in `SearXNGProvider.search()` and one Pydantic field here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Schemas
# =============================================================================

# Subset of SearXNG `topic` parameter values we exercise.
# SearXNG also accepts: images, video, files, social media, map, music.
VALID_TOPICS = frozenset({
    "general", "news", "science", "it", "images", "video",
    "files", "social media", "map", "music",
})

# Subset of SearXNG `categories` we exercise.
# See SearXNG settings.yml -> categories.general / news / science / it / etc.
VALID_CATEGORIES = frozenset({
    "general", "news", "science", "it", "images", "videos",
    "social media", "maps", "music",
})

# Valid SearXNG `language` values per SearXNG docs.
# 'auto' tells SearXNG to detect from query; 'all' bypasses filter.
VALID_LANGUAGES = frozenset({"auto", "en", "zh-CN", "all"})
VALID_TIME_RANGES = frozenset({"day", "week", "month", "year"})

# U0 ★ (2026-08-10 验证 v34 + v35): SearXNG 1.x engine names 用空格分隔
# (e.g. "semantic scholar" 而不是 "semantic_scholar")。客户端传 underscore
# 版本会被 SearXNG 静默 fallback 到默认 engines,导致 engines= 整个被忽略。
# Keep in lock-step with deploy/searxng/settings.yml.
SearXNG_CONFIGURED_ENGINES = frozenset({
    "bing", "brave", "chinaso news",
    "arxiv", "openalex", "semantic scholar", "pubmed",
    "wikipedia", "wikidata",
    "360search", "sogou",
    "duckduckgo", "startpage", "mojeek", "qwant",
})


# =============================================================================
# P2: Chinese EDR vendor site: whitelist
# =============================================================================
# Why: EDR market research for Chinese vendors (Qihoo 360 / Sangfor / NSFOCUS /
# DBAPP / Venustech / Qi An Xin / Topsec / Nsfocus / DBCloud) is dominated by
# academic arxiv hits when using general engines. Injecting `site:` operators
# forces SearXNG to surface vendor-domain content — arxiv alone cannot answer
# "what is X product roadmap in 2024".
#
# Applied as a SECONDARY intent so the LLM-driven primary intent still works
# freely; the whitelist ensures the brief gets at least one vendor-source hit.
# We auto-trigger on Chinese locale (≥30% CJK chars in brief) OR explicit
# vendor tokens in the brief (case-insensitive).

CN_EDR_VENDOR_DOMAINS = frozenset({
    # Curated short list — site: queries must fit in ≤120 chars combined
    # with the base token(s). Keep ≤6 to stay under SearXNG URL limit.
    "qihoo.com",
    "sangfor.com",
    "nsfocus.com",
    "qax.com.cn",
    "qianxin.com",
    "venustech.com.cn",
})

_VENDOR_NAME_TOKENS = (
    "qihoo", "奇安信", "qianxin", "sangfor", "深信服", "nsfocus", "绿盟",
    "dbappsecurity", "dbapp", "安恒", "venustech", "启明星辰", "topsec",
    "360", "天融信", "天融信", "奇安信",
)


def _brief_is_chinese_cyber(brief: str) -> bool:
    """Return True if brief looks Chinese-cyber-market focused.

    Triggers site: whitelist injection when ANY of:
      - ≥30% CJK characters in the brief (CN locale)
      - explicit CN vendor tokens appear
      - keywords: '厂商' / '中国' / '国内' / '本土' / '本土厂商'
    """
    if not brief:
        return False
    cn = sum(1 for ch in brief if "\u4e00" <= ch <= "\u9fff")
    if cn / max(1, len(brief)) >= 0.30:
        return True
    if any(tok in brief for tok in _VENDOR_NAME_TOKENS):
        return True
    if any(kw in brief for kw in ("厂商", "中国", "国内", "本土")):
        return True
    return False


def _vendor_site_query(brief: str, sub_topic_question: str) -> str:
    """Build a SearXNG query that constrains results to CN vendor domains.

    `SearXNG` understands `site:` natively; the result is one OR over our
    whitelist domains with the brief's most distinctive tokens.

    Examples:
      _vendor_site_query("EDR ...", "EDR ... market size")
        → '(EDR OR 终端安全) (site:qihoo.com OR site:sangfor.com OR ...)

    """
    base_tokens = []
    for tok in ("EDR", "终端安全", "终端检测", "endpoint detection"):
        if tok in brief or tok in sub_topic_question:
            base_tokens.append(tok)
    if not base_tokens:
        base_tokens.append("EDR")
    base_expr = " OR ".join(base_tokens[:3])
    site_expr = " OR ".join(f"site:{d}" for d in sorted(CN_EDR_VENDOR_DOMAINS))
    # Wrap each site: clause in parens so SearXNG parses as OR correctly
    site_expr = f"({site_expr})"
    return f"({base_expr}) {site_expr}"


class QueryIntent(BaseModel):
    """One SearXNG search profile = one SearchQuery to issue."""

    queries: list[str] = Field(min_length=1, max_length=3)
    topic: str = "general"
    language: str = "auto"
    categories: list[str] = Field(default_factory=lambda: ["general"])
    engines: list[str] = Field(default_factory=lambda: ["bing", "wikipedia", "arxiv"])
    time_range: Optional[str] = None
    max_results: int = Field(default=10, ge=5, le=50)
    expected_yield: str = "best-effort"

    @field_validator("topic")
    @classmethod
    def _topic_in_set(cls, v: str) -> str:
        if v not in VALID_TOPICS:
            raise ValueError(f"topic={v!r} not in SearXNG-valid set {sorted(VALID_TOPICS)}")
        return v

    @field_validator("language")
    @classmethod
    def _lang_in_set(cls, v: str) -> str:
        if v not in VALID_LANGUAGES:
            raise ValueError(f"language={v!r} not in SearXNG-valid set {sorted(VALID_LANGUAGES)}")
        return v

    @field_validator("time_range")
    @classmethod
    def _time_range_in_set(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_TIME_RANGES:
            raise ValueError(f"time_range={v!r} not in SearXNG-valid set {sorted(VALID_TIME_RANGES)}")
        return v

    @field_validator("queries", mode="before")
    @classmethod
    def _queries_sane(cls, v):
        """Run BEFORE the schema-level min/max length check so we can drop
        blank entries first. Returns a list of ≤3 non-blank strings."""
        if not isinstance(v, list):
            return v  # let Pydantic raise the type error
        cleaned = []
        for q in v:
            if not isinstance(q, str):
                return v  # let Pydantic raise the type error
            if len(q) > 200:
                raise ValueError(f"query too long ({len(q)} chars): {q[:80]!r}...")
            s = q.strip()
            if s:
                cleaned.append(s)
        return cleaned[:3]  # hard cap at 3 after blanks dropped

    @field_validator("categories")
    @classmethod
    def _cats_sane(cls, v: list[str]) -> list[str]:
        for c in v:
            if c not in VALID_CATEGORIES:
                raise ValueError(f"category={c!r} not in SearXNG-valid set {sorted(VALID_CATEGORIES)}")
        # de-dup preserving order
        seen = set()
        out = []
        for c in v:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    @field_validator("engines")
    @classmethod
    def _engines_sane(cls, v: list[str]) -> list[str]:
        for e in v:
            if e not in SearXNG_CONFIGURED_ENGINES:
                raise ValueError(
                    f"engine={e!r} not configured in SearXNG. Configured: {sorted(SearXNG_CONFIGURED_ENGINES)}"
                )
        seen = set()
        out = []
        for e in v:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out


class ExecutionPlan(BaseModel):
    """An ordered set of intents to be issued in sequence by plan_v2 pipeline."""

    version: int = 1
    rationale: str = ""
    intents: list[QueryIntent] = Field(min_length=1, max_length=4)
    source: str = "deterministic"   # 'deterministic' | 'llm' | 'cache'

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# =============================================================================
# Helpers
# =============================================================================

def _strip_json(text: str) -> str:
    """Strip code fences and leading/trailing text — keep only JSON object."""
    text = text.strip()
    if text.startswith("```"):
        # Strip opening fence line.
        nl = text.find("\n")
        if nl > 0:
            text = text[nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    # Find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text


def _detect_locale(brief: str) -> str:
    """Rough locale detection — Chinese chars ≥30% → zh-CN, else en.

    No NLP dependency; just char-class counting.
    """
    if not brief:
        return "en"
    cn = sum(1 for ch in brief if "\u4e00" <= ch <= "\u9fff")
    return "zh-CN" if cn / max(1, len(brief)) >= 0.30 else "en"


def _sanitize_query_for_searxng(q: str) -> str:
    """In-process sanitizer — mirrors search_providers._sanitize_query.

    Kept duplicated here so this module has zero cross-import with
    search_providers (avoids a cycle through __init__.py if you import
    query_constructor at module top-level from the pipeline).
    """
    if not q:
        return q
    out = q
    # Drop punctuation SearXNG chokes on, keeping - _ . ' , ? !
    import re
    out = re.sub(r"[:\(\)\[\]\{\}\\/\u2014\u2013\u2018\u2019\u201C\u201D\u00B7\u2026\u00A0]",
                 " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > 120:
        out = out[:120].rstrip()
    return out or q


def _apply_indicator_hollows(
    queries: list[str], ontology: Optional[Any] = None,
) -> list[str]:
    """Disambiguate queries using ontology.indicator_hollows.

    Used by the 4-layer plan (post-architecture-v2). The ontology's
    `indicator_hollows.<key>.exclude_terms` and `positive_terms` are
    appended to each query as AND NOT / AND filters, so SearXNG drops
    polysemy hits (e.g. EDR = Endothelium-Dependent Relaxation) and
    reinforces the intended sense (EDR = Endpoint Detection and Response).

    For an EDR + cybersecurity brief, each query becomes:
      "<original> 终端检测响应 NOT \"Endothelium\""
    (bounded to 200 chars to fit SearXNG URL limit).

    If ontology is None, queries pass through unchanged.
    """
    if not ontology or not queries:
        return queries

    hollows = getattr(ontology, "indicator_hollows", None)
    if not hollows:
        return queries

    out: list[str] = []
    for q in queries:
        # Pick the hollow whose positive_terms overlap the query
        chosen = None
        q_lower = q.lower()
        for hk, hv in hollows.items():
            positives = hv.get("positive_terms", []) if isinstance(hv, dict) else []
            if any(t.lower() in q_lower for t in positives):
                chosen = (hk, hv)
                break
        if not chosen:
            out.append(q)
            continue
        hk, hv = chosen
        positives = hv.get("positive_terms", [])
        excludes = hv.get("exclude_terms", [])
        if not positives and not excludes:
            out.append(q)
            continue
        # Append first positive (Chinese if any) to disambiguate, then
        # first 2 English exclude terms as NOT.
        extras: list[str] = []
        cn_pos = next((p for p in positives if any("\u4e00" <= c <= "\u9fff" for c in p)), None)
        if cn_pos:
            extras.append(cn_pos)
        for ex in excludes[:2]:
            # Wrap multi-word in quotes for SearXNG NOT semantics
            if " " in ex:
                extras.append(f'NOT "{ex}"')
            else:
                extras.append(f"NOT {ex}")
        augmented = f"{q} " + " ".join(extras)
        if len(augmented) > 200:
            augmented = augmented[:200].rstrip()
        out.append(augmented)
    return out


def _dimension_to_default_engines(dimension_id: Optional[str]) -> tuple[list[str], list[str], str]:
    """Return (engines, categories, topic) per-dimension defaults.

    Single source of truth — used both in deterministic fallback and to
    sanity-check LLM output (LLM overrides freely; we only fall back).

    P0 fix: drop arxiv/openalex from the default engines for vendor /
    market-research dimensions. Academic engines pollute results with
    EDR-disambiguation noise (Early Data Release, Energy Demand Reduction,
    Event Data Recorder) that has nothing to do with cybersecurity. The
    academic engines are still allowed in LLM output but only when the
    dimension is purely descriptive (performance / ethics).
    """
    if dimension_id == "market_size":
        return (
            # U0 ★ (2026-08-10 验证 v34 + v35): SearXNG 1.x engine name 是带空格的
            # "semantic scholar"(不是 "semantic_scholar")。SearXNG 1.x 收到
            # 不识别的 engine name 静默 fallback 到默认 engines (arxiv),
            # 导致 engines= 整个被忽略。中国 EDR site: 召回必须靠 360search;
            # bing 在 zh-CN site:qianxin.com query 翻译成英文"EDR market size"
            # 返 emarketer live commerce。360search (so.com) 对中国厂商域名
            # 完美命中(实测 7/7 qianxin EDR 结果),bing 作为英文学术/全球冗余。
            # P0 fix: dropped arxiv/wikidata from primary; bing + 360search +
            # chinaso + wikipedia cover vendor / news / market data; brave as fallback.
            # 临时绕过 brave 限流(v24 实测 brave:Suspended too many requests):
            # 加 semantic scholar 兜底学术源。
            ["bing", "360search", "chinaso news", "wikipedia", "semantic scholar"],
            ["general", "news"],
            "general",
        )
    if dimension_id == "adoption":
        return (
            ["bing", "360search", "chinaso news", "wikipedia", "semantic scholar"],
            ["general", "news"],
            "general",
        )
    if dimension_id == "regulation":
        return (
            ["bing", "360search", "chinaso news"],
            ["news"],
            "news",
        )
    if dimension_id == "performance":
        # Pure technical/academic — keep arxiv/openalex here.
        return (
            ["arxiv", "openalex", "semantic scholar"],
            ["science", "general"],
            "science",
        )
    if dimension_id == "ethics":
        return (
            ["wikipedia", "wikidata", "pubmed"],
            ["general", "science"],
            "general",
        )
    # Default / context — broader sweep but bias toward news/vendor sources.
    # U0 ★:加 360search — 中国厂商 site: 召回关键源
    return (
        ["bing", "360search", "chinaso news", "wikipedia", "brave"],
        ["general", "news"],
        "general",
    )


def _deterministic_fallback(brief: str, sub_topic: Any) -> ExecutionPlan:
    """Deterministic fallback when LLM fails.

    Goals:
      - Strictly better than the legacy `SearchQuery(queries=[st.question], topic="general", max_results=5)`.
      - Use SearXNG filters so bing/chinaso can answer vendor questions and arxiv
        still surfaces academic context (matches user's earlier diagnostic).
      - Locale-aware (zh-CN if brief is mostly Chinese).
      - dimension_id aware (engines/categories match dimension).
      - P2 fix: when brief is Chinese-cyber focused, emit a SECOND intent that
        constrains results to CN vendor domains (site:qihoo.com OR ...). This
        compensates for arxiv dominating when no constraint is set.
    """
    locale = _detect_locale(brief)
    engines, categories, topic = _dimension_to_default_engines(getattr(sub_topic, "dimension_id", None))
    question = getattr(sub_topic, "question", "") or brief
    sanitized = _sanitize_query_for_searxng(question)

    intents: list[QueryIntent] = [
        QueryIntent(
            queries=[sanitized],
            topic=topic,
            language=locale,
            categories=categories,
            engines=engines,
            time_range=None,
            max_results=10,
            expected_yield="deterministic-best-effort",
        ),
    ]

    # P2: Chinese-cyber brief → emit a secondary site:-constrained intent so
    # SearXNG surfaces at least some CN vendor content. Without this, all
    # EUs come from arxiv + doi.org which is the wrong source for market
    # research (academic papers on "EDR" = Early Data Release / Energy
    # Demand Reduction, NOT endpoint security).
    if _brief_is_chinese_cyber(brief):
        vendor_q = _vendor_site_query(brief, question)
        # Skip _sanitize_query_for_searxng — `site:` operator requires colons,
        # parens are needed for OR-grouping, and the query is hand-crafted.
        # Truncate to 120 chars to stay within SearXNG's limit (we keep
        # only the most important site: clauses if it overflows).
        if len(vendor_q) > 120:
            vendor_q = vendor_q[:120].rstrip()
        intents.append(QueryIntent(
            queries=[vendor_q],
            topic="general",
            language="zh-CN",
            categories=["general", "news"],
            # U0 ★:加 360search — 中国厂商 site: 召回关键源(v34 实测 360search
            # 返 7/7 qianxin EDR 结果,bing 翻译坏了返 emarketer live commerce)。
            # SearXNG 1.x engine name: "semantic scholar" (空格) — 下划线版本
            # 被 SearXNG 静默 fallback 到默认 engines,会让整个 engines= 失效。
            # 临时绕过 brave 限流(供应商站查询 v24):bing + chinaso + wikipedia
            # 仍能命中厂商官网 / 新闻源。
            engines=["bing", "360search", "chinaso news", "wikipedia", "semantic scholar"],
            time_range=None,
            max_results=15,
            expected_yield="CN vendor site:-whitelist",
        ))

    return ExecutionPlan(
        rationale="deterministic fallback (LLM failed or disabled)",
        intents=intents,
        source="deterministic",
    )


# =============================================================================
# Cache (in-process, TTL 1h)
# =============================================================================

@dataclass
class _PlanCacheEntry:
    plan: ExecutionPlan
    expires_at: float

_CACHE: dict[tuple[str, str], _PlanCacheEntry] = {}
_CACHE_TTL_SECONDS = 3600

# Master switch — set `OPEN_DEEP_RESEARCH_NO_QC=1` to disable QueryConstructor
# entirely (legacy `SearchQuery(queries=[st.question], topic="general", max_results=5)`
# behavior). Useful for back-to-back regression runs. Read each call so
# env-var changes during tests are honored.
def _qc_disabled() -> bool:
    return bool(os.environ.get("OPEN_DEEP_RESEARCH_NO_QC"))


def _brief_hash(brief: str) -> str:
    return "b-" + hashlib.sha256((brief or "").encode("utf-8")).hexdigest()[:16]


def _sub_topic_hash(sub_topic: Any) -> str:
    # P0 fix: parens around EVERY getattr call so the `or ""` fallbacks
    # only apply to their own field, not silently consume the whole
    # subsequent string concat via Python's higher-precedence `+`.
    # Pre-fix bug: 3 distinct FakeST(title="performance", ...) instances
    # all produced the same hash because their `id="st-fake"` was truthy
    # and short-circuited the entire expression to just `"st-fake"`.
    seed = (
        (getattr(sub_topic, "id", "") or "")
        + "|"
        + (getattr(sub_topic, "title", "") or "")
        + "|"
        + (getattr(sub_topic, "question", "") or "")
        + "|"
        + (getattr(sub_topic, "dimension_id", "") or "")
    )
    return "st-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _cache_get(key: tuple[str, str]) -> Optional[ExecutionPlan]:
    import time
    e = _CACHE.get(key)
    if e is None:
        return None
    if e.expires_at < time.time():
        _CACHE.pop(key, None)
        return None
    p = e.plan.model_copy(deep=True)
    p.source = "cache"
    return p


def _cache_put(key: tuple[str, str], plan: ExecutionPlan) -> None:
    import time
    _CACHE[key] = _PlanCacheEntry(
        plan=plan.model_copy(deep=True),
        expires_at=time.time() + _CACHE_TTL_SECONDS,
    )


def invalidate_cache() -> int:
    n = len(_CACHE)
    _CACHE.clear()
    return n


# =============================================================================
# LLM construction
# =============================================================================

async def construct_with_llm(
    brief: str,
    sub_topic: Any,
    *,
    llm: Optional[Any] = None,
    config: Optional[Any] = None,
    temperature: float = 0.0,
) -> ExecutionPlan:
    """Issue one LLM call to translate (brief, sub_topic) → ExecutionPlan JSON.

    `llm` may be None, in which case we lazily call `get_llm(role="query_constructor")`.
    `config` may be None; defaults are read from `Configuration.from_runnable_config()`.
    """
    from open_deep_research.configuration import Configuration
    from open_deep_research.llm import get_llm, get_prompt, get_prompt_version, _get_langfuse

    system_prompt = get_prompt("query_constructor")
    prompt_version = get_prompt_version("query_constructor")
    if llm is None:
        if config is None:
            configurable = Configuration.from_runnable_config(None)
            model_name = (configurable.research_model
                          or configurable.summarization_model
                          or "minimax:MiniMax-M3")
            config = {"configurable": {"model": model_name}}
        llm = get_llm(config=config, tags=["langsmith:nostream"])

    # Build the user message — structured, not freeform prose.
    user_msg = (
        f"research_brief:\n```\n{brief}\n```\n\n"
        f"sub_topic.id: {getattr(sub_topic, 'id', '')}\n"
        f"sub_topic.title: {getattr(sub_topic, 'title', '')}\n"
        f"sub_topic.question: {getattr(sub_topic, 'question', '')}\n"
        f"sub_topic.dimension_id: {getattr(sub_topic, 'dimension_id', '')}\n"
        f"sub_topic.expected_entities: {getattr(sub_topic, 'expected_entities', [])}\n\n"
        f"Return JSON only."
    )

    logger.info(
        "query_constructor.construct: brief_hash=%s sub_topic=%s dimension=%s locale=%s",
        _brief_hash(brief),
        getattr(sub_topic, "title", ""),
        getattr(sub_topic, "dimension_id", None),
        _detect_locale(brief),
    )

    # Optional Langfuse span for observability (matches the codebase pattern).
    lf = _get_langfuse()
    raw = None
    try:
        if lf is not None:
            tracer = lf._otel_tracer
            with tracer.start_as_current_span("query_constructor"):
                lf.update_current_span(metadata={
                    "prompt_role": "query_constructor",
                    "prompt_version": prompt_version,
                    "dimension_id": getattr(sub_topic, "dimension_id", None),
                })
                raw = await _ainvoke_llm(llm, system_prompt, user_msg, temperature)
        else:
            raw = await _ainvoke_llm(llm, system_prompt, user_msg, temperature)
    except Exception as e:
        logger.warning("query_constructor LLM call failed: %s", e)
        return _deterministic_fallback(brief, sub_topic)

    if not isinstance(raw, str) or not raw.strip():
        logger.warning("query_constructor LLM returned empty; falling back")
        return _deterministic_fallback(brief, sub_topic)

    try:
        text = _strip_json(raw)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object, got {type(data).__name__}")
        # Tag source; Pydantic will raise if schema violated.
        data["source"] = "llm"
        plan = ExecutionPlan(**data)
        return plan
    except Exception as e:
        logger.warning("query_constructor LLM JSON invalid (%s); raw=%r", e, raw[:300])
        return _deterministic_fallback(brief, sub_topic)


async def _ainvoke_llm(llm: Any, system: str, user: str, temperature: float) -> str:
    """Wrap a LangChain chat model invoke.

    Tries `ainvoke(messages)` first, falls back to sync `invoke` on thread if absent.
    Returns the message content string.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    if hasattr(llm, "ainvoke"):
        try:
            out = await llm.ainvoke(msgs, temperature=temperature)
        except TypeError:
            # Some LangChain LLMs don't accept temperature kwarg.
            out = await llm.ainvoke(msgs)
    else:
        # Fall back to sync invoke in a thread.
        import asyncio
        def _sync():
            try:
                return llm.invoke(msgs, temperature=temperature)
            except TypeError:
                return llm.invoke(msgs)
        out = await asyncio.get_event_loop().run_in_executor(None, _sync)
    if hasattr(out, "content"):
        return out.content
    return str(out)


# =============================================================================
# Public entry point
# =============================================================================

async def construct(
    brief: str,
    sub_topic: Any,
    *,
    primary_provider: Optional[Any] = None,
    config: Optional[Any] = None,
) -> ExecutionPlan:
    """Top-level entry point — return ExecutionPlan for one (brief, sub_topic).

    Honors `OPEN_DEEP_RESEARCH_NO_QC=1` (legacy behavior) and the in-process cache.
    """
    if _qc_disabled():
        # Legacy: no extras, topic=general, max_results=5.
        legacy_intent = QueryIntent(
            queries=[_sanitize_query_for_searxng(getattr(sub_topic, "question", "") or brief)],
            topic="general",
            language="auto",
            categories=["general"],
            engines=["bing", "360search", "wikipedia", "arxiv"],
            time_range=None,
            max_results=5,
            expected_yield="legacy-baseline",
        )
        return ExecutionPlan(
            rationale="OPEN_DEEP_RESEARCH_NO_QC=1 → legacy behavior",
            intents=[legacy_intent],
            source="deterministic",
        )

    key = (_brief_hash(brief), _sub_topic_hash(sub_topic))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        plan = await construct_with_llm(brief, sub_topic, config=config)
    except Exception as e:
        # Defence in depth — construct_with_llm already catches internally
        # and returns a deterministic fallback, but if the inner dispatch
        # itself raises (e.g. cold-import error, rate-limit panic), we still
        # want a usable plan rather than a stack trace.
        logger.warning(
            "construct_with_llm raised for sub_topic=%s: %s; "
            "falling back to deterministic profile",
            getattr(sub_topic, "title", "?"), e,
        )
        plan = _deterministic_fallback(brief, sub_topic)
    _cache_put(key, plan)
    return plan


# =============================================================================
# Projection to SearchQuery
# =============================================================================

def apply_to_search_query(
    intent: QueryIntent,
    *,
    run_id: Optional[str] = None,
    sub_topic: Any = None,
) -> Any:
    """Project one QueryIntent + sub_topic metadata into a SearchQuery.

    `SearchQuery` lives in search_providers; we import lazily to avoid a
    top-level cycle if you ever flip the package init order.
    """
    from open_deep_research.search_providers import SearchQuery

    extras: dict[str, Any] = {
        "engines": intent.engines,
        "categories": intent.categories,
        "language": intent.language,
    }
    # P4 fix: SearXNG with time_range='year' returns 0 results across the
    # board (independent of language / engines) — verified by curl probes:
    #   language=auto + engines=bing + time_range=year  → 0 results
    #   language=auto + engines=bing + (no time_range)  → 10 results
    #   language=en   + engines=bing,wikipedia + time_range=year → 0 results
    #   language=en   + engines=bing,wikipedia + (no time_range) → 10 results
    # 'month' and 'day' work fine for narrow queries; only 'year' is broken.
    # Auto-drop time_range='year' unconditionally so result counts recover.
    # (Bug surfaced by EDR brief v3 run: market_size/adoption/regulation
    #  sub_topics returned 0 EU — root cause was this 0-result fetch.)
    if intent.time_range is not None:
        # Drop time_range if it's 'year' (always) OR if it's 'zh-CN + non-news
        # engines' (the original P3 case — kept for backwards compat).
        engines_set = set(intent.engines or [])
        news_capable = bool(engines_set & {"bing_news", "chinaso news", "google news", "duckduckgo news"})
        if intent.time_range == "year":
            logger.warning(
                "dropping time_range=year for SearXNG (time_range=year returns 0 results regardless of language/engines)"
            )
        elif intent.language == "zh-CN" and not news_capable:
            logger.warning(
                "dropping time_range=%s for SearXNG (zh-CN + non-news engines = 0 results)",
                intent.time_range,
            )
        else:
            extras["time_range"] = intent.time_range

    return SearchQuery(
        queries=list(intent.queries),
        topic=intent.topic,
        max_results=intent.max_results,
        run_id=run_id,
        research_topic=(getattr(sub_topic, "title", None) if sub_topic is not None else None),
        extras=extras,
    )


def to_search_query(
    plan: ExecutionPlan,
    *,
    run_id: Optional[str] = None,
    sub_topic: Any = None,
) -> list[Any]:
    """Project a full ExecutionPlan → list[SearchQuery] (one per intent)."""
    return [
        apply_to_search_query(intent, run_id=run_id, sub_topic=sub_topic)
        for intent in plan.intents
    ]


# =============================================================================
# Provider / brief-kind classifier (cheap utility, no LLM)
# =============================================================================

def configured_providers_for(provider: Any) -> dict[str, Any]:
    """Return a {provider_name: capability_summary} dict for logging/metadata.

    Today we only have SearXNGProvider; kept as a fixed schema for forward
    compat (TavilyProvider may join later).
    """
    name = getattr(provider, "name", "unknown")
    if name == "searxng":
        return {
            "name": "searxng",
            "engines_configured": sorted(SearXNG_CONFIGURED_ENGINES),
            "supports_extras": True,
        }
    return {"name": name, "supports_extras": False}


__all__ = [
    "QueryIntent",
    "ExecutionPlan",
    "construct",
    "construct_with_llm",
    "apply_to_search_query",
    "to_search_query",
    "configured_providers_for",
    "invalidate_cache",
    "SearXNG_CONFIGURED_ENGINES",
    "VALID_TOPICS",
    "VALID_CATEGORIES",
    "VALID_LANGUAGES",
    "VALID_TIME_RANGES",
]
