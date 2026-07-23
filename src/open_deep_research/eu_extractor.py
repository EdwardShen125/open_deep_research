"""Phase 2.2: Deterministic EU extractor.

## Why deterministic, not LLM-based?

The v1 path was: tavily_search → LLM `summarize_webpage` → single string.
That single string then fed `compress_research`, which collapsed it into
another string via another LLM call. Two LLM hops in series is where:

- verbatim quotes vanish
- numeric values drift / round-trip
- entity relations get rewritten (Kompyte ⇆ Crayon ownership)

We replace the post-search LLM hop with this deterministic extractor. The
Tavily `summary` (which is itself an LLM summary, but at least has clear
"what the page says" grounding) is parsed into structured EvidenceUnits
that we can later cite. The writer still uses an LLM, but it now has a
typed evidence graph to draw from.

## Algorithm

For each Tavily/SearXNG/Crawl4AI search result dict:

  1. **Source registration**: upsert into `evidence.sources` via DAO so
     Phase 1.1 page-level filtering is applied (B-anchor compliance).
  2. **Sentence split**: split the result's `content` (or summary) on
     CJK + ASCII sentence boundaries.
  3. **EU per sentence**: each sentence becomes one EU candidate.
     - claim = sentence
     - quote = verbatim sentence (≤ 200 chars)
     - source_url / source_id / source_title from the result
     - numbers mined from the sentence
     - entities mined via a small keyword dict + heuristics
  4. **Confidence heuristic**: pages with more numbers / entities /
     specific dates → higher confidence (this is a weak signal; the
     real signal comes from Phase 3a verifier).
  5. **Dedup by content_hash**: same source + same sentence → one EU.
  6. **Domain gate**: if `page_level=False`, mark the EU with
     `confidence = min(confidence, 0.4)` and `extraction_method =
     "domain_only"`. The writer can still use it, but the verifier
     will downgrade.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from open_deep_research.evidence_units import (
    EvidenceUnit, NumberBinding, EntityRef, dedup_eus, extract_numbers,
)
from open_deep_research.sources_dao import SourcesDAO, SourceRecord, canonicalize_url


# =============================================================================
# Source tier classifier (P0 source_tier 真实分级)
# =============================================================================
# 4 个 tier (从 SourceTier Literal):
#   primary   — peer-reviewed 研究 / 官方财报 / 监管文件 (arxiv / SEC / 政府)
#   secondary — 行业新闻 / 媒体 / 厂商博客 (techcrunch / reuters / vendor blog)
#   tertiary  — 维基 / 百科 (wikipedia / britannica)
#   ugc       — 用户内容 / 论坛 (reddit / twitter / 个人博客)
#
# 数据准确性导向 (Runbook v1 §3.2): arxiv=primary 是 EDR 市场调研的最佳源
# (peer-reviewed 研究数据), wiki=tertiary 权重低, 论坛最低。
# 新增域名请加 _TIER_RULES,_classify_source_tier 自动匹配。

_TIER_RULES: list[tuple[str, str]] = [
    # (substring, tier) — 长 substring 排前 (避免 'gov' 抢 'reuters.com' 的 'co')
    # primary: peer-reviewed / 官方 / 监管
    ("arxiv.org", "primary"),
    ("europa.eu", "primary"),
    ("sec.gov", "primary"),
    (".gov", "primary"),
    (".edu", "primary"),
    # secondary: 行业媒体 / 厂商 blog / 主流新闻
    ("reuters.com", "secondary"),
    ("bloomberg.com", "secondary"),
    ("wsj.com", "secondary"),
    ("ft.com", "secondary"),
    ("nytimes.com", "secondary"),
    ("bbc.com", "secondary"),
    ("bbc.co.uk", "secondary"),
    ("cnn.com", "secondary"),
    ("theguardian.com", "secondary"),
    ("forbes.com", "secondary"),
    ("techcrunch.com", "secondary"),
    ("wired.com", "secondary"),
    ("zdnet.com", "secondary"),
    ("cnet.com", "secondary"),
    ("venturebeat.com", "secondary"),
    ("darkreading.com", "secondary"),
    ("threatpost.com", "secondary"),
    ("securityweek.com", "secondary"),
    ("cyberscoop.com", "secondary"),
    ("theregister.com", "secondary"),
    ("infosecurity-magazine.com", "secondary"),
    ("gartner.com", "secondary"),
    ("forrester.com", "secondary"),
    ("idc.com", "secondary"),
    ("mckinsey.com", "secondary"),
    # tertiary: 百科 / 知识库
    ("wikipedia.org", "tertiary"),
    ("wikimedia.org", "tertiary"),
    ("britannica.com", "tertiary"),
    ("baike.baidu.com", "tertiary"),
    # ugc: 论坛 / 社交 / 个人博客
    ("reddit.com", "ugc"),
    ("quora.com", "ugc"),
    ("twitter.com", "ugc"),
    ("x.com", "ugc"),
    ("facebook.com", "ugc"),
    ("linkedin.com", "ugc"),
    ("medium.com", "ugc"),
    ("substack.com", "ugc"),
]


def _classify_source_tier(url: str) -> str:
    """基于 url 域名真实分级 primary/secondary/tertiary/ugc。

    未知域名 -> 'secondary' (默认中位),保守可被下游 verifier 重新评估。
    """
    if not url:
        return "secondary"
    host = urlsplit(url).hostname or ""
    host = host.lower()
    for substr, tier in _TIER_RULES:
        if substr in host:
            return tier
    # 启发式兜底:未知 .org / .com 视为 secondary
    if host.endswith(".org") or host.endswith(".com") or host.endswith(".net"):
        return "secondary"
    return "secondary"


# =============================================================================
# Sentence splitter — handles CJK and ASCII punctuation
# =============================================================================

_SENTENCE_END_RE = re.compile(
    r"""
    (?<=[.!?])        \s+     # ASCII terminal + whitespace
    | (?<=[。！？])            # CJK terminal (zero-width split is fine)
    | (?<=[.!?。！？])"(?=\s) # quote after terminal punct
    | \n                      # hard line break
    """,
    re.VERBOSE,
)


def split_sentences(text: str) -> list[str]:
    """Split `text` into sentences, returning the cleaned non-empty pieces."""
    if not text:
        return []
    norm = text.replace("\n", "。")  # treat hard newlines as CJK sentence end
    parts = _SENTENCE_END_RE.split(norm)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 5 and not any(c.isalpha() for c in p):
            continue
        out.append(p)
    return out


# =============================================================================
# Entity mining (cheap heuristics — Phase 3a rule-2 re-verifies)
# =============================================================================

# Lightweight lexicon for the seed corpus. Phase 3a verifier adds a
# full NER layer. For extraction we just need *candidate* entities.
_ENTITY_LEXICON = {
    "company": [
        # CI vendor landscape
        "Crayon", "Klue", "Kompyte", "Autobound", "IndustryLens",
        "Contify", "Valona", "Comintelli", "Watchmycompetitor",
        "AlphaSense", "Tegus", "ZoomInfo", "6sense",
        "DataRobot",
        # Generic enterprise software
        "Salesforce", "HubSpot", "Microsoft", "Oracle", "SAP",
    ],
    "product": [
        # CI products
        "Kompyte vs Crayon and Klue", "Klue vs Crayon",
        "Competitive Intelligence Tools",
    ],
    "metric": [
        "TAM", "SAM", "SOM", "ARR", "MAU", "DAU",
    ],
    "person": [],   # populated as needed
}


def mine_entities(text: str, extra_terms: Optional[Iterable[str]] = None) -> list[EntityRef]:
    """Return entity references mentioned in `text` based on a lexicon.

    De-duplicated by (name, entity_type); case-sensitive matching (CI corpus
    uses proper-noun capitalization consistently).
    """
    out: list[EntityRef] = []
    seen = set()
    for etype, names in _ENTITY_LEXICON.items():
        for n in names:
            if n in text and (n, etype) not in seen:
                seen.add((n, etype))
                # Capture binding hints if available in same sentence.
                extra = {}
                # Special-case: "X acquired by Y" pattern
                m = re.search(
                    rf"{re.escape(n)}\s*(?:被\s*)?(?:by|was acquired by|acquired by)\s+([A-Z][\w\-]+)",
                    text,
                )
                if m:
                    extra["acquired_by"] = m.group(1)
                out.append(EntityRef(name=n, entity_type=etype, extra=extra))
    if extra_terms:
        for t in extra_terms:
            if t and t in text and (t, "company") not in seen:
                seen.add((t, "company"))
                out.append(EntityRef(name=t, entity_type="company"))
    return out


# =============================================================================
# Confidence heuristic
# =============================================================================

def _confidence_for(text: str, *, page_level: bool) -> float:
    """Cheap confidence score.

    Heuristics (additive, clipped to [0.15, 0.95]):
      - page_level=False           → -0.30 (downgrade)
      - contains a NumberBinding   → +0.10 (numeric grounding)
      - contains a year 19YY/20YY  → +0.05 (temporal grounding)
      - contains a currency word   → +0.05 (financial anchor)
      - length 80–500 chars        → +0.10 (substantial)
    """
    score = 0.55
    if not page_level:
        score -= 0.30
    if extract_numbers(text):
        score += 0.10
    if re.search(r"\b(19|20)\d{2}\b", text):
        score += 0.05
    if any(w in text for w in ("美元", "USD", "usd", "$", "€", "CNY", "元", "RMB")):
        score += 0.05
    ln = len(text)
    if 80 <= ln <= 500:
        score += 0.10
    return max(0.15, min(0.95, score))


# =============================================================================
# Extraction entry point
# =============================================================================

def extract_from_search_result(
    result: dict[str, Any],
    *,
    run_id: Optional[str] = None,
    sources_dao: Any = None,
    research_topic: Optional[str] = None,
    dimension_id: Optional[str] = None,
) -> list[EvidenceUnit]:
    """Convert one Tavily/SearXNG search result dict to one or more EUs.

    `result` keys used: 'url' (required), 'title', 'content' / 'summary',
    'raw_content' (preferred for full text), 'score'.

    Returns the list of new EUs (deduped by content_hash). If `sources_dao`
    is provided, the source row is registered first so the page-level flag
    is sourced from PG.
    """
    url = result.get("url")
    if not url:
        return []
    title = result.get("title")
    score = result.get("score")

    # 1. Source registration (idempotent — DAO upserts by url_hash)
    source_id: Optional[int] = None
    page_level = True  # optimistic — DAO may downgrade
    if sources_dao is not None:
        rec = SourceRecord.from_raw(
            url=url,
            title=title,
            provider=result.get("provider", "tavily"),
            provider_query=result.get("query"),
            provider_score=score,
            provider_payload={"score": score, "raw_keys": list(result.keys())},
            run_id=run_id,
            research_topic=research_topic,
        )
        try:
            source_id = sources_dao.upsert(rec)
        except Exception:
            # Don't fail the whole extraction — degrade gracefully.
            source_id = None
        # Re-read to get canonical page_level flag from PG classification.
        fetched = sources_dao.get_by_url(url)
        if fetched is not None:
            page_level = fetched.page_level

    # 2. Pick the candidate text to chunk into sentences.
    text_block = (
        result.get("raw_content")
        or result.get("summary")
        or result.get("content")
        or ""
    )
    if not text_block:
        # Fall back to title as a single-statement sentence so we still
        # capture *one* EU — better than dropping the source entirely.
        text_block = title or ""
    sentences = split_sentences(text_block)
    if not sentences and text_block:
        sentences = [text_block.strip()]

    eus: list[EvidenceUnit] = []
    # P0 source_tier 真实分级: 一次分类,所有 sentence EU 共享
    src_tier = _classify_source_tier(url)
    for sent in sentences:
        quote = (sent[:200] + "…") if len(sent) > 200 else sent
        nums = extract_numbers(sent)
        ents = mine_entities(sent)
        conf = _confidence_for(sent, page_level=page_level)
        method = "tavily_summary" if page_level else "domain_only"
        eu = EvidenceUnit(
            claim=sent,
            quote=quote,
            source_url=url,
            source_title=title,
            source_id=source_id,
            numbers=nums,
            entities=ents,
            confidence=conf,
            extraction_method=method,
            run_id=run_id,
            dimension_id=dimension_id,
            source_tier=src_tier,
        )
        eus.append(eu)
    return dedup_eus(eus)


def extract_from_search_results(
    results: Iterable[dict[str, Any]],
    *,
    run_id: Optional[str] = None,
    sources_dao: Any = None,
    research_topic: Optional[str] = None,
    dimension_id: Optional[str] = None,
) -> list[EvidenceUnit]:
    """Extract EUs from an iterable of search result dicts.

    ``dimension_id`` is stamped onto every EU produced so downstream
    PG persistence has a non-NULL dimension_id for query (run, dimension).
    """
    out: list[EvidenceUnit] = []
    for r in results:
        out.extend(extract_from_search_result(
            r, run_id=run_id, sources_dao=sources_dao,
            research_topic=research_topic, dimension_id=dimension_id,
        ))
    return dedup_eus(out)


# =============================================================================
# Re-export for convenience
# =============================================================================

__all__ = [
    "split_sentences", "mine_entities",
    "extract_from_search_result", "extract_from_search_results",
    "extract_numbers",
    "_classify_source_tier",  # P0 source_tier 真实分级
]
