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
]
