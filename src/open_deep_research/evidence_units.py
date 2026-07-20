"""Phase 2.1: EvidenceUnit (EU) — the atomic evidence type for Plan v2.

## Why

v1 compress_research collapses everything into a `Summary` (summary: str,
key_excerpts: str). The writer then produces claims without any binding back
to which source the claim came from. This causes the v1 baseline defects:

- kompyte is written as "independent CI product" with no link to any EU that
  established ownership — fabrication gap.
- "$300/yr, $20K-$40K/yr" pricing numbers appear without any binding to the
  URL where they live — numeric gap.
- "8/59 references are domain-level URLs" — no per-claim URL enforcement.

EvidenceUnit is the **only** object the writer may cite. Each EU has:

- a stable id (assigned by `evidence_units` PG sequence + stored EU.content_hash)
- a *minimal* claim (one sentence)
- an optional verbatim quote — the proof the claim is grounded
- the URL it came from (via `source_id` → `evidence.sources`)
- a numeric binding (extracted from the source)
- an entity reference list
- a confidence score the extractor assigns

EUs go into:
- in-process memory as `ClaimUnit` dataclasses during a single run
- `evidence.evidence_units` table for cross-run persistence (Phase 2.3)

## Plan v2 EU schema (this module defines the Python side)

    EvidenceUnit:
        id: EU id, integer or UUID string
        claim: str                         # the minimal factual statement
        quote: Optional[str]               # ≤ 200 char verbatim from source
        source_id: Optional[int]           # → evidence.sources(id)
        source_url: str                    # denormalized for in-process use
        source_title: Optional[str]
        numbers: list[NumberBinding]       # numeric anchors
        entities: list[EntityRef]          # entity anchors (e.g. "Kompyte")
        confidence: float                  # 0..1 — extractor self-rated
        extraction_method: str             # 'tavily_summary' / 'manual' / 'crawl4ai'
        extracted_at: datetime
        extractions_run_id: Optional[str]  # LangGraph run_id

Supporting types:

    NumberBinding:
        text: str            # e.g. "30-60 亿美元"
        value_min: Optional[float]
        value_max: Optional[float]
        unit: Optional[str]
        is_estimated: bool   # true when text contains "约" / "估算" / "around"

    EntityRef:
        name: str
        entity_type: str     # 'company' / 'product' / 'metric' / 'person'
        extra: dict          # type-specific (e.g. {"acquired_by": "Crayon"})

The extractor (Phase 2.2) populates these. The verifier (Phase 3a) reads them.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


# =============================================================================
# NumberBinding
# =============================================================================

_NUMERIC_RE = re.compile(
    r"""
    (?P<neg>[-−])?                          # optional minus
    (?P<value>\d[\d,]*(?:\.\d+)?)           # first number
    (?:\s*(?:[\-–~])\s*                     # range connector
        (?P<value2>\d[\d,]*(?:\.\d+)?)
    )?
    \s*
    (?P<unit>[万亿千百百]|[%％])?             # trailing unit (Chinese magnitude OR %)
    """,
    re.VERBOSE,
)
_CHINESE_UNITS = {
    "万": 1e4,
    "亿": 1e8,
    "百": 1e2,
    "千": 1e3,
    "%": 1.0,
    "％": 1.0,
}


def _to_float(num_str: str) -> Optional[float]:
    if not num_str:
        return None
    s = num_str.replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def _scale(num: Optional[float], unit: Optional[str]) -> Optional[float]:
    if num is None:
        return None
    if not unit:
        return num
    return num * _CHINESE_UNITS.get(unit, 1.0)


@dataclass
class NumberBinding:
    """One numeric anchor extracted from a piece of evidence."""

    text: str                          # e.g. "30-60 亿美元" / "约 8-10 亿美元"
    value_min: Optional[float] = None  # scaled to base unit (e.g. USD)
    value_max: Optional[float] = None
    unit: Optional[str] = None         # 'USD' / 'RMB' / 'USD/year' / '' (no scaling)
    is_estimated: bool = False         # set if '约' / '估算' / 'rough' in text

    # -- parsing helpers --
    @classmethod
    def from_text(cls, text: str) -> "NumberBinding":
        """Heuristically extract a single range from CJK / English text.

        For multi-number text, this returns the FIRST match. Use
        `extract_numbers` for a full sweep.
        """
        m = _NUMERIC_RE.search(text or "")
        if not m:
            return cls(text=text, is_estimated=_has_estimate_marker(text))
        v1 = _to_float(m.group("value"))
        u1 = m.group("unit") or ""
        v2 = _to_float(m.group("value2"))
        # Apply the trailing unit only if it's a magnitude unit; for % we
        # don't multiply (it stays as a percentage).
        is_pct = u1 in ("%", "％")
        u2 = u1  # unit tied to value1 or value2 same in our regex
        vmin = v1 if v1 is not None else None
        vmax = (_to_float(m.group("value2")) if m.group("value2") else vmin)
        if vmin is not None and not is_pct:
            vmin = _scale(vmin, u1) if vmin is not None else None
            if vmax is not None and vmax == vmin:
                # same value, no second number → leave scaled value
                pass
            if vmax is not None and v2 is not None:
                vmax = _scale(v2, u1)
        if vmin is not None and vmax is not None and vmax < vmin:
            vmin, vmax = vmax, vmin

        # Decide semantic unit. Heuristic: presence of "美元" → USD,
        # "人民币" / "元" → RMB, otherwise leave None.
        if "美元" in text:
            unit = "USD"
        elif "人民币" in text or "人民币" in text:
            unit = "RMB"
        elif u1 in _CHINESE_UNITS and not is_pct:
            # ambiguous CNY unless context says otherwise — keep raw unit
            unit = u1
        else:
            unit = None
        return cls(
            text=text.strip(),
            value_min=vmin,
            value_max=vmax,
            unit=unit,
            is_estimated=_has_estimate_marker(text),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _has_estimate_marker(text: str) -> bool:
    if not text:
        return False
    return any(m in text for m in (
        "约", "估算", "大概", "大约", "大致", "估计",
        "approximately", "approximately", "around", "roughly", "estimated",
    ))


def extract_numbers(text: str) -> list[NumberBinding]:
    """Return every range discovered in `text`, ordered by appearance."""
    out: list[NumberBinding] = []
    for m in _NUMERIC_RE.finditer(text or ""):
        # Use the matched span as the number text
        span = m.group(0).strip()
        out.append(NumberBinding.from_text(span))
    return out


# =============================================================================
# EntityRef
# =============================================================================

@dataclass
class EntityRef:
    """An entity anchor: company / product / person / metric."""
    name: str
    entity_type: str               # 'company' / 'product' / 'metric' / 'person'
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# EvidenceUnit
# =============================================================================

# A short, stable id (12 hex chars) for EUs that aren't yet persisted.
def make_eu_id(content_hash: str) -> str:
    return f"eu-{content_hash[:12]}"


@dataclass
class EvidenceUnit:
    """The atomic evidence object that the writer may cite.

    Invariants:
      - `claim` is non-empty and ≤ 500 chars (we truncate longer).
      - If `quote` is set, it is verbatim from `source_url` (no edits).
      - If `source_id` is set, it points to a real `evidence.sources` row.
      - `confidence` ∈ [0.0, 1.0].
    """
    claim: str
    source_url: str
    quote: Optional[str] = None
    source_id: Optional[int] = None
    source_title: Optional[str] = None
    numbers: list[NumberBinding] = field(default_factory=list)
    entities: list[EntityRef] = field(default_factory=list)
    confidence: float = 0.5
    extraction_method: str = "tavily_summary"
    extracted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: Optional[str] = None
    id: Optional[str] = None  # populated by upsert

    def __post_init__(self):
        if not self.claim or not self.claim.strip():
            raise ValueError("EvidenceUnit.claim must be non-empty")
        if not self.source_url or not self.source_url.strip():
            raise ValueError("EvidenceUnit.source_url must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        # Truncate oversized claims with an ellipsis marker.
        if len(self.claim) > 500:
            self.claim = self.claim[:497] + "..."
        self.id = self.id or make_eu_id(self.content_hash)

    # -- derived --
    @property
    def content_hash(self) -> str:
        """Deterministic content hash for dedup & cross-run lookup.

        We strip `extracted_at` and `run_id` so that EUs extracted twice
        at slightly different times for the same source → same claim/quote
        still dedup to a single row in `evidence.evidence_units`.
        """
        payload = {
            "claim": (self.claim or "").strip(),
            "quote": (self.quote or "").strip() if self.quote else None,
            "url": (self.source_url or "").strip(),
            "title": (self.source_title or "").strip() if self.source_title else None,
            "numbers": [n.text for n in self.numbers],
            "entities": sorted(
                (e.name, e.entity_type) for e in self.entities
            ),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # -- (de)serialize --
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["numbers"] = [n.to_dict() for n in self.numbers]
        d["entities"] = [e.to_dict() for e in self.entities]
        d["extracted_at"] = self.extracted_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceUnit":
        numbers = [NumberBinding(**n) for n in d.get("numbers", [])]
        entities = [EntityRef(**e) for e in d.get("entities", [])]
        kwargs = dict(d)
        kwargs["numbers"] = numbers
        kwargs["entities"] = entities
        ea = kwargs.get("extracted_at")
        if isinstance(ea, str):
            kwargs["extracted_at"] = datetime.fromisoformat(ea)
        return cls(**kwargs)

    # -- builders --
    @classmethod
    def from_search_summary(
        cls,
        *,
        claim: str,
        quote: str,
        source_url: str,
        source_title: Optional[str] = None,
        source_id: Optional[int] = None,
        run_id: Optional[str] = None,
        confidence: float = 0.6,
        text_context: Optional[str] = None,
        entities: Optional[Iterable[EntityRef]] = None,
    ) -> "EvidenceUnit":
        """Build an EU from a Tavily summary/extract line.

        `text_context` (typically the full search summary paragraph) is
        used to mine `numbers` automatically.
        """
        nums = extract_numbers(text_context or quote or claim)
        ents = list(entities or [])
        return cls(
            claim=claim,
            quote=quote,
            source_url=source_url,
            source_title=source_title,
            source_id=source_id,
            numbers=nums,
            entities=ents,
            confidence=confidence,
            extraction_method="tavily_summary",
            run_id=run_id,
        )


# =============================================================================
# Collection helpers
# =============================================================================

def dedup_eus(eus: Iterable[EvidenceUnit]) -> list[EvidenceUnit]:
    """Remove EUs with identical content_hash (stable, cross-run dedup)."""
    seen: dict[str, EvidenceUnit] = {}
    for eu in eus:
        seen[eu.content_hash] = eu
    return list(seen.values())


def eus_as_dicts(eus: Iterable[EvidenceUnit]) -> list[dict]:
    return [eu.to_dict() for eu in eus]
