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
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Iterable, Optional
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from open_deep_research.evidence.schema import EvidenceUnitV2


# =============================================================================
# NumberBinding
# =============================================================================
# Numeric anchor extraction — handles CJK + English magnitudes
# =============================================================================

# English magnitude suffixes (case-insensitive). Order matters for the
# alternation — longer tokens must come first so 'billion' beats 'b'.
_EN_MAGNITUDE = {
    "billion": 1e9,
    "million": 1e6,
    "trillion": 1e12,
    "thousand": 1e3,
    "bn": 1e9,
    "mn": 1e6,
    "m": 1e6,
    "k": 1e3,
    "b": 1e9,
}

# English currency unit keywords (used for NumberBinding.unit hint).
_EN_CURRENCY = {
    "USD": ("usd", "us$", "$", "dollar", "dollars"),
    "EUR": ("eur", "€", "euro", "euros"),
    "GBP": ("gbp", "£", "pound", "pounds"),
    "CAD": ("cad", "c$", "canadian dollar", "canadian dollars"),
    "RMB": ("rmb", "cny", "yuan", "renminbi"),
}

_NUMERIC_RE = re.compile(
    r"""
    (?P<neg>[-−])?                          # optional minus
    (?P<value>\d[\d,]*(?:\.\d+)?)           # first number
    (?:\s*(?:[\-–~])\s*                     # range connector
        (?P<value2>\d[\d,]*(?:\.\d+)?)
    )?
    \s*
    (?P<unit>[万亿千百]|[%％])?             # trailing CJK magnitude OR %
    """,
    re.VERBOSE,
)
# Secondary pass: English magnitude that follows a number but isn't a CJK unit.
# Anchored to lookbehind for a number to avoid false hits like
# "the company has million-dollar contracts".
_EN_MAGNITUDE_RE = re.compile(
    r"""
    (?P<num>\d[\d,]*(?:\.\d+)?)             # the number (already matched once)
    \s*
    (?P<mag>billion|million|trillion|thousand|bn|mn|[mbk])\b
    """,
    re.IGNORECASE | re.VERBOSE,
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


def _scale_en(num: Optional[float], mag: Optional[str]) -> Optional[float]:
    if num is None or not mag:
        return num
    return num * _EN_MAGNITUDE.get(mag.lower(), 1.0)


def _detect_en_unit(text: str) -> Optional[str]:
    """Detect a currency unit from English keywords (case-insensitive).
    Returns one of USD/EUR/GBP/CAD/RMB or None.
    """
    if not text:
        return None
    low = text.lower()
    for canonical, variants in _EN_CURRENCY.items():
        for v in variants:
            if v in low:
                return canonical
    return None


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
    def from_text(cls, text: str, *, context: Optional[str] = None) -> "NumberBinding":
        """Heuristically extract a single range from CJK / English text.

        For multi-number text, this returns the FIRST match. Use
        `extract_numbers` for a full sweep.

        `context`: optional broader string to scan for magnitude / currency
        markers that fall *outside* the numeric span itself. For example,
        when `extract_numbers` is sweeping a sentence, the span may be just
        "62" while the magnitude "million USD" sits in the surrounding
        words. Pass the original sentence as `context` so we can find it.
        """
        m = _NUMERIC_RE.search(text or "")
        if not m:
            return cls(text=text, is_estimated=_has_estimate_marker(text))
        scan_text = context if context is not None else (text or "")
        v1 = _to_float(m.group("value"))
        u1 = m.group("unit") or ""
        v2 = _to_float(m.group("value2"))
        # Apply the trailing unit only if it's a magnitude unit; for % we
        # don't multiply (it stays as a percentage).
        is_pct = u1 in ("%", "％")
        vmin = v1 if v1 is not None else None
        vmax = (_to_float(m.group("value2")) if m.group("value2") else vmin)
        if vmin is not None and not is_pct:
            vmin = _scale(vmin, u1) if vmin is not None else None
            if vmax is not None and vmax == vmin:
                # same value, no second number → leave scaled value
                pass
            if vmax is not None and v2 is not None:
                vmax = _scale(v2, u1)
            # English-magnitude pass — apply if a number was captured
            # without a CJK unit (e.g. "62 million USD"). Scan the
            # broader context so we find magnitude words that follow the
            # span (the span itself may be just "62"). Scale BOTH ends
            # unconditionally — the magnitude word applies to every
            # number in the span (range or single).
            #
            # Guard: skip scaling for year-like values (1900-2099) since
            # Tavily content often has "March 2024, brings ..." where a
            # bare 2024 sits next to currency keywords without actually
            # being a financial figure.
            if not u1 and v1 is not None and not (1900 <= v1 <= 2099):
                em = _EN_MAGNITUDE_RE.search(scan_text)
                if em:
                    mag = em.group("mag")
                    vmin = _scale_en(vmin, mag)
                    if vmax is not None:
                        vmax = _scale_en(vmax, mag)
        if vmin is not None and vmax is not None and vmax < vmin:
            vmin, vmax = vmax, vmin

        # Decide semantic unit. Heuristic: presence of "美元" → USD,
        # "人民币" / "元" → RMB, English currency keywords via
        # `_detect_en_unit`, otherwise leave None or CJK raw unit.
        if "美元" in scan_text:
            unit = "USD"
        elif "人民币" in scan_text:
            unit = "RMB"
        elif (en_unit := _detect_en_unit(scan_text)) is not None:
            unit = en_unit
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
    """Return every range discovered in `text`, ordered by appearance.

    `from_text` is called with the entire source string so the
    English-magnitude / currency detection regexes can scan past the
    numeric span itself (e.g. "62 million USD" → span "62" plus
    context "million USD").
    """
    out: list[NumberBinding] = []
    for m in _NUMERIC_RE.finditer(text or ""):
        # Use the matched span as the number text, but pass the whole
        # source string so `_EN_MAGNITUDE_RE` / `_detect_en_unit` can
        # find magnitude/currency markers *after* the span.
        span = m.group(0).strip()
        out.append(NumberBinding.from_text(span, context=text))
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

    def to_v2(self, *, run_id: str) -> "EvidenceUnitV2":
        """Convert to Phase 3 (= Runbook v1) Pydantic EvidenceUnitV2.

        Phase 3 设计依据: notes/phase1-pr-design-rationale.md 决策 D1。
        桥接 dataclass → Pydantic,让 phase1 PR 不破坏现有 18 个测试
        (test_evidence_units.py)。

        字段映射:
            claim     → claim
            quote     → source_span (≤200 字符的整句,直接当 span)
            source_url→ source_url
            source_title→ source_title
            numbers[0]→ norm_value (取第一个数值绑定作为代表)
            numbers[0].text → unit hint (启发式:含 USD / 美元 → "USD", etc.)
            entities[]→ entities (扁平化 name)
            confidence→ (丢弃,改为三道闸回填 span_verified/numeric_drift/entailment)
            extraction_method → extractor_model
            content_hash → content_hash (跨 run dedup 锚)
            source_domain ← urlsplit(source_url).hostname
            source_tier ← 默认 "tertiary" (阶段 3 才按白名单升级)
            claim_type ← 默认 "attribute" (deterministic extractor 不区分)
            dimension_id ← None (阶段 3 才接 planner)
            value_as_of ← None (deterministic 不解析时间锚)
            published_at ← None (Tavily 没给)
            span_start / span_end ← None (阶段 2 闸 1 才有)
            span_verified / numeric_drift / entailment_verdict ← 默认未校验
        """
        from open_deep_research.evidence.schema import EvidenceUnitV2, SourceTier

        # claim_type 启发:有 numbers → numeric;否则 default attribute
        if self.numbers:
            claim_type = "numeric"
        else:
            claim_type = "attribute"

        # norm_value + unit 从 numbers[0] 取
        norm_value: Optional[Decimal] = None
        unit: Optional[str] = None
        if self.numbers:
            n0 = self.numbers[0]
            if n0.value_min is not None:
                norm_value = Decimal(str(n0.value_min))
            unit = n0.unit

        # source_tier 默认 tertiary(阶段 3 白名单升级)
        # 实际 production 抽取器应该根据 source_domain 决定,这里保守走 default
        source_tier: SourceTier = "tertiary"

        # run_id 转换:str → UUID(str)
        from uuid import UUID as _UUID, uuid5, NAMESPACE_DNS
        # 非 UUID 字符串(如 'r-20260723041315')用 uuid5 派生稳定 UUID,
        # 让 Pydantic EvidenceUnitV2.run_id: UUID 字段校验通过。
        if isinstance(run_id, _UUID):
            rid = run_id
        else:
            try:
                rid = _UUID(run_id)
            except (ValueError, TypeError, AttributeError):
                rid = uuid5(NAMESPACE_DNS, str(run_id))

        return EvidenceUnitV2(
            run_id=rid,
            dimension_id=None,
            claim=self.claim,
            claim_type=claim_type,
            entities=[e.name for e in self.entities],
            norm_value=norm_value,
            unit=unit,
            value_as_of=None,
            source_url=self.source_url,
            source_domain=urlsplit(self.source_url).hostname or "",
            source_title=self.source_title,
            published_at=None,
            source_tier=source_tier,
            source_span=self.quote or self.claim,  # 旧 quote 是 ≤200 字符的整句
            span_start=None,
            span_end=None,
            extractor_model=self.extraction_method,
            extracted_at=self.extracted_at,
            span_verified=False,
            numeric_drift=False,
            entailment_verdict=None,
            entailment_score=None,
            claim_id=None,
            content_hash=self.content_hash,
        )

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
