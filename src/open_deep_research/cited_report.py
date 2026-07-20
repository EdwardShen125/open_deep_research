"""Phase 2.3: chain-of-citation schema + prompt + parser.

## Why

v1's writer node produces a single `final_report: str` — and that string
contains freeform prose with inline numeric references like "[1]" that
don't actually point at anything queryable. The v1 baseline anchors
(`docs/v1_defects.md`) call this out specifically: kompyte ownership is
asserted without a citation, valuation numbers have no backing source
binding, page-level URLs end up as domain-only references.

Plan v2 introduces chain-of-citation: every claim in the writer's output
carries an explicit `eu_id` reference into the EvidenceUnit pool
produced upstream by `eu_extractor`. The writer is therefore a *constraint
solver*, not a free-form author: it may only assemble paragraphs whose
sentences reference EUs that exist.

## Data model

    CitedClaim:
        text: str                        # the assertion sentence
        eu_ids: list[str]                # one or more cited EU IDs
        numbers: list[NumberBinding]     # numeric anchors in the claim
        confidence: float                # writer's self-rated confidence

    CitedSection:
        heading: str                     # the section title (中文 ok)
        claims: list[CitedClaim]         # ordered list

    CitedReport:
        title: str
        sections: list[CitedSection]
        unresolved_eu_ids: list[str]     # ids cited but not found in pool
        orphan_claim_text: list[str]     # sentences with no citations

The validator `validate_cited_report()` flags:
  - sections with **no citations** (red flag — unclaim-anchored prose)
  - claim text mentions a NumberBinding not bound to an EU with that number
  - claim asserts ownership/relation ("被收购","acquired by","owned by") but
    cites only one EU (gap-A risk) — the verifier escalates later
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional

from open_deep_research.evidence_units import EvidenceUnit, NumberBinding


# =============================================================================
# CitedClaim / Section / Report
# =============================================================================

@dataclass
class CitedClaim:
    text: str
    eu_ids: list[str] = field(default_factory=list)
    numbers: list[NumberBinding] = field(default_factory=list)
    confidence: float = 0.5
    # Optional metadata for downstream consumers
    rationale: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["numbers"] = [n.to_dict() for n in self.numbers]
        return d


@dataclass
class CitedSection:
    heading: str
    claims: list[CitedClaim] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "claims": [c.to_dict() for c in self.claims],
        }


@dataclass
class CitedReport:
    title: str
    sections: list[CitedSection] = field(default_factory=list)
    unresolved_eu_ids: list[str] = field(default_factory=list)
    orphan_claim_text: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "unresolved_eu_ids": self.unresolved_eu_ids,
            "orphan_claim_text": self.orphan_claim_text,
        }

    def to_markdown(self) -> str:
        """Render to markdown — the form that downstream final_report wants."""
        lines = [f"# {self.title}", ""]
        for sec in self.sections:
            lines.append(f"## {sec.heading}")
            lines.append("")
            for c in sec.claims:
                cite = " ".join(f"[{i}]" for i in c.eu_ids) if c.eu_ids else "[UNSOURCED]"
                lines.append(f"{c.text} {cite}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"


# =============================================================================
# Chain-of-citation prompt
# =============================================================================

CITED_REPORT_PROMPT = """You are a research report writer. Every claim you make MUST
include an EU id reference. You may ONLY write claims you can ground in the
EvidenceUnit pool provided below. If a fact is not in an EU, do not write it.

## EvidenceUnit pool

Each EU has:
  - id           : stable id, e.g. eu-a1b2c3d4
  - claim        : the assertion as it appears in the source
  - source_url   : the page it came from
  - confidence   : extraction confidence (0..1)
  - numbers      : numeric bindings (text + scaled value)
  - entities     : entity references (companies, products)

```
{eu_pool_block}
```

## Output format (strict JSON)

{{
  "title": "<report title>",
  "sections": [
    {{
      "heading": "<section heading>",
      "claims": [
        {{
          "text": "<one assertion sentence>",
          "eu_ids": ["<eu-id-from-pool>", ...],   # ≥1 required
          "numbers": [{{"text": "<verbatim number text>", "value_min": <float|null>, "value_max": <float|null>, "unit": "<e.g. USD>", "is_estimated": <bool>}}],
          "confidence": <0.0..1.0>,
          "rationale": "<why these EU ids>"
        }}
      ]
    }}
  ]
}}

## Rules

1. Every claim MUST cite at least one EU from the pool.
2. If you cannot ground a fact in any EU, omit it.
3. Numeric values quoted in a claim must have at least one binding EU
   whose `numbers` list contains a NumberBinding overlapping the value.
4. Use the EU `quote` field as your verbatim source-text inspiration;
   do not paraphrase the underlying source beyond what the EU preserves.
5. Do not write claims about ownership ("被收购 / acquired by / owned by")
   without citing ≥2 EUs from independent domains. (Phase 3a verifier
   enforces this; you can preempt by doing it yourself.)
6. The final JSON must be parseable. Do NOT write Markdown. Output only JSON.
"""


def render_eu_pool(eus: Iterable[Any]) -> str:
    """Format EUs as a JSON-ish block the LLM can scan.

    Accepts both `EvidenceUnit` instances and plain dicts (for
    cross-run serialization where state storage may have flattened
    them to dicts).
    """
    out_lines = []
    for eu in eus:
        if isinstance(eu, dict):
            d = {
                "id": eu.get("id", ""),
                "claim": eu.get("claim", ""),
                "source_url": eu.get("source_url", ""),
                "confidence": round(float(eu.get("confidence", 0.0) or 0.0), 2),
                "numbers": [n.get("text", "") if isinstance(n, dict) else str(n)
                            for n in (eu.get("numbers", []) or [])],
                "entities": sorted(
                    (e.get("name") if isinstance(e, dict) else str(e))
                    for e in (eu.get("entities", []) or [])
                ),
                "quote": eu.get("quote", ""),
            }
        else:
            d = {
                "id": eu.id,
                "claim": eu.claim,
                "source_url": eu.source_url,
                "confidence": round(eu.confidence, 2),
                "numbers": [n.text for n in eu.numbers],
                "entities": sorted(e.name for e in eu.entities),
                "quote": eu.quote,
            }
        out_lines.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(out_lines)


# =============================================================================
# Parser: take a raw LLM response and produce CitedReport + diagnostics
# =============================================================================

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> Optional[str]:
    """Pull the first JSON object out of a possibly fenced/free-form response."""
    if not text:
        return None
    # First try a strict fence
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1)
    # Then try a brace-balanced span starting at the first `{`
    if "{" in text:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _parse_numberbinding(d: dict) -> NumberBinding:
    return NumberBinding(
        text=d.get("text", ""),
        value_min=d.get("value_min"),
        value_max=d.get("value_max"),
        unit=d.get("unit"),
        is_estimated=bool(d.get("is_estimated", False)),
    )


def _parse_claim(d: dict) -> CitedClaim:
    return CitedClaim(
        text=d.get("text", ""),
        eu_ids=list(d.get("eu_ids", []) or []),
        numbers=[_parse_numberbinding(n) for n in d.get("numbers", []) or []],
        confidence=float(d.get("confidence", 0.5)),
        rationale=d.get("rationale", "") or "",
    )


def parse_cited_report(raw_response: str) -> tuple[CitedReport, list[str]]:
    """Parse LLM output JSON → CitedReport. Returns (report, warnings)."""
    warnings: list[str] = []
    payload = _extract_json(raw_response or "")
    if not payload:
        warnings.append("no JSON object found in response")
        return CitedReport(title=""), warnings
    try:
        obj = json.loads(payload)
    except Exception as e:
        warnings.append(f"JSON parse error: {e}")
        return CitedReport(title=""), warnings

    title = obj.get("title", "")
    sections = [
        CitedSection(heading=s.get("heading", ""), claims=[_parse_claim(c) for c in s.get("claims", []) or []])
        for s in obj.get("sections", []) or []
    ]
    return CitedReport(title=title, sections=sections), warnings


# =============================================================================
# Validator
# =============================================================================

# Phrasing that asserts ownership / acquisition / relation. Used by validator
# to flag single-source claims and "high-risk relation" claims.
_RELATION_TERMS = (
    "收购", "被收购", "收购方", "旗下", "归",
    "acquired by", "acquired", "owns", "owned by", "subsidiary",
)


def validate_cited_report(
    report: CitedReport,
    eu_pool: list[EvidenceUnit],
    *,
    require_min_eu_per_claim: int = 1,
    flag_relations_with_single_source: bool = True,
) -> list[str]:
    """Return a list of human-readable validation issues.

    The list is **not** an exception; callers decide whether to surface or
    regenerate. Each entry looks like `[WARN] section "X" claim "Y" ...`.
    """
    issues: list[str] = []
    known_eu_ids = {eu.id for eu in eu_pool if eu.id}

    # ---- 1. unresolved EU ids ----
    cited = set()
    for sec in report.sections:
        for c in sec.claims:
            cited.update(c.eu_ids)
    report.unresolved_eu_ids = sorted(cited - known_eu_ids)

    # ---- 2. orphan claims + missing EU ids ----
    for sec in report.sections:
        for c in sec.claims:
            if not c.text.strip():
                continue
            if not c.eu_ids:
                issues.append(
                    f'[WARN] section "{sec.heading}" claim "{c.text[:60]}..." '
                    f'has NO eu_ids'
                )
                report.orphan_claim_text.append(c.text)
            elif any(i not in known_eu_ids for i in c.eu_ids):
                bad = [i for i in c.eu_ids if i not in known_eu_ids]
                issues.append(
                    f'[WARN] section "{sec.heading}" cites unknown EU(s) {bad}'
                )
            elif len(c.eu_ids) < require_min_eu_per_claim:
                issues.append(
                    f'[WARN] section "{sec.heading}" claim "{c.text[:60]}..." '
                    f'has only {len(c.eu_ids)} EU(s), expected >= '
                    f'{require_min_eu_per_claim}'
                )
            # -- relation claim heuristic --
            if flag_relations_with_single_source and any(
                term in c.text.lower() for term in _RELATION_TERMS
            ):
                if len(c.eu_ids) < 2:
                    issues.append(
                        f'[WARN] section "{sec.heading}" claim "{c.text[:60]}..." '
                        f'asserts ownership but cites only {len(c.eu_ids)} EU(s) '
                        f'(gap-A risk: needs ≥2 independent domains)'
                    )

    # ---- 3. numeric coverage check ----
    # For each claim NumberBinding, look for any cited EU whose own
    # NumberBinding matches within 5% (numeric) or as substring (text).
    def _matches(claim_nb: NumberBinding, eu_nb: NumberBinding) -> bool:
        if not claim_nb.text or not eu_nb.text:
            return False
        if claim_nb.text in eu_nb.text or eu_nb.text in claim_nb.text:
            return True
        if claim_nb.value_min is None and claim_nb.value_max is None:
            return False
        for cv in (claim_nb.value_min, claim_nb.value_max):
            if cv is None:
                continue
            for ev in (eu_nb.value_min, eu_nb.value_max):
                if ev is None or ev == 0:
                    continue
                if abs(ev - cv) / max(abs(ev), 1e-9) < 0.05:
                    return True
        return False

    for sec in report.sections:
        for c in sec.claims:
            if not c.numbers:
                continue
            cited_eus = [eu for eu in eu_pool if eu.id in c.eu_ids]
            for nb in c.numbers:
                if nb.value_min is None and nb.value_max is None and not nb.text:
                    continue
                ok = any(
                    any(_matches(nb, eu_nb) for eu_nb in eu.numbers)
                    for eu in cited_eus
                )
                if not ok:
                    issues.append(
                        f'[WARN] section "{sec.heading}" claim "{c.text[:60]}..." '
                        f'quotes number "{nb.text}" but no cited EU contains a '
                        f'matching NumberBinding'
                    )
    return issues
