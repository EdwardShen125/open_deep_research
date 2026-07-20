"""Phase 3a: Verifier engine.

Plan v2 rule set:
  Rule 1 — Numeric binding: every numeric assertion in a CitedClaim must
           correspond to a NumberBinding in at least one cited EU whose
           numeric range overlaps within 5% (or textually contains the
           number's text).
  Rule 2 — Entity relation validation: claims asserting ownership /
           acquisition / relation between named entities must be backed by
           ≥2 EUs from *independent* domains (different `source_url` host).
           The verifier cross-checks the EntityRef `acquired_by` field.
  Rule 3 — High-risk claim cross-source: numeric or ownership claims with
           `confidence ≥ 0.7` flagged "high-risk" must have ≥2 EUs across
           independent domains, otherwise the verifier marks them for
           human review.

Plus, from the EU pool itself:
  Rule C — Chinese number binding: when a NumberBinding's text contains
           Chinese characters (亿/万/%), the overlap test must respect
           unit conversion (1 亿 = 100 M).

The verifier:
- takes a `CitedReport`, `eu_pool`, and optional `domain_graph` (for Rule 2)
- emits a list of `VerificationIssue` records (severity, anchor_id, etc.)
- provides aggregation stats for the acceptance report

Design choice: deterministic, no LLM. Any "LLM should weigh in" step is
delegated to a downstream regeneration step (Phase 3a.4) — Verifier just
*flags* issues, never rewrites.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from open_deep_research.cited_report import (
    CitedReport, validate_cited_report, _RELATION_TERMS,
)
from open_deep_research.evidence_units import EvidenceUnit, NumberBinding


# =============================================================================
# Issue model
# =============================================================================

@dataclass
class VerificationIssue:
    rule_id: str             # 'rule_1' / 'rule_2' / 'rule_3' / 'rule_c'
    severity: str            # 'critical' / 'high' / 'medium' / 'low'
    anchor_id: Optional[str] # v1 baseline anchor this issue maps to (if any)
    section: str
    claim_text: str
    detail: str
    affected_eu_ids: list[str] = field(default_factory=list)
    # Free-form context (sources, numbers, etc.)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Helpers
# =============================================================================

def host_of(url: str) -> str:
    """Lowercased hostname (no scheme/port)."""
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def distinct_domains(urls: Iterable[str]) -> set[str]:
    return {host_of(u) for u in urls if u}


def _claims_to_section(report: CitedReport) -> list[tuple[str, Any]]:
    return [(s.heading, c) for s in report.sections for c in s.claims]


def _claimed_numbers(claim) -> list[NumberBinding]:
    return list(claim.numbers or [])


# =============================================================================
# Rule 1 — numeric binding
# =============================================================================

def rule_1_numeric_binding(
    report: CitedReport,
    eu_pool: list[EvidenceUnit],
) -> list[VerificationIssue]:
    """Re-run the validator's number check and convert warnings into issues."""
    issues: list[VerificationIssue] = []
    eu_index = {eu.id: eu for eu in eu_pool if eu.id}

    def _numeric_overlap(claim_nb: NumberBinding, eu_nb: NumberBinding) -> bool:
        if not claim_nb.text or not eu_nb.text:
            return False
        if claim_nb.text in eu_nb.text or eu_nb.text in claim_nb.text:
            return True
        for cv in (claim_nb.value_min, claim_nb.value_max):
            if cv is None:
                continue
            for ev in (eu_nb.value_min, eu_nb.value_max):
                if ev is None or ev == 0:
                    continue
                if abs(ev - cv) / max(abs(ev), 1e-9) < 0.05:
                    return True
        return False

    for sec_heading, claim in _claims_to_section(report):
        if not claim.numbers:
            continue
        cited_eus = [eu_index[i] for i in claim.eu_ids if i in eu_index]
        for nb in claim.numbers:
            ok = any(
                any(_numeric_overlap(nb, eu_nb) for eu_nb in eu.numbers)
                for eu in cited_eus
            )
            if not ok:
                issues.append(VerificationIssue(
                    rule_id="rule_1",
                    severity="high",
                    anchor_id=None,
                    section=sec_heading,
                    claim_text=claim.text[:200],
                    detail=(
                        f'Quoted number "{nb.text}" (value_min={nb.value_min}, '
                        f'value_max={nb.value_max}, unit={nb.unit}, '
                        f'estimated={nb.is_estimated}) has no overlap with '
                        f'NumberBindings in any cited EU. '
                        f'(Phase 3a rule 1 — numeric binding)'
                    ),
                    affected_eu_ids=list(claim.eu_ids),
                    context={"claimed_number": nb.text},
                ))
    return issues


# =============================================================================
# Rule 2 — entity relation / ownership
# =============================================================================

_KNOWN_ENTITY_RISK = {
    # Entities whose mentions in v1 corpus historically trigger A-class anchors:
    # Plan v2 requires ≥2 independent-domains to back any claim about them.
    "kompyte": "A1_kompyte_ownership",
    "crayon":  "A1_kompyte_ownership",
    "klue":    "A2_klue_algorithmia_fabrication",
    "algorithmia": "A2_klue_algorithmia_fabrication",
    "alphasense": "A2_klue_algorithmia_fabrication",
}


def rule_2_entity_relation(
    report: CitedReport,
    eu_pool: list[EvidenceUnit],
    *,
    relation_terms: tuple[str, ...] = _RELATION_TERMS,
    high_risk_terms: tuple[str, ...] = (
        "被收购", "收购", "收购方", "旗下", "归",
        "acquired", "owned by", "subsidiary", "merged with",
    ),
    known_entity_risk: dict[str, str] = _KNOWN_ENTITY_RISK,
) -> list[VerificationIssue]:
    """Ownership / acquisition claims must be backed by ≥2 independent-domain EUs.

    Two triggers:
      (a) The claim text asserts ownership/acquisition (`relation_terms`).
      (b) The claim mentions a `known_entity_risk` entity (e.g. Kompyte).
          Even when no ownership assertion is present, claims about such
          entities are flagged for cross-source verification — Plan v2
          says any such mention is high-risk.
    "Independent" = different `host_of(source_url)`.
    """
    issues: list[VerificationIssue] = []
    eu_index = {eu.id: eu for eu in eu_pool if eu.id}

    for sec_heading, claim in _claims_to_section(report):
        text_lower = claim.text.lower()
        # (a) explicit-relation trigger
        any_relation = any(t.lower() in text_lower for t in relation_terms)
        # (b) implicit-entity trigger
        matched_known_entity: Optional[str] = None
        matched_anchor: Optional[str] = None
        for ent, anchor in known_entity_risk.items():
            if ent in text_lower:
                matched_known_entity = ent
                matched_anchor = anchor
                break
        if not (any_relation or matched_known_entity):
            continue

        cited = [eu_index[i] for i in claim.eu_ids if i in eu_index]
        domains = distinct_domains(eu.source_url for eu in cited)
        # Determine severity: high-risk phrases escalate.
        is_high_risk = any(t.lower() in text_lower for t in high_risk_terms)
        if not any_relation and matched_known_entity:
            # Implicit trigger — lower severity to 'high' (still needs review).
            severity = "high"
        else:
            severity = "critical" if is_high_risk else "high"
        if len(domains) < 2:
            anchor = matched_anchor if matched_anchor else (
                "A1_kompyte_ownership" if "kompyte" in text_lower else None
            )
            trigger = (
                "explicit ownership assertion"
                if any_relation else
                f"implicit mention of known-risk entity ({matched_known_entity})"
            )
            issues.append(VerificationIssue(
                rule_id="rule_2",
                severity=severity,
                anchor_id=anchor,
                section=sec_heading,
                claim_text=claim.text[:200],
                detail=(
                    f"Claim ({trigger}) cites {len(cited)} EU(s) from "
                    f"{len(domains)} distinct domain(s): {sorted(domains)}. "
                    f"Independent cross-domain verification requires ≥2. "
                    f"(Phase 3a rule 2 — entity relation)"
                ),
                affected_eu_ids=list(claim.eu_ids),
                context={
                    "domains": sorted(domains),
                    "high_risk": is_high_risk,
                    "trigger": trigger,
                },
            ))
    return issues


# =============================================================================
# Rule 3 — high-risk cross-source
# =============================================================================

def rule_3_high_risk_xsource(
    report: CitedReport,
    eu_pool: list[EvidenceUnit],
    *,
    high_risk_threshold: float = 0.7,
) -> list[VerificationIssue]:
    """For high-confidence numeric claims without cross-domain citation,
    flag them as 'high-risk' — they'll be candidates for human review or
    explicit downgrade."""
    issues: list[VerificationIssue] = []
    eu_index = {eu.id: eu for eu in eu_pool if eu.id}
    for sec_heading, claim in _claims_to_section(report):
        if claim.confidence < high_risk_threshold:
            continue
        if not claim.numbers and not any(
            t.lower() in claim.text.lower() for t in _RELATION_TERMS
        ):
            continue
        cited = [eu_index[i] for i in claim.eu_ids if i in eu_index]
        domains = distinct_domains(eu.source_url for eu in cited)
        if len(domains) < 2:
            issues.append(VerificationIssue(
                rule_id="rule_3",
                severity="medium",
                anchor_id=None,
                section=sec_heading,
                claim_text=claim.text[:200],
                detail=(
                    f"high-confidence claim (confidence={claim.confidence}) "
                    f"lacks cross-domain backing ({len(domains)} domains). "
                    f"(Phase 3a rule 3 — high-risk cross-source)"
                ),
                affected_eu_ids=list(claim.eu_ids),
                context={"confidence": claim.confidence, "domains": sorted(domains)},
            ))
    return issues


# =============================================================================
# Rule C — Chinese number binding sanity
# =============================================================================

def rule_c_chinese_numbers(
    report: CitedReport,
    eu_pool: list[EvidenceUnit],
) -> list[VerificationIssue]:
    """Pin-point claims with Chinese magnitude numerals whose EU pool has
    only English / unparseable counterparts — independent of Rule 1."""
    issues: list[VerificationIssue] = []
    eu_index = {eu.id: eu for eu in eu_pool if eu.id}
    cn_unit_re = re.compile(r"[万亿千百%％]")

    for sec_heading, claim in _claims_to_section(report):
        # Pull CN units from claim text + claim numbers
        cjk_in_text = cn_unit_re.search(claim.text)
        cjk_in_nums = any(cn_unit_re.search(nb.text or "") for nb in claim.numbers)
        if not (cjk_in_text or cjk_in_nums):
            continue
        cited = [eu_index[i] for i in claim.eu_ids if i in eu_index]
        # Heuristic: each cited EU should have at least one CN unit in any
        # NumberBinding OR the EU's claim text. If not, flag for review.
        for eu in cited:
            if cn_unit_re.search(eu.claim or ""):
                continue
            if any(cn_unit_re.search(nb.text or "") for nb in eu.numbers):
                continue
            issues.append(VerificationIssue(
                rule_id="rule_c",
                severity="medium",
                anchor_id=None,
                section=sec_heading,
                claim_text=claim.text[:200],
                detail=(
                    f"Claim uses CN magnitude numerals but cited EU "
                    f"{eu.id} has no CN number binding — possible "
                    f"translation drift. (Phase 3a rule C — CN numbers)"
                ),
                affected_eu_ids=[eu.id],
                context={"eu_id": eu.id},
            ))
    return issues


# =============================================================================
# Top-level verifier
# =============================================================================

@dataclass
class VerificationResult:
    issues: list[VerificationIssue]
    by_rule: dict[str, int]
    by_severity: dict[str, int]
    anchors_triggered: dict[str, int]
    passes: bool

    def to_dict(self) -> dict:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "by_rule": self.by_rule,
            "by_severity": self.by_severity,
            "anchors_triggered": self.anchors_triggered,
            "passes": self.passes,
            "issue_count": len(self.issues),
        }


def verify(
    report: CitedReport,
    eu_pool: list[EvidenceUnit],
    *,
    fail_threshold: int = 0,
    relation_terms: tuple[str, ...] = _RELATION_TERMS,
    high_risk_threshold: float = 0.7,
    relation_high_risk_terms: tuple[str, ...] = (
        "被收购", "收购", "收购方", "旗下", "归",
        "acquired", "owned by", "subsidiary", "merged with",
    ),
) -> VerificationResult:
    """Run all Phase-3a rules and return aggregated issues."""
    issues: list[VerificationIssue] = []
    issues.extend(rule_1_numeric_binding(report, eu_pool))
    issues.extend(rule_2_entity_relation(
        report, eu_pool,
        relation_terms=relation_terms,
        high_risk_terms=relation_high_risk_terms,
    ))
    issues.extend(rule_3_high_risk_xsource(
        report, eu_pool, high_risk_threshold=high_risk_threshold,
    ))
    issues.extend(rule_c_chinese_numbers(report, eu_pool))

    by_rule: dict[str, int] = defaultdict(int)
    by_severity: dict[str, int] = defaultdict(int)
    by_anchor: dict[str, int] = defaultdict(int)
    for i in issues:
        by_rule[i.rule_id] += 1
        by_severity[i.severity] += 1
        if i.anchor_id:
            by_anchor[i.anchor_id] += 1

    return VerificationResult(
        issues=issues,
        by_rule=dict(by_rule),
        by_severity=dict(by_severity),
        anchors_triggered=dict(by_anchor),
        passes=len(issues) <= fail_threshold,
    )
