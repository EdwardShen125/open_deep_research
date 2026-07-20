"""Phase 4: Planner v2 — topic decomposition + dependency graph.

## Why a separate module

Plan v2 calls the planner an "independent Module" — it doesn't depend on
v2's data flow (no EvidenceUnit requirement) and can run before the
research phase. Goal: turn a research_brief into a topologically
ordered list of sub-topics, each with:

  - a short title
  - a clear dependency relation to other sub-topics
  - a recommended search-API choice (Tavily / SearXNG / dedicated)
  - a parallelism hint (independent → can fan out, has_dep → must wait)
  - expected entities / keywords (used by Phase 2 extractor)

This module is intentionally framework-free: it accepts a research_brief
string and returns a `PlannerPlan`, which can be passed to the supervisor
or run as a standalone dry-run.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


# =============================================================================
# Models
# =============================================================================

@dataclass
class SubTopic:
    """One leaf in the research plan.

    `id` is a content-hash of (title, topic) so identical sub-topics from
    different decompositions dedup when we feed the plan into the
    supervisor's ConductResearch tool later.
    """
    title: str
    question: str
    depends_on: list[str] = field(default_factory=list)
    search_api: str = "tavily"          # 'tavily' / 'searxng' / 'crawl4ai'
    parallelism: str = "fan_out"        # 'fan_out' / 'serial' / 'optional'
    expected_entities: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    rationale: str = ""
    id: Optional[str] = None

    def __post_init__(self):
        if not self.title or not self.question:
            raise ValueError("SubTopic.title and .question must be non-empty")
        # Stable id: hash of (title, question)
        seed = (self.title + "|" + self.question).strip().lower()
        self.id = "st-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlannerPlan:
    title: str
    sub_topics: list[SubTopic] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Parallel-execution waves — `waves[i]` is a list of sub-topic IDs
    # that can run concurrently in wave i; waves must execute in order.
    waves: list[list[str]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "sub_topics": [s.to_dict() for s in self.sub_topics],
            "created_at": self.created_at.isoformat(),
            "waves": self.waves,
            "notes": self.notes,
        }


# =============================================================================
# Decomposition heuristics — deterministic, LLMs are optional
# =============================================================================

# Sentences / clauses that often indicate a sub-topic in research briefs.
_CLAUSE_SPLIT_RE = re.compile(
    r"""
    [,;:。;]                # ASCII punctuation as topic boundary
    | \s+(?:以及|并且|以及|和|and|also)    # connective words
    | \?                    # question marks
    | \n                    # paragraph breaks
    """,
    re.VERBOSE,
)


def _entities_in(text: str) -> list[str]:
    """Pull capitalized token(s) from `text` as a weak entity hint."""
    return list(set(re.findall(r"\b[A-Z][\w\-]{2,}\b", text)))


def _keywords_in(text: str) -> list[str]:
    """Pull obvious keywords: words with 4+ chars, lowercased."""
    seen = set()
    out = []
    for w in re.findall(r"[\w\u4e00-\u9fff]{2,}", text.lower()):
        if w in seen or len(w) < 2:
            continue
        if w in (
            "the", "and", "are", "for", "with", "this", "that",
            "into", "from", "such", "你", "我", "他", "的",
        ):
            continue
        seen.add(w)
        out.append(w)
    return out[:8]


def _split_into_clauses(brief: str) -> list[str]:
    """Split a multi-clause brief into candidate sub-topics."""
    if not brief:
        return []
    parts = _CLAUSE_SPLIT_RE.split(brief)
    cleaned = []
    for p in parts:
        s = p.strip(" .,")
        if len(s) >= 4:
            cleaned.append(s)
    if not cleaned:
        cleaned = [brief.strip()]
    return cleaned


def _question_for(clause: str) -> str:
    """Turn a clause into a research question if it isn't already one."""
    if clause.endswith("?") or clause.endswith("？") or clause.endswith("吗"):
        return clause
    return f"How does {clause} compare in the current market?"


# =============================================================================
# Plan construction (deterministic)
# =============================================================================

def plan_from_brief(
    brief: str,
    *,
    title: Optional[str] = None,
    max_subtopics: int = 6,
) -> PlannerPlan:
    """Produce a deterministic PlannerPlan from a research brief.

    The output is intentionally conservative (≤ max_subtopics) and may
    run as a dry-run before the supervisor decomposes via LLM.
    """
    clauses = _split_into_clauses(brief)
    sub_topics: list[SubTopic] = []
    if not clauses:
        clauses = [brief or "(empty brief)"]
    # First sub-topic is always the broad context; subsequent ones are
    # the fine-grained clauses and depend on the context.
    first = SubTopic(
        title="context",
        question=f"What is the context of: {clauses[0]}",
        depends_on=[],
        search_api="tavily",
        parallelism="fan_out",
        expected_entities=_entities_in(clauses[0]),
        expected_keywords=_keywords_in(clauses[0]),
        rationale="establishes baseline before sub-questions",
    )
    sub_topics.append(first)
    for c in clauses[1:max_subtopics]:
        sub_topics.append(SubTopic(
            title=c[:40],
            question=_question_for(c),
            depends_on=[first.id],
            search_api="tavily",
            parallelism="fan_out",
            expected_entities=_entities_in(c),
            expected_keywords=_keywords_in(c),
            rationale=f"detail from brief clause: {c}",
        ))
    # Compute waves: Wave 0 = no-dependency topics; Wave 1+ = topics
    # whose deps have been scheduled in earlier waves.
    waves: list[list[str]] = []
    assigned: set[str] = set()
    while len(assigned) < len(sub_topics):
        wave: list[str] = []
        for s in sub_topics:
            if s.id in assigned:
                continue
            if all(d in assigned for d in s.depends_on):
                wave.append(s.id)
        if not wave:
            # Cycle or unreached — add remaining as their own wave.
            wave = [s.id for s in sub_topics if s.id not in assigned]
        waves.append(wave)
        assigned.update(wave)

    return PlannerPlan(
        title=title or "Planner v2 plan",
        sub_topics=sub_topics,
        waves=waves,
        notes=f"Deterministic decomposition into {len(sub_topics)} sub-topic(s) "
              f"across {len(waves)} wave(s).",
    )


# =============================================================================
# Plan validation
# =============================================================================

@dataclass
class PlanValidationIssue:
    severity: str
    kind: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_plan(plan: PlannerPlan) -> list[PlanValidationIssue]:
    """Detect dependency cycles and unknown references."""
    issues: list[PlanValidationIssue] = []
    ids = {s.id for s in plan.sub_topics if s.id}
    by_id = {s.id: s for s in plan.sub_topics if s.id}
    # Unresolved deps
    for s in plan.sub_topics:
        for d in s.depends_on:
            if d not in ids:
                issues.append(PlanValidationIssue(
                    severity="high", kind="unresolved_dep",
                    detail=f"SubTopic {s.id} ({s.title}) depends_on unknown {d!r}"
                ))
    # Cycle detection (DFS)
    visiting: set[str] = set()
    done: set[str] = set()
    def visit(nid: str):
        if nid in done:
            return
        if nid in visiting:
            issues.append(PlanValidationIssue(
                severity="critical", kind="cycle",
                detail=f"Dependency cycle detected at {nid}",
            ))
            return
        visiting.add(nid)
        if nid in by_id:
            for d in by_id[nid].depends_on:
                visit(d)
        visiting.discard(nid)
        done.add(nid)
    for s in plan.sub_topics:
        if s.id:
            visit(s.id)
    # Wave membership completeness
    scheduled = {sid for w in plan.waves for sid in w}
    if scheduled != ids:
        missing = ids - scheduled
        issues.append(PlanValidationIssue(
            severity="medium", kind="wave_incomplete",
            detail=f"Sub-topics missing from waves: {sorted(missing)}"
        ))
    return issues
