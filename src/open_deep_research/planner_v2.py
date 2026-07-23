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
    dimension_id: Optional[str] = None  # 市场维度:market_size/adoption/regulation/performance/ethics
    id: Optional[str] = None

    def __post_init__(self):
        if not self.title or not self.question:
            raise ValueError("SubTopic.title and .question must be non-empty")
        # Stable id: hash of (title, question, dimension_id)
        # 把 dimension 纳入 hash 避免不同 dimension 但 title 相同的 sub_topic 撞 id
        seed = (self.title + "|" + self.question + "|" + (self.dimension_id or "")).strip().lower()
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
# 维度注册表 — Runbook v1 5 个市场调研维度,每个 dimension 独立 search query
# =============================================================================
# 5 维度覆盖 market research 的标准五视角:
#   - market_size    : 市场规模 / 营收 / CAGR / 预测
#   - adoption       : 采用率 / 用户基数 / 部署 / 客户数
#   - regulation     : 法规 / 合规 / 标准 / 政策 / 法律框架
#   - performance    : 性能基准 / 技术能力 / 对比
#   - ethics         : 伦理 / 偏见 / 隐私 / 社会影响
#
# 每个 dimension 的 query 模板拿 brief 作变量替换;5 个 sub_topic 并行 wave 0。
# =============================================================================

DIMENSION_TEMPLATES: dict[str, str] = {
    "market_size":  "market size, revenue, CAGR, forecast, and growth rate of: {brief}",
    "adoption":     "adoption rate, user base, deployment share, and customer count of: {brief}",
    "regulation":   "regulation, compliance, standards, policy, and legal framework for: {brief}",
    "performance":  "performance benchmarks, technical capabilities, and comparison of: {brief}",
    "ethics":       "ethics, bias, privacy concerns, and societal impact of: {brief}",
}

DIMENSION_ORDER: list[str] = ["market_size", "adoption", "regulation", "performance", "ethics"]


def _plan_dimensions(
    brief: str,
    *,
    max_dim: int,
) -> list[SubTopic]:
    """Build one SubTopic per dimension (Wave 0)."""
    short = brief.strip() or "(unspecified topic)"
    subs: list[SubTopic] = []
    for dim in DIMENSION_ORDER[:max_dim]:
        q = DIMENSION_TEMPLATES[dim].format(brief=short)
        subs.append(SubTopic(
            title=dim,
            question=q,
            depends_on=[],
            search_api="searxng",       # SearXNG fallback 已是默认;显式标注便于调度
            parallelism="fan_out",
            expected_entities=_entities_in(short),
            expected_keywords=_keywords_in(short) + [dim],
            rationale=f"dimension-driven search for: {dim}",
            dimension_id=dim,
        ))
    return subs


def plan_from_brief(
    brief: str,
    *,
    title: Optional[str] = None,
    max_subtopics: int = 6,
    mode: str = "dimensions",  # 'dimensions' (推荐) | 'clauses' (旧行为)
) -> PlannerPlan:
    """Produce a deterministic PlannerPlan from a research brief.

    Modes
    -----
    - ``'dimensions'`` (default): 5 维度并行,每个 dimension 一个 search,
      保证 EU 落到正确的 dimension_id(数据准确性导向)。``max_subtopics``
      上限 = 5 + 1(context 可选)。

    - ``'clauses'``: 旧行为,按 brief 子句拆分 sub_topic,dimension 全 None。
      保留向后兼容。
    """
    sub_topics: list[SubTopic] = []

    if mode == "dimensions":
        # Wave 0: 5 维度并行(优先,可被 max_subtopics 截断)
        max_dim = min(max(0, max_subtopics - 1), len(DIMENSION_ORDER))
        sub_topics.extend(_plan_dimensions(brief, max_dim=max_dim))
        # 可选: 1 个 context 子句作 Wave 1(依赖所有 dimension)
        clauses = _split_into_clauses(brief)
        if clauses and len(sub_topics) < max_subtopics:
            dep_ids = [s.id for s in sub_topics if s.id is not None]
            sub_topics.append(SubTopic(
                title="context",
                question=f"What is the context of: {clauses[0]}",
                depends_on=dep_ids,
                search_api="searxng",
                parallelism="serial",
                expected_entities=_entities_in(clauses[0]),
                expected_keywords=_keywords_in(clauses[0]),
                rationale="establishes baseline across all dimensions",
                dimension_id=None,  # context 不归属任何维度
            ))
    else:
        # 旧 clause-based 行为,dimension_id 全 None
        clauses = _split_into_clauses(brief)
        if not clauses:
            clauses = [brief or "(empty brief)"]
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

    dim_n = sum(1 for s in sub_topics if s.dimension_id)
    return PlannerPlan(
        title=title or "Planner v2 plan",
        sub_topics=sub_topics,
        waves=waves,
        notes=(
            f"{mode} decomposition into {len(sub_topics)} sub-topic(s) "
            f"({dim_n} dimensioned) across {len(waves)} wave(s)."
        ),
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
