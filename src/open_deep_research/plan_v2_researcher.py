"""Plan v2 thin graph — wraps run_pipeline_resumable as a single LangGraph node.

This is the canonical entry point for `uvx langgraph dev` and any other
caller that expects a LangGraph-compatible StateGraph. The internal
node delegates to run_pipeline_resumable (the production 5-stage
pipeline: setup → extract → verify → merge → write) — the SAME path
that /api/_run_pipeline_background and run_odr.py use.

§3 (rectification_plan.md):deep_researcher graph is the v9 灾难源
(via _render_eu_digest eating 19,955 EU). This file replaces it as
the recommended entry; the legacy graph remains available for
backward compatibility but is no longer the default.

Usage:
    from open_deep_research.plan_v2_researcher import plan_v2_researcher
    graph = plan_v2_researcher.compile()
    result = await graph.ainvoke({"query": "..."})
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)


class PlanV2State(TypedDict, total=False):
    """LangGraph state shape — flat, single round-trip."""
    query: str
    run_id: str
    max_subtopics: int
    title: str
    primary: Any
    fallback: Any
    result: Any  # PlanV2RunResult
    error: str


async def plan_v2_node(state: PlanV2State) -> PlanV2State:
    """The only node — delegates to run_pipeline_resumable.

    Any exception is captured into state["error"] rather than re-raised,
    so the LangGraph run terminates with a typed result instead of a
    traceback. The 5-stage pipeline itself uses W5 fail-fast and will
    raise LLMGateError on NLI auth failures; we propagate that as an
    error string here for the langgraph client.
    """
    from open_deep_research.staged_runner import run_pipeline_resumable
    try:
        result = await run_pipeline_resumable(
            query=state["query"],
            run_id=state.get("run_id") or "",
            primary=state.get("primary"),
            fallback=state.get("fallback"),
            max_subtopics=state.get("max_subtopics", 4),
            title=state.get("title", "Plan v2 Report"),
        )
        return {**state, "result": result, "error": result.error or ""}
    except Exception as e:
        logger.exception("[plan_v2_node] run_pipeline_resumable failed")
        return {**state, "error": f"{type(e).__name__}: {e}"}


def build_plan_v2_researcher() -> Any:
    """Build a LangGraph StateGraph with a single node.

    Uses langgraph.graph.StateGraph when available; falls back to
    a no-op object with .ainvoke if langgraph isn't installed.
    """
    try:
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(PlanV2State)
        g.add_node("plan_v2", plan_v2_node)
        g.add_edge(START, "plan_v2")
        g.add_edge("plan_v2", END)
        return g
    except ImportError:
        logger.warning(
            "[plan_v2_researcher] langgraph not installed; "
            "falling back to plain callable (no graph features)"
        )

        class _PlainGraph:
            async def ainvoke(self, state: dict) -> dict:
                return await plan_v2_node(state)

        return _PlainGraph()


# Module-level entry point for `uvx langgraph dev`
plan_v2_researcher = build_plan_v2_researcher()


__all__ = ["plan_v2_researcher", "plan_v2_node", "PlanV2State", "build_plan_v2_researcher"]