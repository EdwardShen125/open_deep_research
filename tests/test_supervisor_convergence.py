"""Tests for supervisor convergence — the hard-cap that prevents
supervisor_tools from running indefinitely when MiniMax-M3 keeps
emitting ConductResearch calls without ever calling ResearchComplete.

Reproduces the EDR v4 bug where supervisor_tools ran 45+ times in a
720s timeout because research_iterations reset on every round-trip
and the strict `>` comparison never fired.

These tests cover three things:
1. SupervisorState.research_iterations accumulates across supervisor
   nodes (operator.add reducer, not last-write-wins).
2. supervisor_tools returns END when research_iterations reaches the
   hard cap (max_researcher_iterations - 1) — even when supervisor
   keeps emitting ConductResearch tool calls.
3. supervisor_tools returns END on the configured exceeded limit (>6).
4. supervisor node writes delta=1 (not state.get+1) so the reducer
   doesn't double-count.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolCall
from langgraph.types import Command


# ---------------------------------------------------------------------------
# Test 1: SupervisorState schema declares the reducer.
# ---------------------------------------------------------------------------

def test_supervisor_state_research_iterations_has_add_reducer():
    """The TypedDict must mark research_iterations with operator.add so
    the counter accumulates. Without this Annotated[operator.add] the
    LangGraph default reducer is last-write-wins and the value resets
    to 0 on every Command.update round-trip."""
    from open_deep_research.state import SupervisorState

    import typing

    hints = typing.get_type_hints(SupervisorState, include_extras=True)
    ri_hint = hints.get("research_iterations")

    assert ri_hint is not None, (
        "SupervisorState.research_iterations must be annotated; without "
        "an Annotated[int, ...] reducer the LangGraph default behavior "
        "resets the counter on each round-trip (see EDR v4 bug — 45+ "
        "supervisor loops before timeout)."
    )
    # The annotation must carry a reducer metadata. We accept either
    # operator.add or any callable that returns a + b for ints.
    from typing import get_args, get_origin

    metadata_args = []
    for arg in get_args(ri_hint):
        if get_origin(arg) is not None:
            continue
        # Plain callables like operator.add land here.
        metadata_args.append(arg)

    assert any(callable(m) for m in metadata_args), (
        "research_iterations annotation must include a reducer callable "
        "as its second Annotated argument."
    )


# ---------------------------------------------------------------------------
# Test 2: supervisor_tools hard-cap forces END when budget is hit.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_tools_force_end_at_hard_cap(monkeypatch, caplog):
    """When research_iterations reaches max_researcher_iterations - 1,
    supervisor_tools must return Command(goto=END) regardless of what
    tool calls the supervisor emitted. Without this guard the
    supervisor ran 45+ times in EDR v4 before the 720s curl timeout
    killed the run."""
    from open_deep_research import deep_researcher
    from open_deep_research.configuration import Configuration

    config = {
        "configurable": {
            "max_researcher_iterations": 6,
            "max_concurrent_research_units": 3,
        }
    }
    cfg = Configuration.from_runnable_config(config)
    state = {
        "supervisor_messages": [
            AIMessage(
                content="still researching",
                tool_calls=[
                    ToolCall(name="ConductResearch", args={}, id="call-1"),
                ],
            )
        ],
        "research_brief": "EDR",
        "research_iterations": 5,  # max - 1 = hard cap
    }

    with caplog.at_level(logging.WARNING):
        result = await deep_researcher.supervisor_tools(state, config)

    assert result.goto == "__end__", (
        f"At research_iterations=5 with max=6, supervisor_tools must "
        f"force END. Got goto={result.goto}."
    )
    assert any(
        "forcing END at research_iterations=5" in rec.message
        for rec in caplog.records
    ), "A warning should be logged when the hard-cap fires."


@pytest.mark.asyncio
async def test_supervisor_tools_force_end_at_exceeded_limit():
    """research_iterations > max_researcher_iterations also forces END."""
    from open_deep_research import deep_researcher
    from open_deep_research.configuration import Configuration

    config = {
        "configurable": {
            "max_researcher_iterations": 6,
            "max_concurrent_research_units": 3,
        }
    }
    state = {
        "supervisor_messages": [
            AIMessage(
                content="more research",
                tool_calls=[
                    ToolCall(name="ConductResearch", args={}, id="call-x"),
                ],
            )
        ],
        "research_brief": "EDR",
        "research_iterations": 10,  # > max
    }

    result = await deep_researcher.supervisor_tools(state, config)
    assert result.goto == "__end__"


@pytest.mark.asyncio
async def test_supervisor_tools_exits_on_research_complete():
    """When supervisor emits ResearchComplete, supervisor_tools exits
    immediately without reaching the iteration cap."""
    from open_deep_research import deep_researcher
    from open_deep_research.configuration import Configuration

    config = {
        "configurable": {
            "max_researcher_iterations": 6,
            "max_concurrent_research_units": 3,
        }
    }
    state = {
        "supervisor_messages": [
            AIMessage(
                content="done",
                tool_calls=[
                    ToolCall(name="ResearchComplete", args={}, id="call-c"),
                ],
            )
        ],
        "research_brief": "EDR",
        "research_iterations": 2,  # well below cap
    }

    result = await deep_researcher.supervisor_tools(state, config)
    assert result.goto == "__end__"


# ---------------------------------------------------------------------------
# Test 3: Supervisor node writes delta=1 (not state.get+1).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_node_writes_delta_one(monkeypatch):
    """The supervisor Command.update must contain research_iterations=1
    (a delta), NOT state.get(...) + 1. With the operator.add reducer
    declared in state.py, writing the current + 1 would double-count
    and halve the effective budget."""
    from open_deep_research import deep_researcher

    # Stub the resolved chat model directly so ainvoke returns a fake
    # AIMessage. We monkeypatch _resolve_chat_model at the module that
    # deep_researcher actually uses (create_configurable_model imports
    # it into open_deep_research.__init__, so we patch both names).
    fake_response = AIMessage(content="plan")

    async def _ainvoke(messages, config=None, **kwargs):
        return fake_response

    fake_chain = MagicMock()
    fake_chain.ainvoke = _ainvoke
    fake_chain.bind_tools = MagicMock(return_value=fake_chain)
    fake_chain.with_retry = MagicMock(return_value=fake_chain)
    fake_chain.with_config = MagicMock(return_value=fake_chain)

    # Patch _resolve_chat_model wherever it's imported.
    monkeypatch.setattr(
        "open_deep_research._resolve_chat_model",
        lambda config=None: fake_chain,
    )

    config = {
        "configurable": {
            "research_model": "openai/o3-mini",
            "max_researcher_iterations": 6,
            "max_concurrent_research_units": 3,
        }
    }

    state = {
        "supervisor_messages": [AIMessage(content="seed")],
        "research_iterations": 99,  # any value
    }

    result = await deep_researcher.supervisor(state, config)

    # The Command update must carry research_iterations=1, not 100.
    update = result.update
    assert update["research_iterations"] == 1, (
        f"supervisor must write delta=1 (reducer adds it). Got "
        f"research_iterations={update['research_iterations']}. "
        f"Writing state.get+1 doubles the count."
    )


# ---------------------------------------------------------------------------
# Test 4: E2E simulation — supervisor is forced to END after hard cap.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_loop_terminates_within_budget(monkeypatch):
    """End-to-end: simulate a supervisor that never emits
    ResearchComplete, only ConductResearch. Loop the supervisor →
    supervisor_tools cycle until either (a) it terminates via the hard
    cap or (b) we exceed 2x the configured budget. The hard cap must
    fire before the over-budget case."""
    from open_deep_research import deep_researcher
    from open_deep_research.configuration import Configuration

    config = {
        "configurable": {
            "max_researcher_iterations": 4,
            "max_concurrent_research_units": 2,
        }
    }
    cfg = Configuration.from_runnable_config(config)

    # supervisor_tools reads raw_notes from get_notes_from_tool_calls;
    # stub it to avoid the researcher subgraph dependency.
    monkeypatch.setattr(
        deep_researcher,
        "get_notes_from_tool_calls",
        lambda msgs: [m.content for m in msgs if isinstance(m, AIMessage)],
    )

    # Track research_iterations across calls (with operator.add reducer
    # semantics).
    iterations = [0]

    def supervisor_step(state, config):
        iterations[0] += 1
        return Command(
            goto="supervisor_tools",
            update={
                "supervisor_messages": [
                    AIMessage(
                        content=f"iter {iterations[0]}",
                        tool_calls=[
                            ToolCall(
                                name="ConductResearch", args={}, id="t"
                            )
                        ],
                    )
                ],
                "research_iterations": 1,  # delta
            },
        )

    # Now manually run the supervisor_tools loop with operator.add
    # reducer semantics.
    max_iter = cfg.max_researcher_iterations
    state: dict[str, Any] = {
        "supervisor_messages": [],
        "research_brief": "test",
        "research_iterations": 0,
    }

    terminated = False
    for _ in range(max_iter * 3):  # generous bound
        cmd = supervisor_step(state, config)
        state["supervisor_messages"].extend(cmd.update["supervisor_messages"])
        state["research_iterations"] += cmd.update["research_iterations"]
        tools_result = await deep_researcher.supervisor_tools(state, config)
        if tools_result.goto == "__end__":
            terminated = True
            break
        # update state with the supervisor_tools update payload (notes,
        # research_brief) but keep accumulating research_iterations via
        # operator.add semantics.
        for k, v in tools_result.update.items():
            state[k] = v
        state["research_iterations"] += 0  # supervisor_tools doesn't write iterations

    assert terminated, (
        f"supervisor_tools failed to terminate after 3x max budget "
        f"({max_iter * 3} iterations). research_iterations final = "
        f"{state['research_iterations']}. This reproduces the EDR v4 bug."
    )
    # Must terminate within max_researcher_iterations iterations
    assert state["research_iterations"] <= max_iter, (
        f"research_iterations={state['research_iterations']} exceeds "
        f"max={max_iter} before END was reached."
    )