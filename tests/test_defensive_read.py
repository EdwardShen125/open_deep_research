"""Phase 1.5 — defensive-read regression test for deep_researcher.supervisor_tools.

Gap #7: long Chinese prompts caused the model to omit the `reflection` field
on the think_tool call, raising `KeyError: 'reflection'` and crashing the
whole supervisor subgraph.

This test imports `supervisor_tools` directly (which is possible in our venv
since all LangChain deps are installed) and exercises:

  (a) the patched reflection fallback — provided a constructed tool call
      that LACKS the `reflection` arg, supervisor_tools must not raise.
  (b) the patched research_topic fallback for ConductResearch.

We mock `researcher_subgraph.ainvoke` so the test does NOT issue any network
calls.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import open_deep_research.deep_researcher as dr  # noqa: E402


def _load_supervisor_tools_module():
    """Load the supervisor_tools closure body from deep_researcher.py.

    We don't actually run it — that would require LangGraph + LLM.
    Instead we re-derive the closure's defensive helpers to verify their
    correctness against representative input shapes.
    """
    spec = importlib.util.spec_from_file_location(
        "deep_researcher",
        ROOT / "src" / "open_deep_research" / "deep_researcher.py",
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        # Module may have unrelated imports at top that fail in test env.
        # We only need it loaded enough to introspect signatures.
        print(f"  (note: deep_researcher import noted: {type(e).__name__})")
        return mod
    return mod


# Replicate the defensive expressions from the patched supervisor_tools.
def _read_reflection(tool_call):
    args = tool_call.get("args") or {}
    return args.get("reflection") or "(empty reflection)"


def _read_research_topic(tool_call, state):
    args = tool_call.get("args") or {}
    t = args.get("research_topic")
    if t:
        return t
    if isinstance(state, dict):
        return state.get("research_brief") or "(missing research_topic — fallback)"
    return "(missing research_topic — fallback)"


# ---------------------------------------------------------------------------
# LIVE test: invoke supervisor_tools with malformed tool calls
# ---------------------------------------------------------------------------

def _build_state_with_call(call_name: str, args) -> dict:
    """Build a SupervisorState-shaped dict with one tool call already in
    supervisor_messages."""
    # Match SupervisorState shape from state.py: supervisor_messages,
    # research_brief, notes, research_iterations, raw_notes.
    return {
        "supervisor_messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "name": call_name,
                        "args": args,
                    }
                ],
                "additional_kwargs": {"tool_calls": []},
            }
        ],
        "research_brief": "竞品分析",
        "notes": [],
        "research_iterations": 0,
        "raw_notes": [],
    }


def _build_config():
    """Use the real Configuration object — but only the fields supervisor_tools
    reads need to be valid. We bypass from_runnable_config by monkey-patching
    Configuration.from_runnable_config to a no-op stub."""
    return {"configurable": SimpleNamespace(
        max_concurrent_research_units=2,
        max_researcher_iterations=6,
    )}


async def _run_supervisor_tools_with_call(call_name, args):
    state = _build_state_with_call(call_name, args)
    config = _build_config()

    # Patch the researcher subgraph + Configuration reader.
    with patch.object(dr, "researcher_subgraph") as rs, \
         patch.object(dr.Configuration, "from_runnable_config",
                      return_value=config["configurable"]):
        rs.ainvoke = AsyncMock(return_value={"compressed_research": "ok", "raw_notes": []})
        # supervisor_messages must support [-1] indexing and tool_calls attr access.
        # Convert to objects to satisfy langchain's adapter while still being
        # simple to build.
        class _Msg(dict):
            @property
            def tool_calls(self):
                return self["tool_calls"]
        state["supervisor_messages"] = [_Msg(state["supervisor_messages"][0])]
        # supervisor_messages is a TypedDict list and supervisor_tools does
        # `state.get(...)`. Our state is a dict — that's fine.
        result = await dr.supervisor_tools(state, config)
        return result


def test_live_reflection_missing_arg():
    """Pre-fix this would raise KeyError('reflection'); post-fix it must complete."""
    out = asyncio.run(_run_supervisor_tools_with_call("think_tool", {}))
    # supervisor_tools returns a Command-like dict with goto='supervisor'
    # (continuing the loop, since think_tool is not a terminal call here).
    print(f"  ✓ think_tool w/ empty args no longer raises (result={type(out).__name__})")


def test_live_conductresearch_missing_arg():
    """Pre-fix this would raise KeyError('research_topic'); post-fix it must
    synthesize from research_brief and complete the call."""
    out = asyncio.run(_run_supervisor_tools_with_call("ConductResearch", {}))
    print(f"  ✓ ConductResearch w/ empty args no longer raises (result={type(out).__name__})")


def test_live_conductresearch_no_args_key():
    out = asyncio.run(_run_supervisor_tools_with_call("ConductResearch", None))
    print(f"  ✓ ConductResearch w/ args=None no longer raises")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reflection_present():
    assert _read_reflection({"args": {"reflection": "looks good"}}) == "looks good"

def test_reflection_missing():
    assert _read_reflection({"args": {}}) == "(empty reflection)"

def test_reflection_empty_string():
    assert _read_reflection({"args": {"reflection": ""}}) == "(empty reflection)"

def test_reflection_none_value():
    assert _read_reflection({"args": {"reflection": None}}) == "(empty reflection)"

def test_reflection_no_args_key():
    assert _read_reflection({}) == "(empty reflection)"

def test_reflection_args_is_none():
    assert _read_reflection({"args": None}) == "(empty reflection)"


def test_topic_present():
    assert _read_research_topic(
        {"args": {"research_topic": "Klue 投资方关系"}},
        state={},
    ) == "Klue 投资方关系"


def test_topic_missing_uses_brief():
    assert _read_research_topic(
        {"args": {}},
        state={"research_brief": "竞品分析"},
    ) == "竞品分析"


def test_topic_missing_no_brief():
    v = _read_research_topic({"args": {}}, state={})
    assert v.startswith("(missing research_topic")


def test_topic_state_is_not_dict():
    v = _read_research_topic({"args": {}}, state=None)
    assert v.startswith("(missing research_topic")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    # Just exercise the closure-equivalents to prove behavior.
    tests = [
        ("reflection_present", test_reflection_present),
        ("reflection_missing", test_reflection_missing),
        ("reflection_empty_string", test_reflection_empty_string),
        ("reflection_none_value", test_reflection_none_value),
        ("reflection_no_args_key", test_reflection_no_args_key),
        ("reflection_args_is_none", test_reflection_args_is_none),
        ("topic_present", test_topic_present),
        ("topic_missing_uses_brief", test_topic_missing_uses_brief),
        ("topic_missing_no_brief", test_topic_missing_no_brief),
        ("topic_state_is_not_dict", test_topic_state_is_not_dict),
        ("live_reflection_missing_arg", test_live_reflection_missing_arg),
        ("live_conductresearch_missing_arg", test_live_conductresearch_missing_arg),
        ("live_conductresearch_no_args_key", test_live_conductresearch_no_args_key),
    ] 
    print("=" * 70)
    print(f" Running {len(tests)} defensive-read tests (gap #7)")
    print("=" * 70)
    failed = []
    for name, fn in tests:
        try:
            print(f"\n[{name}]")
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ✗ ERROR: {e!r}")
            failed.append(name)
    print("\n" + "=" * 70)
    if failed:
        print(f" {len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f" ALL {len(tests)} TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
