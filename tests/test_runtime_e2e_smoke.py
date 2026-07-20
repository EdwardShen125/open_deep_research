"""Runtime end-to-end smoke test.

Verifies the Plan v2 four fields (``evidence_units`` / ``cited_report`` /
``verification`` / ``url_compliance``) are populated when the full LangGraph
pipeline runs against a live ``langgraph dev`` server.

This is an *integration* test (not unit): it requires:

  - ``langgraph dev`` server reachable at ``http://127.0.0.1:2024``
  - ``MINIMAX_API_KEY`` (or another configured LLM provider) available
  - ``TAVILY_API_KEY`` available
  - network egress to those APIs

Default pytest behavior is **skip** unless ``--smoke`` is passed OR
``RUNTIME_E2E_SMOKE=1`` is set in the environment. This avoids burning
~$0.10 of tokens + ~5 minutes of wall time on every ``pytest tests/`` run.

Run explicitly with::

    cd /root/open_deep_research
    source .venv/bin/activate
    pytest tests/test_runtime_e2e_smoke.py --smoke -v

Or via the helper script::

    python tests/test_runtime_e2e_smoke.py --skip-server-check
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LANGGRAPH_URL = os.environ.get("LANGGRAPH_URL", "http://127.0.0.1:2024")
QUESTION = "What was Klue's total funding raised?"
THREAD_PATH = "/threads"
ASSISTANTS_PATH = "/assistants/search"
RUN_STREAM_PATH_TEMPLATE = "/threads/{thread_id}/runs/stream"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _post(path: str, body: dict, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(
        LANGGRAPH_URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _server_reachable() -> bool:
    try:
        with urllib.request.urlopen(LANGGRAPH_URL + "/ok", timeout=2) as resp:
            return b"true" in resp.read()
    except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
        return False


def _required_env_keys_present() -> list[str]:
    """Return list of env vars that are required but missing.

    Looks in the current process env first, then falls back to the
    project's `.env` file (the canonical key source for the langgraph
    dev workflow — the API never reaches it as process env when the
    server was started by an unrelated shell).
    """
    missing: list[str] = []
    for key in ("MINIMAX_API_KEY", "TAVILY_API_KEY"):
        if not os.environ.get(key):
            if not _read_env_file_key(ROOT / ".env", key):
                missing.append(key)
    return missing


def _read_env_file_key(env_path: Path, key: str) -> str:
    """Read a single KEY=... entry from a dotenv file. Empty on miss."""
    if not env_path.exists():
        return ""
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def _run_query(question: str, timeout: float = 600.0) -> dict:
    """Submit a run and stream values until end. Return the final state.

    Raises ``RuntimeError`` if the stream doesn't terminate within ``timeout``.
    """
    asst = _post(ASSISTANTS_PATH, {"limit": 1})
    asst_id = asst[0]["assistant_id"]

    thread = _post(THREAD_PATH, {})
    thread_id = thread["thread_id"]

    body = {
        "assistant_id": asst_id,
        "input": {"messages": [{"role": "user", "content": question}]},
        "config": {
            "configurable": {
                "search_api": "tavily",
                "max_concurrent_research_units": 1,
                "max_researcher_iterations": 2,
                "max_react_tool_calls": 3,
            }
        },
        "stream_mode": "values",
    }

    req = urllib.request.Request(
        LANGGRAPH_URL + RUN_STREAM_PATH_TEMPLATE.format(thread_id=thread_id),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    final_state: dict[str, Any] | None = None
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            if line.startswith("event:"):
                et = line[6:].strip()
                if et in ("end", "done"):
                    break
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                evt = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(evt, dict) and evt.get("final_report") is not None:
                final_state = evt
            if time.time() - start > timeout:
                raise RuntimeError(f"stream timed out after {timeout}s")

    if final_state is None:
        raise RuntimeError("stream ended without a state carrying final_report")
    return final_state


# ---------------------------------------------------------------------------
# smoke tests
# ---------------------------------------------------------------------------


def _smoke_enabled() -> bool:
    """Decide whether to actually run. Default skip."""
    if os.environ.get("RUNTIME_E2E_SMOKE") == "1":
        return True
    if "--smoke" in sys.argv:
        return True
    return False


def test_server_reachable() -> None:
    """Pre-flight: server must be up at LANGGRAPH_URL."""
    if not _smoke_enabled():
        print("  ⏭ skipped (pass --smoke or set RUNTIME_E2E_SMOKE=1)")
        return
    assert _server_reachable(), f"langgraph dev not reachable at {LANGGRAPH_URL}/ok"


def test_required_env_keys_present() -> None:
    if not _smoke_enabled():
        print("  ⏭ skipped")
        return
    missing = _required_env_keys_present()
    assert not missing, f"missing env vars: {missing}"


def test_pipeline_populates_plan_v2_fields() -> None:
    """End-to-end: query the agent and assert all 4 Plan v2 fields are populated.

    Assertions (loose, to avoid flakiness on real LLM runs):
      - evidence_units: list, len > 0
      - cited_report:   dict, has 'sections' with len ≥ 1
      - verification:   dict, has 'issues' key
      - url_compliance: list (may be empty if all cited URLs are page-level)
      - final_report:   non-empty markdown
    """
    if not _smoke_enabled():
        print("  ⏭ skipped (pass --smoke or set RUNTIME_E2E_SMOKE=1)")
        return

    assert _server_reachable(), f"server not reachable at {LANGGRAPH_URL}"
    missing = _required_env_keys_present()
    assert not missing, f"missing env vars: {missing}"

    print(f"  → running query: {QUESTION!r}")
    t0 = time.time()
    state = _run_query(QUESTION)
    elapsed = time.time() - t0
    print(f"  → completed in {elapsed:.1f}s")

    eu = state.get("evidence_units")
    assert isinstance(eu, list), f"evidence_units should be list, got {type(eu)}"
    assert len(eu) > 0, "evidence_units empty — EU pipeline regression"
    print(f"  ✓ evidence_units: {len(eu)} EUs")

    cr = state.get("cited_report")
    assert isinstance(cr, dict), f"cited_report should be dict, got {type(cr)}"
    secs = cr.get("sections") or []
    assert len(secs) >= 1, f"cited_report.sections should have ≥1, got {len(secs)}"
    total_claims = sum(len(s.get("claims") or []) for s in secs)
    print(f"  ✓ cited_report: {len(secs)} sections, {total_claims} cited claims")

    v = state.get("verification")
    assert isinstance(v, dict), f"verification should be dict, got {type(v)}"
    assert "issues" in v, "verification should contain 'issues' key"
    print(f"  ✓ verification: {len(v.get('issues') or [])} issues")

    uc = state.get("url_compliance")
    assert isinstance(uc, list), f"url_compliance should be list, got {type(uc)}"
    print(f"  ✓ url_compliance: {len(uc)} issues (empty if all URLs are page-level)")

    fr = state.get("final_report")
    assert isinstance(fr, str) and len(fr) > 100, (
        f"final_report should be non-trivial markdown, got len={len(fr) if isinstance(fr, str) else 'N/A'}"
    )
    print(f"  ✓ final_report: {len(fr)} chars")

    # Regression guards — must NOT see the v1 error fallback
    assert not fr.lstrip().startswith('{"error"'), (
        "final_report fell through to legacy error JSON — EU pipeline broken"
    )
    print("  ✓ final_report is real markdown (not legacy error fallback)")


def test_supervisor_forwards_evidence_units() -> None:
    """The fix for Bug #2: supervisor's state must carry evidence_units.

    Before the fix, ``supervisor_messages`` was wired but ``evidence_units``
    was dropped at the subgraph boundary. This test guards against regression.
    """
    if not _smoke_enabled():
        print("  ⏭ skipped")
        return

    assert _server_reachable(), f"server not reachable at {LANGGRAPH_URL}"
    missing = _required_env_keys_present()
    assert not missing, f"missing env vars: {missing}"

    state = _run_query(QUESTION)
    eu = state.get("evidence_units")
    assert isinstance(eu, list), "evidence_units must be a list on final state"
    assert len(eu) > 0, "evidence_units empty → supervisor didn't forward EUs"

    # Every EU must have a non-empty source_url and a stable id
    bad = [e for e in eu if not (isinstance(e, dict) and e.get("source_url") and e.get("id"))]
    assert not bad, f"{len(bad)} EUs missing source_url or id"

    # URLs must span ≥2 distinct domains for cross-domain verification
    domains = {
        "/".join(e["source_url"].split("/")[:3])
        for e in eu
        if isinstance(e, dict) and e.get("source_url")
    }
    assert len(domains) >= 2, f"EU pool should span ≥2 domains, got {len(domains)}"
    print(f"  ✓ evidence_units: {len(eu)} EUs across {len(domains)} domains")


# ---------------------------------------------------------------------------
# CLI runner (no pytest required)
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("server_reachable", test_server_reachable),
        ("required_env_keys_present", test_required_env_keys_present),
        ("pipeline_populates_plan_v2_fields", test_pipeline_populates_plan_v2_fields),
        ("supervisor_forwards_evidence_units", test_supervisor_forwards_evidence_units),
    ]
    print("=" * 70)
    print(f" Runtime E2E smoke ({len(tests)} tests)")
    print(f" Server: {LANGGRAPH_URL}")
    print(f" Smoke enabled: {_smoke_enabled()} (pass --smoke to run live)")
    print("=" * 70)

    failed: list[str] = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed.append(name)

    print("\n" + "=" * 70)
    if failed:
        print(f" {len(failed)}/{len(tests)} FAILED: {failed}")
        return 1
    if not _smoke_enabled():
        print(f" {len(tests)} smoke tests SKIPPED (no live run)")
        print(" Run with --smoke or RUNTIME_E2E_SMOKE=1 to execute against langgraph dev")
    else:
        print(f" ALL {len(tests)} RUNTIME E2E SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())