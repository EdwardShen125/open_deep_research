"""LangGraph runtime v2 integration tests.

Verifies that the v2 patches in `deep_researcher.py` are wired correctly:
  - the main graph compiles
  - EU extraction helper works against Tavily-style strings
  - the EU field flows through AgentState
  - final_report_generation does NOT crash on empty EU pool
  - final_report_generation attaches cited_report / verification / url_compliance
    when given a parseable writer response (mocked LLM)
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import open_deep_research.deep_researcher as dr  # noqa: E402
from open_deep_research.state import AgentState  # noqa: E402
from open_deep_research.evidence_units import EvidenceUnit  # noqa: E402


# ---------------------------------------------------------------------------
# _parse_tavily_observation
# ---------------------------------------------------------------------------

def test_parse_tavily_observation_extracts_eus():
    """Realistic Tavily tool output → multiple EUs.

    Source bodies are deliberately padded past the 200-char content floor
    introduced by Phase 2.5 (Tavily noise filter) — short stubs don't reach
    the EU extractor anymore.
    """
    obs = (
        "Search results: \n\n"
        "--- SOURCE 1: Klue Battlecards ---\n"
        "URL: https://klue.com/product/battlecards\n\n"
        "SUMMARY:\n"
        "Klue targets battlecards for B2B sales enablement. Pricing starts at "
        "$20K per year and scales with seat count and feature tier. Klue "
        "competes with Crayon and Kompyte in the competitive intelligence "
        "market, and is widely adopted among enterprise revenue teams in "
        "North America and Western Europe for weekly battlecard rollouts.\n\n"
        "---------------------------------------------------------------\n"
        "\n"
        "--- SOURCE 2: Crayon vs Klue ---\n"
        "URL: https://crayon.co/vs-klue\n\n"
        "SUMMARY:\n"
        "Crayon focuses on enterprise competitive intelligence with broad "
        "global coverage and analyst-grade summarization. Crayon acquired "
        "Kompyte in 2022 to add automated change detection and alerting. "
        "Crayon pricing is $20K-$40K per year depending on tracked competitors, "
        "and the platform integrates with Salesforce, Slack, and Highspot.\n\n"
        "---------------------------------------------------------------\n"
    )
    eus = dr._parse_tavily_observation(obs, run_id="test")
    assert len(eus) >= 4, f"expected ≥4 EUs, got {len(eus)}: {eus}"
    urls = {eu.source_url for eu in eus}
    assert "https://klue.com/product/battlecards" in urls
    assert "https://crayon.co/vs-klue" in urls
    print(f"  ✓ Tavily observation → {len(eus)} EUs across {len(urls)} URL(s)")


def test_parse_tavily_observation_handles_no_url():
    eus = dr._parse_tavily_observation("no URLs here")
    assert eus == []
    print("  ✓ Tavily observation with no URL → 0 EUs")


def test_parse_tavily_observation_handles_empty_string():
    assert dr._parse_tavily_observation("") == []
    assert dr._parse_tavily_observation(None) == []    # type: ignore
    print("  ✓ Tavily observation empty/None → 0 EUs (no crash)")


def test_parse_tavily_observation_mines_numbers_and_entities():
    obs = (
        "Search results: \n\n"
        "--- SOURCE 1: Klue VS Crayon ---\n"
        "URL: https://klue.com/vs-crayon\n\n"
        "SUMMARY:\n"
        "Klue 2022 Series C 估值约 8-10 亿美元,主战场在北美与西欧; Crayon 约 "
        "5 亿美元估值,2018 年完成 B 轮。两家平台均围绕 sales battlecards 与 "
        "competitive enablement 展开差异化竞争,目标客户为中大型 B2B 企业的 "
        "产品营销与销售赋能团队。Klue 在自动化数据采集与摘要方面投入更多, "
        "Crayon 偏向人工策展与企业级合规。\n\n"
        "---------------------------------------------------------------\n"
    )
    eus = dr._parse_tavily_observation(obs)
    # Every EU should have either numbers or entities anchored.
    has_numeric_anchor = any(len(eu.numbers) > 0 for eu in eus)
    has_entity_anchor = any(len(eu.entities) > 0 for eu in eus)
    assert has_numeric_anchor, "expected numeric anchor for '亿美元' mentions"
    assert has_entity_anchor, "expected entity anchor for Klue/Crayon"
    print(f"  ✓ Numeric + entity anchoring on Tavily observation ({len(eus)} EUs)")


# ---------------------------------------------------------------------------
# Final report generation — with mocked LLM
# ---------------------------------------------------------------------------

def _config():
    """Patch-friendly config. `Configuration.from_runnable_config`
    is mocked in each test so we can pass an arbitrary dict-configurable."""
    return {"configurable": {"final_report_model": "minimax:MiniMax-M3"}}


def _state(eu_pool=None, raw_notes=None, cited_response=None):
    """Build a minimal AgentState-shaped dict.

    `cited_response` (str) is what the mocked LLM returns — should be
    the body of parseable JSON.
    """
    return {
        "messages": [],
        "research_brief": "competitive intelligence market overview",
        "notes": raw_notes or [],
        "raw_notes": raw_notes or [],
        "evidence_units": eu_pool or [],
        "cited_report": None,
        "verification": None,
        "url_compliance": [],
    }


def _mock_configurable():
    """Build a dict that mimics Configuration shape (final_report_generation
    accesses attributes like final_report_model)."""
    return SimpleNamespace(
        final_report_model="minimax:MiniMax-M3",
        final_report_model_max_tokens=8192,
        max_structured_output_retries=2,
    )


def test_final_report_generation_with_empty_eu_pool_falls_back_legacy():
    """Empty EU pool: writer still runs, final_report gets raw LLM string."""
    state = _state(eu_pool=[])
    fake_msg = MagicMock(content="Legacy prose report from v1.")
    fake_model = MagicMock()
    fake_model.with_config.return_value.ainvoke = AsyncMock(return_value=fake_msg)
    with patch.object(dr, "configurable_model", fake_model), \
         patch.object(dr.Configuration, "from_runnable_config",
                      return_value=_mock_configurable()):
        out = asyncio.run(dr.final_report_generation(state, _config()))
    # The writer returned literal text "Legacy prose report from v1."
    # — parse_cited_report may still return a CitedReport shell (with
    # title="" and no sections) if the content has no JSON object.
    # We assert that the legacy LLM prose string ends up as final_report.
    assert out["final_report"] == "Legacy prose report from v1."
    cited = out.get("cited_report")
    if cited is not None:
        assert not cited.get("sections"), (
            f"cited_report should be empty/absent, got: {cited}"
        )
    print("  ✓ final_report_generation falls back to legacy prose when no EUs")


def test_final_report_generation_attaches_cited_report_and_verification():
    eu = EvidenceUnit(
        id="eu-abcdef123456",
        claim="Klue 估值约 8-10 亿美元,Crayon 估值约 5 亿美元.",
        source_url="https://klue.com/vs-crayon",
        source_title="Klue vs Crayon",
    ).to_dict()
    state = _state(eu_pool=[eu])

    cited_json = json.dumps({
        "title": "CI Market 2026",
        "sections": [{
            "heading": "Overview",
            "claims": [
                {
                    "text": "Klue 估值约 8-10 亿美元.",
                    "eu_ids": ["eu-abcdef123456"],
                    "numbers": [{"text": "8-10 亿美元", "value_min": 8,
                                 "value_max": 10, "unit": "USD",
                                 "is_estimated": True}],
                    "confidence": 0.85,
                    "rationale": "matched EU claim",
                }
            ]
        }]
    })
    fake_msg = MagicMock(content=cited_json)
    fake_model = MagicMock()
    fake_model.with_config.return_value.ainvoke = AsyncMock(return_value=fake_msg)
    with patch.object(dr, "configurable_model", fake_model), \
         patch.object(dr.Configuration, "from_runnable_config",
                      return_value=_mock_configurable()):
        out = asyncio.run(dr.final_report_generation(state, _config()))

    assert "CI Market 2026" in out["final_report"], (
        f"final_report should contain cited title, got: {out['final_report'][:200]}"
    )
    assert out["cited_report"] is not None
    assert out["verification"] is not None
    cr = out["cited_report"]
    assert cr["title"] == "CI Market 2026"
    assert len(cr["sections"]) == 1
    print(f"  ✓ final_report_generation attaches cited_report + verification")


def test_final_report_generation_propagates_url_compliance():
    """Domain-only URL claim → url_compliance has at least one issue."""
    eu = EvidenceUnit(
        id="eu-domain1",
        claim="Crayon home page",
        source_url="https://www.crayon.co",   # domain-only
        source_title="Crayon home",
    ).to_dict()
    state = _state(eu_pool=[eu])
    cited_json = json.dumps({
        "title": "Crayon context",
        "sections": [{
            "heading": "Crayon",
            "claims": [{
                "text": "Crayon is a CI vendor.",
                "eu_ids": ["eu-domain1"],
                "numbers": [],
                "confidence": 0.7,
                "rationale": "single EU grounding",
            }]
        }]
    })
    fake_msg = MagicMock(content=cited_json)
    fake_model = MagicMock()
    fake_model.with_config.return_value.ainvoke = AsyncMock(return_value=fake_msg)
    with patch.object(dr, "configurable_model", fake_model), \
         patch.object(dr.Configuration, "from_runnable_config",
                      return_value=_mock_configurable()):
        out = asyncio.run(dr.final_report_generation(state, _config()))
    assert isinstance(out["url_compliance"], list)
    high = [u for u in out["url_compliance"] if u.get("severity") == "high"]
    assert high, f"expected high-severity URL issue, got: {out['url_compliance']}"
    print(f"  ✓ final_report_generation flagged {len(high)} URL compliance issue(s)")


# ---------------------------------------------------------------------------
# AgentState field additions
# ---------------------------------------------------------------------------

def test_agent_state_has_plan_v2_fields():
    fields = AgentState.__annotations__
    for f in ("evidence_units", "cited_report", "verification", "url_compliance"):
        assert f in fields, f"missing Plan v2 field on AgentState: {f}"
    print(f"  ✓ AgentState has all 4 Plan v2 fields: "
          f"{[k for k in fields if k not in ('messages',)]}")


# ---------------------------------------------------------------------------
# Main graph compiles
# ---------------------------------------------------------------------------

def test_main_graph_compiles_with_v2_state():
    """The deep_researcher_builder must compile after the v2 patches."""
    from open_deep_research.deep_researcher import deep_researcher_builder
    g = deep_researcher_builder.compile()
    # Inspect the nodes in the compiled graph
    nodes = list(g.nodes.keys())
    assert "clarify_with_user" in nodes
    assert "write_research_brief" in nodes
    assert "research_supervisor" in nodes
    assert "final_report_generation" in nodes
    print(f"  ✓ main graph compiles, nodes: {nodes}")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("parse_tavily_observation_extracts_eus",
         test_parse_tavily_observation_extracts_eus),
        ("parse_tavily_observation_handles_no_url",
         test_parse_tavily_observation_handles_no_url),
        ("parse_tavily_observation_handles_empty_string",
         test_parse_tavily_observation_handles_empty_string),
        ("parse_tavily_observation_mines_numbers_and_entities",
         test_parse_tavily_observation_mines_numbers_and_entities),
        ("final_report_generation_with_empty_eu_pool_falls_back_legacy",
         test_final_report_generation_with_empty_eu_pool_falls_back_legacy),
        ("final_report_generation_attaches_cited_report_and_verification",
         test_final_report_generation_attaches_cited_report_and_verification),
        ("final_report_generation_propagates_url_compliance",
         test_final_report_generation_propagates_url_compliance),
        ("agent_state_has_plan_v2_fields",
         test_agent_state_has_plan_v2_fields),
        ("main_graph_compiles_with_v2_state",
         test_main_graph_compiles_with_v2_state),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} LangGraph runtime v2 integration tests")
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
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed.append(name)
    print("\n" + "=" * 70)
    if failed:
        print(f" {len(failed)}/{len(tests)} FAILED: {failed}")
        sys.exit(1)
    print(f" ALL {len(tests)} LANGGRAPH RUNTIME V2 TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
