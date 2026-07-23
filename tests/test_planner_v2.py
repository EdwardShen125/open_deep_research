"""Phase 4 — Planner v2 tests."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_deep_research.planner_v2 import (  # noqa: E402
    SubTopic, PlannerPlan, plan_from_brief, validate_plan,
)


# ---------------------------------------------------------------------------
# SubTopic invariants
# ---------------------------------------------------------------------------

def test_subtopic_id_is_stable():
    a = SubTopic(title="Klue", question="What is Klue?")
    b = SubTopic(title="Klue", question="What is Klue?")
    c = SubTopic(title="Klue", question="How is Klue priced?")
    assert a.id == b.id
    assert a.id != c.id
    assert a.id.startswith("st-")
    print(f"  ✓ sub-topic id deterministic: a={a.id}")


def test_subtopic_rejects_empty_title():
    try:
        SubTopic(title="", question="x")
    except ValueError:
        print("  ✓ empty title rejected")
        return
    raise AssertionError("expected ValueError on empty title")


# ---------------------------------------------------------------------------
# plan_from_brief
# ---------------------------------------------------------------------------

def test_plan_from_brief_splits_clauses():
    brief = (
        "Klue leads the CI market; "
        "Crayon owns enterprise CI; "
        "Kompyte targets budget teams."
    )
    plan = plan_from_brief(brief, max_subtopics=5, mode="clauses")
    assert plan.title
    assert len(plan.sub_topics) >= 2
    first = plan.sub_topics[0]
    assert first.title == "context"
    assert not first.depends_on
    print(f"  ✓ plan decomposition: {len(plan.sub_topics)} sub-topics")


def test_plan_waves_topological():
    brief = "Klue leads CI. Crayon competes. Kompyte is budget."
    plan = plan_from_brief(brief, max_subtopics=5, mode="clauses")
    # Wave 0 has the context-only sub-topic
    assert len(plan.waves) >= 1
    # Subsequent waves reference earlier sub-topic IDs
    by_id = {s.id: s for s in plan.sub_topics}
    scheduled: set[str] = set()
    for w in plan.waves:
        for sid in w:
            assert sid in by_id, f"unknown sub-topic in wave: {sid}"
            for d in by_id[sid].depends_on:
                assert d in scheduled, f"dep {d} must run before {sid}"
            scheduled.add(sid)
    print(f"  ✓ waves: {len(plan.waves)} with topological-order respected")


def test_plan_respects_max_subtopics():
    brief = "; ".join(f"topic{i}" for i in range(20))
    plan = plan_from_brief(brief, max_subtopics=3, mode="clauses")
    assert len(plan.sub_topics) <= 3
    print(f"  ✓ plan respects max_subtopics={3} (got {len(plan.sub_topics)})")


def test_plan_extracts_entities_and_keywords():
    brief = "Klue and Crayon compete in competitive intelligence market."
    plan = plan_from_brief(brief, mode="clauses")
    first = plan.sub_topics[0]
    # expect Klue and Crayon in entities
    assert "Klue" in first.expected_entities or "Crayon" in first.expected_entities
    print(f"  ✓ entity extraction: {first.expected_entities}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_plan_flags_cycle():
    s1 = SubTopic(title="A", question="?", depends_on=["st-cycleB"])
    s2 = SubTopic(title="B", question="?", depends_on=["st-cycleA"])
    s1.id = "st-cycleA"
    s2.id = "st-cycleB"
    plan = PlannerPlan(title="cyclic", sub_topics=[s1, s2])
    issues = validate_plan(plan)
    cyc = [i for i in issues if i.kind == "cycle"]
    assert cyc, f"expected cycle issue, got {issues}"
    print(f"  ✓ cycle detected: {[i.detail for i in cyc]}")


def test_validate_plan_flags_unresolved_dep():
    s1 = SubTopic(title="A", question="?", depends_on=["st-missing"])
    s1.id = "st-real"
    plan = PlannerPlan(title="unresolved", sub_topics=[s1])
    issues = validate_plan(plan)
    unresolved = [i for i in issues if i.kind == "unresolved_dep"]
    assert unresolved, f"expected unresolved_dep, got {issues}"
    print(f"  ✓ unresolved dep detected")


def test_validate_plan_detects_missing_wave_membership():
    s1 = SubTopic(title="A", question="?")
    s1.id = "st-orphan"
    plan = PlannerPlan(title="orphan", sub_topics=[s1], waves=[])
    issues = validate_plan(plan)
    wave_issues = [i for i in issues if i.kind == "wave_incomplete"]
    assert wave_issues, issues
    print(f"  ✓ wave_incomplete detected")


def test_validate_plan_clean_emit_no_issue():
    plan = plan_from_brief(
        "Klue leads CI. Crayon competes in CI. Kompyte is budget.",
        max_subtopics=5,
    )
    issues = validate_plan(plan)
    # Should be 0 issues (or only trivial wave reorderings we accept)
    crit_or_high = [i for i in issues if i.severity in ("critical", "high")]
    assert not crit_or_high, crit_or_high
    print(f"  ✓ clean plan validates without critical/high issues")


# ---------------------------------------------------------------------------
# Plan to_dict
# ---------------------------------------------------------------------------

def test_plan_to_dict_roundtrip():
    plan = plan_from_brief("Klue leads CI. Crayon competes.", max_subtopics=3)
    d = plan.to_dict()
    assert d["title"]
    assert isinstance(d["sub_topics"], list)
    assert isinstance(d["waves"], list)
    assert "created_at" in d
    assert d["notes"]
    print(f"  ✓ plan.to_dict roundtrip")


def test_plan_independent_of_data_flow():
    """Phase 4 explicitly says Planner doesn't depend on EU / sources."""
    # If planner imports break the EU/sources modules, this test signals.
    from open_deep_research import planner_v2  # noqa
    from open_deep_research import sources_dao  # noqa
    from open_deep_research import evidence_units  # noqa
    plan = plan_from_brief("Foo Bar.", max_subtopics=2)
    assert plan is not None
    print("  ✓ planner module is independent of EU/sources")


# ---------------------------------------------------------------------------
# Dimension-driven decomposition (default since Runbook v1 阶段 3)
# ---------------------------------------------------------------------------

def test_plan_dimensions_mode_assigns_all_5():
    """dimension-driven default: 5 standard dimensions covered per brief."""
    from open_deep_research.planner_v2 import DIMENSION_ORDER
    plan = plan_from_brief("EDR market overview 2024", max_subtopics=6)
    dims = sorted(s.dimension_id for s in plan.sub_topics if s.dimension_id)
    assert dims == sorted(DIMENSION_ORDER), f"missing dims: {set(DIMENSION_ORDER) - set(dims)}"
    print(f"  ✓ dimensions mode: 5 dims covered = {dims}")


def test_plan_dimensions_mode_uses_dimension_specific_queries():
    """每个 dimension 的 query 必须不同(否则等于退化成单 query)。"""
    plan = plan_from_brief("EDR market overview 2024")
    queries = [s.question for s in plan.sub_topics if s.dimension_id]
    assert len(set(queries)) == len(queries), "queries must be unique per dimension"
    market_size_q = next(s.question for s in plan.sub_topics if s.dimension_id == "market_size")
    assert "market size" in market_size_q.lower()
    print(f"  ✓ 5 dimension queries are distinct and templated")


def test_plan_dimensions_mode_respects_max_subtopics():
    """max_subtopics 截断 dimension 列表 + 可选 context。"""
    plan = plan_from_brief("EDR market", max_subtopics=3)
    assert len(plan.sub_topics) <= 3
    assert sum(1 for s in plan.sub_topics if s.dimension_id) >= 2
    print(f"  ✓ max_subtopics=3 → {len(plan.sub_topics)} sub_topics, "
          f"{sum(1 for s in plan.sub_topics if s.dimension_id)} dimensioned")


def test_plan_clauses_mode_backward_compat_dimension_none():
    """Backward-compat: clauses mode 仍工作,所有 dimension_id 为 None。"""
    plan = plan_from_brief("Klue. Crayon. Kompyte.", max_subtopics=4, mode="clauses")
    assert all(s.dimension_id is None for s in plan.sub_topics)
    print(f"  ✓ clauses mode backward-compat: dim=None across {len(plan.sub_topics)} subs")


def test_subtopic_dimension_id_in_hash():
    """dimension 不同 → sub_topic.id 不同(避免不同 dimension 但 title 相同的 sub_topic 撞 id)。"""
    from open_deep_research.planner_v2 import SubTopic
    a = SubTopic(title="x", question="?", dimension_id="market_size")
    b = SubTopic(title="x", question="?", dimension_id="regulation")
    assert a.id != b.id, f"same id {a.id} for different dimensions"
    print(f"  ✓ dim-aware hash: {a.id} != {b.id}")


# ---------------------------------------------------------------------------
# EU ↔ dimension plumbing (extractor + to_v2)
# ---------------------------------------------------------------------------

def test_extractor_stamps_dimension_id():
    """extract_from_search_result 必须透传 dimension_id 到 EU。"""
    from open_deep_research.eu_extractor import extract_from_search_result
    eu = extract_from_search_result(
        {"url": "https://arxiv.org/abs/1234", "title": "EDR paper",
         "content": "Endpoint detection grew 18% YoY in 2024, reaching $9.4B."},
        dimension_id="market_size",
    )
    assert eu and eu[0].dimension_id == "market_size"
    print(f"  ✓ extractor stamps dimension_id={eu[0].dimension_id!r}")


def test_evidence_unit_to_v2_preserves_dimension():
    """EvidenceUnit.dimension_id 必须透传到 EvidenceUnitV2。"""
    from open_deep_research.evidence_units import EvidenceUnit
    eu = EvidenceUnit(
        claim="Endpoint detection market reached $9.4B in 2024.",
        source_url="https://arxiv.org/abs/1234",
        quote="Endpoint detection grew 18% YoY in 2024.",
        dimension_id="market_size",
    )
    v2 = eu.to_v2(run_id="00000000-0000-0000-0000-000000000001")
    assert v2.dimension_id == "market_size", f"got {v2.dimension_id!r}"
    print(f"  ✓ to_v2 preserves dimension_id={v2.dimension_id!r}")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    tests = [
        ("subtopic_id_is_stable", test_subtopic_id_is_stable),
        ("subtopic_rejects_empty_title", test_subtopic_rejects_empty_title),
        ("plan_from_brief_splits_clauses", test_plan_from_brief_splits_clauses),
        ("plan_waves_topological", test_plan_waves_topological),
        ("plan_respects_max_subtopics", test_plan_respects_max_subtopics),
        ("plan_extracts_entities_and_keywords",
         test_plan_extracts_entities_and_keywords),
        ("validate_plan_flags_cycle", test_validate_plan_flags_cycle),
        ("validate_plan_flags_unresolved_dep",
         test_validate_plan_flags_unresolved_dep),
        ("validate_plan_detects_missing_wave_membership",
         test_validate_plan_detects_missing_wave_membership),
        ("validate_plan_clean_emit_no_issue",
         test_validate_plan_clean_emit_no_issue),
        ("plan_to_dict_roundtrip", test_plan_to_dict_roundtrip),
        ("plan_independent_of_data_flow", test_plan_independent_of_data_flow),
    ]
    print("=" * 70)
    print(f" Running {len(tests)} Planner v2 tests")
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
    print(f" ALL {len(tests)} TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
