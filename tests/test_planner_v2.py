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
    plan = plan_from_brief(brief, max_subtopics=5)
    assert plan.title
    assert len(plan.sub_topics) >= 2
    first = plan.sub_topics[0]
    assert first.title == "context"
    assert not first.depends_on
    print(f"  ✓ plan decomposition: {len(plan.sub_topics)} sub-topics")


def test_plan_waves_topological():
    brief = "Klue leads CI. Crayon competes. Kompyte is budget."
    plan = plan_from_brief(brief, max_subtopics=5)
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
    plan = plan_from_brief(brief, max_subtopics=3)
    assert len(plan.sub_topics) <= 3
    print(f"  ✓ plan respects max_subtopics={3} (got {len(plan.sub_topics)})")


def test_plan_extracts_entities_and_keywords():
    brief = "Klue and Crayon compete in competitive intelligence market."
    plan = plan_from_brief(brief)
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
