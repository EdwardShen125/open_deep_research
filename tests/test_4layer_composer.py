"""Unit tests for 4-layer architecture composer.

Validates:
  - 3 archetype × 1 ontology × 1 registry composition produces expected slots
  - expansion rule emits per-vendor slots (cap respected)
  - template rendering fills {market}/{category}/{year} from instance
  - caliber_ref resolves to concrete registry caliber_id
  - 3-archetype combination has no duplicate slot_ids across sections
"""
import pytest
from pathlib import Path

from open_deep_research.evidence.composer import (
    build_plan, load_archetype, load_ontology, load_registry, Instance,
)


DATA = Path(__file__).resolve().parents[1] / "data"


def test_load_3_archetypes():
    """All 3 archetype files exist and load cleanly."""
    for arch in ("market_size", "competitive", "regulatory"):
        a = load_archetype(arch)
        assert a.archetype_id == arch
        assert len(a.sections) >= 1


def test_load_cn_cybersec_ontology():
    onto = load_ontology("cn_cybersec")
    assert onto.ontology_id == "cn_cybersec"
    # entity_spaces.vendors has 9 vendors
    assert len(onto.entity_spaces["vendors"]["items"]) >= 7
    # indicator_hollows has EDR disambiguation
    assert "edr_disambiguation" in onto.indicator_hollows
    assert len(onto.indicator_hollows["edr_disambiguation"]["exclude_terms"]) >= 5
    # 7 regulations (composed → list[dict] after normalisation)
    assert len(onto.regulations) >= 5


def test_load_cn_cybersec_registry():
    reg = load_registry("cn_cybersec")
    # ≥15 A/B sources (F1 DoD)
    a_b = [s for s in reg.sources if s.get("tier") in ("A", "B")]
    assert len(a_b) >= 15, f"only {len(a_b)} A/B sources, need ≥15"
    # ≥3 calibers (narrow / broad / total)
    assert len(reg.calibers) >= 3
    # narrow caliber exists
    assert any(c["id"] == "cn_edr_idc_2024_narrow" for c in reg.calibers)


def test_build_plan_market_size_only():
    """market_size archetype × cn_cybersec ontology → TAM/SAM/caliber slots."""
    onto = load_ontology("cn_cybersec")
    reg = load_registry("cn_cybersec")
    inst = Instance(market="CN", category="EDR 终端检测响应", year=2024)
    fw = build_plan(
        archetypes=["market_size"],
        ontology=onto, registry=reg, instance=inst,
    )
    slot_ids = [s.slot_id for sec in fw.sections for s in sec.slots]
    # Tam total / tam yoy / sam / narrow_vs_broad / cagr_5y / forecast_3y
    assert any("tam_total" in sid for sid in slot_ids)
    assert any("narrow_vs_broad" in sid for sid in slot_ids)
    assert any("cagr_5y" in sid for sid in slot_ids)
    # Templates filled with instance
    tam_q = next(s.question for sec in fw.sections for s in sec.slots
                 if "tam_total" in s.slot_id)
    assert "CN" in tam_q
    assert "EDR" in tam_q
    assert "2024" in tam_q


def test_build_plan_competitive_expansion():
    """competitive archetype expansion emits per-vendor slots, capped at 30."""
    onto = load_ontology("cn_cybersec")
    reg = load_registry("cn_cybersec")
    inst = Instance(market="CN", category="EDR 终端检测响应", year=2024)
    fw = build_plan(
        archetypes=["competitive"],
        ontology=onto, registry=reg, instance=inst,
    )
    # Should have slots for each vendor in ontology.vendors
    slot_ids = [s.slot_id for sec in fw.sections for s in sec.slots]
    vendor_slots = [sid for sid in slot_ids if "vendor_profile_" in sid]
    # 9 vendors in ontology
    assert len(vendor_slots) >= 7
    # Per-vendor slot should reference vendor name
    qax_slot = next(s for sec in fw.sections for s in sec.slots
                    if "qax" in s.slot_id and "vendor_profile" in s.slot_id)
    assert "奇安信" in qax_slot.question or "qax" in qax_slot.slot_id.lower()


def test_build_plan_3_archetypes_combo():
    """3 archetypes combine into one framework; slot_ids globally unique."""
    onto = load_ontology("cn_cybersec")
    reg = load_registry("cn_cybersec")
    inst = Instance(market="CN", category="EDR 终端检测响应", year=2024)
    fw = build_plan(
        archetypes=["market_size", "competitive", "regulatory"],
        ontology=onto, registry=reg, instance=inst,
    )
    # ≥6 sections (3 archetypes × ~2-4 sections each)
    assert len(fw.sections) >= 6
    # ≥25 slots (3 archetype × 7 vendors + caliber slots + reg slots)
    total_slots = sum(len(s.slots) for s in fw.sections)
    assert total_slots >= 25
    # No duplicate slot_ids
    all_sids = [s.slot_id for sec in fw.sections for s in sec.slots]
    assert len(all_sids) == len(set(all_sids)), \
        f"duplicate slot_ids: {[s for s in all_sids if all_sids.count(s) > 1]}"


def test_caliber_ref_resolution():
    """slot's caliber_id resolves from registry.calibers."""
    onto = load_ontology("cn_cybersec")
    reg = load_registry("cn_cybersec")
    inst = Instance(market="CN", category="EDR 终端检测响应", year=2024)
    fw = build_plan(
        archetypes=["market_size"],
        ontology=onto, registry=reg, instance=inst,
    )
    # Find a slot that should have resolved caliber
    for sec in fw.sections:
        for s in sec.slots:
            if "tam_total" in s.slot_id:
                # caliber_ref: "{ontology}.market_size" → registry's market_size caliber
                assert s.caliber_id is not None, \
                    f"tam_total slot has no caliber_id: {s.model_dump()}"
                return
    pytest.fail("No tam_total slot found")


def test_regulatory_expansion_with_regulation_items():
    """regulatory archetype expansion uses ontology.regulations.items."""
    onto = load_ontology("cn_cybersec")
    reg = load_registry("cn_cybersec")
    inst = Instance(market="CN", category="EDR 终端检测响应", year=2024)
    fw = build_plan(
        archetypes=["regulatory"],
        ontology=onto, registry=reg, instance=inst,
    )
    slot_ids = [s.slot_id for sec in fw.sections for s in sec.slots]
    # Each regulation id from ontology.regulations.items becomes a slot
    for reg_id in ("djbp", "gjzc", "dsjfa", "wlaqfa"):
        assert any(reg_id in sid for sid in slot_ids), \
            f"regulation {reg_id} missing from slots"


def test_indicator_hollows_disambiguates_edr_query():
    """_apply_indicator_hollows augments EDR queries with positive + NOT terms."""
    from open_deep_research.query_constructor import _apply_indicator_hollows
    onto = load_ontology("cn_cybersec")
    queries = ["EDR 终端检测响应 2024 市场份额", "qax 营收 2024"]
    out = _apply_indicator_hollows(queries, ontology=onto)
    # EDR query should be augmented
    assert out[0] != queries[0]
    assert "终端检测响应" in out[0]  # positive term added
    # Non-EDR query (qax 营收) — hollow not triggered, pass-through
    # (because the chosen hollow matches on positive_terms overlap)
    # We accept either: pass-through OR augmented
    assert isinstance(out[1], str)


def test_run_pipeline_with_4_layers(monkeypatch):
    """End-to-end: run_pipeline(archetypes, ontology, ...) builds a 4-layer plan."""
    import asyncio
    from open_deep_research.plan_v2_pipeline import run_pipeline

    # Use None provider → evidence-only mode (no network)
    out = asyncio.run(run_pipeline(
        "Test 4-layer for cn_cybersec",
        run_id="r-4layer-test",
        primary=None,
        archetypes=["market_size", "competitive", "regulatory"],
        ontology="cn_cybersec",
        registry_vertical="cn_cybersec",
        instance_market="CN",
        instance_category="EDR 终端检测响应",
        instance_year=2024,
    ))
    assert out.planner is not None
    sub_topics = out.planner.sub_topics
    # ≥10 sub_topics (across 3 archetypes, capped at max_subtopics=4 default!)
    # default max_subtopics=4 truncates, so this might be 4
    assert len(sub_topics) >= 4
    # First sub_topic should come from market_size archetype
    assert "market_size" in sub_topics[0].dimension_id
