"""Single-source-of-truth W1 (deterministic plan builder).

集中 plan_v2_pipeline 和 staged_runner 两处曾经各自实现 W1 hook 的代码,
确保:
  - 4-layer (archetypes+ontology) → build_plan()
  - legacy vertical= → load_framework(vertical)
  - 都没有 → LLM plan_from_brief()(绝不静默套 us_livecommerce)

R1 整改 (task book v2 / R1 §): staged_runner 曾经硬编码
load_framework("us_livecommerce"),导致任意非直播电商题都被错位。
改后所有 W1 都走 build_deterministic_plan,参数来自 ctx/state,
绝不在此模块出现硬编码 vertical 字面量。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from open_deep_research.evidence.framework import Framework, load_framework
from open_deep_research.planner_v2 import PlannerPlan, SubTopic, plan_from_brief

logger = logging.getLogger(__name__)


def build_deterministic_plan(
    query: str,
    *,
    vertical: Optional[str] = None,
    archetypes: Optional[list[str]] = None,
    ontology: Optional[str] = None,
    registry_vertical: Optional[str] = None,
    instance_brand: Optional[str] = None,
    instance_market: Optional[str] = None,
    instance_category: Optional[str] = None,
    instance_year: Optional[int] = None,
    max_subtopics: int = 4,
) -> tuple[PlannerPlan, Framework | None, Any]:
    """Build a deterministic plan from the 4-layer / vertical inputs.

    Returns (plan, framework, onto) — framework and onto may be None when
    the LLM-planner fallback path is taken. Callers use framework for
    downstream binding (e.g. indicator_hollows), onto for W2b.

    Priority:
      1. archetypes + ontology → 4-layer build_plan
      2. vertical alone → legacy load_framework(vertical)
      3. neither → LLM plan_from_brief (explicit "no framework" log)

    No path silently uses "us_livecommerce".
    """
    framework: Framework | None = None
    onto: Any = None

    if archetypes and ontology:
        # 4-layer path
        from open_deep_research.evidence.composer import (
            Instance,
            Ontology,
            build_plan,
            load_ontology,
            load_registry,
        )
        try:
            onto = load_ontology(ontology)
            reg_v = registry_vertical or ontology
            reg = load_registry(reg_v)
            inst = Instance(
                brand=instance_brand,
                market=instance_market,
                category=instance_category,
                year=instance_year,
            )
            framework = build_plan(
                archetypes=archetypes,
                ontology=onto,
                registry=reg,
                instance=inst,
                vertical_id=ontology,
            )
            plan = _framework_to_plan(
                framework,
                max_subtopics=max_subtopics,
                note=(
                    f"[W1] from 4-layer build (archetypes={archetypes}, "
                    f"ontology={ontology})"
                ),
            )
            logger.info(
                "[W1] 4-layer plan built: archetypes=%s ontology=%s "
                "registry=%s instance=%s → sections=%d slots=%d",
                archetypes, ontology, reg_v,
                inst.model_dump(exclude_none=True),
                len(framework.sections),
                sum(len(s.slots) for s in framework.sections),
            )
            return plan, framework, onto
        except FileNotFoundError as e:
            logger.warning(
                "[W1] 4-layer build failed (%s); falling back to LLM planner",
                e,
            )
            plan = plan_from_brief(query, max_subtopics=max_subtopics)
            return plan, None, None

    if vertical:
        # Legacy vertical= path — back-compat for us_livecommerce callers.
        try:
            framework = load_framework(vertical)
            plan = _framework_to_plan(
                framework,
                max_subtopics=max_subtopics,
                note=f"[W1] from framework {vertical}.yaml",
            )
            logger.info(
                "[W1] framework loaded: vertical=%s sections=%d slots=%d",
                vertical, len(framework.sections),
                sum(len(s.slots) for s in framework.sections),
            )
            return plan, framework, None
        except FileNotFoundError:
            logger.info(
                "[W1] no framework yaml for %s; falling back to LLM planner",
                vertical,
            )
            plan = plan_from_brief(query, max_subtopics=max_subtopics)
            return plan, None, None

    # No deterministic framework — explicit LLM plan (no silent fallback)
    logger.info(
        "[W1] LLM plan (no framework) — query=%s",
        query[:80],
    )
    plan = plan_from_brief(query, max_subtopics=max_subtopics)
    return plan, None, None


def _framework_to_plan(
    framework: Framework,
    *,
    max_subtopics: int,
    note: str,
) -> PlannerPlan:
    """Flatten Framework.sections[].slots[] → PlannerPlan.sub_topics."""
    subs: list[SubTopic] = []
    for sec in framework.sections:
        for slot in sec.slots:
            subs.append(
                SubTopic(
                    id=slot.slot_id,
                    title=slot.question,
                    question=slot.question,
                    search_api="searxng",
                    parallelism="fan_out",
                    expected_entities=[],
                    expected_keywords=[],
                    rationale=slot.notes or "",
                    dimension_id=sec.section_id,
                )
            )
    total_slots = sum(len(s.slots) for s in framework.sections)
    subs = subs[:max_subtopics]
    return PlannerPlan(
        title=framework.title,
        sub_topics=subs,
        waves=[],
        notes=f"{note}, {len(subs)}/{total_slots} slots",
    )