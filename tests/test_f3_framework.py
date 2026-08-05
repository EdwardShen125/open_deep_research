"""F3 · vertical YAML 骨架框架验收测试

按 SPEC §5 F3:
- schema 校验
- 种子 vertical ≥6 个 section、≥25 个 slot
- 每个 quantitative slot 必须声明 caliber_id 和 required_tier_min(校验强制)
- build_skeleton 含正确数量的命名 section/slot
- 空骨架 slot_fill_rate==0
- 填 1 槽后指标更新
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from open_deep_research.evidence.framework import (
    Framework,
    Report,
    Section,
    Slot,
    build_skeleton,
    load_framework,
)


FRAMEWORK_PATH = Path("data/frameworks/us_livecommerce.yaml")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def framework() -> Framework:
    return load_framework("us_livecommerce", base_dir=FRAMEWORK_PATH.parent)


@pytest.fixture
def skeleton(framework: Framework) -> Report:
    return build_skeleton(framework)


# -----------------------------------------------------------------------------
# 验收 1: schema 校验
# -----------------------------------------------------------------------------

class TestSchemaValidation:
    def test_loads_yaml(self, framework: Framework) -> None:
        assert isinstance(framework, Framework)
        assert framework.vertical_id == "us_livecommerce"

    def test_min_section_count(self, framework: Framework) -> None:
        # plan F3 DoD:≥ 6 section
        assert len(framework.sections) >= 6

    def test_min_slot_count(self, framework: Framework) -> None:
        # plan F3 DoD:≥ 25 slot
        total = sum(len(s.slots) for s in framework.sections)
        assert total >= 25

    def test_all_sections_named(self, framework: Framework) -> None:
        for sec in framework.sections:
            assert sec.section_id
            assert sec.title

    def test_all_slots_have_question(self, framework: Framework) -> None:
        for sec in framework.sections:
            for slot in sec.slots:
                assert slot.question
                assert len(slot.question) >= 5


class TestQuantitativeConstraints:
    """每个 quantitative slot 必须声明 caliber_id 和 required_tier_min。"""

    def test_all_quantitative_slots_have_caliber(self, framework: Framework) -> None:
        for sec in framework.sections:
            for slot in sec.slots:
                if slot.expected_claim_type == "quantitative":
                    assert slot.caliber_id, (
                        f"slot {slot.slot_id} is quantitative but has no caliber_id"
                    )

    def test_all_quantitative_slots_have_tier_min(self, framework: Framework) -> None:
        for sec in framework.sections:
            for slot in sec.slots:
                if slot.expected_claim_type == "quantitative":
                    assert slot.required_tier_min, (
                        f"slot {slot.slot_id} is quantitative but has no required_tier_min"
                    )

    def test_validator_rejects_quantitative_without_caliber(self) -> None:
        with pytest.raises(ValidationError, match="caliber_id"):
            Slot(
                slot_id="bad_slot",
                question="What is the market size?",
                expected_claim_type="quantitative",
                required_tier_min="B",
            )

    def test_validator_rejects_quantitative_without_tier(self) -> None:
        with pytest.raises(ValidationError, match="required_tier_min"):
            Slot(
                slot_id="bad_slot",
                question="What is the market size?",
                expected_claim_type="quantitative",
                caliber_id="some_caliber",
            )

    def test_validator_rejects_comparative_without_axis(self) -> None:
        with pytest.raises(ValidationError, match="comparison_axis"):
            Slot(
                slot_id="bad_cmp",
                question="Compare US and China",
                expected_claim_type="comparative",
            )


# -----------------------------------------------------------------------------
# 验收 2: build_skeleton 正确
# -----------------------------------------------------------------------------

class TestBuildSkeleton:
    def test_skeleton_has_correct_sections(self, framework: Framework, skeleton: Report) -> None:
        assert len(skeleton.sections) == len(framework.sections)
        # 命名 section
        for orig, built in zip(framework.sections, skeleton.sections):
            assert built.section_id == orig.section_id
            assert built.title == orig.title

    def test_skeleton_has_correct_slots(self, framework: Framework, skeleton: Report) -> None:
        orig_total = sum(len(s.slots) for s in framework.sections)
        built_total = sum(len(s.slots) for s in skeleton.sections)
        assert orig_total == built_total

    def test_skeleton_copies_slot_details(self, skeleton: Report) -> None:
        # 检查第一槽的字段传递正确
        first_slot = skeleton.sections[0].slots[0]
        assert first_slot.slot_id
        assert first_slot.expected_claim_type


# -----------------------------------------------------------------------------
# 验收 3: slot_fill_rate
# -----------------------------------------------------------------------------

class TestSlotFillRate:
    def test_empty_skeleton_zero_fill(self, skeleton: Report) -> None:
        # plan F3 DoD:空骨架 slot_fill_rate == 0
        assert skeleton.slot_fill_rate() == 0.0

    def test_fill_one_slot_updates_metric(self, skeleton: Report) -> None:
        # plan F3 DoD:填 1 槽后指标更新
        skeleton.mark_filled(skeleton.sections[0].slots[0].slot_id)
        total = sum(len(s.slots) for s in skeleton.sections)
        assert skeleton.slot_fill_rate() == pytest.approx(1.0 / total)
        assert skeleton.slot_fill_rate() > 0.0

    def test_fill_all_slots_one_fill(self, skeleton: Report) -> None:
        for sec in skeleton.sections:
            for slot in sec.slots:
                skeleton.mark_filled(slot.slot_id)
        assert skeleton.slot_fill_rate() == 1.0

    def test_filled_and_unfilled_helpers(self, skeleton: Report) -> None:
        target = skeleton.sections[0].slots[0].slot_id
        skeleton.mark_filled(target)
        assert target in skeleton.filled_slot_ids()
        assert skeleton.unfilled_slot_ids() and target not in skeleton.unfilled_slot_ids()

    def test_get_slot(self, skeleton: Report) -> None:
        target = skeleton.sections[0].slots[0]
        got = skeleton.get_slot(target.slot_id)
        assert got is not None
        assert got.slot_id == target.slot_id
        assert skeleton.get_slot("nonexistent") is None

    def test_empty_report_fill_rate_is_zero(self) -> None:
        empty = Report(vertical_id="x", title="x", sections=[])
        assert empty.slot_fill_rate() == 0.0


# -----------------------------------------------------------------------------
# 验收 4: 跨 section slot_id 唯一
# -----------------------------------------------------------------------------

class TestSlotIdUniqueness:
    def test_within_section(self) -> None:
        with pytest.raises(ValidationError, match="duplicate slot_id"):
            Section(
                section_id="sec",
                title="Sec",
                slots=[
                    Slot(
                        slot_id="dup",
                        question="first",
                        expected_claim_type="qualitative",
                    ),
                    Slot(
                        slot_id="dup",
                        question="second",
                        expected_claim_type="qualitative",
                    ),
                ],
            )

    def test_across_sections(self) -> None:
        with pytest.raises(ValidationError, match="duplicate slot_id"):
            Framework(
                vertical_id="v",
                title="V",
                sections=[
                    Section(
                        section_id="sec_a",
                        title="A",
                        slots=[
                            Slot(
                                slot_id="shared",
                                question="first",
                                expected_claim_type="qualitative",
                            ),
                        ],
                    ),
                    Section(
                        section_id="sec_b",
                        title="B",
                        slots=[
                            Slot(
                                slot_id="shared",
                                question="second",
                                expected_claim_type="qualitative",
                            ),
                        ],
                    ),
                ],
            )


# -----------------------------------------------------------------------------
# 验收 5: cross-check with F2 caliber registry
# -----------------------------------------------------------------------------

class TestCrossFramework:
    def test_all_caliber_ids_referenced_exist_in_f2(self, framework: Framework) -> None:
        """所有 quantitative slot 引用的 caliber_id 都应在 F2 中存在。"""
        from open_deep_research.evidence.caliber_registry import (
            load_caliber_registry,
        )
        calibers = load_caliber_registry(
            "us_livecommerce", base_dir="data/registry/calibers"
        )
        for sec in framework.sections:
            for slot in sec.slots:
                if slot.caliber_id:
                    assert calibers.get(slot.caliber_id) is not None, (
                        f"slot {slot.slot_id} references unknown caliber_id "
                        f"{slot.caliber_id}"
                    )


# -----------------------------------------------------------------------------
# 验收 6: loader 错误处理
# -----------------------------------------------------------------------------

class TestLoaderErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_framework("nonexistent", base_dir=tmp_path)
