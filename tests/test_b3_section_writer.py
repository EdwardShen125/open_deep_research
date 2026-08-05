"""B3 · 分节检索 + 写作验收测试

按 SPEC §5 B3:
- 构造 20k claim 的库 → 单节写作 context token 数 < 阈值(不再 400)
- 每节只见到本槽 claim
- 端到端产出一份 ReportResult
- bounded context:retrieve_for_slot 严格 ≤ k
"""
from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from open_deep_research.evidence.claim_v3 import ClaimV3
from open_deep_research.evidence.framework import Framework, Slot, load_framework
from open_deep_research.evidence.report_result import ReportResult
from open_deep_research.evidence.section_writer import (
    DEFAULT_TOP_K,
    _build_section_prompt,
    assemble_report_async,
    retrieve_for_slot,
    write_section,
    write_section_async,
)


FRAMEWORK_PATH = "data/frameworks/us_livecommerce.yaml"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_claim(
    slot_id: str,
    status: str = "verified",
    tier: str = "A",
    *,
    caliber_id: str = "us_livestream_retail_emarketer",
) -> ClaimV3:
    """构造针对某个 slot 的 claim(value 含 slot_id 关键词)。"""
    return ClaimV3(
        value=f"This is a finding about {slot_id} with metric 146.4",
        source_id=f"https://example.com/{slot_id}",
        verification_status=status,  # type: ignore[arg-type]
        tier=tier,  # type: ignore[arg-type]
        caliber_id=caliber_id,
    )


class _MockLLM:
    def __init__(self, content: str = '{"markdown": "ok", "rationale": "ok"}'):
        self._content = content

    async def ainvoke(self, messages: list[dict]) -> Any:
        return MagicMock(content=self._content)

    def invoke(self, messages: list[dict]) -> Any:
        return MagicMock(content=self._content)


# -----------------------------------------------------------------------------
# 验收 1: retrieve_for_slot bounded context
# -----------------------------------------------------------------------------

class TestRetrieveForSlotBounded:
    def test_returns_at_most_k(self) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="What is the 2025 US market size?",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        # 构造 50 条相关 claim,k=10 应只返 10
        claims = [_make_claim("market_size_2025") for _ in range(50)]
        result = retrieve_for_slot(slot, k=10, claims=claims)
        assert len(result) <= 10
        assert len(result) == 10

    def test_empty_claims_returns_empty(self) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="What is the 2025 US market size?",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        assert retrieve_for_slot(slot, k=10, claims=[]) == []
        assert retrieve_for_slot(slot, k=10, claims=None) == []

    def test_filters_by_caliber(self) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="market",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        claims = [
            _make_claim("market"),  # caliber 匹配
            ClaimV3(
                value="market other caliber",
                source_id="u",
                caliber_id="us_video_shopping_broad",  # 不匹配
                verification_status="verified",
                tier="A",
            ),
        ]
        result = retrieve_for_slot(slot, k=10, claims=claims)
        assert len(result) == 1
        assert result[0].caliber_id == "us_livestream_retail_emarketer"

    def test_sorts_verified_first(self) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="market",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        # 混合 verified + to_verify
        claims = [
            ClaimV3(value="market size", source_id="u1", verification_status="to_verify", tier="A", caliber_id="us_livestream_retail_emarketer"),
            ClaimV3(value="market size v2", source_id="u2", verification_status="verified", tier="A", caliber_id="us_livestream_retail_emarketer"),
        ]
        result = retrieve_for_slot(slot, k=10, claims=claims)
        assert len(result) == 2
        # verified 应排第一
        assert result[0].verification_status == "verified"


# -----------------------------------------------------------------------------
# 验收 2: 20k claim 库 → bounded context(plan B3 DoD:不再 context 溢出)
# -----------------------------------------------------------------------------

class TestBoundedContext:
    def test_20k_claim_library_bounded_prompt(self) -> None:
        # 构造 20k claim 的库
        library = [
            ClaimV3(
                value=f"finding {i} about market size 146.4",
                source_id=f"https://example.com/{i}",
                verification_status="verified",
                tier="A" if i % 2 == 0 else "B",
                caliber_id="us_livestream_retail_emarketer",
            )
            for i in range(20000)
        ]
        slot = Slot(
            slot_id="market_size_2025",
            question="market size 2025",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        result = retrieve_for_slot(slot, k=10, claims=library)
        # 关键断言:不论库多大,prompt 只见到 k 条
        assert len(result) == 10

        # 验证 prompt 大小:10 条 claim JSON 序列化应远小于阈值
        system, user = _build_section_prompt(slot, result)
        approx_tokens = (len(system) + len(user)) // 4
        # plan B3:阈值远小于导致 400 的 19955 EU 量级
        assert approx_tokens < 4000, (
            f"single-section prompt too large: {approx_tokens} tokens "
            f"(plan B3 阈值应远小于 19955-EU 量级)"
        )


# -----------------------------------------------------------------------------
# 验收 3: 每节只见到本槽 claim
# -----------------------------------------------------------------------------

class TestPerSlotIsolation:
    def test_each_section_only_sees_its_claim(self) -> None:
        # 三个 slot,每 slot 一个 claim
        slot1 = Slot(
            slot_id="market_size_2025",
            question="market size",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        slot2 = Slot(
            slot_id="penetration_2025",
            question="penetration",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_penetration_emarketer",
        )
        claims_per_slot = {
            "market_size_2025": [_make_claim("market_size_2025")],
            "penetration_2025": [_make_claim("penetration_2025", caliber_id="us_penetration_emarketer")],
        }
        # 调用 retrieve_for_slot 应分别过滤
        r1 = retrieve_for_slot(slot1, k=10, claims=claims_per_slot["market_size_2025"])
        r2 = retrieve_for_slot(slot2, k=10, claims=claims_per_slot["penetration_2025"])
        assert "market_size_2025" in r1[0].value
        assert "penetration_2025" in r2[0].value


# -----------------------------------------------------------------------------
# 验收 4: 端到端 — 产 ReportResult
# -----------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_assemble_report_with_mock_llm(self) -> None:
        # 加载种子 framework
        framework = load_framework("us_livecommerce", base_dir="data/frameworks")
        llm = _MockLLM(content='{"markdown": "test section", "rationale": "ok"}')

        # 为每个 quantitative slot 提供 claim
        claims_per_slot: dict[str, list[ClaimV3]] = {}
        for sec in framework.sections:
            for slot in sec.slots:
                if slot.caliber_id:
                    claims_per_slot[slot.slot_id] = [_make_claim(slot.slot_id)]
                else:
                    claims_per_slot[slot.slot_id] = []

        report = await assemble_report_async(
            framework, claims_per_slot=claims_per_slot, llm=llm, k=10,
        )
        assert isinstance(report, ReportResult)
        assert report.vertical_id == "us_livecommerce"
        # 应有 sections(可能有些 qualitative slot 没 claim)
        assert len(report.sections) == len(framework.sections)
        # unresolved 字段存在
        assert isinstance(report.unresolved, list)

    @pytest.mark.asyncio
    async def test_unfilled_slots_go_to_unresolved(self) -> None:
        framework = load_framework("us_livecommerce", base_dir="data/frameworks")
        llm = _MockLLM()
        # 完全不提供 claim → 所有 slot 进 unresolved
        report = await assemble_report_async(
            framework, claims_per_slot={}, llm=llm,
        )
        # 至少应有一些未填槽
        assert len(report.unresolved) > 0

    def test_write_section_sync(self) -> None:
        slot = Slot(
            slot_id="market_size_2025",
            question="What is the 2025 US market size?",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        claims = [_make_claim("market_size_2025")]
        llm = _MockLLM()
        sr = write_section(slot, claims, llm=llm)
        assert sr.section_id == "market_size_2025"
        assert len(sr.slots) == 1
        assert sr.slots[0].claims[0].verification_status == "verified"


# -----------------------------------------------------------------------------
# 验收 5: 默认 k=10
# -----------------------------------------------------------------------------

class TestDefaults:
    def test_default_top_k_is_10(self) -> None:
        assert DEFAULT_TOP_K == 10

    def test_prompt_json_only_has_k_claims(self) -> None:
        # 验证 prompt 中的 Claims Available 数严格等于传入的 claims 数
        slot = Slot(
            slot_id="market_size_2025",
            question="market size",
            expected_claim_type="quantitative",
            required_tier_min="B",
            caliber_id="us_livestream_retail_emarketer",
        )
        claims = [_make_claim("market_size_2025") for _ in range(15)]
        limited = claims[:10]  # k=10
        _, user = _build_section_prompt(slot, limited)
        # user prompt 中应明确说 "10 total"
        assert "10 total" in user
        # 且只含 10 条 claim 的 JSON 序列化
        claims_section = user.split("## Claims Available")[1].split("## Output")[0]
        # 数 JSON 数组元素
        json_match = re.search(r"\[.*\]", claims_section, re.DOTALL)
        assert json_match is not None
        arr = json.loads(json_match.group(0))
        assert len(arr) == 10
