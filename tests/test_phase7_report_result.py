"""阶段 5 = Runbook v1 阶段 5: ReportResult 结构化输出 + fallback 信号硬化测试。

验收:
1. ReportResult.ok 字段是硬信号 — ok=False 时调用方必须知道失败
2. status 枚举区分 ok / partial / fallback_used / failed
3. fallback_used 状态携带 failure 元信息但 body 仍可读
4. ClaimStats.from_claim_list 推导出 grade_dist 与 has_conflict
5. ReportResult 序列化进 LangGraph state 不破(向后兼容 final_report: str)

不验证:
- 真 PG 集成测试 — pgvector 缺失(阶段 7 一起处理)
- deep_researcher 端到端 — 需要真 writer LLM
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import pytest

from open_deep_research.evidence import (
    ClaimStats,
    ClaimV2,
    Failure,
    ReportResult,
    ReportSection,
    is_report_success,
)


# =============================================================================
# 1. ReportResult 基础结构
# =============================================================================


class TestReportResultBasics:
    def test_default_construction(self):
        r = ReportResult(ok=True, status="ok")
        assert r.ok is True
        assert r.status == "ok"
        assert r.body_markdown == ""
        assert r.failures == []
        assert r.warnings == []

    def test_from_markdown_and_status_ok(self):
        r = ReportResult.from_markdown_and_status("# My Report", status="ok")
        assert r.ok is True
        assert r.status == "ok"
        assert r.body_markdown == "# My Report"

    def test_from_markdown_and_status_partial(self):
        """partial: writer 部分成功 / verifier flag 了 critical 但 body 完整"""
        r = ReportResult.from_markdown_and_status(
            "# Report",
            status="partial",
            warnings=["1 critical issue"],
        )
        assert r.ok is True
        assert r.status == "partial"
        assert "1 critical issue" in r.warnings

    def test_from_markdown_and_status_fallback(self):
        """fallback_used: writer 失败, 用 _render_eu_digest 兜底"""
        r = ReportResult.from_markdown_and_status(
            "# Digest",
            status="fallback_used",
            failures=[
                Failure(stage="write", error_type="ReadTimeout", error_message="network")
            ],
        )
        assert r.ok is True  # 兜底也算"产出"
        assert r.status == "fallback_used"
        assert is_report_success(r) is False  # 但 success=False
        assert len(r.failures) == 1
        assert r.failures[0].stage == "write"

    def test_from_markdown_and_status_failed(self):
        """failed: 严重错误, body 可能为空"""
        r = ReportResult.from_markdown_and_status(
            "",
            status="failed",
            failures=[
                Failure(stage="write", error_type="ValueError", error_message="bad input"),
                Failure(stage="merge", error_type="ConnectionError", error_message="db down"),
            ],
        )
        assert r.ok is False
        assert r.status == "failed"
        assert r.has_failures is True
        assert is_report_success(r) is False
        assert len(r.failures) == 2


class TestIsReportSuccess:
    """is_report_success: 给上层用, 只看 status=='ok'."""

    def test_ok_returns_true(self):
        r = ReportResult.from_markdown_and_status("body", status="ok")
        assert is_report_success(r) is True

    def test_partial_returns_false(self):
        r = ReportResult.from_markdown_and_status("body", status="partial")
        # partial 不算 success — 调用方需要看 warnings
        assert is_report_success(r) is False

    def test_fallback_returns_false(self):
        r = ReportResult.from_markdown_and_status("body", status="fallback_used")
        assert is_report_success(r) is False

    def test_failed_returns_false(self):
        r = ReportResult.from_markdown_and_status("body", status="failed")
        assert is_report_success(r) is False


# =============================================================================
# 2. P0 验收: fallback 伪装成功拦截
# =============================================================================


class TestFallbackDisguise:
    """核心验收: 调用方看到 fallback_used 时不能误以为调研成功。

    之前的 bug: writer LLM 失败时把错误字符串写到 final_report,
    调用方看到非空字符串就以为"调研成功了"。
    整改: ReportResult.status='fallback_used' 显式标出 + is_report_success() == False。
    """

    def test_fallback_used_status_with_full_body(self):
        """fallback_used 时 body 仍非空 (digest 兜底), 但 status 暴露真相。"""
        r = ReportResult.from_markdown_and_status(
            "# Raw Evidence Digest\n\n- eu_count=17\n...",
            status="fallback_used",
            warnings=["writer LLM exhausted retries; rendered EU digest fallback"],
        )
        # 旧 API 视角: body 非空 → "有报告"
        assert r.body_markdown != ""
        # 新 API 视角: status 暴露真相
        assert r.status == "fallback_used"
        assert is_report_success(r) is False

    def test_failed_status_with_empty_body(self):
        r = ReportResult.from_markdown_and_status(
            "",
            status="failed",
            failures=[Failure(stage="write", error_type="OOM", error_message="mem")],
        )
        assert r.body_markdown == ""
        assert r.ok is False
        assert r.has_failures is True

    def test_partial_status_distinguishable_from_ok(self):
        """partial 必须 ≠ ok — 否则调用方不会去看 verifier warnings。"""
        ok_r = ReportResult.from_markdown_and_status("# R", status="ok")
        part_r = ReportResult.from_markdown_and_status("# R", status="partial", warnings=["critical flagged"])
        assert ok_r.status != part_r.status
        assert is_report_success(ok_r) is True
        assert is_report_success(part_r) is False


# =============================================================================
# 3. ClaimStats 与 grade 分布
# =============================================================================


def _make_claim(rid: Any, *, grade: str, indep: int = 1, primary: int = 0, conflict: bool = False) -> ClaimV2:
    """便利构造 — ClaimV2 + grade 字段。"""
    return ClaimV2(
        claim_id=str(uuid4()),
        run_id=rid,
        dimension_id=f"d-{grade}",
        canonical_claim=f"test claim grade {grade}",
        claim_type="numeric",
        entities=["X"],
        eu_count=indep,
        independent_source_count=indep,
        primary_source_count=primary,
        grade=grade,
        grade_reason=f"test grade {grade}",
        has_conflict=conflict,
    )


class TestClaimStats:
    def test_empty_claims_returns_zero_stats(self):
        stats = ClaimStats.from_claim_list([])
        assert stats.total_claims == 0
        assert stats.primary_claims == 0
        assert stats.unverified_claims == 0

    def test_grade_distribution_pct(self):
        rid = uuid4()
        claims = [
            _make_claim(rid, grade="A", indep=2, primary=1),
            _make_claim(rid, grade="A", indep=3, primary=2),
            _make_claim(rid, grade="B", indep=1, primary=1),
            _make_claim(rid, grade="C", indep=1),
            _make_claim(rid, grade="D", indep=0),
        ]
        stats = ClaimStats.from_claim_list(claims)
        assert stats.total_claims == 5
        assert stats.primary_claims == 2
        assert stats.secondary_claims == 1
        assert stats.tertiary_claims == 1
        assert stats.unverified_claims == 1
        assert stats.grade_dist_pct == {
            "A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2,
        }

    def test_conflict_counter(self):
        rid = uuid4()
        claims = [
            _make_claim(rid, grade="A", conflict=True),
            _make_claim(rid, grade="A"),
            _make_claim(rid, grade="C", conflict=True),
        ]
        stats = ClaimStats.from_claim_list(claims)
        assert stats.has_conflict == 2

    def test_eu_count_propagates(self):
        from open_deep_research.evidence import EvidenceUnitV2
        from datetime import datetime, timezone
        rid = uuid4()
        eus = [
            EvidenceUnitV2(
                eu_id=str(uuid4()), run_id=rid, dimension_id="d1",
                claim="test claim 1", claim_type="attribute", entities=["X"],
                norm_value=None,
                source_url=f"https://example.com/{i}",
                source_domain="example.com",
                source_tier="primary",
                source_span="some sufficiently long span text content here",
                extractor_model="test-model",
                extracted_at=datetime.now(timezone.utc),
                span_verified=True, numeric_drift=False,
                entailment_verdict="entailed", entailment_score=0.9,
            )
            for i in range(3)
        ]  # noqa: E501
        stats = ClaimStats.from_claim_list([], eus=eus)
        assert stats.total_eus == 3
        assert stats.usable_eus == 3
        assert stats.unique_sources == 3
        assert stats.unique_primary_sources == 3


# =============================================================================
# 4. JSON 序列化与 LangGraph state 兼容
# =============================================================================


class TestReportResultSerialization:
    def test_model_dump_mode_json(self):
        """ReportResult 序列化进 LangGraph state 时不应报错。"""
        r = ReportResult.from_markdown_and_status(
            "# body",
            status="partial",
            failures=[Failure(stage="write", error_type="X", error_message="y")],
        )
        dumped = r.model_dump(mode="json")
        assert isinstance(dumped, dict)
        assert dumped["ok"] is True
        assert dumped["status"] == "partial"
        assert len(dumped["failures"]) == 1

    def test_failures_serializable(self):
        f = Failure(stage="write", error_type="X", error_message="y")
        d = f.model_dump(mode="json")
        assert d["stage"] == "write"
        assert isinstance(d["timestamp"], str)  # datetime → ISO string

    def test_attach_to_report_result(self):
        stats = ClaimStats(total_claims=5, primary_claims=2)
        r = ReportResult(
            ok=True, status="ok",
            body_markdown="body",
            claim_stats=stats,
        )
        assert r.claim_stats.primary_claims == 2


# =============================================================================
# 5. Markdown 渲染 — 包含 stats / failures 信息
# =============================================================================


class TestReportMarkdown:
    def test_to_markdown_with_warnings_includes_warning_block(self):
        r = ReportResult.from_markdown_and_status(
            "body",
            status="partial",
            warnings=["verifier flagged 1 critical issue"],
        )
        md = r.to_markdown_with_warnings()
        assert "⚠️ Warnings" in md
        assert "verifier flagged 1 critical issue" in md
        assert md.endswith("body")  # body 在最后

    def test_to_markdown_with_stats_block(self):
        stats = ClaimStats(total_claims=10, primary_claims=4, secondary_claims=3, tertiary_claims=2, unverified_claims=1)
        r = ReportResult(ok=True, status="ok", body_markdown="body", claim_stats=stats)
        md = r.to_markdown_with_warnings()
        assert "📊 Evidence Stats" in md
        assert "Total Claims: 10" in md
        assert "A:4" in md

    def test_to_markdown_with_failures_block(self):
        r = ReportResult.from_markdown_and_status(
            "body",
            status="failed",
            failures=[Failure(stage="write", error_type="Timeout", error_message="30s wait")],
        )
        md = r.to_markdown_with_warnings()
        assert "❌ Failures" in md
        assert "Timeout" in md


# =============================================================================
# 6. ReportSection 基础
# =============================================================================


class TestReportSection:
    def test_section_metadata(self):
        s = ReportSection(
            section_id="s1",
            title="Kompyte 营收",
            body_markdown="2024 年 ARR 是 $12M",
            claim_ids=["c1"],
            eu_ids=["e1", "e2"],
            grade="A",
            confidence=0.92,
        )
        assert s.grade == "A"
        assert s.confidence == 0.92
        assert s.claim_ids == ["c1"]
        assert s.eu_ids == ["e1", "e2"]

    def test_attach_to_result(self):
        s1 = ReportSection(section_id="s1", title="A")
        s2 = ReportSection(section_id="s2", title="B")
        r = ReportResult(ok=True, status="ok", sections=[s1, s2])
        assert len(r.sections) == 2
