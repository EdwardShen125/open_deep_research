"""A1 · ClaimV3 schema + DB migration 验收测试

按 SPEC §5 A1:
- migration up/down 幂等(文件语法 + DROP IF EXISTS 存在)
- 旧行回填后 verification_status=='to_verify'(default 满足)
- 新增列非空约束正确
- 单测覆盖序列化
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from open_deep_research.evidence.claim_v3 import (
    ClaimV3,
    GateResults,
    Tier,
    VerificationStatus,
    EntailVerdict,
)
from open_deep_research.evidence.schema import EvidenceUnitV2, ClaimType


MIGRATION_PATH = Path("migrations/004_claim_v3_columns.sql")


# -----------------------------------------------------------------------------
# 验收 1: ClaimV3 基本构造
# -----------------------------------------------------------------------------

class TestClaimV3Basics:
    def test_minimal_construct(self) -> None:
        c = ClaimV3(value="EDR is growing", source_id="https://example.com/x")
        assert c.value == "EDR is growing"
        assert c.verification_status == "to_verify"  # default
        assert c.gate_results.span is None
        assert c.tier is None
        assert c.caliber_id is None

    def test_explicit_tier(self) -> None:
        c = ClaimV3(value="x", source_id="u", tier="A")
        assert c.tier == "A"

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClaimV3(value="x", source_id="u", tier="Z")  # type: ignore[arg-type]

    def test_invalid_verification_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ClaimV3(
                value="x", source_id="u",
                verification_status="maybe",  # type: ignore[arg-type]
            )


# -----------------------------------------------------------------------------
# 验收 2: GateResults 与 verification_status 一致性
# -----------------------------------------------------------------------------

class TestGateConsistency:
    def test_verified_status_no_constraint(self) -> None:
        # verified 不强制 entail (A3a + A3b 都过才 verified,但 schema 只校验 failed_gate)
        c = ClaimV3(
            value="x", source_id="u",
            verification_status="verified",
            gate_results=GateResults(span=True, drift=0.01, entail="entailed"),
        )
        assert c.verification_status == "verified"

    def test_failed_gate_requires_contradicted(self) -> None:
        # failed_gate 时 entail 必须是 contradicted
        with pytest.raises(ValidationError, match="contradicted"):
            ClaimV3(
                value="x", source_id="u",
                verification_status="failed_gate",
                gate_results=GateResults(span=True, entail="entailed"),
            )

    def test_failed_gate_with_contradicted_passes(self) -> None:
        c = ClaimV3(
            value="x", source_id="u",
            verification_status="failed_gate",
            gate_results=GateResults(span=True, entail="contradicted"),
        )
        assert c.verification_status == "failed_gate"
        assert c.gate_results.entail == "contradicted"

    def test_to_verify_with_anything(self) -> None:
        c = ClaimV3(
            value="x", source_id="u",
            verification_status="to_verify",
            gate_results=GateResults(span=False, entail="unknown"),
        )
        assert c.verification_status == "to_verify"


# -----------------------------------------------------------------------------
# 验收 3: numeric claim 必须有 norm_value
# -----------------------------------------------------------------------------

class TestNumericConstraint:
    def test_numeric_with_norm_value(self) -> None:
        c = ClaimV3(
            value="146.4B USD in 2025",
            source_id="u",
            claim_type="numeric",
            norm_value=146.4,
            unit="USD_B",
        )
        assert c.norm_value == 146.4

    def test_numeric_without_norm_value_rejected(self) -> None:
        with pytest.raises(ValidationError, match="norm_value"):
            ClaimV3(value="146.4B", source_id="u", claim_type="numeric")


# -----------------------------------------------------------------------------
# 验收 4: from_eu() 转换
# -----------------------------------------------------------------------------

class TestFromEU:
    def _make_eu(self, **overrides) -> EvidenceUnitV2:
        defaults = {
            "run_id": uuid4(),
            "claim": "US live commerce market size in 2025 is $146.4B",
            "claim_type": "numeric",
            "entities": ["us_live_commerce"],
            "norm_value": Decimal("146.4"),
            "unit": "USD_B",
            "value_as_of": date(2025, 12, 31),
            "source_url": "https://www.emarketer.com/chart/123",
            "source_domain": "emarketer.com",
            "source_title": "US Live Commerce 2025 Forecast",
            "published_at": datetime(2025, 3, 15, tzinfo=timezone.utc),
            "source_tier": "primary",
            "source_span": "US live commerce is forecast to reach $146.4B in 2025.",
            "extractor_model": "gpt-4o-mini",
        }
        defaults.update(overrides)  # type: ignore[arg-type]
        return EvidenceUnitV2(**defaults)  # type: ignore[arg-type]

    def test_from_eu_default_status_to_verify(self) -> None:
        eu = self._make_eu()
        c = ClaimV3.from_eu(eu)
        assert c.value == eu.claim
        assert c.source_id == str(eu.eu_id)
        assert c.tier is None
        assert c.caliber_id is None
        assert c.verification_status == "to_verify"
        assert c.gate_results.span is None

    def test_from_eu_infers_verified_when_eu_usable(self) -> None:
        # eu.usable = span_verified && !numeric_drift && entail ∈ {entailed, partial}
        eu = self._make_eu(
            span_verified=True,
            numeric_drift=False,
            entailment_verdict="entailed",
        )
        c = ClaimV3.from_eu(eu)
        assert c.verification_status == "verified"
        assert c.gate_results.span is True
        assert c.gate_results.entail == "entailed"

    def test_from_eu_infers_failed_gate_on_contradicted(self) -> None:
        eu = self._make_eu(
            span_verified=True,
            numeric_drift=False,
            entailment_verdict="contradicted",
        )
        c = ClaimV3.from_eu(eu)
        assert c.verification_status == "failed_gate"
        assert c.gate_results.entail == "contradicted"

    def test_from_eu_maps_partial_to_entailed(self) -> None:
        eu = self._make_eu(
            span_verified=True,
            numeric_drift=False,
            entailment_verdict="partial",
        )
        c = ClaimV3.from_eu(eu)
        assert c.gate_results.entail == "entailed"

    def test_from_eu_maps_unverifiable_to_unknown(self) -> None:
        eu = self._make_eu(
            span_verified=False,
            numeric_drift=False,
            entailment_verdict="unverifiable",
        )
        c = ClaimV3.from_eu(eu)
        assert c.gate_results.entail == "unknown"

    def test_from_eu_with_explicit_overrides(self) -> None:
        eu = self._make_eu()
        c = ClaimV3.from_eu(
            eu,
            tier="A",
            caliber_id="us_livestream_retail_emarketer",
            embedding_model="BGE-M3",
            origin_chain=["emarketer.com -> reuters.com"],
        )
        assert c.tier == "A"
        assert c.caliber_id == "us_livestream_retail_emarketer"
        assert c.embedding_model == "BGE-M3"
        assert c.origin_chain == ["emarketer.com -> reuters.com"]

    def test_from_eu_norm_value_decimal_to_float(self) -> None:
        eu = self._make_eu(norm_value=Decimal("678.5"))
        c = ClaimV3.from_eu(eu)
        assert c.norm_value == 678.5
        assert isinstance(c.norm_value, float)


# -----------------------------------------------------------------------------
# 验收 5: to_pg_update_dict() 序列化
# -----------------------------------------------------------------------------

class TestPGSerialization:
    def test_default_to_pg(self) -> None:
        c = ClaimV3(value="x", source_id="u")
        d = c.to_pg_update_dict()
        assert d["tier"] is None
        assert d["caliber_id"] is None
        assert d["verification_status"] == "to_verify"
        assert isinstance(d["gate_results"], str)  # JSON string
        gate_dict = json.loads(d["gate_results"])
        assert gate_dict == {"span": None, "drift": None, "entail": None}
        assert d["origin_chain"] == []
        assert d["embedding_model"] is None

    def test_filled_to_pg(self) -> None:
        c = ClaimV3(
            value="x", source_id="u",
            tier="B",
            caliber_id="us_video_shopping_broad",
            verification_status="verified",
            gate_results=GateResults(span=True, drift=0.02, entail="entailed"),
            origin_chain=["a.com", "b.com"],
            embedding_model="BGE-M3",
        )
        d = c.to_pg_update_dict()
        assert d["tier"] == "B"
        assert d["caliber_id"] == "us_video_shopping_broad"
        assert d["verification_status"] == "verified"
        gate_dict = json.loads(d["gate_results"])
        assert gate_dict == {"span": True, "drift": 0.02, "entail": "entailed"}
        assert d["origin_chain"] == ["a.com", "b.com"]
        assert d["embedding_model"] == "BGE-M3"


# -----------------------------------------------------------------------------
# 验收 6: migration 文件存在 + 语法结构正确
# -----------------------------------------------------------------------------

class TestMigrationFile:
    def test_migration_file_exists(self) -> None:
        assert MIGRATION_PATH.exists(), f"migration not found: {MIGRATION_PATH}"

    def test_migration_adds_required_columns(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        for col in [
            "tier",
            "caliber_id",
            "verification_status",
            "gate_results",
            "origin_chain",
            "embedding_model",
        ]:
            assert col in sql, f"migration missing column: {col}"

    def test_migration_has_default_for_verification_status(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        # plan A1 DoD:旧行回填后 verification_status='to_verify'
        assert "DEFAULT 'to_verify'" in sql, (
            "migration must DEFAULT 'to_verify' for verification_status "
            "to satisfy plan DoD '旧行回填后 verification_status==\"to_verify\"'"
        )

    def test_migration_creates_indexes(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        for idx in ["idx_eu_tier", "idx_eu_verification", "idx_eu_caliber"]:
            assert idx in sql, f"migration missing index: {idx}"

    def test_migration_uses_if_not_exists_for_idempotency(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        # plan A1 DoD:migration up/down 幂等
        assert "ADD COLUMN IF NOT EXISTS" in sql
        assert "CREATE INDEX IF NOT EXISTS" in sql

    def test_migration_has_drop_if_exists_in_comments(self) -> None:
        # plan A1 DoD:迁移可上可下
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        assert "DROP COLUMN IF EXISTS" in sql, (
            "migration must document DROP COLUMN IF EXISTS in down section"
        )

    def test_migration_targets_evidence_evidence_unit(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        assert "evidence.evidence_unit" in sql

    def test_gate_results_is_jsonb(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        assert "gate_results JSONB" in sql

    def test_origin_chain_is_text_array(self) -> None:
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        assert "origin_chain TEXT[]" in sql


# -----------------------------------------------------------------------------
# 验收 7: 与 plan § 共享上下文一致(verify 模块符号完整)
# -----------------------------------------------------------------------------

class TestModuleSurface:
    def test_exports(self) -> None:
        from open_deep_research.evidence import claim_v3
        for sym in ["Tier", "VerificationStatus", "EntailVerdict", "GateResults", "ClaimV3"]:
            assert hasattr(claim_v3, sym), f"missing export: {sym}"
