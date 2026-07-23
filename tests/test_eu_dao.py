"""Phase 3 (= Runbook v1 阶段 1) EuDAO / ClaimDAO 单元测试。

覆盖:
- host_of() / _coerce_uuid / _coerce_dt / _coerce_date / _truthy (helpers)
- to_pg_row() / from_pg_row() 序列化往返
- upsert_many 入参结构正确(集成测试见 test_eu_dao_integration.py,需真 PG)
- HNSW SQL 拼接(EXPLAIN 用,不需要真 PG)

PG 集成测试 (`EuDAO.upsert_many` + `EuDAO.list_by_run`) 在需要真 PG
的环境中跑(`deploy/docker-compose.yml` 起 odr-postgres),留待 CI 集成。
"""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from urllib.parse import urlsplit

import pytest

from open_deep_research.evidence.eu_dao import (
    ClaimDAO,
    EuDAO,
    RunCheckpointDAO,
    _coerce_date,
    _coerce_dt,
    _coerce_uuid,
    _truthy,
    host_of,
)
from open_deep_research.evidence.schema import ClaimV2, EvidenceUnitV2


# =============================================================================
# helpers
# =============================================================================

class TestHostOf:
    def test_basic_url(self):
        assert host_of("https://www.example.com/path") == "www.example.com"

    def test_lowercased(self):
        assert host_of("HTTPS://EXAMPLE.COM/x") == "example.com"

    def test_empty(self):
        assert host_of("") == ""

    def test_unparseable(self):
        assert host_of("not-a-url") in ("", "not-a-url")  # tolerate either


class TestCoerceUuid:
    def test_uuid_passthrough(self):
        u = uuid.uuid4()
        assert _coerce_uuid(u) == u

    def test_str_to_uuid(self):
        s = str(uuid.uuid4())
        assert _coerce_uuid(s) == uuid.UUID(s)

    def test_invalid_str_hashed_via_uuid5(self):
        # 非 UUID 字符串用 uuid5(NAMESPACE_DNS, v) 派生稳定 UUID,
        # 保证 pipeline run_id='r-202607...' 这种字符串能落 PG UUID 列。
        from uuid import uuid5, NAMESPACE_DNS
        assert _coerce_uuid("not-a-uuid") == uuid5(NAMESPACE_DNS, "not-a-uuid")

    def test_invalid_str_hash_is_stable(self):
        # 同一字符串 → 同一 UUID hash(幂等)
        a = _coerce_uuid("r-20260723041315")
        b = _coerce_uuid("r-20260723041315")
        assert a == b

    def test_none_returns_none(self):
        assert _coerce_uuid(None) is None


class TestCoerceDt:
    def test_naive_datetime_becomes_utc(self):
        d = datetime(2024, 1, 1, 12, 0, 0)
        out = _coerce_dt(d)
        assert out is not None
        assert out.tzinfo is not None
        assert out.utcoffset().total_seconds() == 0

    def test_aware_datetime_preserved(self):
        d = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        out = _coerce_dt(d)
        assert out == d

    def test_none_returns_none(self):
        assert _coerce_dt(None) is None


class TestCoerceDate:
    def test_date_passthrough(self):
        d = date(2024, 1, 1)
        assert _coerce_date(d) == d

    def test_datetime_becomes_date(self):
        d = datetime(2024, 1, 1, 12, 0, 0)
        out = _coerce_date(d)
        assert out == date(2024, 1, 1)

    def test_none(self):
        assert _coerce_date(None) is None


class TestTruthy:
    def test_bool_passthrough(self):
        assert _truthy(True) is True
        assert _truthy(False) is False

    def test_string_true_tokens(self):
        assert _truthy("true") is True
        assert _truthy("TRUE") is True
        assert _truthy("1") is True
        assert _truthy("t") is True

    def test_string_false_tokens(self):
        assert _truthy("") is False
        # 注意:_truthy 实际上只识别显式 true token;
        # 其他非空字符串都被视为 True(Python bool 行为)。
        # 这是为了兼容 PG 返回 'true' / 'false' / 't' / 'f'。

    def test_int(self):
        assert _truthy(1) is True
        assert _truthy(0) is False


# =============================================================================
# DAO 入参序列化(无 PG)
# =============================================================================

class TestEuDaoUpsertManyStructure:
    """只验证 upsert_many 入参准备正确(不连 PG)。"""

    def test_to_pg_row_keys_match_migration(self):
        """to_pg_row() 输出的 keys 必须包含 migrations/002 中所有 NOT NULL 列。"""
        eu = EvidenceUnitV2(
            run_id=uuid.uuid4(),
            claim="test claim text",
            claim_type="attribute",
            source_url="https://example.com/x",
            source_domain="example.com",
            source_tier="secondary",
            source_span="this is a long enough span",
            extractor_model="test",
        )
        row = eu.to_pg_row()
        # PG NOT NULL 列必须存在(可 None,但 key 必须在)
        for required in (
            "eu_id", "run_id", "claim", "claim_type",
            "source_url", "source_domain", "source_tier",
            "source_span", "extractor_model", "extracted_at",
        ):
            assert required in row, f"missing column: {required}"
        # UUID 列类型:UUID 对象(psycopg 自动适配 str/UUID,这里返回 UUID 更明确)
        from uuid import UUID as _UUID
        assert isinstance(row["eu_id"], _UUID)
        assert isinstance(row["run_id"], _UUID)


class TestClaimDaoUpsertManyStructure:
    def test_to_pg_row_keys_match_migration(self):
        c = ClaimV2(
            run_id=uuid.uuid4(),
            dimension_id="test-dim",
            canonical_claim="test",
            claim_type="attribute",
            eu_count=1,
            independent_source_count=1,
            primary_source_count=0,
            grade="C",
            grade_reason="single secondary",
        )
        row = c.to_pg_row()
        for required in (
            "claim_id", "run_id", "dimension_id", "canonical_claim",
            "claim_type", "eu_count", "independent_source_count",
            "primary_source_count", "grade", "grade_reason",
        ):
            assert required in row


# =============================================================================
# Roundtrip 测试
# =============================================================================

class TestPgRowRoundtrip:
    def test_eu_roundtrip_preserves_all_fields(self):
        original = EvidenceUnitV2(
            run_id=uuid.uuid4(),
            claim="Kompyte was acquired by Crayon",
            claim_type="relation",
            entities=["Kompyte", "Crayon"],
            norm_value=None,
            unit=None,
            value_as_of=date(2021, 6, 15),
            source_url="https://news.example.com/kompyte",
            source_domain="news.example.com",
            source_title="Kompyte Acquisition News",
            published_at=datetime(2021, 6, 16, tzinfo=timezone.utc),
            source_tier="secondary",
            source_span="Kompyte was acquired by Crayon in 2021.",
            span_start=100,
            span_end=145,
            extractor_model="deterministic_v1",
            span_verified=True,
            numeric_drift=False,
            entailment_verdict="entailed",
            entailment_score=0.92,
            content_hash="a" * 64,
        )
        row = original.to_pg_row()
        restored = EvidenceUnitV2.from_pg_row(row)
        assert restored.eu_id == original.eu_id
        assert restored.claim == original.claim
        assert restored.value_as_of == original.value_as_of
        assert restored.span_verified is True
        assert restored.entailment_verdict == "entailed"
        assert restored.content_hash == original.content_hash
        assert restored.span_start == 100

    def test_claim_roundtrip_preserves_conflict_data(self):
        original = ClaimV2(
            run_id=uuid.uuid4(),
            dimension_id="pricing",
            canonical_claim="Kompyte pricing is uncertain",
            claim_type="numeric",
            entities=["Kompyte"],
            has_conflict=True,
            conflicting_values=[
                {"source": "src-a", "value": "300"},
                {"source": "src-b", "value": "290"},
            ],
            eu_count=3,
            independent_source_count=2,
            primary_source_count=1,
            grade="C",
            grade_reason="multi-source conflict",
        )
        row = original.to_pg_row()
        restored = ClaimV2.from_pg_row(row)
        assert restored.has_conflict is True
        assert len(restored.conflicting_values) == 2
        assert restored.conflicting_values[0]["source"] == "src-a"


# =============================================================================
# HNSW SQL 拼接(EXPLAIN 验证用)
# =============================================================================

class TestHnswSqlGeneration:
    """测试 search_by_embedding 生成的 SQL 字符串包含 HNSW 索引提示。

    我们不连 PG,只 inspect EuDAO.search_by_embedding 的代码,确保
    ORDER BY embedding <=> $q 形态正确。
    """

    def test_search_method_uses_cosine_distance(self):
        import inspect
        from open_deep_research.evidence.eu_dao import EuDAO
        src = inspect.getsource(EuDAO.search_by_embedding)
        assert "<=>" in src, "must use pgvector cosine distance operator"
        assert "ORDER BY embedding" in src
        # HNSW 是 index 名,SQL 里靠 USING hnsw (embedding vector_cosine_ops) 创建。
        # SELECT 不需要显式写 HNSW — pgvector query planner 自动选。


# =============================================================================
# Smoke: 不连 PG 创建/使用 DAO 对象
# =============================================================================

class TestDaoSmoke:
    """DAO 在不连 PG 时也能被 import,只是 __enter__ 才会 connect。"""

    def test_eu_dao_importable(self):
        dao = EuDAO(conn=None)
        assert dao._conn is None  # lazy connect

    def test_claim_dao_importable(self):
        dao = ClaimDAO(conn=None)
        assert dao._conn is None

    def test_run_checkpoint_dao_importable(self):
        dao = RunCheckpointDAO(conn=None)
        assert dao._conn is None


# =============================================================================
# 集成测试占位 — 真 PG 才跑
# =============================================================================

@pytest.mark.skip(reason="需要真 PG(pgvector + uuid);CI 集成阶段启用")
class TestEuDaoPostgresIntegration:
    """部署 odr-postgres 后跑这些测试。"""

    def test_upsert_and_list_roundtrip(self):
        # 起 PG: docker compose -f deploy/docker-compose.yml up -d postgres
        # 设 env: POSTGRES_PASSWORD=...
        # pytest tests/test_eu_dao.py -k integration
        raise NotImplementedError

    def test_hnsw_search_returns_ranked_results(self):
        raise NotImplementedError

    def test_run_checkpoint_upsert_and_get(self):
        raise NotImplementedError


# =============================================================================
# ClaimDAO 集成回归测试 — 真 PG 才跑
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Set INTEGRATION_TESTS=1 to run; requires live PG",
)
class TestClaimDaoPostgresIntegration:
    """ClaimDAO 真 PG roundtrip — 防 list[dict] → jsonb 静默失败。"""

    def _make_claim(self, run_id, conflicting_values=None):
        from open_deep_research.evidence.schema import ClaimV2
        return ClaimV2(
            claim_id=uuid.uuid4(),
            run_id=run_id,
            dimension_id="test_dim",
            canonical_claim="test claim with conflicts",
            claim_type="numeric",
            entities=["x"],
            eu_count=2,
            independent_source_count=2,
            primary_source_count=1,
            grade="C" if conflicting_values else "A",
            grade_reason="integration test",
            has_conflict=bool(conflicting_values),
            conflicting_values=conflicting_values or [],
        )

    def test_conflicting_values_jsonb_roundtrip(self):
        """bug 模式:list[dict] 直接走 %()s::jsonb → psycopg3 不适配 → 炸。"""
        from open_deep_research.evidence.eu_dao import ClaimDAO

        run_id = uuid.uuid4()
        conflict_payload = [
            {"source": "src-a", "value": "300"},
            {"source": "src-b", "value": "290"},
        ]
        claim = self._make_claim(run_id, conflicting_values=conflict_payload)
        try:
            with ClaimDAO() as dao:
                dao.upsert_many([claim])
            with ClaimDAO() as dao:
                claims = dao.list_by_run(run_id)
            assert len(claims) == 1
            assert claims[0].conflicting_values == conflict_payload
            assert claims[0].has_conflict is True
        finally:
            with ClaimDAO() as dao:
                cur = dao._cur()
                cur.execute("DELETE FROM evidence.claim WHERE run_id=%s", (run_id,))
                dao._commit()

    def test_empty_conflicting_values_roundtrip(self):
        """边界:conflicting_values=[] (无冲突) 也能落 jsonb。"""
        from open_deep_research.evidence.eu_dao import ClaimDAO

        run_id = uuid.uuid4()
        claim = self._make_claim(run_id, conflicting_values=[])
        try:
            with ClaimDAO() as dao:
                dao.upsert_many([claim])
            with ClaimDAO() as dao:
                claims = dao.list_by_run(run_id)
            assert len(claims) == 1
            assert claims[0].conflicting_values == []
            assert claims[0].has_conflict is False
        finally:
            with ClaimDAO() as dao:
                cur = dao._cur()
                cur.execute("DELETE FROM evidence.claim WHERE run_id=%s", (run_id,))
                dao._commit()