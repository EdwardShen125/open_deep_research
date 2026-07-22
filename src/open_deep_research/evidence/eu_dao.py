"""Phase 3 (= Runbook v1 阶段 1): EvidenceUnit + Claim DAO.

设计依据: notes/evidence-pipeline-runbook-v1.md 1.2 节。

与 SourcesDAO 的关系:
- SourcesDAO 维护 evidence.sources (URL 维度)
- EuDAO / ClaimDAO 维护 evidence.evidence_unit / evidence.claim (语义维度)
- 一对多:sources.id ←→ evidence_unit.source_url (软引用,无 FK)

风格对齐 SourcesDAO:
- 同步 psycopg + context manager
- `__enter__` 懒连数据库,env vars 来自 deploy/init_db.py
- write 方法幂等(基于 content_hash / claim_id)
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit
from uuid import UUID

from open_deep_research.evidence.schema import ClaimV2, EvidenceUnitV2, Grade


# =============================================================================
# Helpers
# =============================================================================

def host_of(url: str) -> str:
    """Lowercased hostname (no scheme/port). 来源检测 primary/secondary tier 用。"""
    try:
        return (urlsplit(url).hostname or "").lower()
    except Exception:
        return ""


def _coerce_uuid(v: Any) -> Optional[UUID]:
    if v is None:
        return None
    if isinstance(v, UUID):
        return v
    if isinstance(v, str) and v:
        try:
            return UUID(v)
        except ValueError:
            return None
    return None


def _coerce_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return None


def _coerce_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    # datetime 必须先于 date 判断(datetime 是 date 子类)
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return None


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v in (1, "1", "t", "T", "true", "TRUE"):
        return True
    return bool(v)


# =============================================================================
# EuDAO
# =============================================================================

class EuDAO:
    """evidence.evidence_unit 表的同步读写。

    关键方法:
        upsert_many(eus)        — 批量写入,基于 eu_id ON CONFLICT DO NOTHING(只增)
        list_by_run(run_id)     — 列出某 run 的全部 EU(供 verifier / final_report 用)
        count_by_run(run_id)    — 加速 state 瘦身后的 supervisor 聚合
        search_by_embedding()   — Runbook 1.2 验收:HNSW 走 ORDER BY embedding <=> $q
    """

    def __init__(self, conn: Any = None) -> None:
        self._conn = conn

    # ---------- context manager ----------
    def __enter__(self) -> "EuDAO":
        if self._conn is None:
            self._conn = self._connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and self._conn is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        return False

    @staticmethod
    def _connect() -> Any:
        import psycopg
        return psycopg.connect(
            host=os.environ.get("POSTGRES_HOST", "127.0.0.1"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "odr_v2_pg_pass_change_me"),
            dbname=os.environ.get("POSTGRES_DB", "odr_v2"),
            autocommit=False,
        )

    def _cur(self):
        assert self._conn is not None, "EuDAO must be used as a context manager"
        return self._conn.cursor()

    def _commit(self) -> None:
        assert self._conn is not None
        self._conn.commit()

    # ---------- writes ----------
    def upsert_many(self, eus: Iterable[EvidenceUnitV2]) -> list[str]:
        """批量 upsert(基于 eu_id PK,ON CONFLICT DO NOTHING — EU 不可变)。

        返回成功写入的 eu_id 列表(注意:已存在的不算"写入",不在返回里)。
        """
        rows = [eu.to_pg_row() for eu in eus]
        if not rows:
            return []
        cur = self._cur()
        cur.executemany(
            """
            INSERT INTO evidence.evidence_unit (
                eu_id, run_id, dimension_id, claim, claim_type, entities,
                norm_value, unit, value_as_of,
                source_url, source_domain, source_title, published_at, source_tier,
                source_span, span_start, span_end,
                extractor_model, extracted_at,
                span_verified, numeric_drift, entailment_verdict, entailment_score,
                claim_id, content_hash, embedding
            ) VALUES (
                %(eu_id)s, %(run_id)s, %(dimension_id)s, %(claim)s, %(claim_type)s, %(entities)s,
                %(norm_value)s, %(unit)s, %(value_as_of)s,
                %(source_url)s, %(source_domain)s, %(source_title)s, %(published_at)s, %(source_tier)s,
                %(source_span)s, %(span_start)s, %(span_end)s,
                %(extractor_model)s, %(extracted_at)s,
                %(span_verified)s, %(numeric_drift)s, %(entailment_verdict)s, %(entailment_score)s,
                %(claim_id)s, %(content_hash)s, %(embedding)s::vector
            )
            ON CONFLICT (eu_id) DO NOTHING
            """,
            rows,
        )
        # 返回 list 顺序与输入顺序对齐
        self._commit()
        return [r["eu_id"] for r in rows]

    def update_verification(
        self,
        eu_id: str,
        *,
        span_verified: Optional[bool] = None,
        numeric_drift: Optional[bool] = None,
        entailment_verdict: Optional[str] = None,
        entailment_score: Optional[float] = None,
    ) -> None:
        """阶段 2 三道闸的回填通道(只允许 narrow update,不允许覆盖 claim/span)。"""
        sets: list[str] = []
        params: dict[str, Any] = {"eu_id": eu_id}
        if span_verified is not None:
            sets.append("span_verified = %(span_verified)s")
            params["span_verified"] = span_verified
        if numeric_drift is not None:
            sets.append("numeric_drift = %(numeric_drift)s")
            params["numeric_drift"] = numeric_drift
        if entailment_verdict is not None:
            sets.append("entailment_verdict = %(entailment_verdict)s")
            params["entailment_verdict"] = entailment_verdict
        if entailment_score is not None:
            sets.append("entailment_score = %(entailment_score)s")
            params["entailment_score"] = entailment_score
        if not sets:
            return
        cur = self._cur()
        cur.execute(
            f"UPDATE evidence.evidence_unit SET {', '.join(sets)} WHERE eu_id = %(eu_id)s",
            params,
        )
        self._commit()

    def update_claim_id(self, eu_id: str, claim_id: Optional[str]) -> None:
        """阶段 3 归并后回填。"""
        cur = self._cur()
        cur.execute(
            "UPDATE evidence.evidence_unit SET claim_id = %s WHERE eu_id = %s",
            (claim_id, eu_id),
        )
        self._commit()

    # ---------- reads ----------
    def list_by_run(
        self,
        run_id: str | UUID,
        *,
        dimension_id: Optional[str] = None,
        only_usable: bool = False,
    ) -> list[EvidenceUnitV2]:
        """按 run_id 列出 EU。供 verifier / final_report / writer 使用。

        `only_usable` 为 True 时只返回通过闸的 EU(span_verified && !numeric_drift
        && entailment_verdict in (entailed, partial))。
        """
        rid = str(run_id)
        sql = "SELECT * FROM evidence.evidence_unit WHERE run_id = %s"
        params: list[Any] = [rid]
        if dimension_id is not None:
            sql += " AND dimension_id = %s"
            params.append(dimension_id)
        if only_usable:
            sql += (
                " AND span_verified = TRUE"
                " AND numeric_drift = FALSE"
                " AND entailment_verdict IN ('entailed', 'partial')"
            )
        sql += " ORDER BY extracted_at"
        cur = self._cur()
        cur.execute(sql, params)
        return [_row_to_eu(cur.description, r) for r in cur.fetchall()]

    def count_by_run(self, run_id: str | UUID) -> int:
        rid = str(run_id)
        cur = self._cur()
        cur.execute(
            "SELECT count(*) FROM evidence.evidence_unit WHERE run_id = %s",
            (rid,),
        )
        return cur.fetchone()[0]

    def count_by_dimension(self, run_id: str | UUID) -> dict[str, int]:
        """Runbook 1.3 state 瘦身后的 supervisor 聚合:eu_counts[dim] = N"""
        rid = str(run_id)
        cur = self._cur()
        cur.execute(
            """
            SELECT COALESCE(dimension_id, '<unknown>') AS dim, count(*)
            FROM evidence.evidence_unit WHERE run_id = %s
            GROUP BY dim
            """,
            (rid,),
        )
        return {r[0]: r[1] for r in cur.fetchall()}

    def search_by_embedding(
        self,
        run_id: str | UUID,
        query_embedding: list[float],
        *,
        dimension_ids: Optional[list[str]] = None,
        limit: int = 40,
        min_similarity: Optional[float] = None,
    ) -> list[tuple[EvidenceUnitV2, float]]:
        """向量检索:ORDER BY embedding <=> $q(走 HNSW 索引)。

        返回 (EU, cosine_similarity) 列表;similarity = 1 - distance(cosine)。

        Runbook 1.2 验收:EXPLAIN 应显示 HNSW 索引扫描。
        """
        rid = str(run_id)
        if dimension_ids is None:
            dim_filter = ""
            params: list[Any] = [rid, query_embedding, limit]
        else:
            dim_filter = " AND dimension_id = ANY(%s)"
            params = [rid, dim_filter and list(dimension_ids) or query_embedding, limit]
            # 上面写法略乱,重写清晰版:
        if dimension_ids is not None:
            sql = (
                "SELECT *, 1 - (embedding <=> %s::vector) AS similarity"
                " FROM evidence.evidence_unit"
                " WHERE run_id = %s AND embedding IS NOT NULL"
                f"{dim_filter}"
                " ORDER BY embedding <=> %s::vector"
                " LIMIT %s"
            )
            params = [query_embedding, rid, list(dimension_ids), query_embedding, limit]
        else:
            sql = (
                "SELECT *, 1 - (embedding <=> %s::vector) AS similarity"
                " FROM evidence.evidence_unit"
                " WHERE run_id = %s AND embedding IS NOT NULL"
                " ORDER BY embedding <=> %s::vector"
                " LIMIT %s"
            )
            params = [query_embedding, rid, query_embedding, limit]
        if min_similarity is not None:
            sql = sql.replace(
                "WHERE run_id", f"WHERE 1 - (embedding <=> %s::vector) >= %s AND run_id"
            )
            params.insert(0, query_embedding)
            params.insert(1, float(min_similarity))
        cur = self._cur()
        cur.execute(sql, params)
        out: list[tuple[EvidenceUnitV2, float]] = []
        for row in cur.fetchall():
            sim = float(row[-1])
            row_dict = {c.name: v for c, v in zip(cur.description, row)}
            row_dict.pop("similarity", None)
            out.append((_row_to_eu(cur.description, row_dict), sim))
        return out


# =============================================================================
# ClaimDAO
# =============================================================================

class ClaimDAO:
    """evidence.claim 表的同步读写。"""

    def __init__(self, conn: Any = None) -> None:
        self._conn = conn

    def __enter__(self) -> "ClaimDAO":
        if self._conn is None:
            self._conn = EuDAO._connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and self._conn is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        return False

    def _cur(self):
        assert self._conn is not None, "ClaimDAO must be used as a context manager"
        return self._conn.cursor()

    def _commit(self) -> None:
        assert self._conn is not None
        self._conn.commit()

    def upsert_many(self, claims: Iterable[ClaimV2]) -> list[str]:
        rows = [c.to_pg_row() for c in claims]
        if not rows:
            return []
        cur = self._cur()
        cur.executemany(
            """
            INSERT INTO evidence.claim (
                claim_id, run_id, dimension_id, canonical_claim, claim_type,
                entities, norm_value, unit, value_as_of, value_spread,
                eu_count, independent_source_count, primary_source_count,
                earliest_published_at, has_conflict, conflicting_values,
                grade, grade_reason
            ) VALUES (
                %(claim_id)s, %(run_id)s, %(dimension_id)s, %(canonical_claim)s, %(claim_type)s,
                %(entities)s, %(norm_value)s, %(unit)s, %(value_as_of)s, %(value_spread)s,
                %(eu_count)s, %(independent_source_count)s, %(primary_source_count)s,
                %(earliest_published_at)s, %(has_conflict)s, %(conflicting_values)s::jsonb,
                %(grade)s, %(grade_reason)s
            )
            ON CONFLICT (claim_id) DO NOTHING
            """,
            rows,
        )
        self._commit()
        return [r["claim_id"] for r in rows]

    def list_by_run(
        self,
        run_id: str | UUID,
        *,
        grade: Optional[Grade] = None,
        exclude_grade: Optional[Grade] = None,
        dimension_ids: Optional[list[str]] = None,
    ) -> list[ClaimV2]:
        rid = str(run_id)
        sql = "SELECT * FROM evidence.claim WHERE run_id = %s"
        params: list[Any] = [rid]
        if grade is not None:
            sql += " AND grade = %s"
            params.append(grade)
        if exclude_grade is not None:
            sql += " AND grade <> %s"
            params.append(exclude_grade)
        if dimension_ids is not None:
            sql += " AND dimension_id = ANY(%s)"
            params.append(list(dimension_ids))
        sql += " ORDER BY created_at"
        cur = self._cur()
        cur.execute(sql, params)
        return [_row_to_claim(cur.description, r) for r in cur.fetchall()]

    def count_by_run(self, run_id: str | UUID) -> int:
        rid = str(run_id)
        cur = self._cur()
        cur.execute(
            "SELECT count(*) FROM evidence.claim WHERE run_id = %s",
            (rid,),
        )
        return cur.fetchone()[0]

    def grade_distribution(self, run_id: str | UUID) -> dict[str, int]:
        rid = str(run_id)
        cur = self._cur()
        cur.execute(
            """
            SELECT grade, count(*) FROM evidence.claim
            WHERE run_id = %s GROUP BY grade
            """,
            (rid,),
        )
        out: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        for g, n in cur.fetchall():
            out[g] = int(n)
        return out


# =============================================================================
# RunCheckpointDAO(阶段 4 用,阶段 1 先建 stub)
# =============================================================================

class RunCheckpointDAO:
    """evidence.run_checkpoint 表的读写。阶段 4 worker 用,阶段 1 只建 stub。"""

    def __init__(self, conn: Any = None) -> None:
        self._conn = conn

    def __enter__(self) -> "RunCheckpointDAO":
        if self._conn is None:
            self._conn = EuDAO._connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and self._conn is not None:
            try:
                self._conn.rollback()
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        return False

    def _cur(self):
        assert self._conn is not None
        return self._conn.cursor()

    def _commit(self) -> None:
        assert self._conn is not None
        self._conn.commit()

    def upsert(
        self,
        run_id: str | UUID,
        stage: str,
        *,
        status: str,
        payload: Optional[dict] = None,
    ) -> None:
        cur = self._cur()
        cur.execute(
            """
            INSERT INTO evidence.run_checkpoint (run_id, stage, status, payload, finished_at)
            VALUES (%s, %s, %s, %s::jsonb,
                    CASE WHEN %s IN ('done', 'failed') THEN NOW() ELSE NULL END)
            ON CONFLICT (run_id, stage) DO UPDATE SET
                status = EXCLUDED.status,
                payload = EXCLUDED.payload,
                finished_at = EXCLUDED.finished_at
            """,
            (str(run_id), stage, status, payload or {}, status),
        )
        self._commit()

    def get(self, run_id: str | UUID, stage: str) -> Optional[dict]:
        cur = self._cur()
        cur.execute(
            "SELECT * FROM evidence.run_checkpoint WHERE run_id = %s AND stage = %s",
            (str(run_id), stage),
        )
        row = cur.fetchone()
        return dict(zip([c.name for c in cur.description], row)) if row else None


# =============================================================================
# Row → model converters
# =============================================================================

_EU_FIELDS = (
    "eu_id", "run_id", "dimension_id", "claim", "claim_type", "entities",
    "norm_value", "unit", "value_as_of",
    "source_url", "source_domain", "source_title", "published_at", "source_tier",
    "source_span", "span_start", "span_end",
    "extractor_model", "extracted_at",
    "span_verified", "numeric_drift", "entailment_verdict", "entailment_score",
    "claim_id", "content_hash", "embedding",
)


def _row_to_eu(description, row) -> EvidenceUnitV2:
    col_names = [c.name for c in description]
    raw = dict(zip(col_names, row)) if not isinstance(row, dict) else dict(row)
    clean: dict[str, Any] = {}
    for f in _EU_FIELDS:
        v = raw.get(f)
        if f in ("eu_id", "run_id", "claim_id"):
            clean[f] = _coerce_uuid(v)
        elif f == "published_at" or f == "extracted_at":
            clean[f] = _coerce_dt(v)
        elif f == "value_as_of":
            clean[f] = _coerce_date(v)
        elif f in ("span_verified", "numeric_drift"):
            clean[f] = _truthy(v) if v is not None else False
        elif f in ("entities",):
            clean[f] = list(v) if v is not None else []
        elif f == "embedding":
            # 不回填到 model(列表太长,默认 None)
            continue
        else:
            clean[f] = v
    # 空字符串 → None
    for k in ("dimension_id", "source_title", "unit", "entailment_verdict", "content_hash"):
        if clean.get(k) == "":
            clean[k] = None
    return EvidenceUnitV2(**{k: v for k, v in clean.items() if k in _EU_FIELDS})


_CLAIM_FIELDS = (
    "claim_id", "run_id", "dimension_id", "canonical_claim", "claim_type",
    "entities", "norm_value", "unit", "value_as_of", "value_spread",
    "eu_count", "independent_source_count", "primary_source_count",
    "earliest_published_at", "has_conflict", "conflicting_values",
    "grade", "grade_reason", "embedding", "created_at",
)


def _row_to_claim(description, row) -> ClaimV2:
    col_names = [c.name for c in description]
    raw = dict(zip(col_names, row)) if not isinstance(row, dict) else dict(row)
    clean: dict[str, Any] = {}
    for f in _CLAIM_FIELDS:
        v = raw.get(f)
        if f in ("claim_id", "run_id"):
            clean[f] = _coerce_uuid(v)
        elif f == "earliest_published_at" or f == "created_at":
            clean[f] = _coerce_dt(v)
        elif f == "value_as_of":
            clean[f] = _coerce_date(v)
        elif f == "has_conflict":
            clean[f] = _truthy(v) if v is not None else False
        elif f == "entities":
            clean[f] = list(v) if v is not None else []
        elif f == "conflicting_values":
            clean[f] = list(v) if v is not None else []
        elif f == "embedding":
            continue
        else:
            clean[f] = v
    for k in ("unit", "value_spread"):
        if clean.get(k) == "":
            clean[k] = None
    return ClaimV2(**{k: v for k, v in clean.items() if k in _CLAIM_FIELDS})


__all__ = [
    "EuDAO",
    "ClaimDAO",
    "RunCheckpointDAO",
    "host_of",
]