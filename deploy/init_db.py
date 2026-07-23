"""Phase 0.5+ DB init - 创建 evidence/langfuse schema + 按顺序应用 migrations/*.sql。

Run (host):
    POSTGRES_HOST=127.0.0.1 POSTGRES_USER=postgres POSTGRES_PASSWORD=... \
    POSTGRES_DB=odr_v2 .venv/bin/python deploy/init_db.py

Run (docker compose):
    docker compose -f deploy/docker-compose.yml run --rm api \
      .venv/bin/python /app/deploy/init_db.py

幂等:可反复执行。migrations 应用通过 schema_migrations 表记名。
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import psycopg

ROOT = pathlib.Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "migrations"

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "odr_v2_pg_pass_change_me")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "odr_v2")

RETRY = 5  # docker compose 启动 PG 还没就绪时重试


def wait_for_pg() -> None:
    """Retry-until-ready, 用于 docker compose up 后 PG 可能没就绪。"""
    last = None
    for i in range(RETRY):
        try:
            with psycopg.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
                dbname=POSTGRES_DB,
                connect_timeout=3,
            ) as c:
                with c.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return
        except Exception as e:
            last = e
            print(f"  PG not ready (attempt {i+1}/{RETRY}): {e}")
            time.sleep(2)
    raise SystemExit(f"PG unreachable after {RETRY} retries: {last}")


def ensure_schemas(cur) -> None:
    for schema in ["evidence", "langfuse"]:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    print("  schemas: evidence, langfuse")


def ensure_migration_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence.schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def applied_set(cur) -> set[str]:
    cur.execute("SELECT filename FROM evidence.schema_migrations")
    return {r[0] for r in cur.fetchall()}


def apply_migration(cur, path: pathlib.Path) -> None:
    sql = path.read_text()
    if not sql.strip():
        return
    cur.execute(sql)
    cur.execute(
        "INSERT INTO evidence.schema_migrations (filename) VALUES (%s)",
        (path.name,),
    )
    print(f"  applied: {path.name}")


def main() -> int:
    print(f"==> init_db")
    print(f"  PG host: {POSTGRES_HOST}:{POSTGRES_PORT}")
    print(f"  DB: {POSTGRES_DB}  user: {POSTGRES_USER}")
    wait_for_pg()
    with psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    ) as conn:
        with conn.cursor() as cur:
            ensure_schemas(cur)
            ensure_migration_table(cur)
            done = applied_set(cur)
            files = sorted(MIGRATIONS_DIR.glob("*.sql"))
            if not files:
                print("  no migrations/*.sql found")
            for f in files:
                if f.name in done:
                    print(f"  skip (already applied): {f.name}")
                else:
                    apply_migration(cur, f)
        conn.commit()
    print("\n[OK] Database ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
