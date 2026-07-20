"""Phase 0.5 DB init - 创建 evidence/langfuse schema.

Run: docker exec -e PGPASSWORD=... odr-postgres psql ...
or via docker exec with python in postgres container.
"""
import psycopg
import os

POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "odr_v2_pg_pass_change_me")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "odr_v2")

conn_str = f"host=127.0.0.1 port=5432 user={POSTGRES_USER} password={POSTGRES_PASSWORD} dbname={POSTGRES_DB}"

with psycopg.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user, version()")
        row = cur.fetchone()
        print(f"  DB: {row[0]}")
        print(f"  User: {row[1]}")
        print(f"  Version: {row[2].split(',')[0]}")

        cur.execute("SELECT schema_name FROM information_schema.schemata ORDER BY schema_name")
        schemas = [r[0] for r in cur.fetchall()]
        print(f"\n  Existing schemas: {schemas}")

        for schema in ['evidence', 'langfuse']:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            print(f"  Created schema: {schema}")

        conn.commit()

print("\n[OK] Database ready")
