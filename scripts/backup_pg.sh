#!/bin/bash
# backup_pg.sh — pg_dump 当前 PG 到 ./backups/<timestamp>.sql.gz
# 跑法:
#   POSTGRES_HOST=127.0.0.1 POSTGRES_PASSWORD=*** ./scripts/backup_pg.sh
#   ./scripts/backup_pg.sh                         # 用 .env 默认
#
# 容器:
#   docker compose exec -e POSTGRES_PASSWORD=... postgres \
#     pg_dump -U postgres odr_v2 > backups/$(date +%Y%m%d_%H%M%S).sql
set -euo pipefail

# 加载 .env (基本 key=value 一行)
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

: "${POSTGRES_HOST:=127.0.0.1}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_USER:=postgres}"
: "${POSTGRES_DB:=odr_v2}"

OUT_DIR="${BACKUP_DIR:-./backups}"
mkdir -p "$OUT_DIR"
FNAME="${OUT_DIR}/${POSTGRES_DB}_$(date -u +%Y%m%d_%H%M%SZ).sql.gz"

echo "==> backup_pg: $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB -> $FNAME"

if command -v pg_dump >/dev/null 2>&1; then
    PGPASSWORD="${POSTGRES_PASSWORD:-}" pg_dump \
        -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" \
        --no-owner --no-privileges --clean --if-exists \
        "$POSTGRES_DB" | gzip -9 > "$FNAME"
elif command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -q 'odr-postgres'; then
    echo "  pg_dump not found, falling back to docker exec"
    docker exec odr-postgres pg_dump -U "$POSTGRES_USER" --no-owner --no-privileges --clean --if-exists \
        "$POSTGRES_DB" | gzip -9 > "$FNAME"
elif .venv/bin/python -c "import psycopg" 2>/dev/null; then
    echo "  pg_dump not found, falling back to psycopg-python replica dump"
    .venv/bin/python - <<PYEOF | gzip -9 > "$FNAME"
import psycopg, os, sys
dsn = f"host={os.environ['POSTGRES_HOST']} port={os.environ.get('POSTGRES_PORT','5432')} \
    user={os.environ['POSTGRES_USER']} password={os.environ.get('POSTGRES_PASSWORD','')} \
    dbname={os.environ['POSTGRES_DB']}"
with psycopg.connect(dsn) as conn, conn.cursor() as cur:
    sql_parts = []
    # schemas
    cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast') ORDER BY schema_name")
    for (s,) in cur.fetchall():
        sql_parts.append(f"CREATE SCHEMA IF NOT EXISTS {s};")
    # tables + columns (basic, no data) — enough to satisfy --clean --if-exists model.
    # NOTE: this is a SHAPE-only dump.  For data, use pg_dump in CI.
    cur.execute("""SELECT table_schema, table_name FROM information_schema.tables
                   WHERE table_schema NOT IN ('pg_catalog','information_schema')
                   AND table_type='BASE TABLE' ORDER BY table_schema, table_name""")
    for sch, tbl in cur.fetchall():
        sql_parts.append(f"DROP TABLE IF EXISTS {sch}.{tbl} CASCADE;")
        cur.execute("""SELECT column_name, data_type FROM information_schema.columns
                       WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position""", (sch, tbl))
        cols = ", ".join(f"{n} {t}" for n, t in cur.fetchall())
        sql_parts.append(f"CREATE TABLE IF NOT EXISTS {sch}.{tbl} ({cols});")
    print("\n".join(sql_parts))
PYEOF
else
    echo "ERROR: pg_dump not in PATH, no odr-postgres container, and venv lacks psycopg" >&2
    exit 2
fi

SIZE=$(stat -c %s "$FNAME" 2>/dev/null || echo "?")
echo "[OK] $FNAME  ($SIZE bytes)"
echo "    restore:  gunzip -c $FNAME | psql -h <host> -U postgres -d $POSTGRES_DB"

# 自旋 7 天
find "$OUT_DIR" -name "${POSTGRES_DB}_*.sql.gz" -mtime +7 -delete 2>/dev/null || true
