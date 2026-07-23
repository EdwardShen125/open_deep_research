#!/bin/bash
# restore_pg.sh — 把 backups/*.sql.gz 解到目标 PG。
# 跑法:
#   POSTGRES_HOST=127.0.0.1 POSTGRES_PASSWORD=*** \
#     ./scripts/restore_pg.sh backups/odr_v2_20260723_120000.sql.gz
#   ./scripts/restore_pg.sh                             # 用最新的 backup
set -euo pipefail

if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

: "${POSTGRES_HOST:=127.0.0.1}"
: "${POSTGRES_PORT:=5432}"
: "${POSTGRES_USER:=postgres}"
: "${POSTGRES_DB:=odr_v2}"

BACKUP_DIR="${BACKUP_DIR:-./backups}"

if [ $# -eq 0 ]; then
    FILE=$(ls -1t "$BACKUP_DIR"/${POSTGRES_DB}_*.sql.gz 2>/dev/null | head -1 || true)
    if [ -z "${FILE:-}" ]; then
        echo "ERROR: no backups found in $BACKUP_DIR" >&2
        exit 2
    fi
    echo "==> using latest backup: $FILE"
else
    FILE="$1"
fi

if [ ! -f "$FILE" ]; then
    echo "ERROR: $FILE not found" >&2
    exit 2
fi

echo "==> restore_pg: $FILE -> $POSTGRES_HOST:$POSTGRES_PORT/$POSTGRES_DB"

read -p "  confirm DROP+RECREATE on $POSTGRES_DB? (type 'yes'): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "aborted"
    exit 1
fi

gunzip -c "$FILE" | PGPASSWORD="${POSTGRES_PASSWORD:-}" psql \
    -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" \
    -d "$POSTGRES_DB" -v ON_ERROR_STOP=1
echo "[OK] restored"
