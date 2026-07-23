#!/bin/sh
# Multi-mode entrypoint for the API image.
#
# Modes (default = api):
#   api          run uvicorn for open_deep_research.api.server
#   langgraph    run `langgraph dev` for Studio
#   pipeline     run a one-shot pipeline_run for $QUERY (set QUERY + MODE env)
#   bash         drop to a shell
#
# Env wiring:
#   - Loads /app/.env or any ${ENV_FILE} if present (python-dotenv-compatible,
#     done via a small inline python here so we don't add a runtime dep).
set -eu

PORT="${PORT:-2024}"
HOST="${HOST:-0.0.0.0}"

# Best-effort .env loader (don't fail if file missing / no python or jq)
load_env() {
    env_file="$1"
    [ -f "$env_file" ] || return 0
    while IFS='=' read -r key val; do
        case "$key" in
            ''|\#*) continue ;;
        esac
        # strip optional quoting
        val=$(printf '%s' "$val" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//")
        export "$key"="$val" >/dev/null 2>&1 || true
    done < "$env_file"
}

load_env "/app/.env"
[ -n "${ENV_FILE:-}" ] && load_env "$ENV_FILE"

cd /app
export PYTHONPATH=/app/src

mode="${1:-api}"
shift || true

case "$mode" in
    api)
        echo "[entrypoint] starting uvicorn on $HOST:$PORT"
        exec .venv/bin/uvicorn open_deep_research.api.server:app \
            --host "$HOST" --port "$PORT" --log-level "${LOG_LEVEL:-info}" \
            "$@"
        ;;
    langgraph)
        echo "[entrypoint] starting langgraph dev (Studio)"
        exec .venv/bin/langgraph dev --host "$HOST" --port "$PORT" "$@"
        ;;
    pipeline)
        if [ -z "${QUERY:-}" ]; then
            echo "[entrypoint] QUERY env var required for pipeline mode" >&2
            exit 2
        fi
        echo "[entrypoint] running one-shot pipeline_run for: $QUERY"
        exec .venv/bin/python -c "
import asyncio, sys, os
sys.path.insert(0, '/app/src')
from open_deep_research.plan_v2_pipeline import run_pipeline
MODE = os.environ.get('MODE', 'evidence-only')
rid = os.environ.get('RUN_ID') or None
r = asyncio.run(run_pipeline(query=os.environ['QUERY'], run_id=rid, max_subtopics=int(os.environ.get('MAX_SUBTOPICS','3'))))
print('=== pipeline done ===')
print('run_id:', r.run_id)
print('error:', r.error)
print('n_eus:', len(r.evidence_units or []))
print('n_claims:', len(r.claims or []))
print('claim_grade_dist:', r.claim_grade_dist)
print('gate_stats:', r.gate_stats)
"
        ;;
    bash|shell)
        exec /bin/sh "$@"
        ;;
    *)
        echo "[entrypoint] unknown mode: $mode" >&2
        exit 2
        ;;
esac
