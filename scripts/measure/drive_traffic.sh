#!/usr/bin/env bash
# Drive sustained decode traffic against a live pi-phase service on a3-22.
#
# Why this exists next to phase_smoke_client.py: that client sends a fixed
# number of requests once.  A profiler window needs the engine to stay busy for
# the whole window, and `max_tokens` is capped by `max_model_len` (a 2048-token
# model length rejects an 8192-token request with HTTP 400 -- which silently
# yields an idle engine and a perf.data with zero samples), so sustained load
# has to come from several sequential requests.
#
# Usage:
#   scripts/measure/drive_traffic.sh --base-url URL --model NAME --outdir DIR \
#       --prompt-tokens 128 --max-tokens 1700 --rounds 4 \
#       [--concurrency 1] [--requests 1] [--cpus 120,121] [--seed 1024] \
#       [--tag traffic] [--timeout 900]
#
# Each round writes its own client_summary.json under <outdir>/rNN/, and the
# script prints one line per round so the caller can see progress in a log
# while it runs detached.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CLIENT="$SCRIPT_DIR/../phase_smoke_client.py"

BASE_URL=""
MODEL=""
OUTDIR=""
PROMPT_TOKENS=128
MAX_TOKENS=1700
ROUNDS=4
CONCURRENCY=1
REQUESTS=1
CPUS="120,121"
SEED=1024
TAG=traffic
TIMEOUT=900

while (($#)); do
    case "$1" in
        --base-url) BASE_URL=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --outdir) OUTDIR=$2; shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS=$2; shift 2 ;;
        --max-tokens) MAX_TOKENS=$2; shift 2 ;;
        --rounds) ROUNDS=$2; shift 2 ;;
        --concurrency) CONCURRENCY=$2; shift 2 ;;
        --requests) REQUESTS=$2; shift 2 ;;
        --cpus) CPUS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --tag) TAG=$2; shift 2 ;;
        --timeout) TIMEOUT=$2; shift 2 ;;
        *) echo "drive_traffic: unknown argument $1" >&2; exit 2 ;;
    esac
done

[[ -n $BASE_URL && -n $MODEL && -n $OUTDIR ]] || {
    echo "drive_traffic: --base-url, --model and --outdir are required" >&2
    exit 2
}

START=$(date +%s)
for round in $(seq 1 "$ROUNDS"); do
    rdir="$OUTDIR/r$(printf '%02d' "$round")"
    mkdir -p "$rdir"
    taskset -c "$CPUS" python3 "$CLIENT" \
        --base-url "$BASE_URL" --model "$MODEL" --outdir "$rdir" \
        --requests "$REQUESTS" --concurrency "$CONCURRENCY" \
        --prompt-tokens "$PROMPT_TOKENS" --max-tokens "$MAX_TOKENS" \
        --seed "$((SEED + round))" --timeout "$TIMEOUT" \
        --tag "${TAG}-r${round}" >"$rdir/client.log" 2>&1
    rc=$?
    ok=$(python3 - "$rdir/client_summary.json" <<'PY' 2>/dev/null || echo "?"
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
    print(doc.get("n_ok", 0))
except Exception:
    print(0)
PY
)
    echo "[drive_traffic] round $round/$ROUNDS rc=$rc ok=$ok elapsed=$(( $(date +%s) - START ))s"
done
echo "[drive_traffic] done in $(( $(date +%s) - START ))s"
