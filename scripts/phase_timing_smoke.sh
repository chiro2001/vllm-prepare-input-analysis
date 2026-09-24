#!/usr/bin/env bash
# One command that reproduces the whole minimal phase-timing flow on a3-22.
#
#   phase_timing_smoke.sh [--run-id ID] [--model-key K] [--profile-mode M]
#                         [--warmup-requests N] [--requests N]
#                         [--prompt-tokens N] [--max-tokens N]
#                         [--async-scheduling on|off]
#                         [--pystack-interval-us N] [--keep] [--dry-run]
#
# profile-mode:
#   liteprofiler   (default) LiteProfiler armed at the upstream pystack
#                  interval (1000 us).  Numbers are directly comparable with
#                  the historical HIST_PROJECT runs, but the sampler costs
#                  ~2.6 ms per decode step on 0.8B TP1 -- see the report.
#   liteprofiler-quiet
#                  same instrumentation with the pystack sampler pushed past
#                  the end of the run.  Use this for CPU-side analysis.
#   none           no instrumentation at all; the throughput/TTFT control arm.
#
# It acquires the chip3 lock, runs the service, collects, releases the lock,
# and prints the phase table.  Nothing outside $PROJECT_ROOT and the chip3
# CPU slice is touched.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PREPARE_INPUT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

RUN_ID=""
MODEL_KEY=qwen35-08b
PROFILE_MODE=liteprofiler
WARMUP_REQUESTS=1
REQUESTS=1
PROMPT_TOKENS=128
MAX_TOKENS=64
ASYNC_SCHEDULING=on
CLIENT_CONCURRENCY=1
KEEP=0
DRY_RUN=0

die() { printf 'phase_timing_smoke: ERROR: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,28p' "$0"; exit 0; }

while (($#)); do
    case "$1" in
        --run-id) RUN_ID=$2; shift 2 ;;
        --model-key) MODEL_KEY=$2; shift 2 ;;
        --profile-mode) PROFILE_MODE=$2; shift 2 ;;
        --warmup-requests) WARMUP_REQUESTS=$2; shift 2 ;;
        --requests) REQUESTS=$2; shift 2 ;;
        --concurrency) CLIENT_CONCURRENCY=$2; shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS=$2; shift 2 ;;
        --max-tokens) MAX_TOKENS=$2; shift 2 ;;
        --async-scheduling) ASYNC_SCHEDULING=$2; shift 2 ;;
        --pystack-interval-us) PYSTACK_OVERRIDE=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) die "unknown argument: $1" ;;
    esac
done

case "$PROFILE_MODE" in
    liteprofiler) PHASE_TIMING=on; PYSTACK=${PYSTACK_OVERRIDE:-1000} ;;
    liteprofiler-quiet) PHASE_TIMING=on; PYSTACK=${PYSTACK_OVERRIDE:-0} ;;
    none) PHASE_TIMING=off; PYSTACK=${PYSTACK_OVERRIDE:-0} ;;
    *) die "--profile-mode must be liteprofiler | liteprofiler-quiet | none" ;;
esac

[[ -n $RUN_ID ]] || RUN_ID="smoke-${MODEL_KEY}-${PROFILE_MODE}-$(date -u +%Y%m%dT%H%MZ)"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"

LAUNCH_ARGS=(
    "$SCRIPT_DIR/launch_phase_service.sh"
    --run-id "$RUN_ID"
    --model-key "$MODEL_KEY"
    --phase-timing "$PHASE_TIMING"
    --pystack-interval-us "$PYSTACK"
    --cpu-pinning on
    --warmup-requests "$WARMUP_REQUESTS"
    --async-scheduling "$ASYNC_SCHEDULING"
    --requests "$REQUESTS"
    --concurrency "$CLIENT_CONCURRENCY"
    --prompt-tokens "$PROMPT_TOKENS"
    --max-tokens "$MAX_TOKENS"
)
((KEEP)) && LAUNCH_ARGS+=(--keep)

if ((DRY_RUN)); then
    printf 'run_id=%s\nrun_dir=%s\nprofile_mode=%s\n' "$RUN_ID" "$RUN_DIR" "$PROFILE_MODE"
    printf '%q ' "$SCRIPT_DIR/chip3_lock.sh" run --owner service_bringup \
        --purpose "phase timing smoke ($PROFILE_MODE)" --wait 3600 -- "${LAUNCH_ARGS[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$PROJECT_ROOT/logs"
LOG="$PROJECT_ROOT/logs/${RUN_ID}.log"
"$SCRIPT_DIR/chip3_lock.sh" run --owner service_bringup \
    --purpose "phase timing smoke ($PROFILE_MODE)" --wait 3600 -- \
    "${LAUNCH_ARGS[@]}" 2>&1 | tee "$LOG"

if [[ -f $RUN_DIR/lite-profiler/phase_timing.txt ]]; then
    printf -- '--- phase timing (%s) ---\n' "$RUN_DIR/lite-profiler/phase_timing.txt"
    cat "$RUN_DIR/lite-profiler/phase_timing.txt"
else
    printf -- '--- no phase timing (profile-mode=%s) ---\n' "$PROFILE_MODE"
    cat "$RUN_DIR/requests/client_summary.json" 2>/dev/null || true
fi
