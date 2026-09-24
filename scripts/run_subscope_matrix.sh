#!/usr/bin/env bash
# Run the A/B/C instrumentation-overhead ladder for one scenario.
#
# Arms (two independent dimensions: mount/probe state x pystack on/off):
#
#   arm  mode         pystack  purpose
#   ---  -----------  -------  -------------------------------------------------
#   A    baseline     off      stock image, stock runner (reference)
#   B    mounted-off  off      instrumented files mounted, PI_SUBSCOPE=off
#   C    on           off      probes active
#   D1   mounted-off  1000 us  pystack sampler at the upstream default
#   D2   mounted-off  off      repeat of B, so run-to-run noise is measurable
#   C2   on           off      like C plus the second-level probes inside
#                              AscendAttentionMetadataBuilder.build (mounts
#                              instrument/patched/attention_v1.py too); use it
#                              to explain the biggest single sub-scope
#
# Consequences, all on the *same* metric (median wall time of the LiteProfiler
# `prepare input` scope from scripts/parse_phase_timing.py):
#
#   B  - A   = cost of bind-mounting / importing the probe module (expect ~0)
#   C  - B   = cost of the probes themselves  (the <=5% budget)
#   D1 - D2  = cost of the pystack sampler inside the prepare-input window,
#              i.e. exactly what pollutes the historical 55-59% numbers
#   D2 - B   = run-to-run noise floor (both arms are nominally identical)
#
# and the history-corrected estimate of the *model-runner* cost is
#   corrected_prepare = D1_median - (D1 - D2)_median
# which is what docs/09-historical-data-caveat.md quotes.
#
# The container of `--keep-mode` (default C) is left running so that a profiler
# stage can attach to the engine-core TID recorded in
# runs/<id>/run_manifest.json without paying for another service start.
#
# Usage:
#   run_subscope_matrix.sh --scenario s1|s2|s3 [--modes A,B,C,D1,D2]
#                          [--keep-mode C] [--port-base 18100]
#                          [--extra-serve-args "..."] [--extra-launch-args "..."]
#
# Scenarios (fixed by the task statement):
#   s1  B=1  ISL=128 decode  FULL_DECODE_ONLY
#   s2  B=64 ISL=128 decode  FULL_DECODE_ONLY
#   s3  B=1  ISL=32768 chunked prefill, max-num-batched-tokens=2048
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PREPARE_INPUT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
STAMP=${PI_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}

SCENARIO=""
MODES="A,B,C,D1,D2"
KEEP_MODE=""
PORT_BASE=18100
EXTRA_SERVE_ARGS=""
EXTRA_LAUNCH_ARGS=""

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while (($#)); do
    case "$1" in
        --scenario) SCENARIO=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --keep-mode) KEEP_MODE=$2; shift 2 ;;
        --keep-last) KEEP_MODE=last; shift ;;
        --port-base) PORT_BASE=$2; shift 2 ;;
        --extra-serve-args) EXTRA_SERVE_ARGS=$2; shift 2 ;;
        --extra-launch-args) EXTRA_LAUNCH_ARGS=$2; shift 2 ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ $KEEP_MODE == last ]]; then
    KEEP_MODE=${MODES##*,}
fi

case "$SCENARIO" in
    s1) REQUESTS=1;  CONCURRENCY=1;  MAX_NUM_SEQS=1;  PROMPT=128;   MAXTOK=64
        MAXBATCH=2048; WARMUP=2; DESC="B=1 ISL=128 decode" ;;
    s2) REQUESTS=64; CONCURRENCY=64; MAX_NUM_SEQS=64; PROMPT=128;   MAXTOK=64
        MAXBATCH=2048; WARMUP=2; DESC="B=64 ISL=128 decode" ;;
    s3) REQUESTS=1;  CONCURRENCY=1;  MAX_NUM_SEQS=1;  PROMPT=32768; MAXTOK=4
        MAXBATCH=2048; WARMUP=0; DESC="B=1 ISL=32768 chunked prefill" ;;
    *) echo "ERROR: --scenario must be s1|s2|s3" >&2; exit 2 ;;
esac

# arm -> "launch mode|pystack interval us|run-id slug|attn probes"
arm_spec() {
    case "$1" in
        A|a)   echo "baseline|0|baseline|off" ;;
        B|b)   echo "mounted-off|0|mounted-off|off" ;;
        C|c)   echo "on|0|on|off" ;;
        C2|c2) echo "on|0|on-attn|on" ;;
        D1|d1) echo "mounted-off|1000|mounted-off-pystack|off" ;;
        D2|d2) echo "mounted-off|0|mounted-off-2|off" ;;
        *)     echo "" ;;
    esac
}

RUN_IDS=()
INDEX=0
for arm in ${MODES//,/ }; do
    spec=$(arm_spec "$arm")
    [[ -n $spec ]] || { echo "ERROR: unknown arm $arm (use A,B,C,D1,D2)" >&2; exit 2; }
    IFS='|' read -r LAUNCH_MODE PYSTACK SLUG ATTN <<<"$spec"
    RUN_ID="sub-${SCENARIO}-${SLUG}-${STAMP}"
    RUN_IDS+=("$arm=$RUN_ID")
    PORT=$((PORT_BASE + INDEX))
    INDEX=$((INDEX + 1))

    extra_args=()
    if [[ -n $KEEP_MODE ]] && [[ ${arm,,} == "${KEEP_MODE,,}" ]]; then
        extra_args+=(--keep)
    fi
    if [[ -n $EXTRA_SERVE_ARGS ]]; then
        extra_args+=(--extra-serve-args "$EXTRA_SERVE_ARGS")
    fi
    if [[ -n $EXTRA_LAUNCH_ARGS ]]; then
        extra_args+=($EXTRA_LAUNCH_ARGS)
    fi

    echo "=== arm=$arm scenario=$SCENARIO ($DESC) mode=$LAUNCH_MODE run_id=$RUN_ID port=$PORT pystack_interval_us=$PYSTACK attn_probes=$ATTN"
    "$SCRIPT_DIR/launch_subscope_service.sh" \
        --run-id "$RUN_ID" --mode "$LAUNCH_MODE" --attn-probes "$ATTN" \
        --model-key qwen35-08b --chip 3 --port "$PORT" \
        --requests "$REQUESTS" --concurrency "$CONCURRENCY" \
        --max-num-seqs "$MAX_NUM_SEQS" --prompt-tokens "$PROMPT" \
        --max-tokens "$MAXTOK" --max-num-batched-tokens "$MAXBATCH" \
        --warmup-requests "$WARMUP" --cudagraph-mode FULL_DECODE_ONLY \
        --pystack-interval-us "$PYSTACK" \
        "${extra_args[@]+"${extra_args[@]}"}"

    run_dir="$PROJECT_ROOT/runs/$RUN_ID"
    if [[ -f $run_dir/pi-subscope/raw.csv ]]; then
        mkdir -p "$PROJECT_ROOT/data/subscope/$RUN_ID"
        cp -f "$run_dir/pi-subscope/raw.csv" "$PROJECT_ROOT/data/subscope/$RUN_ID/raw.csv"
        [[ -f $run_dir/pi-subscope/probe_notes.json ]] && \
            cp -f "$run_dir/pi-subscope/probe_notes.json" \
                  "$PROJECT_ROOT/data/subscope/$RUN_ID/probe_notes.json"
        python3 "$SCRIPT_DIR/parse_subscope.py" \
            "$PROJECT_ROOT/data/subscope/$RUN_ID/raw.csv" \
            --outdir "$PROJECT_ROOT/data/subscope/$RUN_ID" \
            --engine-steps "$run_dir/lite-profiler/phase_timing.csv" \
            --attn-breakdown \
            >"$PROJECT_ROOT/data/subscope/$RUN_ID/parse.log" 2>&1 || \
            echo "WARNING: parse_subscope.py failed, see data/subscope/$RUN_ID/parse.log" >&2
    fi
done

echo "=== ladder complete for scenario $SCENARIO"
compare_args=()
for entry in "${RUN_IDS[@]}"; do
    compare_args+=(--run-id "$entry")
done
python3 "$SCRIPT_DIR/compare_overhead.py" \
    --scenario "$SCENARIO" --stamp "$STAMP" \
    --modes "$MODES" \
    "${compare_args[@]+"${compare_args[@]}"}" || true
