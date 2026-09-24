#!/usr/bin/env bash
# Launch the phase service with the prepare_input sub-scope probes.
#
# This is a thin wrapper around scripts/launch_phase_service.sh: it composes
# the opt-in `PREPARE_INPUT_EXTRA_DOCKER_ARGS` hook so that the stock launcher
# stays untouched for every other experiment.
#
# Modes (the *mount/probe* dimension of the overhead ladder):
#   baseline     no bind mount at all  -> stock image behaviour
#   mounted-off  instrumented files mounted, PI_SUBSCOPE=off
#                (isolates the cost of the mount / import, expected ~0)
#   on           instrumented files mounted, PI_SUBSCOPE=on
#                (writes /runmeta/pi-subscope/raw.csv)
#
# The other confounder of this project, the pystack sampler, is a separate
# dimension and is controlled by the forwarded --pystack-interval-us argument
# (0 = disabled, 1000 = upstream default).  scripts/run_subscope_matrix.sh
# combines the two dimensions into the A/B/C/D1/D2 ladder.
#
# This launcher is the *public* entry point for the prepare_input experiments
# and is meant to be reused by other stages (perf / libkperfx / msprof) so that
# two agents never start two services for the same scenario:
#
#   * it composes the docker command line (mode + PI_SUBSCOPE + optional
#     --extra-docker-args) and records exactly what was mounted;
#   * it forwards every unknown argument to scripts/launch_phase_service.sh;
#   * when the inner launcher returns it writes runs/<run_id>/run_manifest.json
#     with the container name/id, image digest, device, cpuset, workload
#     parameters, **engine-core PID and main-thread TID**, log paths, and
#     ready-to-paste perf / workload-driver commands.
#
# Reuse recipe for a profiler agent (no second service start needed):
#
#   1. bring the service up and leave it running (no workload is generated):
#        scripts/launch_subscope_service.sh --run-id ID --mode baseline \
#            --serve-only --model-key qwen35-08b --chip 3 ...
#   2. read runs/ID/run_manifest.json:
#        jq -r .engine_core.tid   runs/ID/run_manifest.json
#        jq -r .handoff.attach_perf runs/ID/run_manifest.json
#   3. attach perf/libkperfx to that TID (host pid namespace), then drive the
#      measured traffic with the .handoff.drive_workload command, which also
#      tells you how to arm LiteProfiler via /start_profile + /stop_profile.
#   4. stop it with .handoff.stop_container.
#
# TID discovery in one line: `npu-smi info -t proc-mem -i CARD -c SUBCHIP`
# gives the engine-core host PID, and because vLLM v1 runs
# NPUModelRunner.execute_model on the process main thread (and LiteProfiler
# records threading.get_native_id()), TID == PID.  scripts/run_manifest.py
# performs this lookup and stores both, plus the full thread list with comm so
# you can exclude acl_thread / release_thread / ZMQ threads.
#
# Usage:
#   scripts/launch_subscope_service.sh --run-id ID --mode baseline|mounted-off|on \
#       [--extra-docker-args "..."] [any further launch_phase_service.sh argument]
#
# The remaining arguments are forwarded verbatim, so e.g.
#   scripts/launch_subscope_service.sh --run-id s1 --mode on \
#       --model-key qwen35-08b --chip 3 --requests 1 --concurrency 1 \
#       --prompt-tokens 128 --max-tokens 64 --max-num-seqs 1 \
#       --cudagraph-mode FULL_DECODE_ONLY --pystack-interval-us 0
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PREPARE_INPUT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
INSTRUMENT_DIR=${PI_INSTRUMENT_DIR:-$PROJECT_ROOT/instrument}

RUN_ID=""
MODE=""
EXTRA_ARGS=""
ATTN_PROBES=off
FORWARDED=()

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while (($#)); do
    case "$1" in
        --run-id) RUN_ID=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        --extra-docker-args) EXTRA_ARGS=$2; shift 2 ;;
        --attn-probes) ATTN_PROBES=$2; shift 2 ;;
        -h|--help) usage ;;
        *) FORWARDED+=("$1"); shift ;;
    esac
done

[[ -n $RUN_ID ]] || { echo "ERROR: --run-id is required" >&2; exit 2; }
[[ $MODE =~ ^(baseline|mounted-off|on)$ ]] || {
    echo "ERROR: --mode must be baseline|mounted-off|on" >&2; exit 2; }

RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"

# The container writes /runmeta/pi-subscope/raw.csv; /runmeta is $RUN_DIR.
mkdir -p "$RUN_DIR/pi-subscope"

PATCHED="$INSTRUMENT_DIR/patched/model_runner_v1.py"
HELPER="$INSTRUMENT_DIR/pi_subscope.py"

case "$MODE" in
    baseline)
        export PREPARE_INPUT_EXTRA_DOCKER_ARGS="${EXTRA_ARGS:-}"
        ;;
    mounted-off|on)
        [[ $ATTN_PROBES == on || $ATTN_PROBES == off ]] || {
            echo "ERROR: --attn-probes must be on|off" >&2; exit 2; }
        MOUNTED=("$PATCHED" "$HELPER")
        if [[ $ATTN_PROBES == on ]]; then
            MOUNTED+=(
                "$INSTRUMENT_DIR/patched/attention_v1.py"
                "$INSTRUMENT_DIR/patched/sfa_v1.py"
                "$INSTRUMENT_DIR/patched/gdn_attn.py"
                "$INSTRUMENT_DIR/patched/gdn_attn_builder.py"
            )
        fi
        for f in "${MOUNTED[@]}"; do
            [[ -f $f ]] || { echo "ERROR: missing $f" >&2; exit 2; }
        done
        extra=(
            -v "$PATCHED:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro"
            -v "$HELPER:/vllm-workspace/vllm-ascend/vllm_ascend/worker/pi_subscope.py:ro"
        )
        if [[ $ATTN_PROBES == on ]]; then
            extra+=(
                -v "$INSTRUMENT_DIR/patched/attention_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py:ro"
                -v "$INSTRUMENT_DIR/patched/sfa_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/attention/sfa_v1.py:ro"
                -v "$INSTRUMENT_DIR/patched/gdn_attn.py:/vllm-workspace/vllm/vllm/v1/attention/backends/gdn_attn.py:ro"
                -v "$INSTRUMENT_DIR/patched/gdn_attn_builder.py:/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn_attn_builder.py:ro"
            )
        fi
        if [[ $MODE == on ]]; then
            extra+=(
                -e PI_SUBSCOPE=on
                -e PI_SUBSCOPE_LOG=/runmeta/pi-subscope/raw.csv
                -e PI_SUBSCOPE_META=/runmeta/pi-subscope/probe_notes.json
                -e "PI_SUBSCOPE_DUMP_EVERY=${PI_SUBSCOPE_DUMP_EVERY:-25}"
            )
        else
            extra+=(-e PI_SUBSCOPE=off)
        fi
        if [[ -n ${EXTRA_ARGS:-} ]]; then
            extra+=("$EXTRA_ARGS")
        fi
        # Word-split by design: the hook consumes a flat string.
        printf -v flat '%s ' "${extra[@]}"
        export PREPARE_INPUT_EXTRA_DOCKER_ARGS="${flat% }"
        ;;
esac

# Record what was mounted, before the container starts.
{
    printf 'mode=%s\n' "$MODE"
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'instrument_dir=%s\n' "$INSTRUMENT_DIR"
    printf 'PI_SUBSCOPE=%s\n' "${PI_SUBSCOPE:-}"
    printf 'attn_probes=%s\n' "$ATTN_PROBES"
    printf 'extra_docker_args=%s\n' "$PREPARE_INPUT_EXTRA_DOCKER_ARGS"
    for f in "${MOUNTED[@]:-$PATCHED}"; do
        [[ -f $f ]] && sha256sum "$f"
    done
    date -u +%Y-%m-%dT%H:%M:%SZ
} >"$RUN_DIR/pi-subscope/launch.env.txt"

# `bash <script>` rather than `./<script>`: the launcher's mode bit has been
# lost by at least one editor round-trip in this project, and that failure mode
# (126 Permission denied) is indistinguishable from a real launch error.
set +e
bash "$SCRIPT_DIR/launch_phase_service.sh" --run-id "$RUN_ID" \
    "${FORWARDED[@]+"${FORWARDED[@]}"}"
INNER_RC=$?
set -e

# ---------------------------------------------------------------- hand-off
# A dry run never creates a container, so there is no state to hand off.
DRY_RUN=0
for arg in "${FORWARDED[@]+"${FORWARDED[@]}"}"; do
    [[ $arg == --dry-run ]] && DRY_RUN=1
done
if ((DRY_RUN)); then
    exit "$INNER_RC"
fi

# Rebuild the machine-readable manifest from live state.  launch_phase_service
# already writes one, but this second pass runs after the container/CPU/thread
# state settled and is what profiler stages are told to read.
python3 "$SCRIPT_DIR/run_manifest.py" --run-id "$RUN_ID" \
    --project-root "$PROJECT_ROOT" || \
    echo "WARNING: run_manifest.py failed; the run itself returned $INNER_RC" >&2

if ((INNER_RC != 0)); then
    echo "launch_phase_service.sh exited with $INNER_RC" >&2
fi
exit "$INNER_RC"
