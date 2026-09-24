#!/usr/bin/env bash
# Apply (or verify) the prepare_input sub-scope instrumentation.
#
# Preferred mode is *non-invasive*: nothing is written into the container
# image or into the read-only reference checkouts.  Instead the two files are
# bind-mounted over the in-image copies:
#
#   -v $INSTRUMENT/patched/model_runner_v1.py:\
#      /vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro
#   -v $INSTRUMENT/pi_subscope.py:\
#      /vllm-workspace/vllm-ascend/vllm_ascend/worker/pi_subscope.py:ro
#
# scripts/launch_subscope_service.sh composes exactly that command line.
#
# For source-tree work (code review, running the verifier) this script can
# additionally copy the instrumented file into a *writable copy* of the
# vllm-ascend checkout:
#
#   apply.sh --target /path/to/vllm-ascend          # patch a checkout
#   apply.sh --target /path/to/vllm-ascend --revert # restore it
#
# Usage:
#   apply.sh                        # verify hashes + print the mount args
#   apply.sh --docker-args          # print only the -v/-e arguments
#   apply.sh --target DIR [--revert]
set -euo pipefail

INSTRUMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE="$INSTRUMENT_DIR/baseline/model_runner_v1.py.image"
PATCHED="$INSTRUMENT_DIR/patched/model_runner_v1.py"
HELPER="$INSTRUMENT_DIR/pi_subscope.py"
BASELINE_ATTN="$INSTRUMENT_DIR/baseline/attention_v1.py.image"
PATCHED_ATTN="$INSTRUMENT_DIR/patched/attention_v1.py"

# sha256 of the file as shipped inside local/vllm-ascend-liteprofiler:
# v0.26.0rc1-openeuler (vllm-ascend f2f74a16c + LiteProfiler v1/v2 patch).
EXPECTED_BASELINE_SHA=bf65ca111750da9a411853cc59e2fe756756fa58304f48bd53d370e3811bab1f

TARGET=""
REVERT=0
ONLY_ARGS=0

while (($#)); do
    case "$1" in
        --target) TARGET=$2; shift 2 ;;
        --revert) REVERT=1; shift ;;
        --docker-args) ONLY_ARGS=1; shift ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

verify() {
    local got
    got=$(sha256sum "$BASELINE" | cut -d' ' -f1)
    if [[ $got != "$EXPECTED_BASELINE_SHA" ]]; then
        echo "ERROR: baseline sha256 mismatch" >&2
        echo "  expected $EXPECTED_BASELINE_SHA" >&2
        echo "  got      $got" >&2
        return 1
    fi
    python3 "$INSTRUMENT_DIR/verify_patch.py" "$BASELINE" "$PATCHED" >/dev/null
    python3 "$INSTRUMENT_DIR/verify_patch.py" "$BASELINE_ATTN" "$PATCHED_ATTN" >/dev/null
    for pair in "attention_v1" "sfa_v1" "gdn_attn"; do
        python3 "$INSTRUMENT_DIR/verify_patch.py" \
            "$INSTRUMENT_DIR/baseline/$pair.py.image" \
            "$INSTRUMENT_DIR/patched/$pair.py" >/dev/null
    done
    echo "baseline sha256 ok; patched files are baseline + probes (AST equivalence)"
}

docker_args() {
    printf -- '-v %s:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro ' "$PATCHED"
    printf -- '-v %s:/vllm-workspace/vllm-ascend/vllm_ascend/worker/pi_subscope.py:ro ' "$HELPER"
    printf -- '-e PI_SUBSCOPE=on -e PI_SUBSCOPE_LOG=/runmeta/pi-subscope/raw.csv '
    printf -- '-e PI_SUBSCOPE_META=/runmeta/pi-subscope/probe_notes.json '
    printf -- '-e PI_SUBSCOPE_DUMP_EVERY=%s' "${PI_SUBSCOPE_DUMP_EVERY:-25}"
}

if ((ONLY_ARGS)); then
    docker_args
    exit 0
fi

verify

if [[ -n $TARGET ]]; then
    dest="$TARGET/vllm_ascend/worker/model_runner_v1.py"
    helper_dest="$TARGET/vllm_ascend/worker/pi_subscope.py"
    [[ -f $dest ]] || { echo "ERROR: $dest not found" >&2; exit 2; }
    if ((REVERT)); then
        cp -f "$BASELINE" "$dest"
        rm -f "$helper_dest"
        echo "reverted $dest from $BASELINE and removed $helper_dest"
        exit 0
    fi
    cp -f "$PATCHED" "$dest"
    cp -f "$HELPER" "$helper_dest"
    echo "patched $dest and installed $helper_dest"
    echo "NOTE: keep a copy of the original file or use --revert to restore."
    exit 0
fi

cat <<EOF
Everything checks out.  Mount the instrumentation with either

    PREPARE_INPUT_EXTRA_DOCKER_ARGS="$(docker_args)" \\
        scripts/launch_phase_service.sh --run-id ID ...

or the wrapper that composes it for you

    scripts/launch_subscope_service.sh --run-id ID --mode on ...

(modes: baseline | mounted-off | on).
EOF
