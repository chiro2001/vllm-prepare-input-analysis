#!/usr/bin/env bash
# Revert the prepare_input sub-scope instrumentation.
#
# With bind mounts there is nothing to undo on the host: the container is
# created fresh in every run and the mounts only exist for that container's
# lifetime.  Run this script after the container is gone (or use it to make
# sure it is gone) and, for source-tree checkouts, to restore the pristine
# file from the image baseline.
#
# Usage:
#   revert.sh [--target /path/to/vllm-ascend] [--container NAME]
set -euo pipefail

INSTRUMENT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASELINE="$INSTRUMENT_DIR/baseline/model_runner_v1.py.image"
HELPER="$INSTRUMENT_DIR/pi_subscope.py"

TARGET=""
CONTAINER=""

while (($#)); do
    case "$1" in
        --target) TARGET=$2; shift 2 ;;
        --container) CONTAINER=$2; shift 2 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -n $TARGET ]]; then
    dest="$TARGET/vllm_ascend/worker/model_runner_v1.py"
    [[ -f $dest ]] || { echo "ERROR: $dest not found" >&2; exit 2; }
    cp -f "$BASELINE" "$dest"
    rm -f "$TARGET/vllm_ascend/worker/pi_subscope.py"
    echo "restored $dest from the image baseline"
    sha256sum "$dest"
fi

if [[ -n $CONTAINER ]]; then
    if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        echo "removing container $CONTAINER (mounts disappear with it)"
        docker rm -f "$CONTAINER" >/dev/null
    fi
fi

cat <<'EOF'
Bind mounts are per-container, so a fresh container always starts clean.
Verify with:
    docker run --rm --entrypoint sha256sum IMAGE \
        /vllm-workspace/vllm-ascend/vllm_ascend/worker/pi_subscope.py   # must fail
EOF
