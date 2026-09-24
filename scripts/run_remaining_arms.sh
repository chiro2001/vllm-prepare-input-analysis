#!/usr/bin/env bash
# Chain the remaining measurement arms of the sub-scope study.
#
# Runs, strictly serially on the chip3 lock, each arm as its own service:
#   S1 D1  pystack on,  probes off   (sampler pollution)
#   S1 D2  pystack off, probes off   (noise-floor repeat of B)
#   S1 C2  pystack off, probes on, second-level attention probes mounted
#   S2 A,B,C,D1,D2,C2                (B=64 ISL=128 decode, full ladder)
#
# Every arm keeps its artifacts under runs/<run_id>/ and data/subscope/<run_id>/.
# No container is kept alive except the ones the individual arms ask for, so the
# next arm's preflight (npu-smi "No process in device") succeeds.
#
# Usage:  nohup scripts/run_remaining_arms.sh > logs/remaining-arms.log 2>&1 &
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PREPARE_INPUT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
STAMP=${PI_STAMP2:-S1c}
STAMP_S2=${PI_STAMP_S2:-S2a}

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

require_free_chip() {
    local probe
    probe=$(sudo -n npu-smi info -t proc-mem -i 1 -c 1 2>&1 || true)
    if grep -q "Process id:" <<<"$probe"; then
        echo "ERROR: chip3 still busy: $probe" >&2
        return 1
    fi
}

log "=== S1 remaining arms (D1, D2, C2), stamp=$STAMP"
"$SCRIPT_DIR/run_subscope_matrix.sh" \
    --scenario s1 --modes D1,D2,C2 --port-base 18110

require_free_chip
log "=== S2 full ladder (A,B,C,D1,D2,C2), stamp=$STAMP_S2"
"$SCRIPT_DIR/run_subscope_matrix.sh" \
    --scenario s2 --modes A,B,C,D1,D2,C2 --port-base 18120

log "=== all remaining arms complete"
