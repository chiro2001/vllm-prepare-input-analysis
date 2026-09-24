#!/usr/bin/env bash
# chip3 exclusivity lock for the prepare_input analysis project.
#
# Primitive: an atomic `mkdir` of locks/chip3.holder.  Whoever wins the mkdir
# owns the NPU; everybody else waits or fails closed.  The lease file
# locks/chip3.lease.json records who holds it, why, and since when, so a
# human can always tell whether a stuck holder is dead.
#
# Usage:
#   chip3_lock.sh acquire --owner ID --purpose TEXT [--wait SEC] [--pid PID]
#   chip3_lock.sh status
#   chip3_lock.sh release
#   chip3_lock.sh run --owner ID --purpose TEXT -- CMD [ARGS...]
#
# Exit codes: 0 ok; 1 usage/error; 75 held by someone else (EX_TEMPFAIL).
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PREPARE_INPUT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
LOCK_DIR="$PROJECT_ROOT/locks"
HOLDER_DIR="$LOCK_DIR/chip3.holder"
LEASE_FILE="$LOCK_DIR/chip3.lease.json"

die() { printf 'chip3_lock: ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf 'chip3_lock: %s\n' "$*" >&2; }

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

lease_owner() {
    [[ -f $LEASE_FILE ]] || { printf ''; return; }
    python3 -c 'import json,sys
try:
    print(json.load(open(sys.argv[1])).get("owner",""))
except Exception:
    print("")' "$LEASE_FILE" 2>/dev/null || printf ''
}

cmd_acquire() {
    local owner="" purpose="" wait_s=0 pid="$$"
    while (($#)); do
        case "$1" in
            --owner) owner=$2; shift 2 ;;
            --purpose) purpose=$2; shift 2 ;;
            --wait) wait_s=$2; shift 2 ;;
            --pid) pid=$2; shift 2 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    [[ -n $owner ]] || die "acquire needs --owner"
    [[ -n $purpose ]] || die "acquire needs --purpose"

    mkdir -p "$LOCK_DIR"
    local deadline=$((SECONDS + wait_s))
    while true; do
        if mkdir "$HOLDER_DIR" 2>/dev/null; then
            local host
            host=$(hostname)
            python3 - "$LEASE_FILE" "$owner" "$purpose" "$pid" "$host" <<'PY'
import json, sys, datetime
path, owner, purpose, pid, host = sys.argv[1:6]
doc = {
    "resource": "chip3",
    "device": "/dev/davinci3",
    "owner": owner,
    "purpose": purpose,
    "pid": int(pid),
    "host": host,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cpuset": "120-159",
    "nonce": f"{pid}-{datetime.datetime.now().timestamp()}",
}
with open(path, "w") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
            say "acquired by owner=$owner pid=$pid purpose=$purpose"
            return 0
        fi
        if ((SECONDS >= deadline)); then
            say "held by owner=$(lease_owner) (waited ${wait_s}s)"
            return 75
        fi
        sleep 1
    done
}

cmd_status() {
    if [[ -d $HOLDER_DIR ]]; then
        printf 'HELD\n'
        [[ -f $LEASE_FILE ]] && cat "$LEASE_FILE"
        return 0
    fi
    printf 'FREE\n'
    return 0
}

cmd_release() {
    local owner=""
    while (($#)); do
        case "$1" in
            --owner) owner=$2; shift 2 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    if [[ -n $owner ]]; then
        local current
        current=$(lease_owner)
        if [[ -n $current && $current != "$owner" ]]; then
            die "refusing to release: held by owner=$current, not $owner"
        fi
    fi
    rmdir "$HOLDER_DIR" 2>/dev/null || true
    rm -f "$LEASE_FILE"
    say "released"
}

cmd_run() {
    local owner="" purpose="" wait_s=1800
    while (($#)); do
        case "$1" in
            --owner) owner=$2; shift 2 ;;
            --purpose) purpose=$2; shift 2 ;;
            --wait) wait_s=$2; shift 2 ;;
            --) shift; break ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    (($#)) || die "run needs a command after --"
    local rc=0
    cmd_acquire --owner "$owner" --purpose "$purpose" --wait "$wait_s" || return $?
    "$@" || rc=$?
    cmd_release --owner "$owner" || true
    return $rc
}

main() {
    local cmd=${1:-}
    [[ -n $cmd ]] || die "usage: chip3_lock.sh {acquire|status|release|run} ..."
    shift
    case "$cmd" in
        acquire) cmd_acquire "$@" ;;
        status) cmd_status "$@" ;;
        release) cmd_release "$@" ;;
        run) cmd_run "$@" ;;
        *) die "unknown subcommand: $cmd" ;;
    esac
}

main "$@"
