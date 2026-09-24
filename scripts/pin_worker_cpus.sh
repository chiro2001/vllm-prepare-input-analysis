#!/usr/bin/env bash
# Pin a running vLLM-Ascend worker to the chip's CPU slice, without the
# upstream `enable_cpu_binding` side effects.
#
# Why this exists instead of `--additional-config '{"enable_cpu_binding":true}'`:
# vllm_ascend.cpu_binding.CpuAlloc.bind_npu_irq() calls
# `systemctl stop irqbalance` on the *host* when it finds the service active.
# On a3-22 irqbalance is active+enabled and the host is shared with other
# tenants, so a container must never be allowed to run that path.  With
# `enable_cpu_binding:false` the engine skips all of it, and this script
# reproduces the part we actually want: the same role split the upstream code
# would have produced for a 40-CPU chip slice.
#
#   main (worker forward / engine core)  ->  cpuset minus the last two CPUs
#   acl_thread                           ->  second to last CPU
#   release_thread                       ->  last CPU
#
# For chip3 on a3-22 that is main=122-157, acl=158, release=159.
#
# Usage:
#   pin_worker_cpus.sh --chip N [--main LIST] [--acl LIST] [--rel LIST]
#                      [--pid PID | --container NAME] [--verify] [--dry-run]
set -euo pipefail

die() { printf 'pin_worker_cpus: ERROR: %s\n' "$*" >&2; exit 1; }
say() { printf 'pin_worker_cpus: %s\n' "$*" >&2; }

CHIP=3
MAIN=""
ACL=""
REL=""
PID=""
CONTAINER=""
VERIFY=0
DRY_RUN=0

while (($#)); do
    case "$1" in
        --chip) CHIP=$2; shift 2 ;;
        --main) MAIN=$2; shift 2 ;;
        --acl) ACL=$2; shift 2 ;;
        --rel) REL=$2; shift 2 ;;
        --pid) PID=$2; shift 2 ;;
        --container) CONTAINER=$2; shift 2 ;;
        --verify) VERIFY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ $CHIP =~ ^[0-9]+$ ]] || die "--chip must be a number"
FIRST=$((40 * CHIP))
MEMS=$((CHIP / 2))
CARD=$((CHIP / 2))
SUBCHIP=$((CHIP % 2))
[[ -n $MAIN ]] || MAIN="$((FIRST + 2))-$((FIRST + 37))"
[[ -n $ACL ]] || ACL="$((FIRST + 38))"
[[ -n $REL ]] || REL="$((FIRST + 39))"

resolve_pid() {
    if [[ -n $PID ]]; then
        printf '%s' "$PID"
        return
    fi
    [[ -n $CONTAINER ]] || die "need --pid or --container"
    local from_npu
    # Authoritative: the process npu-smi says is holding this die.
    from_npu=$(sudo -n npu-smi info -t proc-mem -i "$CARD" -c "$SUBCHIP" 2>/dev/null \
        | awk -F: '/Process id/{gsub(/[^0-9]/,"",$2); print $2; exit}')
    # NOTE: the engine core runs as root inside the container, so `kill -0`
    # from this unprivileged shell fails with EPERM for a *live* process.  A
    # bare /proc existence check is the right liveness test here.
    if [[ -n $from_npu && -d /proc/$from_npu ]]; then
        printf '%s' "$from_npu"
        return
    fi
    # Fallback: the container's main pid.  Only valid when the engine core is
    # in the container's first process (uni backend); ambiguous for mp.
    local cpid
    cpid=$(sudo -n docker inspect -f '{{.State.Pid}}' "$CONTAINER" 2>/dev/null || true)
    [[ -n $cpid && $cpid != 0 ]] || die "cannot resolve the NPU process for chip $CHIP"
    printf '%s' "$cpid"
}

threads_named() {
    local pid=$1 name=$2
    sudo -n sh -c "for t in /proc/$pid/task/*; do
        [ -r \"\$t/comm\" ] || continue
        if [ \"\$(cat \"\$t/comm\" 2>/dev/null)\" = \"$name\" ]; then
            basename \"\$t\"
        fi
    done"
}

show_affinity() {
    local pid=$1
    local main_mask acl_mask rel_mask
    main_mask=$(taskset -pc "$pid" 2>/dev/null | sed 's/^.*: //')
    printf '  pid=%s main_thread.cpus=%s\n' "$pid" "$main_mask" >&2
    sudo -n sh -c "for t in /proc/$pid/task/*; do
        comm=\$(cat \"\$t/comm\" 2>/dev/null) || continue
        case \"\$comm\" in
            acl_thread|release_thread)
                printf '  tid=%s comm=%s cpus=%s\n' \"\$(basename \$t)\" \"\$comm\" \
                    \"\$(taskset -pc \$(basename \$t) 2>/dev/null | sed 's/^.*: //')\"
                ;;
        esac
    done" >&2
}

main() {
    local pid
    pid=$(resolve_pid)
    [[ -n $pid ]] || die "no pid"
    say "chip=$CHIP card=$CARD subchip=$SUBCHIP pid=$pid"
    say "target main=$MAIN acl=$ACL release=$REL"
    printf 'pid=%s main=%s acl=%s rel=%s\n' "$pid" "$MAIN" "$ACL" "$REL"

    if ((VERIFY)); then
        show_affinity "$pid"
        return 0
    fi
    if ((DRY_RUN)); then
        say "dry-run: would taskset -acp $MAIN $pid"
        return 0
    fi

    # All threads first; the role-specific ones are carved out afterwards.
    sudo -n taskset -acp "$MAIN" "$pid" >/dev/null
    local tid
    while read -r tid; do
        [[ -n $tid ]] || continue
        sudo -n taskset -pc "$ACL" "$tid" >/dev/null
        say "acl_thread tid=$tid -> $ACL"
    done < <(threads_named "$pid" acl_thread)
    while read -r tid; do
        [[ -n $tid ]] || continue
        sudo -n taskset -pc "$REL" "$tid" >/dev/null
        say "release_thread tid=$tid -> $REL"
    done < <(threads_named "$pid" release_thread)

    show_affinity "$pid"
    say "done"
}

main "$@"
