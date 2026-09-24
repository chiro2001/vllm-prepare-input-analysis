#!/usr/bin/env bash
# Launch a matrix campaign on a3-22 detached from this shell.
#
# A campaign brings up 5 services (~2.5 min each) plus 36 points, so it runs
# for hours.  Running it under nohup keeps it alive across ssh disconnects, and
# the log file is the first thing to look at when something goes wrong.
#
# Usage (on a3-22):
#   scripts/measure/run_campaign.sh start  RUN_ID [--plan FILE] [--only-tags a,b]
#   scripts/measure/run_campaign.sh status RUN_ID
#   scripts/measure/run_campaign.sh tail   RUN_ID [N]
#
set -uo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PLAN="$PROJECT_ROOT/scripts/measure/plans/matrix-main.json"
LOG_DIR="$PROJECT_ROOT/logs"

cmd_start() {
    local run_id=$1; shift
    [[ -n $run_id ]] || { echo "start needs RUN_ID" >&2; return 1; }
    while (($#)); do
        case "$1" in
            --plan) PLAN=$2; shift 2 ;;
            *) echo "unknown arg $1" >&2; return 1 ;;
        esac
    done
    mkdir -p "$LOG_DIR"
    local log="$LOG_DIR/campaign-$run_id.log"
    if [[ -f $LOG ]] && pgrep -f "matrix_run.py --plan .* --run-id $run_id" >/dev/null 2>&1; then
        echo "campaign $run_id already running (log $log)" >&2
        return 1
    fi
    : >"$log"
    nohup setsid python3 "$PROJECT_ROOT/scripts/measure/matrix_run.py" \
        --plan "$PLAN" --run-id "$run_id" "$@" >>"$log" 2>&1 &
    local pid=$!
    echo "campaign $run_id started: pid=$pid log=$log"
}

cmd_status() {
    local run_id=$1
    local log="$LOG_DIR/campaign-$run_id.log"
    if pgrep -f "matrix_run.py --plan .* --run-id $run_id" >/dev/null 2>&1; then
        echo "RUNNING"
    else
        echo "NOT RUNNING"
    fi
    [[ -f $log ]] && tail -n 5 "$log"
    python3 - "$PROJECT_ROOT/data/measure/$run_id/points.json" <<'PY' 2>/dev/null
import json, sys
try:
    recs = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
ok = sum(1 for r in recs if r.get("rc") == 0)
print(f"points recorded: {len(recs)} (rc=0: {ok})")
for r in recs[-3:]:
    s = r.get("summary") or {}
    print(f"  {r.get('tag')}: rc={r.get('rc')} tps={s.get('tps_aggregate_mean')}")
PY
}

cmd_tail() {
    local run_id=$1 n=${2:-40}
    tail -n "$n" "$LOG_DIR/campaign-$run_id.log"
}

main() {
    local cmd=${1:-}
    [[ -n $cmd ]] || { echo "usage: run_campaign.sh {start|status|tail} RUN_ID" >&2; exit 1; }
    shift
    case "$cmd" in
        start) cmd_start "$@" ;;
        status) cmd_status "$@" ;;
        tail) cmd_tail "$@" ;;
        *) echo "unknown subcommand $cmd" >&2; exit 1 ;;
    esac
}

main "$@"
