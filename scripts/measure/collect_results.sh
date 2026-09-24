#!/usr/bin/env bash
# Pull a measurement run back from a3-22 and render the figures locally.
#
# The target owns capture (host TIDs, root perf/libkperfx, the NPU); the dev
# machine owns rendering, because a3-22's system python has neither matplotlib
# nor numpy and we must not install packages into that shared host.
#
# Usage:
#   scripts/measure/collect_results.sh pull  RUN_ID [RUN_ID...]   # rsync down
#   scripts/measure/collect_results.sh plot  RUN_ID [--prefix 03] [--title-suffix S]
#   scripts/measure/collect_results.sh all   RUN_ID
#
# Every artifact stays inside the two allowed trees; raw perf.data is excluded
# from what we keep locally (only summaries/folded stacks/SVG travel back).
set -uo pipefail

LOCAL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE=${REMOTE:-a3-22}
REMOTE_HOME=${REMOTE_HOME:-$(ssh -o ConnectTimeout=20 "$REMOTE" 'echo $HOME')}
REMOTE_ROOT=${REMOTE_ROOT:-$REMOTE_HOME/projects/vllm/prepare-input-phase}

say() { printf '[collect] %s\n' "$*" >&2; }
die() { say "ERROR: $*"; exit 1; }

cmd_pull() {
    local run_id=$1
    [[ -n $run_id ]] || die "pull needs RUN_ID"
    mkdir -p "$LOCAL_ROOT/data/measure"
    say "pulling data/measure/$run_id (excluding perf.data)"
    rsync -a --exclude 'perf.data' --exclude 'perf.data.*' --exclude '*.part' \
        "$REMOTE:$REMOTE_ROOT/data/measure/$run_id" "$LOCAL_ROOT/data/measure/" \
        || die "rsync failed"
    local n
    n=$(find "$LOCAL_ROOT/data/measure/$run_id/points" -name point.json 2>/dev/null | wc -l)
    say "pulled $n point.json file(s)"
}

cmd_plot() {
    local run_id=$1 prefix=03 suffix=""
    shift
    while (($#)); do
        case "$1" in
            --prefix) prefix=$2; shift 2 ;;
            --title-suffix) suffix=$2; shift 2 ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    local dir="$LOCAL_ROOT/data/measure/$run_id"
    [[ -d $dir ]] || die "no such run dir: $dir (run 'pull' first)"
    # `plot_measure.py` wants a flat array of point.json documents.  The run
    # log written by matrix_run.py (`points.json`) deliberately wraps each
    # record with its exit code, so the flat view is always regenerated here.
    local points_json="$dir/points_flat.json"
    say "synthesising the flat point list from points/*/point.json"
    python3 - "$dir" "$points_json" <<'PY'
import json, pathlib, sys
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
records = []
for pj in sorted(root.glob("points/*/point.json")):
    try:
        records.append(json.loads(pj.read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001
        print(f"skip {pj}: {exc}", file=sys.stderr)
out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out} with {len(records)} record(s)", file=sys.stderr)
PY
    [[ -s $points_json ]] || die "no point.json found under $dir/points"
    mkdir -p "$LOCAL_ROOT/figures"
    say "rendering figures with prefix $prefix"
    python3 "$LOCAL_ROOT/scripts/measure/plot_measure.py" \
        --points-json "$points_json" \
        --outdir "$LOCAL_ROOT/figures" \
        --data-outdir "$dir" \
        --prefix "$prefix" ${suffix:+--title-suffix "$suffix"}
}

cmd_all() {
    local run_id=$1
    cmd_pull "$run_id"
    cmd_plot "$run_id"
}

main() {
    local cmd=${1:-}
    [[ -n $cmd ]] || die "usage: collect_results.sh {pull|plot|all} RUN_ID ..."
    shift
    case "$cmd" in
        pull) cmd_pull "$@" ;;
        plot) cmd_plot "$@" ;;
        all) cmd_all "$@" ;;
        *) die "unknown subcommand: $cmd" ;;
    esac
}

main "$@"
