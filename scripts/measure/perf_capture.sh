#!/usr/bin/env bash
#
# perf_capture.sh -- capture the on-CPU profile of ONE host TID and turn it into
# report / hotspot / folded-stack / flamegraph / annotate artifacts.
#
#   bash scripts/measure/perf_capture.sh --tid HOST_TID --seconds S --name NAME \
#        --outdir DIR [--freq 999] [--call-graph fp|dwarf|lbr] [--title T]
#
# Artifacts written into DIR (all rewritten on every run; no leftovers):
#   perf.data              raw capture           (large -- never uploaded)
#   perf-record.log        raw perf record stderr/stdout
#   perf-report.txt        perf report --stdio --no-children --sort dso,symbol
#   hotspots.csv           rank,dso,symbol,self_pct,samples   (top 30, self)
#   folded.txt             perf script | inferno-collapse-perf (on-CPU stacks)
#   flame.svg              inferno-flamegraph, title=NAME
#   annotate-<sym>.txt     perf annotate --stdio -l --symbol <sym>  (top 3)
#   hotspots_srcline.csv   rank,file_line,pct,symbol           (top 20)
#   hotspots_instr.csv     rank,symbol,address,instruction,pct  (top 20)
#   manifest.json          machine readable record (schema below)
#   perf-report.err        stderr of the report/script/annotate steps
#
# Notes that matter on a3-22 (Kunpeng 920B / openEuler 24.03):
#   * perf_event_paranoid=2, so every perf call goes through `sudo -n`.
#   * only the requested TID is sampled (`perf record -t TID`); nothing else on
#     the box is touched.  The timer is a child `sleep S`, so the run stops by
#     itself after ~S seconds -- no kill, no signal games.
#   * the sampling rate is pinned with -F so two machines are comparable.
#
# The target python may live inside a container.  Two things then have to hold
# or the python-level frames disappear (see agents/measurement_profiling/
# PERF_SYMBOLS.md):
#   * CPython >= 3.12 must run with PYTHONPERFSUPPORT=1 (stack trampoline);
#   * perf must be able to read /tmp/perf-<host_tid>.map.  Use `--pid=host`
#     when creating the container, or pass --perf-map <path> here so that the
#     map is linked into place.

set -uo pipefail

SELF="$(basename "$0")"
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ORIG_ARGV=("$@")

TID=""
SECS=""
NAME=""
OUTDIR=""
FREQ=999
CG="fp"
TITLE=""
EVENT="cycles"
PERF_MAP=""
DO_ANNOTATE=1
ANNOT_TOP=3
TOOLS_BIN="${PIC_TOOLS_BIN:-$HOME/projects/vllm/prepare-input-phase/tools/bin}"
SUDO="${PIC_SUDO-sudo -n}"
SUDO_ARR=()
# shellcheck disable=SC2206  (intentional word split: "sudo -n" is two argv items)
[ -n "$SUDO" ] && SUDO_ARR=($SUDO)

STAGE="init"
FAIL_MSG=""

usage() {
    sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --tid TID              host TID to sample (required)
  --seconds S            sampling window in seconds (required)
  --name NAME            run name; becomes the flame graph title (required)
  --outdir DIR           artifact directory, created if missing (required)
  --freq HZ              sample frequency, default 999
  --call-graph MODE      fp | dwarf | lbr | dwarf,32768 ..., default fp
  --title T              override the flame graph title (default: NAME)
  --event E              PMU event, default cycles
  --perf-map PATH        link PATH to /tmp/perf-<tid>.map before recording
                         (CPython trampoline map of a containerised process)
  --annotate-top N       annotate the N hottest symbols, default 3 (0 = skip)
  --no-annotate          same as --annotate-top 0
  -h, --help             this text

Env:
  PIC_TOOLS_BIN  directory holding inferno-collapse-perf / inferno-flamegraph
                 (default ~/projects/vllm/prepare-input-phase/tools/bin)
  PIC_SUDO       command prefix used for perf, default "sudo -n"

Exit code 0 on success.  On failure the artifacts produced so far are kept and
manifest.json records the failing stage in "error" plus "notes".
EOF
}

die_usage() { echo "$SELF: $*" >&2; echo "try --help" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --tid)         [ $# -ge 2 ] || die_usage "--tid needs a value";  TID="$2"; shift 2 ;;
        --seconds)     [ $# -ge 2 ] || die_usage "--seconds needs a value"; SECS="$2"; shift 2 ;;
        --name)        [ $# -ge 2 ] || die_usage "--name needs a value"; NAME="$2"; shift 2 ;;
        --outdir)      [ $# -ge 2 ] || die_usage "--outdir needs a value"; OUTDIR="$2"; shift 2 ;;
        --freq)        [ $# -ge 2 ] || die_usage "--freq needs a value"; FREQ="$2"; shift 2 ;;
        --call-graph)  [ $# -ge 2 ] || die_usage "--call-graph needs a value"; CG="$2"; shift 2 ;;
        --title)       [ $# -ge 2 ] || die_usage "--title needs a value"; TITLE="$2"; shift 2 ;;
        --event)       [ $# -ge 2 ] || die_usage "--event needs a value"; EVENT="$2"; shift 2 ;;
        --perf-map)    [ $# -ge 2 ] || die_usage "--perf-map needs a value"; PERF_MAP="$2"; shift 2 ;;
        --annotate-top) [ $# -ge 2 ] || die_usage "--annotate-top needs a value"; ANNOT_TOP="$2"; shift 2 ;;
        --no-annotate) DO_ANNOTATE=0; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             die_usage "unknown argument: $1" ;;
    esac
done

[ -n "$TID" ]    || die_usage "--tid is required"
[ -n "$SECS" ]   || die_usage "--seconds is required"
[ -n "$NAME" ]   || die_usage "--name is required"
[ -n "$OUTDIR" ] || die_usage "--outdir is required"
case "$TID"  in *[!0-9]*|'') die_usage "--tid must be a numeric host TID" ;; esac
case "$SECS" in *[!0-9]*|'') die_usage "--seconds must be a positive integer" ;; esac
case "$FREQ" in *[!0-9]*|'') die_usage "--freq must be a positive integer" ;; esac
[ "$SECS" -gt 0 ] || die_usage "--seconds must be > 0"
[ "$FREQ" -gt 0 ] || die_usage "--freq must be > 0"
case "$ANNOT_TOP" in *[!0-9]*|'') die_usage "--annotate-top must be >= 0" ;; esac

[ -n "$TITLE" ] || TITLE="$NAME"

# perf's default dwarf stack dump is 8192 B; spell it out so the byte size of
# perf.data does not depend on the perf build.
CG_EFF="$CG"
case "$CG" in dwarf) CG_EFF="dwarf,8192" ;; esac

mkdir -p "$OUTDIR" || { echo "$SELF: cannot create $OUTDIR" >&2; exit 2; }
OUTDIR="$(cd "$OUTDIR" && pwd)"

RECORD_LOG="$OUTDIR/perf-record.log"
ERR_LOG="$OUTDIR/perf-report.err"
TMPDIR_PC="$(mktemp -d "${TMPDIR:-/tmp}/perf_capture.XXXXXX")"
NOTES_FILE="$TMPDIR_PC/notes"
: > "$NOTES_FILE"
: > "$ERR_LOG"

note() { printf '%s\n' "$*" >> "$NOTES_FILE"; }

# ---------------------------------------------------------------- artifacts --
clean_outputs() {
    rm -f "$OUTDIR"/perf.data \
          "$OUTDIR"/perf-report.txt "$OUTDIR"/hotspots.csv \
          "$OUTDIR"/folded.txt "$OUTDIR"/flame.svg "$OUTDIR"/manifest.json \
          "$OUTDIR"/hotspots_srcline.csv "$OUTDIR"/hotspots_instr.csv
    rm -f "$OUTDIR"/annotate-*.txt
}

# The manifest writer is deliberately tolerant: it is also used on the failure
# path, where most of the numbers do not exist yet.
write_manifest() {
    PC_OUTDIR="$OUTDIR" PC_TID="$TID" PC_SECS="$SECS" PC_FREQ="$FREQ" \
    PC_CG="$CG_EFF" PC_CG_REQ="$CG" PC_EVENT="$EVENT" PC_NAME="$NAME" \
    PC_TITLE="$TITLE" PC_STARTED="$STARTED_UTC" \
    PC_SAMPLE_COUNT="${SAMPLE_COUNT:-0}" PC_LOST="${LOST_SAMPLES:-0}" \
    PC_RESOLVED="${RESOLVED_PCT:-}" PC_UNKNOWN="${UNKNOWN_PCT:-}" \
    PC_DATA_BYTES="${DATA_BYTES:-0}" PC_PYFRAMES="${PY_FRAMES:-0}" \
    PC_PYLEAF="${PY_LEAF:-0}" PC_STAGE="$STAGE" PC_ERROR="$FAIL_MSG" \
    PC_PYCOV="${PY_COVERAGE_PCT:-}" PC_PYLEAFP="${PY_LEAF_PCT:-}" \
    PC_UNKFRAME="${UNKFRAME_PCT:-}" PC_MEANFRAMES="${MEAN_FRAMES:-}" \
    PC_UNKFRAME_LINE="${UNKFRAME_LINE_PCT:-}" PC_UNKFRAMES="${UNK_FRAMES:-0}" \
    PC_FOLDSAMPLES="${FOLD_SAMPLES:-0}" \
    PC_RECORD_LOG="$RECORD_LOG" PC_NOTES_FILE="$NOTES_FILE" \
    PC_ANNOT_OK="${ANNOT_OK:-false}" PC_ANNOT_SYMS="${ANNOT_SYMS:-}" \
    PC_ANNOT_NODEBUG="${ANNOT_NODEBUG:-}" PC_CMD_JSON="${CMD_JSON:-[]}" \
    PC_RECORD_JSON="${RECORD_JSON:-[]}" PC_ROWS_TSV="${TMPDIR_PC:-}/rows.txt" \
    PC_TOOLS="$TOOLS_BIN" \
    python3 - <<'PY' || true
import json, os, re, platform, subprocess

def env(n, d=""):
    return os.environ.get(n, d)

def num_int(n, d=0):
    v = env(n, "")
    try:
        return int(v)
    except Exception:
        return d

def num_float(n, d=None):
    v = env(n, "")
    if v == "":
        return d
    try:
        return float(v)
    except Exception:
        return d

def lines(n):
    return [x for x in env(n, "").split("\n") if x]

def jload(n, d):
    try:
        return json.loads(env(n, "") or "null")
    except Exception:
        return d

def tool_version(name, args):
    try:
        out = subprocess.run([name] + args, capture_output=True, text=True,
                             timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception:
        return ""

note_lines = []
if os.path.exists(env("PC_NOTES_FILE")):
    try:
        with open(env("PC_NOTES_FILE"), encoding="utf-8", errors="replace") as fh:
            note_lines = [x.rstrip("\n") for x in fh if x.strip()]
    except Exception:
        note_lines = []

err = env("PC_ERROR")
if err:
    note_lines.append("FAILED at stage %s: %s" % (env("PC_STAGE"), err))

unk = num_float("PC_UNKNOWN")
res = num_float("PC_RESOLVED")

# per-dso self overhead, straight out of the report rows (pct<TAB>samples<TAB>dso<TAB>sym)
dso_counts = {}
rows_total = 0
try:
    with open(env("PC_ROWS_TSV"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            try:
                s = int(parts[1])
            except ValueError:
                continue
            dso_counts[parts[2]] = dso_counts.get(parts[2], 0) + s
            rows_total += s
except Exception:
    pass
top_dso = [d for d, _ in sorted(dso_counts.items(), key=lambda kv: -kv[1])[:5]]
dso_self_pct = {d: round(100.0 * c / rows_total, 2)
                for d, c in sorted(dso_counts.items(), key=lambda kv: -kv[1])[:8]} \
    if rows_total else {}

man = {
    "cmd": jload("PC_CMD_JSON", []),
    "perf_record_cmd": jload("PC_RECORD_JSON", []),
    "tid": num_int("PC_TID", -1),
    "seconds": num_int("PC_SECS", -1),
    "freq": num_int("PC_FREQ", -1),
    "call_graph": env("PC_CG"),
    "call_graph_requested": env("PC_CG_REQ"),
    "event": env("PC_EVENT"),
    "name": env("PC_NAME"),
    "title": env("PC_TITLE"),
    "outdir": env("PC_OUTDIR"),
    "perf_version": tool_version("perf", ["--version"]),
    "kernel_version": platform.release(),
    "sample_count": num_int("PC_SAMPLE_COUNT", 0),
    "lost_samples": num_int("PC_LOST", 0),
    "symbol_quality": {
        "resolved_pct": res,
        "unknown_pct": unk,
        "top_dso": top_dso,
        "dso_self_pct": dso_self_pct,
        "python_frames": {
            "py_frame_lines": num_int("PC_PYFRAMES", 0),
            "py_leaf_samples": num_int("PC_PYLEAF", 0),
            "samples_with_py_frame_pct": num_float("PC_PYCOV"),
            "samples_with_py_leaf_pct": num_float("PC_PYLEAFP"),
            "samples_with_unknown_frame_pct": num_float("PC_UNKFRAME"),
            "unknown_frame_lines": num_int("PC_UNKFRAMES", 0),
            "unknown_frame_line_pct": num_float("PC_UNKFRAME_LINE"),
            "mean_frames_per_sample": num_float("PC_MEANFRAMES"),
            "samples_total": num_int("PC_FOLDSAMPLES", 0),
            "note": ("self/leaf symbolisation can be ~100% while every Python "
                     "caller is still missing; use samples_with_py_frame_pct. "
                     "samples_with_unknown_frame_pct is ~100% whenever the "
                     "unresolved libc bottom frame (__libc_start_call_main) is "
                     "present, so unknown_frame_line_pct is the useful one"),
        },
    },
    "annotate": {
        "symbols": lines("PC_ANNOT_SYMS"),
        "ok": env("PC_ANNOT_OK") == "true",
        "no_debug_info": lines("PC_ANNOT_NODEBUG"),
    },
    "started_utc": env("PC_STARTED"),
    "finished_utc": "",
    "perf_data_bytes": num_int("PC_DATA_BYTES", 0),
    "tools_bin": env("PC_TOOLS"),
    "notes": note_lines,
    "error": err or None,
    "ok": (err == ""),
}
import time
man["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
tmp = os.environ["PC_OUTDIR"] + "/.manifest.json.tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(man, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp, os.path.join(os.environ["PC_OUTDIR"], "manifest.json"))
PY
}

fail() {
    echo "[$SELF] ERROR (stage=$STAGE): $*" >&2
    FAIL_MSG="$*"
    write_manifest
    exit 1
}

# -------------------------------------------------------------- command lines -
CMD_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
            "bash" "$SELF" "${ORIG_ARGV[@]}")"

# --------------------------------------------------------------- validation --
STAGE="validate"
[ -d /proc ] || fail "no /proc"
[ -d "/proc/$TID" ] || fail "TID $TID is not alive (/proc/$TID missing)"
command -v perf >/dev/null 2>&1 || fail "perf not found in PATH"
for tool in inferno-collapse-perf inferno-flamegraph; do
    [ -x "$TOOLS_BIN/$tool" ] || fail "missing $TOOLS_BIN/$tool (set PIC_TOOLS_BIN)"
done
command -v python3 >/dev/null 2>&1 || fail "python3 not found in PATH"

COMM="$(cat "/proc/$TID/comm" 2>/dev/null || echo '?')"
note "target: host TID $TID, comm '$COMM'"
note "perf event '$EVENT', -F $FREQ, -g --call-graph $CG_EFF, window ${SECS}s"

clean_outputs

# ------------------------------------------------------------ optional map ---
STAGE="perf-map"
if [ -n "$PERF_MAP" ]; then
    # The trampoline map is written by the target process, typically root inside
    # a container, so test readability as root (perf itself runs as root).
    if "${SUDO_ARR[@]}" test -r "$PERF_MAP" 2>/dev/null; then
        "${SUDO_ARR[@]}" ln -sfn "$PERF_MAP" "/tmp/perf-$TID.map" \
            || note "WARN: could not link $PERF_MAP to /tmp/perf-$TID.map"
        if "${SUDO_ARR[@]}" test -e "/tmp/perf-$TID.map" 2>/dev/null; then
            note "linked /tmp/perf-$TID.map -> $PERF_MAP ($("${SUDO_ARR[@]}" stat -c %s "$PERF_MAP" 2>/dev/null) bytes)"
        else
            note "WARN: symlink /tmp/perf-$TID.map was not created"
        fi
    else
        note "WARN: --perf-map $PERF_MAP is not readable even as root, ignored"
    fi
fi

# ----------------------------------------------------------------- capture ---
STAGE="record"
RECORD_CMD=("${SUDO_ARR[@]}" perf record -o "$OUTDIR/perf.data" -e "$EVENT"
            -F "$FREQ" -g --call-graph "$CG_EFF" -t "$TID" -- sleep "$SECS")
RECORD_JSON="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
               "${RECORD_CMD[@]}")"
{
    echo "# stage=record utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# cmd: ${RECORD_CMD[*]}"
} >> "$RECORD_LOG"
"${RECORD_CMD[@]}" >> "$RECORD_LOG" 2>&1
RECORD_RC=$?
echo "# perf record exit=$RECORD_RC" >> "$RECORD_LOG"
if [ "$RECORD_RC" -ne 0 ]; then
    note "perf record exit code $RECORD_RC (see perf-record.log)"
fi
if [ ! -s "$OUTDIR/perf.data" ]; then
    STAGE="record"
    fail "perf.data was not produced; last log lines: $(tail -n 3 "$RECORD_LOG" | tr '\n' ' ')"
fi
DATA_BYTES="$(stat -c %s "$OUTDIR/perf.data")"
note "perf.data $(numfmt --to=iec "$DATA_BYTES" 2>/dev/null || echo "$DATA_BYTES B") ($DATA_BYTES bytes)"

# Keep perf.data root owned: `perf report` refuses a file whose owner is neither
# the calling uid nor root, so a root-owned 0644 file stays usable by root and
# by any user, and it is still world readable for packaging.
"${SUDO_ARR[@]}" chmod 0644 "$OUTDIR/perf.data" 2>/dev/null || true
note "perf.data kept root-owned mode 0644 (perf's owner check); raw data is not uploaded"

# ------------------------------------------------------------------ report ---
STAGE="report"
"${SUDO_ARR[@]}" perf report --stdio --no-children --sort dso,symbol \
    -i "$OUTDIR/perf.data" > "$OUTDIR/perf-report.txt" 2>> "$ERR_LOG" \
    || note "WARN: perf report exited non-zero"

"${SUDO_ARR[@]}" perf report --stdio --no-children --sort dso,symbol \
    -F overhead,sample,dso,symbol -t '|' -i "$OUTDIR/perf.data" \
    > "$TMPDIR_PC/self.txt" 2>> "$ERR_LOG" || true

LOST_SAMPLES="$(awk -F: '/Total Lost Samples/ {gsub(/[^0-9]/,"",$2); print $2+0}' \
                "$OUTDIR/perf-report.txt" | head -n 1)"
LOST_SAMPLES="${LOST_SAMPLES:-0}"

# rows.txt: all (self) rows as pct<TAB>samples<TAB>dso<TAB>symbol
awk -F'|' '
    /^[[:space:]]*[0-9]+\.[0-9]+%[[:space:]]*\|/ {
        pct = $1; gsub(/[[:space:]%]/, "", pct)
        samples = $2; gsub(/[[:space:]]/, "", samples)
        dso = $3; gsub(/^[[:space:]]+|[[:space:]]+$/, "", dso)
        sym = $4; gsub(/^[[:space:]]+|[[:space:]]+$/, "", sym)
        sub(/^\[\.\][[:space:]]+/, "", sym)
        sub(/^\[k\][[:space:]]+/, "", sym)
        if (dso == "" || sym == "" || samples == "") next
        printf "%s\t%s\t%s\t%s\n", pct, samples, dso, sym
    }' "$TMPDIR_PC/self.txt" > "$TMPDIR_PC/rows.txt" 2>/dev/null || true

# hotspots.csv is the top-30 by *self samples* (perf's own --sort is
# alphabetical by dso, so the ordering has to be redone here).
{
    echo "rank,dso,symbol,self_pct,samples"
    sort -t"$(printf '\t')" -k2,2nr "$TMPDIR_PC/rows.txt" 2>/dev/null \
        | head -n 30 \
        | awk -F'\t' 'BEGIN{OFS=","} {
              for (i = 3; i <= 4; i++)
                  if ($i ~ /,|"/) { gsub(/"/, "\"\"", $i); $i = "\"" $i "\"" }
              print NR, $3, $4, $1, $2
          }'
} > "$OUTDIR/hotspots.csv" || true

SAMPLE_COUNT="$(awk -F'\t' '{s+=$2} END {print s+0}' "$TMPDIR_PC/rows.txt" 2>/dev/null)"
UNKNOWN_SAMPLES="$(awk -F'\t' '
    $3 == "[unknown]" || $4 == "[unknown]" || $4 ~ /^0[xX][0-9a-fA-F]+$/ {s+=$2}
    END {print s+0}' "$TMPDIR_PC/rows.txt" 2>/dev/null)"
SAMPLE_COUNT="${SAMPLE_COUNT:-0}"
UNKNOWN_SAMPLES="${UNKNOWN_SAMPLES:-0}"
if [ "$SAMPLE_COUNT" -gt 0 ]; then
    UNKNOWN_PCT="$(awk -v u="$UNKNOWN_SAMPLES" -v t="$SAMPLE_COUNT" \
                   'BEGIN {printf "%.2f", 100.0*u/t}')"
    RESOLVED_PCT="$(awk -v u="$UNKNOWN_SAMPLES" -v t="$SAMPLE_COUNT" \
                   'BEGIN {printf "%.2f", 100.0*(t-u)/t}')"
else
    UNKNOWN_PCT="100.00"; RESOLVED_PCT="0.00"
fi

# ------------------------------------------------------------------ folded ---
STAGE="folded"
"${SUDO_ARR[@]}" perf script -i "$OUTDIR/perf.data" > "$TMPDIR_PC/script.txt" 2>> "$ERR_LOG" \
    || note "WARN: perf script exited non-zero"
"$TOOLS_BIN/inferno-collapse-perf" < "$TMPDIR_PC/script.txt" > "$OUTDIR/folded.txt" \
    || note "WARN: inferno-collapse-perf exited non-zero"
if [ ! -s "$OUTDIR/folded.txt" ]; then
    note "WARN: folded.txt is empty"
fi

# Stack-level symbol quality.  The self/leaf symbol of an interpreter sample is
# always inside libpython, so `perf report`'s own "[unknown]" share stays ~0%
# even when every Python caller is lost.  These metrics expose that loss:
# how many samples keep at least one interpreter (py::) frame and how deep the
# average resolved stack is.
#
# Counted per *sample* (perf script separates samples with a blank line), not
# per folded line: folded.txt carries the sample period as its weight, so it is
# cycle weighted, while these numbers are plain sample counts.
awk 'BEGIN { RS = ""; FS = "\n" }
    {
        n++; frames = NF - 1; if (frames < 0) frames = 0
        total_frames += frames
        if (index($0, "py::") > 0) py_samples++
        if (index($0, "[unknown]") > 0) unk_samples++
        if (NF >= 1 && index($NF, "py::") > 0) py_leaf++
        for (i = 2; i <= NF; i++)
            if (index($i, "[unknown]") > 0) unk_frames++
    }
    END {
        printf "#FOLDED_STACKS=%d\n",   n+0
        printf "#FOLDED_SAMPLES=%d\n",  n+0
        printf "#FRAME_SAMPLES=%d\n",   total_frames+0
        printf "#UNK_FRAMES=%d\n",      unk_frames+0
        printf "#PY_SAMPLES=%d\n",      py_samples+0
        printf "#PY_LEAF=%d\n",         py_leaf+0
        printf "#UNKFRAME_SAMPLES=%d\n", unk_samples+0
    }' "$TMPDIR_PC/script.txt" > "$TMPDIR_PC/fold_agg.txt" 2>/dev/null || true

PY_FRAMES="$(grep -c 'py::' "$OUTDIR/folded.txt" 2>/dev/null)"
PY_FRAMES="${PY_FRAMES:-0}"

fold_get() { awk -F= -v k="$1" '$1==k {print $2}' "$TMPDIR_PC/fold_agg.txt" 2>/dev/null | head -n 1; }
FOLD_SAMPLES="$(fold_get '#FOLDED_SAMPLES')"; FOLD_SAMPLES="${FOLD_SAMPLES:-0}"
FRAME_SAMPLES="$(fold_get '#FRAME_SAMPLES')"; FRAME_SAMPLES="${FRAME_SAMPLES:-0}"
PY_SAMPLES="$(fold_get '#PY_SAMPLES')";       PY_SAMPLES="${PY_SAMPLES:-0}"
UNKFRAME_SAMPLES="$(fold_get '#UNKFRAME_SAMPLES')"; UNKFRAME_SAMPLES="${UNKFRAME_SAMPLES:-0}"
FOLD_PY_LEAF="$(fold_get '#PY_LEAF')";        FOLD_PY_LEAF="${FOLD_PY_LEAF:-0}"
UNK_FRAMES="$(fold_get '#UNK_FRAMES')";       UNK_FRAMES="${UNK_FRAMES:-0}"

if [ "$FOLD_SAMPLES" -gt 0 ]; then
    PY_COVERAGE_PCT="$(awk -v a="$PY_SAMPLES" -v t="$FOLD_SAMPLES" 'BEGIN{printf "%.2f",100.0*a/t}')"
    PY_LEAF_PCT="$(awk -v a="$FOLD_PY_LEAF" -v t="$FOLD_SAMPLES" 'BEGIN{printf "%.2f",100.0*a/t}')"
    UNKFRAME_PCT="$(awk -v a="$UNKFRAME_SAMPLES" -v t="$FOLD_SAMPLES" 'BEGIN{printf "%.2f",100.0*a/t}')"
    MEAN_FRAMES="$(awk -v f="$FRAME_SAMPLES" -v t="$FOLD_SAMPLES" 'BEGIN{printf "%.2f",f/t}')"
else
    PY_COVERAGE_PCT="0.00"; PY_LEAF_PCT="0.00"; UNKFRAME_PCT="0.00"; MEAN_FRAMES="0.00"
fi
if [ "${FRAME_SAMPLES:-0}" -gt 0 ]; then
    UNKFRAME_LINE_PCT="$(awk -v a="$UNK_FRAMES" -v t="$FRAME_SAMPLES" 'BEGIN{printf "%.2f",100.0*a/t}')"
else
    UNKFRAME_LINE_PCT="0.00"
fi

# ------------------------------------------------------------- flame graph ---
STAGE="flamegraph"
"$TOOLS_BIN/inferno-flamegraph" \
    --title "$TITLE" \
    --subtitle "tid=$TID  freq=${FREQ}Hz  call-graph=$CG_EFF  window=${SECS}s  samples=${SAMPLE_COUNT}" \
    --notes "perf_capture.sh / $NAME / $(date -u +%Y-%m-%dT%H:%M:%SZ) / unknown=${UNKNOWN_PCT}%" \
    --countname samples \
    "$OUTDIR/folded.txt" > "$OUTDIR/flame.svg" \
    || note "WARN: inferno-flamegraph exited non-zero"

# ---------------------------------------------------------------- annotate ---
STAGE="annotate"
ANNOT_OK="false"
ANNOT_SYMS=""
ANNOT_NODEBUG=""
if [ "$DO_ANNOTATE" = "1" ] && [ "${ANNOT_TOP:-0}" -gt 0 ]; then
    top_syms="$(python3 - "$OUTDIR/hotspots.csv" "$ANNOT_TOP" <<'PY'
import csv, re, sys
want = int(sys.argv[2])
out = []
with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
    for row in csv.DictReader(fh):
        sym = row["symbol"].strip()
        dso = row["dso"].strip()
        if not sym or sym == "[unknown]" or dso == "[unknown]":
            continue
        # [JIT] (perf map) and [kernel.kallsyms] cannot be fed to objdump, and
        # py:: names only exist there, so they are never annotate candidates.
        if dso.startswith("["):
            continue
        if sym.startswith("py::") or sym.startswith("0x"):
            continue
        if re.fullmatch(r"[0-9a-fA-F]{6,}", sym):
            continue
        out.append(sym)
        if len(out) >= want:
            break
print("\n".join(out))
PY
)"
    while IFS= read -r raw; do
        [ -n "$raw" ] || continue
        # "/lib.so (deleted)" -> clean; "[.] sym+0x12" -> sym
        clean="$(printf '%s' "$raw" | sed -e 's/^\[[^]]*\][[:space:]]*//' \
                                           -e 's/[[:space:]]*(deleted)$//' \
                                           -e 's/+0x[0-9a-fA-F]*$//')"
        [ -n "$clean" ] || continue
        safe="$(printf '%s' "$clean" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-80)"
        out="$OUTDIR/annotate-$safe.txt"
        if "${SUDO_ARR[@]}" perf annotate -i "$OUTDIR/perf.data" --stdio -l \
                --symbol "$clean" > "$out" 2>> "$ERR_LOG"; then
            if [ -s "$out" ]; then
                ANNOT_SYMS="${ANNOT_SYMS}${ANNOT_SYMS:+
}${clean}"
                # "has source lines" means perf printed real file:line
                # attributions -- either in its own Sorted summary block or as
                # a `// file.c:123` comment.  A bare "Sorted summary for file"
                # header with no rows (libopenblas) or an operand comment such
                # as `// #0` does NOT count.
                if ! grep -qE '^[[:space:]]*[0-9]+\.[0-9]+[[:space:]]+[^[:space:]:]+:[0-9]+[[:space:]]*$' "$out" \
                   && ! grep -qE '//[[:space:]]*[A-Za-z0-9_./+-]+\.(c|h|cc|cpp|cxx|hpp|pyx|py|S|s|go|rs):[0-9]+' "$out"; then
                    ANNOT_NODEBUG="${ANNOT_NODEBUG}${ANNOT_NODEBUG:+
}${clean}"
                    note "PERF-NOTE-SRC: $clean -- source lines found: none (no debug info for this dso/symbol)"
                else
                    note "PERF-NOTE-SRC: $clean -- source lines found: yes"
                fi
            else
                rm -f "$out"
                note "WARN: perf annotate produced nothing for '$clean'"
            fi
        else
            rm -f "$out"
            note "WARN: perf annotate failed for '$clean'"
        fi
    done <<EOF
$top_syms
EOF
    if [ -n "$ANNOT_SYMS" ]; then
        ANNOT_OK="true"
    fi

    python3 - "$OUTDIR" <<'PY'
import csv, glob, os, re, sys
outdir = sys.argv[1]
asm_re = re.compile(r'^\s*([0-9]+\.[0-9]+)\s*:\s*([0-9a-fA-F]+):\s+(.*?)\s*$')
src_re = re.compile(r'//\s*([^\s]+?):(\d+)\s*$')
sum_re = re.compile(r'^\s*([0-9]+\.[0-9]+)\s+([^\s:]+):(\d+)\s*$')

srcline_rows = []
instr_rows = []
for path in sorted(glob.glob(os.path.join(outdir, "annotate-*.txt"))):
    sym = os.path.basename(path)[len("annotate-"):-len(".txt")]
    asm_lines = {}
    sum_lines = {}
    order = []
    in_summary = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Sorted summary for file" in line:
                in_summary = True
                continue
            if asm_re.match(line):
                in_summary = False
                m = asm_re.match(line)
                pct = float(m.group(1)); addr = m.group(2); rest = m.group(3)
                mm = src_re.search(rest)
                text = src_re.sub("", rest).strip()
                if mm:
                    key = "%s:%s" % (mm.group(1), mm.group(2))
                    asm_lines[key] = asm_lines.get(key, 0.0) + pct
                if pct > 0:
                    instr_rows.append((pct, sym, addr, text))
            elif in_summary:
                m = sum_re.match(line)
                if m:
                    key = "%s:%s" % (m.group(2), m.group(3))
                    if key not in sum_lines:
                        order.append(key)
                    sum_lines[key] = float(m.group(1))
    # perf's own aggregate (Sorted summary) is the authoritative `pct file:line`
    # table; the per-instruction `//` comments are only a fallback for builds
    # whose summary block is missing.
    per_line = sum_lines or asm_lines
    for key in order:
        if key in per_line:
            srcline_rows.append((per_line[key], key, sym))
    for key, pct in per_line.items():
        if key in order:
            continue
        srcline_rows.append((pct, key, sym))

srcline_rows.sort(reverse=True)
with open(os.path.join(outdir, "hotspots_srcline.csv"), "w", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["rank", "file_line", "pct", "symbol"])
    for i, (pct, key, sym) in enumerate(srcline_rows[:20], 1):
        w.writerow([i, key, "%.2f" % pct, sym])

instr_rows.sort(reverse=True)
with open(os.path.join(outdir, "hotspots_instr.csv"), "w", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["rank", "symbol", "address", "instruction", "pct"])
    for i, (pct, sym, addr, text) in enumerate(instr_rows[:20], 1):
        w.writerow([i, sym, addr, text, "%.2f" % pct])
PY
else
    note "annotation disabled (--no-annotate / --annotate-top 0)"
    printf 'rank,file_line,pct,symbol\n' > "$OUTDIR/hotspots_srcline.csv"
    printf 'rank,symbol,address,instruction,pct\n' > "$OUTDIR/hotspots_instr.csv"
fi

[ -s "$OUTDIR/hotspots_srcline.csv" ] || printf 'rank,file_line,pct,symbol\n' > "$OUTDIR/hotspots_srcline.csv"
[ -s "$OUTDIR/hotspots_instr.csv" ]   || printf 'rank,symbol,address,instruction,pct\n' > "$OUTDIR/hotspots_instr.csv"

STAGE="done"
FAIL_MSG=""
write_manifest

rm -rf "$TMPDIR_PC"

cat <<EOF
[$SELF] ok: tid=$TID samples=$SAMPLE_COUNT unknown=${UNKNOWN_PCT}% py_frames=$PY_FRAMES
[$SELF] outdir=$OUTDIR
EOF
exit 0
