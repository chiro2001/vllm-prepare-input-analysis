#!/usr/bin/env bash
# hostnoise_gate.sh - pre-flight gate for any chip3 / CPU-slice measurement.
#
# Answers three questions before a profiling or benchmark run starts:
#   1. is chip3 (/dev/davinci3 == NPU1/Chip1 == Phy-ID 3) really free?
#   2. is the reserved CPU slice (default 120-159) quiet, and if not, *who* is
#      on it?
#   3. are there stray msprof / DevKit / perf collectors left over from an
#      earlier run (they pollute each other and the PMU)?
#
# Usage:
#   hostnoise_gate.sh [--cpus 120-159] [--json OUT.json] [--quiet-mean PCT]
#                      [--sample-s SECONDS] [--chip3 npu1/chip1]
#
# Exit codes: 0 = gate open, 1 = gate closed (noise found), 2 = usage error.
#
# NOTE on npu-smi addressing (learned the hard way on a3-22):
#   `npu-smi info -t proc-mem -i 3` addresses *NPU id* 3 (Phy-ID 6/7, other
#   people's dsv41 jobs).  chip3 is NPU id 1, chip id 1.  Use `-i 1 -c 1`.
set -uo pipefail

CPUS="120-159"
JSON_OUT=""
SAMPLE_S=1.5
QUIET_MEAN=2.0
CHIP_NPU=1
CHIP_CHIP=1

while (($#)); do
	case "$1" in
	--cpus) CPUS=$2; shift 2 ;;
	--json) JSON_OUT=$2; shift 2 ;;
	--sample-s) SAMPLE_S=$2; shift 2 ;;
	--quiet-mean) QUIET_MEAN=$2; shift 2 ;;
	--chip3)
		# accept "npuX/chipY" only; the mapping itself is not guessed here
		CHIP_NPU=${2%%/*}; CHIP_NPU=${CHIP_NPU#npu}
		CHIP_CHIP=${2##*/}; CHIP_CHIP=${CHIP_CHIP#chip}
		shift 2 ;;
	*) echo "usage: $0 [--cpus 120-159] [--json OUT] [--sample-s S]" >&2; exit 2 ;;
	esac
done

tmp=$(mktemp -d /tmp/hostnoise.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

echo "# hostnoise_gate $(date -Is)  host=$(hostname)  cpus=$CPUS  chip3=npu$CHIP_NPU/chip$CHIP_CHIP"

# ---------------------------------------------------------------- 1. chip3
chip3_raw=$(npu-smi info -t proc-mem -i "$CHIP_NPU" -c "$CHIP_CHIP" 2>&1)
chip3_free=0
if grep -q "No process in device" <<<"$chip3_raw"; then
	chip3_free=1
	echo "[chip3]   FREE  (npu-smi info -t proc-mem -i $CHIP_NPU -c $CHIP_CHIP)"
else
	echo "[chip3]   BUSY  (npu-smi info -t proc-mem -i $CHIP_NPU -c $CHIP_CHIP)"
	sed 's/^/          /' <<<"$chip3_raw"
fi
# `npu-smi info -t common -i N` prints one block per chip on that NPU; the
# "Chip ID" line precedes its memory block, so track it and print the HBM rate
# of the chip we care about.
hbm_raw=$(npu-smi info -t common -i "$CHIP_NPU" 2>&1 | awk -v c="$CHIP_CHIP" '
	/^[[:space:]]*Chip ID/ {cid=$NF}
	/HBM Usage Rate/ {if (cid==c) {print $NF; exit}}')
echo "[chip3]   HBM usage rate: ${hbm_raw:-?}%"

# --------------------------------------------------- 2. stray collectors
residue=$(ps -eo pid,comm,args --no-headers 2>/dev/null |
	grep -Ei 'msprof|mindstudio|devkit|(^|/)perf (record|stat|top|trace|annotate)|ascend_toolkit' |
	grep -v grep)
if [[ -z $residue ]]; then
	echo "[residue] none (no msprof/DevKit/perf process)"
else
	echo "[residue] FOUND:"
	sed 's/^/          /' <<<"$residue"
fi

# ------------------------------------------------- 3. CPU slice occupancy
CPUS="$CPUS" SAMPLE_S="$SAMPLE_S" QUIET_MEAN="$QUIET_MEAN" TMPD="$tmp" python3 - <<'PY'
import json, os, re, time

spec = os.environ["CPUS"]
sample_s = float(os.environ["SAMPLE_S"])
quiet_mean = float(os.environ["QUIET_MEAN"])
tmpd = os.environ["TMPD"]

def expand(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

cpus = expand(spec)
cpuset = set(cpus)

def cpu_snapshot():
    d = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if not re.match(r"^cpu\d+ ", line):
                continue
            f = line.split()
            idx = int(f[0][3:])
            v = [int(x) for x in f[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            d[idx] = (sum(v), idle)
    return d

def proc_snapshot():
    d = {}
    hz = os.sysconf("SC_CLK_TCK")
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            stat = open(f"/proc/{pid}/stat").read()
            rp = stat.rindex(")")
            comm = stat[stat.index("(") + 1:rp]
            fields = stat[rp + 2:].split()
            utime, stime = int(fields[11]), int(fields[12])
            nthreads = int(fields[17])
            allowed = open(f"/proc/{pid}/status").read()
            m = re.search(r"^Cpus_allowed_list:\s*(\S+)$", allowed, re.M)
            m2 = re.search(r"^Uid:\s*(\d+)", allowed, re.M)
            d[pid] = {
                "comm": comm, "ticks": utime + stime, "threads": nthreads,
                "allowed": expand(m.group(1)) if m else [],
                "uid": int(m2.group(1)) if m2 else -1,
            }
        except (OSError, ValueError, IndexError):
            continue
    return d, hz

a_cpu = cpu_snapshot()
a_proc, _hz = proc_snapshot()
t0 = time.time()
time.sleep(sample_s)
b_cpu = cpu_snapshot()
b_proc, hz = proc_snapshot()
elapsed = time.time() - t0

per_cpu, hot_cpus = {}, []
for c in cpus:
    if c in a_cpu and c in b_cpu:
        dt = b_cpu[c][0] - a_cpu[c][0]
        di = b_cpu[c][1] - a_cpu[c][1]
        if dt > 0:
            busy = round(100.0 * (dt - di) / dt, 2)
            per_cpu[c] = busy
            if busy > 20.0:
                hot_cpus.append(c)
mean = round(sum(per_cpu.values()) / len(per_cpu), 2) if per_cpu else 0.0
mx = max(per_cpu.values()) if per_cpu else 0.0

offenders = []
for pid, b in b_proc.items():
    a = a_proc.get(pid)
    if not a:
        continue
    dt = b["ticks"] - a["ticks"]
    if dt <= 0:
        continue
    cpu_pct = round(100.0 * dt / hz / elapsed, 2)
    if cpu_pct < 1.0:
        continue
    overlap = cpuset.intersection(b["allowed"])
    offenders.append({
        "pid": int(pid), "comm": b["comm"], "cpu_pct": cpu_pct,
        "threads": b["threads"], "uid": b["uid"],
        "allowed_first_last": [b["allowed"][0], b["allowed"][-1]] if b["allowed"] else None,
        "n_cpus_allowed": len(b["allowed"]),
        "intersects_slice": len(overlap),
        "overlap_first_last": [min(overlap), max(overlap)] if overlap else None,
        "oversubscribes_slice": len(b["allowed"]) > 0 and set(b["allowed"]).issubset(cpuset),
    })
offenders.sort(key=lambda x: -x["cpu_pct"])
on_slice = [o for o in offenders if o["intersects_slice"]]

doc = {
    "cpus": spec, "sample_s": round(elapsed, 3),
    "per_cpu_busy_pct": per_cpu, "mean_busy_pct": mean, "max_busy_pct": mx,
    "hot_cpus": hot_cpus,
    "offenders_on_slice": on_slice[:25],
    "n_offenders_on_slice": len(on_slice),
    "quiet_mean_threshold": quiet_mean,
    "slice_quiet": mean <= quiet_mean and not hot_cpus,
    "quiet_cpus": [c for c in cpus if per_cpu.get(c, 0.0) < 5.0],
}
with open(os.path.join(tmpd, "cpu.json"), "w") as fh:
    json.dump(doc, fh, indent=2)

print(f"[cpu]     slice {spec}: mean {mean:.2f}%  max {mx:.2f}%  "
      f"hot(>20%)={hot_cpus if hot_cpus else 'none'}")
if on_slice:
    print(f"[cpu]     {len(on_slice)} unrelated process(es) with affinity on the slice "
          f"(top 8 below); threshold for 'quiet' is mean<={quiet_mean}% and no hot cpu")
    for o in on_slice[:8]:
        print(f"          pid={o['pid']:<9} cpu={o['cpu_pct']:>7.2f}%  threads={o['threads']:<4} "
              f"uid={o['uid']:<6} cpus_allowed={o['allowed_first_last']} "
              f"overlap={o['overlap_first_last']} fullslice={o['oversubscribes_slice']}  {o['comm']}")
print(f"[cpu]     verdict: {'QUIET' if doc['slice_quiet'] else 'NOISY'}")
q = doc["quiet_cpus"]
if q:
    # compress into ranges for a copy-pasteable taskset spec
    ranges, start, prev = [], q[0], q[0]
    for c in q[1:]:
        if c != prev + 1:
            ranges.append((start, prev))
            start = c
        prev = c
    ranges.append((start, prev))
    spec2 = ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)
    print(f"[cpu]     idle CPUs (<5% busy): {len(q)} of {len(cpus)}  "
          f"-> recommended taskset: {spec2[:160]}")
    print("[cpu]     note: affinity alone does NOT evict other tenants - foreign "
          "processes with a wide Cpus_allowed_list can still land on these CPUs.")
PY
cpu_rc=$([ "$(python3 -c "import json;print(json.load(open('$tmp/cpu.json'))['slice_quiet'])")" = "True" ] && echo 0 || echo 1)

# ------------------------------------------------------------- 4. verdict
gate=0
[[ $chip3_free -eq 1 ]] || gate=1
[[ -z $residue ]] || gate=1
[[ $cpu_rc -eq 0 ]] || gate=1

if [[ -n $JSON_OUT ]]; then
	mkdir -p "$(dirname "$JSON_OUT")"
	JSON_OUT="$JSON_OUT" CHIP3_FREE="$chip3_free" GATE="$gate" TMPD="$tmp" \
		CPUS="$CPUS" CHIP="$CHIP_NPU/$CHIP_CHIP" python3 - <<'PY'
import json, os
cpu = json.load(open(os.path.join(os.environ["TMPD"], "cpu.json")))
doc = {
    "chip3": {"addressed_as": f"npu{os.environ['CHIP'].split('/')[0]}/chip"
                              f"{os.environ['CHIP'].split('/')[1]}",
              "free": os.environ["CHIP3_FREE"] == "1"},
    "cpu": cpu,
    "gate_open": os.environ["GATE"] == "0",
}
with open(os.environ["JSON_OUT"], "w") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")
PY
	echo "[gate]    wrote $JSON_OUT"
fi

if [[ $gate -eq 0 ]]; then
	echo "[gate]    OPEN - safe to take the chip3 lock and measure"
else
	echo "[gate]    CLOSED - fix the items above (or wait) before measuring"
fi
exit "$gate"
