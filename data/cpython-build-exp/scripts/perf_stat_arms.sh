#!/usr/bin/env bash
# perf_stat_arms.sh -- PMU comparison of the three arms (IPC / branch misses /
# frontend-relevant counters) on the same benchmark and the same CPUs.
#
# perf lives on the *host* (the image has no perf), so the container-built
# python binaries are executed on the host.  Verified beforehand that the
# image's glibc (2.38 / openEuler 24.03 SP3) and the host's (2.38 / SP4)
# are the same major version, i.e. the binaries run unmodified.
#
# Usage: STEPS=2000 ROUNDS=3 bash perf_stat_arms.sh
set -uo pipefail

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
cpus=${CPUS:-200-203}
steps=${STEPS:-2000}
rounds=${ROUNDS:-3}
reqs=${REQS:-64}
script=${SCRIPT:-$exp/scripts/pi_sim_bench.py}
mode=${MODE:-interleaved}
outdir=${OUTDIR:-$exp/data/perf}
arms=${ARMS:-"A B C"}
mkdir -p "$outdir"

# NOTE: do NOT name this `GROUPS` -- that is a standard login environment
# variable (indexed array of group ids) and `declare -A GROUPS` then fails
# with "cannot convert indexed to associative array".
declare -A PMU_GROUPS=(
	[g1]="cycles,instructions,branches,branch-misses"
	[g2]="L1-icache-loads,L1-icache-load-misses,iTLB-loads,iTLB-load-misses"
)

echo "# perf_stat_arms $(date -Is) host=$(hostname) cpus=$cpus steps=$steps rounds=$rounds arms=$arms"
for arm in $arms; do
	src="$exp/build-arm$arm/Python-3.12.13"
	[ -x "$src/python" ] || { echo "arm$arm missing"; continue; }
	for g in g1 g2; do
		ev=${PMU_GROUPS[$g]}
		out="$outdir/perf-arm$arm-$g.txt"
		sudo -n perf stat -e "$ev" -o "$out" -- \
			taskset -c "$cpus" env \
			OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
			PYTHONHASHSEED=0 LD_LIBRARY_PATH="$src" \
			PYTHONPATH="$exp/pydeps" \
			"$src/python" "$script" --steps "$steps" --rounds "$rounds" \
			--reqs "$reqs" --mode "$mode" --label "perf-arm$arm-$g" \
			>"$outdir/perf-arm$arm-$g.stdout" 2>&1
		echo "arm$arm $g rc=$? -> $out"
		grep -E "seconds time elapsed|insn per cycle|branch-miss|event counter" "$out" || true
	done
done
echo "# done $(date -Is)"
