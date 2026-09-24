#!/usr/bin/env bash
# run_bench_arms.sh -- run the SAME benchmark under all three CPython arms.
#
# Runs on a3-22 (host).  Every arm is executed with
#   taskset -c $CPUS  OMP_NUM_THREADS=1  LD_LIBRARY_PATH=<build tree>
# and with byte-identical benchmark sources, fixed iteration counts.
#
# Arms are interleaved per round (A,B,C / A,B,C / ...) so that slow host-noise
# drift hits all arms alike instead of being attributed to one build.
#
# Usage:
#   BENCH=pi|dispatch ROUNDS=5 STEPS=2000 bash run_bench_arms.sh
#
set -uo pipefail

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
cpus=${CPUS:-200-203}
rounds=${ROUNDS:-5}
steps=${STEPS:-2000}
iters=${ITERS:-200000}
reqs=${REQS:-64}
bench=${BENCH:-pi}
outdir=${OUTDIR:-$exp/data/bench-$bench}
arms=${ARMS:-"A B C"}

case "$bench" in
pi) script="$exp/scripts/pi_sim_bench.py" ;;
dispatch) script="$exp/scripts/dispatch_micro_bench.py" ;;
*) echo "unknown BENCH=$bench" >&2; exit 2 ;;
esac

mkdir -p "$outdir"
echo "# run_bench_arms  $(date -Is)  host=$(hostname)  cpus=$cpus bench=$bench rounds=$rounds steps=$steps"

for r in $(seq 1 "$rounds"); do
	for arm in $arms; do
		src="$exp/build-arm$arm/Python-3.12.13"
		py="$src/python"
		if [ ! -x "$py" ]; then
			echo "arm$arm: $py missing -- skipped"
			continue
		fi
		tag="$bench-arm$arm-r$r"
		json="$outdir/$tag.json"
		log="$outdir/$tag.log"
		start=$(date +%s.%N)
		if [ "$bench" = pi ]; then
			args=(--steps "$steps" --rounds 1 --reqs "$reqs" --label "arm$arm"
			      --json "$json" --mode "${MODE:-interleaved}")
		else
			args=(--iters "$iters" --rounds 1 --label "arm$arm" --json "$json")
		fi
		taskset -c "$cpus" env \
			OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
			PYTHONHASHSEED=0 LD_LIBRARY_PATH="$src" \
			PYTHONPATH="$exp/pydeps" \
			"$py" "$script" "${args[@]}" \
			>"$log" 2>&1
		rc=$?
		end=$(date +%s.%N)
		echo "round$r arm$arm rc=$rc wall=$(echo "$end - $start" | bc)s json=$json log=$log"
	done
done
echo "# done $(date -Is)"
