#!/usr/bin/env bash
# run_bench_mixed_stock.sh -- interleave the UNMODIFIED image python with the
# three arms in ONE session, round by round (A,B,C,stock / A,B,C,stock / ...).
#
# Why this exists: the standalone stock-image runs are their own session, and
# this host drifts by several percent between sessions (see 06-build-config-
# experiment.md §10.6). Only an intra-session interleaved comparison can
# support the claim "the production interpreter == arm B, not arm A".
#
# Usage: ROUNDS=7 STEPS=4000 bash run_bench_mixed_stock.sh
set -uo pipefail

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
img=${PI_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}
cpus=${CPUS:-200-203}
rounds=${ROUNDS:-7}
steps=${STEPS:-4000}
bench=${BENCH:-pi}
outdir="$exp/data/bench-$bench-mixed"
mkdir -p "$outdir"

case "$bench" in
pi) script=pi_sim_bench.py; args="--steps $steps --rounds 1 --reqs 64" ;;
dispatch) script=dispatch_micro_bench.py; args="--iters ${ITERS:-200000} --rounds 1" ;;
*) echo "unknown BENCH=$bench" >&2; exit 2 ;;
esac

echo "# run_bench_mixed_stock $(date -Is) bench=$bench rounds=$rounds cpus=$cpus"
for r in $(seq 1 "$rounds"); do
	for arm in A B C; do
		src="$exp/build-arm$arm/Python-3.12.13"
		[ -x "$src/python" ] || { echo "arm$arm missing"; continue; }
		taskset -c "$cpus" env \
			OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
			PYTHONHASHSEED=0 LD_LIBRARY_PATH="$src" PYTHONPATH="$exp/pydeps" \
			"$src/python" "$exp/scripts/$script" $args --label "arm$arm" \
			--json "$outdir/${bench}-arm$arm-r$r.json" >/dev/null 2>&1
		echo "round$r arm$arm rc=$?"
	done
	sudo -n docker run --rm \
		--network none --user "$(id -u):$(id -g)" \
		--security-opt seccomp=unconfined --cpuset-cpus "$cpus" \
		-e HOME=/tmp -e USER=REMOTE_USER -e LOGNAME=REMOTE_USER \
		-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
		-e PYTHONHASHSEED=0 \
		-v "$exp/..:/work" -w /work/exp-cpython \
		--entrypoint /usr/local/python3.12.13/bin/python3.12 "$img" \
		"/work/exp-cpython/scripts/$script" $args --label stockimg \
		--json "/work/exp-cpython/data/bench-$bench-mixed/${bench}-stockimg-r$r.json" \
		>/dev/null 2>&1
	echo "round$r stockimg rc=$?"
done
echo "# done $(date -Is)"
