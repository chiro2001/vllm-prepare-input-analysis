#!/usr/bin/env bash
# run_bench_stock_image.sh -- run the SAME benchmarks under the *unmodified
# image python* (no libpython swap at all), so the image's interpreter can be
# compared behaviourally against arm A (switch) and arm B (computed gotos).
#
# This is the decisive test for "which dispatch does the production image
# actually execute?" -- the installed pyconfig.h and the shipped
# libpython3.12.so.1.0 disagree, so a static answer is not enough.
#
set -uo pipefail

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
img=${PI_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}
cpus=${CPUS:-200-203}
rounds=${ROUNDS:-7}
steps=${STEPS:-4000}
iters=${ITERS:-200000}
bench=${BENCH:-pi}
outdir="$exp/data/bench-$bench-stock"
mkdir -p "$outdir"

case "$bench" in
pi) script=pi_sim_bench.py; args="--steps $steps --rounds 1 --reqs 64" ;;
dispatch) script=dispatch_micro_bench.py; args="--iters $iters --rounds 1" ;;
*) echo "unknown BENCH=$bench" >&2; exit 2 ;;
esac

echo "# run_bench_stock_image $(date -Is) bench=$bench rounds=$rounds cpus=$cpus"
for r in $(seq 1 "$rounds"); do
	json="/work/exp-cpython/data/bench-$bench-stock/${bench}-stockimg-r$r.json"
	sudo -n docker run --rm \
		--network none \
		--user "$(id -u):$(id -g)" \
		--security-opt seccomp=unconfined \
		--cpuset-cpus "$cpus" \
		-e HOME=/tmp -e USER=REMOTE_USER -e LOGNAME=REMOTE_USER \
		-e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
		-e PYTHONHASHSEED=0 \
		-v "$exp/..:/work" \
		-w /work/exp-cpython \
		--entrypoint /usr/local/python3.12.13/bin/python3.12 \
		"$img" \
		"/work/exp-cpython/scripts/$script" $args \
		--label stockimg --json "$json" >/dev/null 2>&1
	echo "round$r rc=$? -> ${json}"
done
echo "# done $(date -Is)"
