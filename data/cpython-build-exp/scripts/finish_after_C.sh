#!/usr/bin/env bash
# finish_after_C.sh -- wait for the arm-C (PGO+LTO) build to finish, then run
# every measurement that needs arm C, in the order that matters.
#
# Safe to run in the background under nohup: everything is logged.
set -uo pipefail

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
cd "$exp" || exit 1

echo "# finish_after_C start $(date -Is)"
for i in $(seq 1 240); do          # up to 240 * 15 s = 60 min
	if grep -q "^make_rc=" logs/armC.meta.txt 2>/dev/null; then
		rc=$(sed -n 's/^make_rc=//p' logs/armC.meta.txt)
		echo "# armC build finished rc=$rc at $(date -Is)"
		break
	fi
	sleep 15
done

if ! grep -q "^make_rc=0" logs/armC.meta.txt 2>/dev/null; then
	echo "# armC did NOT finish successfully -- skipping C benches"
	exit 1
fi

echo "# --- C dispatch evidence ---"
objdump -t build-armC/Python-3.12.13/libpython3.12.so.1.0 | grep -c opcode_targets
objdump -d --disassemble=_PyEval_EvalFrameDefault \
	build-armC/Python-3.12.13/libpython3.12.so.1.0 | grep -cE "\bbr\s+x[0-9]"

echo "# --- full three-arm benches ---"
ARMS="A B C" bash scripts/run_all_benches.sh

echo "# --- stock-image control for the same benches ---"
BENCH=dispatch ROUNDS=7 bash scripts/run_bench_stock_image.sh
BENCH=pi ROUNDS=5 bash scripts/run_bench_stock_image.sh

echo "# --- real-workload harness: stock vs C, interleaved, n=3 ---"
ARMS="stock C" REPS=3 bash scripts/run_harness_repeats.sh

echo "# finish_after_C done $(date -Is)"
