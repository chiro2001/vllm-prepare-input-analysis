#!/usr/bin/env bash
# run_harness_repeats.sh -- N arm-interleaved repeats of the real
# prepare_input harness, to separate the build effect from run-to-run noise.
#
#   ARMS="stock A B" REPS=3 bash run_harness_repeats.sh
#
# Each repeat is a fresh container (= fresh interpreter state), 200 steps,
# --preset realmachine --batch 1 --isl 128, CPUs 200-215.
set -uo pipefail

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
arms=${ARMS:-"stock A B"}
reps=${REPS:-3}
steps=${STEPS:-200}

cd "$exp" || exit 1
for r in $(seq 1 "$reps"); do
	for arm in $arms; do
		tag="rep${r}_${arm}"
		PERF=0 TAG="$tag" STEPS="$steps" \
			bash scripts/run_harness_arm.sh "$arm" \
			> "data/harness-rep-${arm}-${r}.log" 2>&1
		echo "rep$r arm=$arm rc=$? tag=$tag"
	done
done

python3 "$exp/scripts/summarize_harness_reps.py" "$exp"
