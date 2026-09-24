#!/usr/bin/env bash
# run_all_benches.sh -- the full measurement sequence for the build-config
# experiment, in the order that matters.
#
#   ARMS="A B C" bash run_all_benches.sh
#
# Sequence (each stage is independent; failures do not abort the rest):
#   1. pi_sim_bench  : prepare_input-like mixed workload, 7 interleaved rounds
#   2. dispatch bench: dispatch-dominated kernels (upper bound of the effect)
#   3. perf stat     : IPC / branch-misses / icache / iTLB for both benches
#
# Everything runs on CPUs 200-203 with OMP_NUM_THREADS=1 and arm-interleaved
# rounds, so host-noise drift hits all arms alike.
set -uo pipefail

exp=${EXP:-/home/REMOTE_USER/projects/vllm/prepare-input-phase/exp-cpython}
arms=${ARMS:-"A B C"}
cpus=${CPUS:-200-203}
rounds=${ROUNDS:-7}
steps=${STEPS:-4000}
iters=${ITERS:-200000}

export EXP=$exp CPUS=$cpus ARMS=$arms

section() { echo; echo "########## $* ##########"; }

section "1/3 pi_sim_bench (mixed prepare_input-like)"
ROUNDS=$rounds STEPS=$steps OUTDIR=$exp/data/bench-pi \
	bash "$exp/scripts/run_bench_arms.sh"

section "2/3 dispatch_micro_bench (dispatch-dominated upper bound)"
BENCH=dispatch ROUNDS=$rounds ITERS=$iters OUTDIR=$exp/data/bench-dispatch \
	bash "$exp/scripts/run_bench_arms.sh"

section "3/3 perf stat (IPC, branch-misses, icache, iTLB)"
STEPS=$steps ROUNDS=3 OUTDIR=$exp/data/perf \
	bash "$exp/scripts/perf_stat_arms.sh"

section "done"
date -Is
