#!/usr/bin/env bash
# 本次采集的复现命令（由 pi_harness.profiling.collect 生成）
set -euo pipefail

# tag=prof_calib_arith cpus=223 image=quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler
# caliber=prof-v1 ts=2026-09-24T02:07:59+0800
REPO=${REPO:-/home/REMOTE_USER/projects/vllm/prepare-input-phase}

# --- 采样前噪声门 ---
cd "$REPO/harness" && python3 -m pi_harness.profiling.hostnoise --cpus 223 --seconds 5.0

# --- 负载（容器内，等价 prof_harness.sh 的入口）---
CIL='python3 -m pi_harness.profiling.microbench --phase arith --threads 1 --warmup 1 --json /tmp/microbench_arith.json'
docker run --rm --network none --cpuset-cpus 223 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PYTHONPATH=/work/harness -v "$REPO:/work" -w /work quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler \
  bash -c "$CIL"

# --- manifest ---
# /home/REMOTE_USER/projects/vllm/prepare-input-phase/data/harness/prof_runs/prof_calib_arith_20260924-020459/manifest.json
