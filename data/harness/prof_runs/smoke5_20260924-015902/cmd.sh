#!/usr/bin/env bash
# 本次采集的复现命令（由 pi_harness.profiling.collect 生成）
set -euo pipefail

# tag=smoke5 cpus=204,206,210,212 image=quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler
# caliber=prof-v1 ts=2026-09-24T01:59:23+0800
REPO=${REPO:-/home/REMOTE_USER/projects/vllm/prepare-input-phase}

# --- 采样前噪声门 ---
cd "$REPO/harness" && python3 -m pi_harness.profiling.hostnoise --cpus 204,206,210,212 --seconds 5.0

# --- 负载（容器内，等价 prof_harness.sh 的入口）---
CIL='python3 -m pi_harness.profiling.microbench --phase arith --iters 6000000 --warmup 0 --json /tmp/mb_smoke.json'
docker run --rm --network none --cpuset-cpus 204,206,210,212 \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PYTHONPATH=/work/harness -v "$REPO:/work" -w /work quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler \
  bash -c "$CIL"

# --- manifest ---
# /home/REMOTE_USER/projects/vllm/prepare-input-phase/data/harness/prof_runs/smoke5_20260924-015902/manifest.json
