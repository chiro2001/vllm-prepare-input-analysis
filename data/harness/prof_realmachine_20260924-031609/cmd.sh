#!/usr/bin/env bash
# 本次采集的复现命令（由 pi_harness.profiling.collect 生成）
set -euo pipefail

# tag=prof_realmachine cpus=201-202,205,207-208,210,212,214 image=quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler
# caliber=prof-v1 ts=2026-09-24T03:33:00+0800
REPO=${REPO:-/home/REMOTE_USER/projects/vllm/prepare-input-phase}

# --- 采样前噪声门 ---
cd "$REPO/harness" && python3 -m pi_harness.profiling.hostnoise --cpus 201-202,205,207-208,210,212,214 --seconds 5.0

# --- 负载（容器内，等价 prof_harness.sh 的入口）---
# 环境变量唯一真源（USER/LOGNAME/TORCHINDUCTOR_CACHE_DIR/... 缺一不可）
source "$REPO/harness/scripts/pi_env.sh"
pi_env_selfcheck || exit 1
CIL='python3 -m pi_harness.runner._replay_main --preset realmachine --batch 1 --isl 128 --osl 64 --steady-seconds 240'
docker run --rm --network none --cpuset-cpus 201-202,205,207-208,210,212,214 \
  --user "$(id -u):$(id -g)" "${PI_DOCKER_ENVS[@]}" \
  -v "$REPO:/work" -v "${PI_MODELS_DIR}:/models:ro" -w /work \
  quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler \
  bash -c "$CIL"

# --- manifest ---
# /home/REMOTE_USER/projects/vllm/prepare-input-phase/data/harness/prof_runs/prof_realmachine_20260924-031609/manifest.json
