#!/usr/bin/env bash
# 无卡容器执行器：在 a3-22 上以「不挂任何 NPU 设备」的方式跑 harness。
#
# 用法:
#   pi-docker.sh <command...>            # 在 /work 下执行命令
#   pi-docker.sh --cpus 200-215 <cmd...> # 覆盖绑核 (默认 200-215)
#   pi-docker.sh --image <img> <cmd...>  # 覆盖镜像
#
# 设计要点:
#   * 绝不传 --device /dev/davinci*（真机测量由别的 agent 在 chip3 上做）
#   * --network none：避免任何外部依赖与网络抖动，loopback 仍可用
#   * TORCH_DEVICE_BACKEND_AUTOLOAD=0：阻止 torch import 时自动加载 torch_npu
#   * 绑核默认 200-215 (NUMA node2)，绝不使用 120-159（真机实验切片）
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pi_env.sh"
pi_env_selfcheck || exit 1

IMAGE="${PI_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}"
CPUS="$PI_CPUS"
HOST_ROOT="${PI_HOST_ROOT:-$HOME/projects/vllm/prepare-input-phase}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --host-root) HOST_ROOT="$2"; shift 2;;
    *) break;;
  esac
done

exec docker run --rm \
  --network none \
  --user "$(id -u):$(id -g)" \
  --cpuset-cpus "$CPUS" \
  "${PI_DOCKER_ENVS[@]}" \
  -v "${HOST_ROOT}:/work" \
  -v "${PI_MODELS_DIR}:/models:ro" \
  -w /work \
  "$IMAGE" \
  bash -c "$*"
