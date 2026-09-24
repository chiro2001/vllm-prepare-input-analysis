#!/usr/bin/env bash
# **唯一的**「无卡容器环境变量」来源。任何自己起 docker 的脚本都必须 source 它，
# 否则会踩下面这个坑，而且症状会误导人（进程 2~3 秒后 exit=1，看起来像"负载跑太快"）。
#
# 坑：镜像里 uid=1003 没有 passwd 条目 → `getpass.getuser()` 抛
#     `KeyError: 'getpwuid(): uid not found: 1003'`
#     → torch inductor `default_cache_dir()` 崩 → `import torch` 失败 → exit 1
#
# 用法（数组形式，保留参数边界）：
#     source .../pi_env.sh
#     docker run ... "${PI_DOCKER_ENVS[@]}" ... <image> bash -c "$CMD"
#
# 也可单独取 CPUS 默认值：`${PI_CPUS:-200-215}`

# a3-22 上真机实验切片是 120-159，**任何无卡 harness 都不许用**。
PI_CPUS="${PI_CPUS:-200-215}"
# 容器内没有 passwd 条目，必须显式给 USER/LOGNAME
PI_USER="${PI_USER:-REMOTE_USER}"
PI_MODELS_DIR="${PI_MODELS_DIR:-$HOME/models}"

PI_DOCKER_ENVS=(
  -e "HOME=/tmp"
  -e "USER=${PI_USER}"
  -e "LOGNAME=${PI_USER}"
  -e "TORCHINDUCTOR_CACHE_DIR=/tmp/ti_cache"
  -e "TRITON_CACHE_DIR=/tmp/triton_cache"
  -e "TORCH_DEVICE_BACKEND_AUTOLOAD=0"
  -e "PI_HARNESS_ROOT=/work/harness"
  -e "PYTHONPATH=/work/harness"
  -e "PYTHONDONTWRITEBYTECODE=1"
)

# 自检：确认关键项都在
pi_env_selfcheck() {
  local missing=0
  for key in USER LOGNAME TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR TORCH_DEVICE_BACKEND_AUTOLOAD; do
    if ! printf '%s\n' "${PI_DOCKER_ENVS[@]}" | grep -q "^${key}="; then
      echo "pi_env.sh: FATAL missing ${key}" >&2
      missing=1
    fi
  done
  return $missing
}
