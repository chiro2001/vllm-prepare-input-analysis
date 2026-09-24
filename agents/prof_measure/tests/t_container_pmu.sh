#!/usr/bin/env bash
# 探测容器内 perf_event_open 权限：不同 capability / seccomp 组合下的 kperfx 结果
set -uo pipefail

IMG=${IMG:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}
CPUS=${CPUS:-200-215}
TOOLS=$HOME/tools

probe() {
  local label="$1"; shift
  echo "### ${label}"
  docker run --rm --network none --cpuset-cpus "$CPUS" "$@" \
    -v "${TOOLS}:/tools:ro" "$IMG" \
    bash -c 'cd /tools/libkperfx && ./kperfx info 2>&1 | sed -n "5,9p"; echo "-- probe --"; ./kperfx probe 2>&1 | head -5' \
    || echo "   (docker run failed)"
}

probe "A default-caps root"
probe "B SYS_ADMIN+seccomp=unconfined root" --cap-add=SYS_ADMIN --security-opt seccomp=unconfined
probe "C privileged root" --privileged
probe "D SYS_ADMIN+seccomp=unconfined user $(id -u)" --cap-add=SYS_ADMIN --security-opt seccomp=unconfined --user "$(id -u):$(id -g)"
