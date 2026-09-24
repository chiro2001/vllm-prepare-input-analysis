#!/usr/bin/env bash
# prof_* 脚本共用的设置与护栏（被 source，不单独执行）。
#
# 口径版本：prof-v1（2026-09-24）
# 规则：
#   * 负载跑在无卡容器里（不挂 NPU、--network none、绑 200-239/360-399、线程 <=16）；
#   * PMU / perf 一律在宿主侧 root 采（sudo -n 免密）；
#   * 绝对不用 120-159（真机实验切片）；
#   * 每次采集都写 manifest（含镜像 digest / 核 / 频率 / SMT / 噪声 / 脚本哈希）。

set -euo pipefail

PROF_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROF_REPO_ROOT="$(cd "${PROF_SCRIPTS_DIR}/../.." && pwd)"
export PROF_REPO_ROOT
export PROF_HARNESS_DIR="${PROF_REPO_ROOT}/harness"
export PROF_DATA_DIR="${PROF_REPO_ROOT}/data/harness"

PI_IMAGE="${PI_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}"
PI_CPUS="${PI_CPUS:-200-215}"
PI_LIBKPERFX="${PI_LIBKPERFX:-$HOME/tools/libkperfx}"
PI_FLAMEGRAPH="${PI_FLAMEGRAPH:-$HOME/tools/cargo/bin/flamegraph}"
PI_PYSPY="${PI_PYSPY:-$HOME/.local/bin/py-spy}"
export PI_IMAGE PI_CPUS PI_LIBKPERFX PI_FLAMEGRAPH PI_PYSPY

prof_log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
prof_die() { printf '[prof_*] 错误: %s\n' "$*" >&2; exit 2; }

# python3 -m pi_harness.profiling.<...>（宿主侧；容器侧用 PYTHONPATH=/work/harness）
prof_py() {
  PYTHONPATH="${PROF_HARNESS_DIR}${PYTHONPATH:+:$PYTHONPATH}" python3 -m "$@"
}

# 禁止踩真机切片 120-159
prof_guard_cpus() {
  local spec="$1" bad=""
  local -a parts=()
  IFS=',' read -r -a parts <<<"$spec"
  local part lo hi c
  for part in "${parts[@]}"; do
    # 非数字（例如 auto）先跳过：由调用方在定型后再校验
    [[ "$part" =~ ^[0-9]+(-[0-9]+)?$ ]] || continue
    if [[ "$part" == *-* ]]; then
      lo="${part%-*}"; hi="${part#*-}"
      for ((c = lo; c <= hi; c++)); do
        if ((c >= 120 && c <= 159)); then bad="$c"; fi
      done
    else
      if ((part >= 120 && part <= 159)); then bad="$part"; fi
    fi
  done
  [[ -z "$bad" ]] || prof_die "cpuset ${spec} 落在真机切片 120-159（含 ${bad}），禁止使用"
}

prof_require_tools() {
  [[ -x /usr/bin/perf ]] || prof_die "找不到 /usr/bin/perf"
  sudo -n true 2>/dev/null || prof_die "sudo -n 不可用：PMU 采集需要 root"
  if [[ ! -f "${PI_LIBKPERFX}/python/kperfx.py" ]]; then
    prof_die "libkperfx 未安装到 ${PI_LIBKPERFX}（见 agents/prof_measure/REPORT.md 的安装命令）"
  fi
}

prof_tool_versions() {
  {
    printf 'perf: %s\n' "$(/usr/bin/perf --version 2>/dev/null | head -1)"
    printf 'libkperfx: %s\n' "$("${PI_LIBKPERFX}/kperfx" --version 2>/dev/null || echo missing)"
    printf 'flamegraph-rs: %s\n' "$([[ -x "${PI_FLAMEGRAPH}" ]] && echo "${PI_FLAMEGRAPH}" || echo missing)"
    printf 'py-spy: %s\n' "$([[ -x "${PI_PYSPY}" ]] && "${PI_PYSPY}" --version 2>/dev/null | head -1 || echo missing)"
    printf 'kernel: %s\n' "$(uname -r)"
    printf 'paranoid: %s\n' "$(cat /proc/sys/kernel/perf_event_paranoid)"
  }
}

# 可选：仓库里若有 scripts/hostnoise_gate.sh 就按本脚本的 cpuset 跑一次
# （env_toolchain 维护；它还会看 chip3/NPU 残留进程，对本任务只是参考信息）
prof_optional_hostnoise_gate() {
  local cpus="${1:-$PI_CPUS}"
  local gate="${PROF_REPO_ROOT}/scripts/hostnoise_gate.sh"
  if [[ -x "$gate" ]]; then
    prof_log "调用 ${gate} --cpus ${cpus}"
    "$gate" --cpus "$cpus" --sample-s "${PROF_NOISE_SECONDS:-1.5}" \
      || prof_log "hostnoise_gate.sh 返回非 0（继续；本脚本自己的噪声门结果在 noise_*.json）"
  fi
}
