#!/usr/bin/env bash
# 920B topdown preset 采集（libkperfx）：四层分量 + 细分，必带 time_running/time_enabled。
#
# 用法
#   bash harness/scripts/prof_topdown.sh [选项] -- <容器内命令...>
#   bash harness/scripts/prof_topdown.sh --split-dir DIR --merge-only       # 只重新汇总
#
# 选项
#   --level l1|full     l1 = 只跑 preset 第 1 组（retiring/bad-spec/frontend/backend）
#                       full = 9 组（每组一次同样的负载，可出 backend/mem/OOO 细分）
#   --mode split|mux    split = 重启负载 N 次，置信度 1.0（默认）
#                       mux   = 多组同开一次采完，跑得快但可能 multiplex
#   --cpus 200-215      绑核（禁止 120-159）
#   --out DIR           直接指定产物目录
#   --tag NAME          产物 tag（默认 prof_topdown）
#   --split-dir DIR     只汇总已有 split_*.json
#
# 产物：<out>/pmu/topdown.json（level1/level2/level3_mem/ooo_stall + 每组置信度）
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prof_lib.sh"

LEVEL="l1"
MODE="split"
CPUS="$PI_CPUS"
TAG="prof_topdown"
OUT=""
SPLIT_DIR=""
MERGE_ONLY=0
CMD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --level) LEVEL="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --split-dir) SPLIT_DIR="$2"; MERGE_ONLY=1; shift 2;;
    --merge-only) MERGE_ONLY=1; shift;;
    -h|--help) sed -n '2,22p' "$0"; exit 0;;
    --) shift; CMD=("$@"); break;;
    *) prof_die "未知参数 $1（容器内命令放在 -- 之后）";;
  esac
done

if [[ "$MERGE_ONLY" == "1" ]]; then
  [[ -n "$SPLIT_DIR" ]] || prof_die "--split-dir 必填"
  prof_py pi_harness.profiling.topdown --split-dir "$SPLIT_DIR" \
    --out "${OUT:-${SPLIT_DIR}/topdown.json}"
  exit 0
fi

[[ ${#CMD[@]} -gt 0 ]] || prof_die "需要 -- <容器内命令>"
prof_guard_cpus "$CPUS"
prof_require_tools
mkdir -p "$PROF_DATA_DIR"

args=( --tag "$TAG" --cpus "$CPUS" --image "$PI_IMAGE" --stages topdown
       --topdown-level "$LEVEL" --topdown-mode "$MODE"
       --noise-seconds "${PROF_NOISE_SECONDS:-3}" --timeout "${PROF_TIMEOUT:-1800}" )
[[ -n "$OUT" ]] && args+=( --out "$OUT" )
args+=( -- "${CMD[@]}" )

prof_log "topdown level=${LEVEL} mode=${MODE} cpus=${CPUS}"
prof_py pi_harness.profiling.collect "${args[@]}"
