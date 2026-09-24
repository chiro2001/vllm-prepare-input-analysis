#!/usr/bin/env bash
# on-CPU 火焰图：优先 flamegraph-rs，装不上退化到内置渲染器；另可出 py-spy（Python 帧）。
#
# 用法
#   # a) 已有 perf.data
#   bash harness/scripts/prof_flamegraph.sh --perfdata /path/perf.data -o out.svg \
#        [--collapsed out.folded] [--tool auto|flamegraph-rs|builtin]
#
#   # b) 现场采一个 PID（宿主 PID；容器进程用 docker inspect -f '{{.State.Pid}}' <cid>）
#   bash harness/scripts/prof_flamegraph.sh --pid 12345 --duration 10 --freq 999 \
#        --callgraph dwarf -o out.svg
#
#   # c) Python 帧（py-spy，带 C 帧用 --native）
#   bash harness/scripts/prof_flamegraph.sh --pyspy-pid 12345 --duration 10 -o py.svg
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prof_lib.sh"

PERFDATA=""
PID=""
PYSPY_PID=""
DURATION=10
FREQ=999
CALLGRAPH=dwarf
OUT=""
COLLAPSED=""
TOOL="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --perfdata) PERFDATA="$2"; shift 2;;
    --pid) PID="$2"; shift 2;;
    --pyspy-pid) PYSPY_PID="$2"; shift 2;;
    --duration) DURATION="$2"; shift 2;;
    --freq) FREQ="$2"; shift 2;;
    --callgraph) CALLGRAPH="$2"; shift 2;;
    -o|--out) OUT="$2"; shift 2;;
    --collapsed) COLLAPSED="$2"; shift 2;;
    --tool) TOOL="$2"; shift 2;;
    -h|--help) sed -n '2,17p' "$0"; exit 0;;
    *) prof_die "未知参数 $1";;
  esac
done

[[ -n "$OUT" ]] || prof_die "-o/--out 必填"
mkdir -p "$(dirname "$OUT")"

if [[ -n "$PYSPY_PID" ]]; then
  [[ -x "$PI_PYSPY" ]] || prof_die "找不到 py-spy（${PI_PYSPY}）"
  prof_log "py-spy 采样 pid=${PYSPY_PID} ${DURATION}s -> ${OUT}"
  sudo -n "$PI_PYSPY" record --pid "$PYSPY_PID" --duration "$DURATION" --rate 199 \
    --format flamegraph -o "$OUT" --native
  prof_log "完成: ${OUT}"
  exit 0
fi

if [[ -z "$PERFDATA" ]]; then
  [[ -n "$PID" ]] || prof_die "需要 --perfdata 或 --pid（或 --pyspy-pid）"
  PERFDATA="/tmp/prof_flamegraph_${PID}_$(date +%s).perf.data"
  prof_log "perf record -F ${FREQ} -g --call-graph ${CALLGRAPH} -p ${PID} ${DURATION}s"
  sudo -n timeout -s INT "$DURATION" perf record -o "$PERFDATA" -F "$FREQ" -g \
    --call-graph "$CALLGRAPH" -p "$PID"
fi

[[ -f "$PERFDATA" ]] || prof_die "找不到 ${PERFDATA}"
COLLAPSED="${COLLAPSED:-${OUT%.svg}.folded}"

prof_py pi_harness.profiling.flamegraph from-perfdata "$PERFDATA" \
  -o "$OUT" --collapsed "$COLLAPSED" --tool "$TOOL" --title "on-CPU $(basename "$PERFDATA")"
prof_log "完成: ${OUT}（折叠栈 ${COLLAPSED}）"
