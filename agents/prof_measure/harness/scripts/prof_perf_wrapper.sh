#!/usr/bin/env bash
# perf 包装器：把 --symfs 注入 flamegraph-rs 内部调用的 `perf script`。
#
# 背景：容器里的 python/torch 动态库在宿主上不存在（路径是 /usr/local/...），
# perf 解析不出符号，热点会退化成裸地址。flamegraph-rs 只暴露 PERF 环境变量，
# 于是用本包装器在 `perf script` 后面补 --force --symfs <容器 DSO 抽取目录>。
#
# 用法（由 pi_harness.profiling.flamegraph / collect.py 自动设置）：
#   sudo env PERF=<本脚本> PERF_SYMFS=/path/symfs <flamegraph> --perfdata x.data -o x.svg
set -uo pipefail

REAL_PERF="${PERF_REAL:-/usr/bin/perf}"
SYMFS="${PERF_SYMFS:-}"

if [[ "${1:-}" == "script" && -n "$SYMFS" && -d "$SYMFS" ]]; then
  shift
  exec "$REAL_PERF" script --force --symfs "$SYMFS" "$@"
fi
exec "$REAL_PERF" "$@"
