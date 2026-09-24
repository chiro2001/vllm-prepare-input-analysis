#!/usr/bin/env bash
# 只把 devdeps_* 的**小文件**拉回本地（避免每次都同步 2.8 MB 的 scope.json 与 ab_*.json）。
#
#   bash harness/probes/pull_data.sh          # 默认拉全部 devdeps_* 元数据
#   bash harness/probes/pull_data.sh --stacks # 连 devdeps_stacks/ 一起拉
set -euo pipefail

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
REMOTE_ROOT='~/projects/vllm/prepare-input-phase'
FILES=(
  devdeps_apilist.md devdeps_apilist.json
  devdeps_matrix.md devdeps_matrix.json
  devdeps_manifest.json devdeps_manifest_host.json devdeps_manifest_container.json
  devdeps_shim_smoke.json devdeps_shim_report.json devdeps_shim_patch_selftest.json
  devdeps_dynamic_hits.json devdeps_imports.json
)
mkdir -p "$LOCAL_REPO/data/harness"
for f in "${FILES[@]}"; do
  rsync -a "a3-22:$REMOTE_ROOT/data/harness/$f" "$LOCAL_REPO/data/harness/" 2>/dev/null || true
done
if [[ "${1:-}" == "--stacks" ]]; then
  rsync -a "a3-22:$REMOTE_ROOT/data/harness/devdeps_stacks/" "$LOCAL_REPO/data/harness/devdeps_stacks/"
fi
ls -la "$LOCAL_REPO/data/harness/" | grep devdeps || true
