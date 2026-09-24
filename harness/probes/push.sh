#!/usr/bin/env bash
# probe_devdeps 专用同步：只推自己名下的路径（harness/probes,**/pi_harness/shim 本地副本）。
#
#   bash harness/probes/push.sh                 # 只推 probes（shim 归 replay_harness，默认不推）
#   bash harness/probes/push.sh push --with-shim     # 需要时显式推 shim 本地副本
#   bash harness/probes/push.sh push --with-report   # 再推 agents/probe_devdeps/REPORT.md
#   bash harness/probes/push.sh pull            # 把 data/harness/devdeps_* 拉回本地
#
# 注意: replay_harness 的 harness/scripts/sync.sh 在本仓库布局下会把 LOCAL_ROOT 算成
# `<repo>/agents`（见 harness/probes/README.md），所以这里用自带的相对路径 rsync。
set -euo pipefail

LOCAL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"   # = preparing-input-phase
LOCAL_HARNESS="$LOCAL_REPO/agents/replay_harness/harness"
REMOTE=a3-22
REMOTE_ROOT='~/projects/vllm/prepare-input-phase'

mode="${1:-push}"
case "$mode" in
  push)
    rsync -a --relative "$LOCAL_HARNESS/./probes" "$REMOTE:$REMOTE_ROOT/harness/"
    for arg in "${@:2}"; do
      case "$arg" in
        --with-shim)
          # shim 的 owner 是 replay_harness；只有显式要求时才覆盖远端版本
          rsync -a --relative "$LOCAL_HARNESS/./pi_harness/shim" "$REMOTE:$REMOTE_ROOT/harness/"
          ;;
      esac
    done
    if [[ "${2:-}" == "--with-report" ]]; then
      rsync -a --relative "$LOCAL_REPO/./agents/probe_devdeps" "$REMOTE:$REMOTE_ROOT/"
    fi
    ;;
  pull)
    rsync -a --relative "$REMOTE:$REMOTE_ROOT/./data/harness/" "$LOCAL_REPO/"
    ;;
  *)
    echo "usage: push.sh [push [--with-report] | pull]" >&2
    exit 2
    ;;
esac
