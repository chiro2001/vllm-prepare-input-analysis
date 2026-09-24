#!/usr/bin/env bash
# 本地 <-> a3-22 同步 helper。本地是本仓库 /home/chiro/projects/vllm/preparing-input-phase，
# 远端是 a3-22:/home/REMOTE_USER/projects/vllm/prepare-input-phase（唯一允许写入的远端位置）。
set -euo pipefail
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REMOTE=a3-22
REMOTE_ROOT='~/projects/vllm/prepare-input-phase'

case "${1:-push}" in
  push)
    rsync -a --relative \
      "$LOCAL_ROOT/./harness" \
      "$LOCAL_ROOT/./docs" \
      "$LOCAL_ROOT/./data" \
      "$LOCAL_ROOT/./agents" \
      "$REMOTE:$REMOTE_ROOT/"
    ;;
  pull)
    shift
    rsync -a "$REMOTE:$REMOTE_ROOT/$1" "$LOCAL_ROOT/$1"
    ;;
  *)
    echo "usage: sync.sh [push|pull <path>]" >&2; exit 2;;
esac
