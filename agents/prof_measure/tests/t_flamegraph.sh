#!/usr/bin/env bash
# 验证：flamegraph-rs 读 perf.data 出 SVG；py-spy 出带 Python 帧的 SVG；perf annotate
set -uo pipefail

IMG=${IMG:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}
CPUS=${CPUS:-200-215}
TOOLS=$HOME/tools
OUT=${OUT:-/tmp/pi-prof-fg}
FG=$TOOLS/cargo/bin/flamegraph
mkdir -p "$OUT"

CIDFILE=$(mktemp -u /tmp/pi-prof-cid.XXXXXX)
docker run -d --rm --cidfile "$CIDFILE" --name pi-prof-fg \
  --network none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${TOOLS}/tests:/tests:ro" "$IMG" \
  bash -c 'python3 /tests/busy.py --seconds 40 --mode mixed' >/dev/null
CID=$(cat "$CIDFILE")
sleep 2
PID=$(docker top "$CID" -eo pid,cmd | awk '/busy\.py/ && !/awk/ {print $1; exit}')
echo "container=$CID pid=$PID"

echo "=== [1] perf record -F 999 -g dwarf (8s) ==="
sudo timeout -s INT 8 perf record -o "$OUT/fg.data" -F 999 -g --call-graph dwarf -p "$PID" 2>&1 | tail -2

echo "=== [2] flamegraph-rs 生成 SVG ==="
sudo "$FG" --perfdata "$OUT/fg.data" -o "$OUT/perf.svg" --title "pi-prof perf cycles" 2>&1 | tail -5
ls -la "$OUT/perf.svg" 2>/dev/null && head -c 200 "$OUT/perf.svg" | head -3

echo "=== [3] py-spy record (6s, python+native 帧) ==="
sudo "$HOME/.local/bin/py-spy" record -p "$PID" --duration 6 --rate 199 \
  --format flamegraph -o "$OUT/pyspy.svg" --nonblocking --native 2>&1 | tail -5
ls -la "$OUT/pyspy.svg" 2>/dev/null

echo "=== [4] py-spy top 采样文本（前 15 行）==="
sudo "$HOME/.local/bin/py-spy" dump --pid "$PID" 2>&1 | head -15

echo "=== [5] perf annotate 热点函数 ==="
sudo perf annotate --stdio -l -i "$OUT/fg.data" --symbol "_PyEval_EvalFrameDefault" 2>&1 | head -20

docker rm -f "$CID" >/dev/null 2>&1
echo done
