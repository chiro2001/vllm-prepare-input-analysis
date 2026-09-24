#!/usr/bin/env bash
# 验证：宿主 root perf record 挂到容器内进程（-p <host pid>）
set -uo pipefail

IMG=${IMG:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}
CPUS=${CPUS:-200-215}
TOOLS=$HOME/tools
OUT=${OUT:-/tmp/pi-prof-attach}
mkdir -p "$OUT"

CIDFILE=$(mktemp -u /tmp/pi-prof-cid.XXXXXX)   # docker 要求 cidfile 不存在
docker run -d --rm --cidfile "$CIDFILE" --name pi-prof-attach \
  --network none --cpuset-cpus "$CPUS" --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${TOOLS}/tests:/tests:ro" "$IMG" \
  bash -c 'python3 /tests/busy.py --seconds 30 --mode mixed' >/dev/null

CID=$(cat "$CIDFILE")
echo "container=$CID"
sleep 2
echo "--- docker top (宿主 PID) ---"
docker top "$CID" -eo pid,ppid,cmd | head -6
PID=$(docker top "$CID" -eo pid,cmd | awk '/busy\.py/ && !/awk/ {print $1; exit}')
echo "target python host pid=$PID"
if [[ -z "${PID:-}" ]]; then echo "FAIL: 找不到目标 PID"; exit 1; fi

echo "--- perf stat (8s) ---"
sudo timeout -s INT 8 perf stat -p "$PID" -e cycles,instructions,branches,branch-misses 2>&1 | tail -12

echo "--- perf record -F 999 -g dwarf (6s) ---"
sudo timeout -s INT 6 perf record -o "$OUT/attach.data" -F 999 -g --call-graph dwarf -p "$PID" 2>&1 | tail -3
ls -la "$OUT/attach.data"

echo "--- perf report top ---"
sudo perf report --stdio -i "$OUT/attach.data" --no-children --percent-limit 1 2>&1 | head -25

echo "--- perf script 样本行 ---"
sudo perf script -i "$OUT/attach.data" 2>&1 | head -12

docker rm -f "$CID" >/dev/null 2>&1
echo done
