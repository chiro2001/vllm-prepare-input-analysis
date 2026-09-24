#!/usr/bin/env bash
# 把一次 run 目录里要交付的产物归集到 data/harness/prof_<tag>_<ts>/ 并生成顶层文件。
#
# 交付约定（plan/COORDINATION.md 第 5 节）：
#   * 原始 perf.data（数百 MB）**不**进交付包，只在 a3-22 留档；
#   * 上传的是符号化后的 hotspots.json / annotate JSON / 火焰图 SVG / 汇总表 / manifest。
#
# 用法
#   bash harness/scripts/prof_collect_artifacts.sh <run_dir> [--tag NAME] [--keep-perfdata]
#
# 产物（data/harness/ 下）
#   prof_<tag>_<ts>/hotspots.json            符号化后的热点函数 top-N
#   prof_<tag>_<ts>/annotate_top.json        top-N 函数的指令级注释
#   prof_<tag>_<ts>/flamegraph_oncpu.svg     on-CPU 火焰图（含 Python 帧）
#   prof_<tag>_<ts>/flamegraph_oncpu.folded  折叠栈（可换渲染器重画）
#   prof_<tag>_<ts>/topdown.json             topdown 全树 + 置信度
#   prof_<tag>_<ts>/count.json               PMU 计数（IPC/cycles/instructions）
#   prof_<tag>_<ts>/summary.{json,csv}       汇总
#   prof_<tag>_<ts>/manifest.json            口径 manifest
#   prof_<tag>_<ts>/noise_{pre,post}.json    噪声门记录
#   prof_<tag>_<ts>/cmd.sh                   复现命令
#   prof_<tag>_<ts>/hotspots_top20.csv       热点 top-20 长表（便于对照真机）
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prof_lib.sh"

RUN_DIR="${1:-}"
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || prof_die "用法: $0 <run_dir> [--tag NAME] [--keep-perfdata]"
shift
TAG_OVERRIDE=""
KEEP_PERFDATA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG_OVERRIDE="$2"; shift 2;;
    --keep-perfdata) KEEP_PERFDATA=1; shift;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) prof_die "未知参数 $1";;
  esac
done

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
BASE="$(basename "$RUN_DIR")"
TAG="${TAG_OVERRIDE:-${BASE%%_[0-9]*}}"
TS="${BASE##*_}"
# 命名：data/harness/<run tag>_<ts>/（run tag 本身已带 prof_ 前缀时不再重复加）
OUT="${PROF_DATA_DIR}/${TAG}_${TS}"
mkdir -p "$OUT"
prof_log "归集 ${RUN_DIR} -> ${OUT}"

copy_if() { [[ -f "$1" ]] && cp -f "$1" "$2" || true; }

copy_if "${RUN_DIR}/pmu/count.json"      "${OUT}/count.json"
copy_if "${RUN_DIR}/pmu/topdown.json"    "${OUT}/topdown.json"
copy_if "${RUN_DIR}/summary.json"        "${OUT}/summary.json"
copy_if "${RUN_DIR}/summary.csv"         "${OUT}/summary.csv"
copy_if "${RUN_DIR}/manifest.json"       "${OUT}/manifest.json"
copy_if "${RUN_DIR}/noise_pre.json"      "${OUT}/noise_pre.json"
copy_if "${RUN_DIR}/noise_post.json"     "${OUT}/noise_post.json"
copy_if "${RUN_DIR}/cmd.sh"              "${OUT}/cmd.sh"
copy_if "${RUN_DIR}/passes/perf/hotspots.json" "${OUT}/hotspots.json"
copy_if "${RUN_DIR}/passes/perf/annotate/annotate_top.json" "${OUT}/annotate_top.json"
copy_if "${RUN_DIR}/passes/perf/flamegraph.svg"    "${OUT}/flamegraph_oncpu.svg"
copy_if "${RUN_DIR}/passes/perf/flamegraph.folded" "${OUT}/flamegraph_oncpu.folded"
copy_if "${RUN_DIR}/passes/perf/pyspy.svg" "${OUT}/flamegraph_pyspy_native.svg"
[[ "$KEEP_PERFDATA" == "1" ]] && copy_if "${RUN_DIR}/passes/perf/perf.data" "${OUT}/perf.data"

# top-20 长表
python3 - "$OUT" <<'PY'
import csv, json, sys
from pathlib import Path

out = Path(sys.argv[1])
hs = out / "hotspots.json"
if hs.is_file():
    d = json.loads(hs.read_text())
    with (out / "hotspots_top20.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "overhead_pct", "dso", "symbol"])
        for i, r in enumerate((d.get("top") or [])[:20], 1):
            w.writerow([i, r.get("overhead_pct"), r.get("dso"), r.get("symbol")])
    print(f"top20 -> {out/'hotspots_top20.csv'}")
PY

# 清单
{
  echo "# 交付清单 prof_${TAG}_${TS}"
  echo "# 来自 run 目录: ${RUN_DIR}"
  echo "# 口径版本: prof-v1"
  echo
  ( cd "$OUT" && ls -la )
} > "${OUT}/FILES.txt"

prof_log "完成。交付目录：${OUT}"
cat "${OUT}/FILES.txt"
