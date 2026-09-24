#!/usr/bin/env bash
# 纯 CPU 标定 microbenchmark 采集：证明「无卡采集链路」本身可信（P4 对照锚点）。
#
# 用法
#   bash harness/scripts/prof_calib.sh [选项]
#
# 选项
#   --phases arith,memcpy,hash   要跑的相位（默认三个都跑）
#   --iters N                    固定迭代数（确定性模式，9 组 topdown pass 活一样）
#   --seconds S                  改成按目标秒数自动估迭代数（非确定性，冒烟用）
#   --threads N                  线程数（默认 1；hash 相位 >2KB 缓冲会释放 GIL 可扩核）
#   --cpus 200-215               绑核（禁止 120-159）
#   --full-topdown               跑 9 组全套 topdown（默认只跑第 1 组 l1）
#   --workload-gen KIND          额外跑 libkperfx 的 C 负载（alu/l1/dram/... 纯 C 锚点）
#   --no-pyspy / --pyspy         是否额外产 Python 帧火焰图（默认 pyspy 开）
#   --tag-prefix PREFIX          产物前缀（默认 prof_calib）
#
# 产物
#   data/harness/prof_runs/prof_calib_<phase>_<ts>/   原始（perf.data / SVG / JSON）
#   data/harness/prof_calib_<phase>_<ts>.json          汇总（IPC / topdown / 热点）
#   data/harness/prof_calib_<phase>_<ts>.csv           长表（metric,value,source,note）
#   data/harness/prof_calib_matrix_<ts>.csv            跨相位矩阵
#   data/harness/prof_calib_flamegraph_sample.svg      示例 on-CPU 火焰图
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prof_lib.sh"

# 固定迭代数（确定性）：所有 pass 做完全一样的活，才能把 9 组 topdown 对齐。
# 数值按 a3-22 容器实测标定到每相位约 4-5 秒。
PHASES="arith,memcpy,hash"
ITERS="${PROF_ITERS:-}"
ITERS_MAP="${PROF_ITERS_MAP:-arith=20000000,memcpy=3000000,hash=350000}"
SECONDS_TARGET=""
THREADS=1
CPUS="$PI_CPUS"
TOPLEVEL="l1"
STAGES="pmu,topdown,perf"
WORKLOAD_GEN=""
PYSPY=1
TAG_PREFIX="prof_calib"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phases) PHASES="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --seconds) SECONDS_TARGET="$2"; shift 2;;
    --iters-map) ITERS_MAP="$2"; shift 2;;
    --auto-iters) ITERS=""; SECONDS_TARGET="${SECONDS_TARGET:-5}"; shift;;
    --threads) THREADS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --full-topdown) TOPLEVEL="full"; shift;;
    --stages) STAGES="$2"; shift 2;;
    --workload-gen) WORKLOAD_GEN="$2"; shift 2;;
    --no-pyspy) PYSPY=0; shift;;
    --pyspy) PYSPY=1; shift;;
    --tag-prefix) TAG_PREFIX="$2"; shift 2;;
    -h|--help) sed -n '2,26p' "$0"; exit 0;;
    *) prof_die "未知参数 $1";;
  esac
done

# --cpus auto：在允许的核池里挑最安静的 N 个（一次选定，所有相位复用同一 cpuset）
if [[ "$CPUS" == "auto" ]]; then
  # 单线程负载也留几个备选核：被别的租户抢核时调度器可以挪走
  NTHREADS=$(( THREADS > 4 ? THREADS : 4 ))
  mkdir -p "$PROF_DATA_DIR"
  PICK_JSON="${PROF_DATA_DIR}/prof_calib_cpupick_$(date +%Y%m%d-%H%M%S).json"
  CPUS="$(prof_py pi_harness.profiling.hostnoise --pick "$NTHREADS" \
            --pool "${PROF_CPU_POOL:-200-239,360-399}" --seconds "${PROF_PICK_SECONDS:-6}" \
            --json "$PICK_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cpuset"])')"
  prof_log "自动选核 -> ${CPUS}（噪声明细 ${PICK_JSON}）"
fi
prof_guard_cpus "$CPUS"
prof_require_tools
prof_optional_hostnoise_gate "$CPUS"
mkdir -p "$PROF_DATA_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
JSONS=()

run_one() {
  local tag="$1"; shift
  local -a collect_args=( --tag "$tag" --cpus "$CPUS" --image "$PI_IMAGE"
                          --stages "$STAGES" --topdown-level "$TOPLEVEL"
                          --timeout "${PROF_TIMEOUT:-1800}" )
  [[ "$PYSPY" == "1" ]] && collect_args+=( --pyspy --duration "${PROF_PYSPY_DURATION:-12}" )
  prof_log "=== ${tag} ==="
  prof_py pi_harness.profiling.collect "${collect_args[@]}" -- "$@"
  local run_dir
  run_dir="$(ls -1dt "${PROF_DATA_DIR}"/prof_runs/"${tag}"_* | head -1)"
  cp "${run_dir}/summary.json" "${PROF_DATA_DIR}/${TAG_PREFIX}_${tag#${TAG_PREFIX}_}_${TS}.json"
  cp "${run_dir}/summary.csv" "${PROF_DATA_DIR}/${TAG_PREFIX}_${tag#${TAG_PREFIX}_}_${TS}.csv"
  JSONS+=( "${PROF_DATA_DIR}/${TAG_PREFIX}_${tag#${TAG_PREFIX}_}_${TS}.json" )
  if [[ ! -f "${PROF_DATA_DIR}/prof_calib_flamegraph_sample.svg" && -f "${run_dir}/passes/perf/flamegraph.svg" ]]; then
    cp "${run_dir}/passes/perf/flamegraph.svg" "${PROF_DATA_DIR}/prof_calib_flamegraph_sample.svg"
    prof_log "示例火焰图 -> ${PROF_DATA_DIR}/prof_calib_flamegraph_sample.svg"
  fi
  prof_log "产物: ${PROF_DATA_DIR}/${TAG_PREFIX}_${tag#${TAG_PREFIX}_}_${TS}.{json,csv}"
}

IFS=',' read -r -a PHASE_ARR <<<"$PHASES"
for phase in "${PHASE_ARR[@]}"; do
  cmd=( python3 -m pi_harness.profiling.microbench --phase "$phase" --threads "$THREADS"
        --warmup "${PROF_WARMUP:-1}" --json "/tmp/microbench_${phase}.json" )
  PHASE_ITERS="$ITERS"
  if [[ -z "$PHASE_ITERS" ]]; then
    for kv in ${ITERS_MAP//,/ }; do
      [[ "${kv%%=*}" == "$phase" ]] && PHASE_ITERS="${kv#*=}"
    done
  fi
  if [[ -n "$PHASE_ITERS" ]]; then
    cmd+=( --iters "$PHASE_ITERS" )
    prof_log "phase=${phase} 固定迭代数 ${PHASE_ITERS}（确定性模式）"
  elif [[ -n "$SECONDS_TARGET" ]]; then
    cmd+=( --target-seconds "$SECONDS_TARGET" )
  fi
  run_one "${TAG_PREFIX}_${phase}" "${cmd[@]}"
done

if [[ -n "$WORKLOAD_GEN" ]]; then
  # 纯 C 锚点：libkperfx 自带的 workload_gen（只读挂载，不进镜像）
  run_one "${TAG_PREFIX}_c_${WORKLOAD_GEN}" \
    --docker-arg -v --docker-arg "${PI_LIBKPERFX}:/opt/libkperfx:ro" \
    "${PROF_LIBKPERFX:-/opt/libkperfx}/build/workload_gen" --kind "$WORKLOAD_GEN"
fi

# 跨相位矩阵
python3 - "$TS" "${JSONS[@]}" <<'PY'
import csv, json, sys
from pathlib import Path

ts, *paths = sys.argv[1:]
rows = []
for p in paths:
    d = json.loads(Path(p).read_text())
    l1 = ((d.get("topdown") or {}).get("level1") or {})
    rows.append({
        "phase": d["tag"].replace("prof_calib_", ""),
        "caliber": d.get("caliber_version"),
        "ipc": (d.get("pmu") or {}).get("ipc"),
        "confidence": (d.get("pmu") or {}).get("confidence"),
        "retiring": l1.get("retiring"),
        "bad_spec": l1.get("bad_spec"),
        "frontend_bound": l1.get("frontend_bound"),
        "backend_bound": l1.get("backend_bound"),
        "top_hot": (d.get("hotspots_top") or [{}])[0].get("symbol"),
        "flamegraph": bool(d.get("artifacts", {}).get("flamegraph_svg")),
        "pyspy": bool(d.get("artifacts", {}).get("pyspy_svg")),
        "run_dir": d.get("out"),
    })
out = Path(f"data/harness/prof_calib_matrix_{ts}.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["phase"])
    w.writeheader()
    w.writerows(rows)
print(f"矩阵 -> {out}")
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
PY
