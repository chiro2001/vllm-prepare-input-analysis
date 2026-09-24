#!/usr/bin/env bash
# 通用无卡 profiling 采集器：给定「容器内命令」就能采 PMU / perf / 火焰图。
#
# 用法
#   bash harness/scripts/prof_harness.sh --tag NAME [选项] -- <容器内命令...>
#
# 接口约定（replay_harness / 其他 agent 只需要满足这两条）
#   1) 起法：把 python 入口 + 参数原样写在 -- 后面，脚本自己套 docker run：
#        docker run -d --network none --cpuset-cpus <CPUS> --user $(id -u):$(id -g) \
#          -e HOME=/tmp -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PYTHONPATH=/work/harness \
#          -v <repo>:/work -w /work <IMG> bash -c 'kill -STOP $$; exec <你的命令>'
#      容器是 STOP 住的：采集脚本挂好 perf/kperfx 之后才 SIGCONT，短负载也不会丢头。
#   2) PID 怎么找：容器 init 的宿主 PID = docker inspect -f '{{.State.Pid}}' <cid>；
#      命令用 exec 接在 bash 后面，PID 不变；kperfx/perf 都用这个 PID（含线程/子进程）。
#      想挂已有进程用 --attach <宿主PID>（例如常驻服务的 worker）。
#
# 常用选项
#   --tag NAME            必填，产物目录 data/harness/prof_runs/<tag>_<ts>/
#   --stages pmu,perf     默认 pmu,perf；加 topdown 就是 pmu,topdown,perf
#   --topdown-level l1|full   l1=1 组（快）；full=9 组（每组重跑一次负载）
#   --cpus 200-215        绑核（禁止 120-159）
#   --perf-freq 999       采样频率
#   --callgraph dwarf|fp|dwarf,fp|none
#   --pyspy               额外产出 Python 帧火焰图
#   --duration S          attach 模式窗口 / py-spy 采样时长
#   --window S            门控模式下每个 pass 精确量 S 秒（量完杀容器；配合负载的
#                         --steady-seconds 使用，可省掉 9 组 topdown 的等待时间）
#   --warmup S            先裸跑 S 秒再上仪器（默认 0 = STOP 门控）
#   --timeout S           单 pass 上限
#   --docker-arg ARG      追加 docker 参数（可重复，如 -v /path:/opt:ro）
#
# 典型调用
#   bash harness/scripts/prof_harness.sh --tag replay_bs8 --stages pmu,topdown,perf -- \
#        python3 -m pi_harness.runner.replay --trace data/harness/trace_bs8.jsonl
#   bash harness/scripts/prof_harness.sh --tag svc --attach 12345 --duration 20
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prof_lib.sh"

TAG=""
STAGES="${PROF_STAGES:-pmu,perf}"
TOPLEVEL="${PROF_TOPDOWN_LEVEL:-l1}"
CPUS="$PI_CPUS"
PERF_FREQ=999
CALLGRAPH=dwarf
PYSPY=""
DURATION=""
WINDOW=""
WARMUP=0
TIMEOUT=1800
ATTACH=""
DOCKER_ARGS=()
CMD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --topdown-level) TOPLEVEL="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --image) PI_IMAGE="$2"; shift 2;;
    --perf-freq) PERF_FREQ="$2"; shift 2;;
    --callgraph) CALLGRAPH="$2"; shift 2;;
    --pyspy) PYSPY="--pyspy"; shift;;
    --duration) DURATION="$2"; shift 2;;
    --window) WINDOW="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --timeout) TIMEOUT="$2"; shift 2;;
    --attach) ATTACH="$2"; shift 2;;
    --pmu-events) PMU_EVENTS="$2"; shift 2;;
    --docker-arg) DOCKER_ARGS+=("$2"); shift 2;;
    --out) OUT="$2"; shift 2;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    --) shift; CMD=("$@"); break;;
    *) prof_die "未知参数 $1（容器内命令必须放在 -- 之后）";;
  esac
done

[[ -n "$TAG" ]] || prof_die "--tag 必填"
[[ -n "$ATTACH" || ${#CMD[@]} -gt 0 ]] || prof_die "需要 -- <容器内命令> 或 --attach PID"
# --cpus auto：挑最安静的 8 个物理核（可能撞上别的租户，噪声明细写进 manifest）
if [[ "$CPUS" == "auto" ]]; then
  mkdir -p "$PROF_DATA_DIR"
  PICK_JSON="${PROF_DATA_DIR}/prof_cpupick_${TAG}_$(date +%Y%m%d-%H%M%S).json"
  CPUS="$(prof_py pi_harness.profiling.hostnoise --pick "${PROF_PICK_N:-8}" \
            --pool "${PROF_CPU_POOL:-200-239,360-399}" --seconds "${PROF_PICK_SECONDS:-6}" \
            --json "$PICK_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cpuset"])')"
  prof_log "自动选核 -> ${CPUS}（噪声明细 ${PICK_JSON}）"
fi
prof_guard_cpus "$CPUS"
prof_require_tools
prof_optional_hostnoise_gate "$CPUS"
mkdir -p "$PROF_DATA_DIR"

prof_log "工具版本：$(prof_tool_versions | tr '\n' ' ')"

args=( --tag "$TAG" --cpus "$CPUS" --image "$PI_IMAGE" --stages "$STAGES"
       --topdown-level "$TOPLEVEL" --perf-freq "$PERF_FREQ" --callgraph "$CALLGRAPH"
       --timeout "$TIMEOUT" )
[[ -n "$PYSPY" ]] && args+=( --pyspy )
[[ -n "$DURATION" ]] && args+=( --duration "$DURATION" )
[[ -n "$WINDOW" ]] && args+=( --window "$WINDOW" )
[[ -n "$ATTACH" ]] && args+=( --attach "$ATTACH" )
[[ -n "${OUT:-}" ]] && args+=( --out "$OUT" )
[[ -n "${PMU_EVENTS:-}" ]] && args+=( --pmu-events "$PMU_EVENTS" )
for d in "${DOCKER_ARGS[@]:-}"; do [[ -n "$d" ]] && args+=( --docker-arg "$d" ); done
[[ "${WARMUP}" != "0" ]] && args+=( --no-gate --warmup "$WARMUP" )
[[ ${#CMD[@]} -gt 0 ]] && args+=( -- "${CMD[@]}" )

prof_log "开始采集 tag=${TAG} cpus=${CPUS} stages=${STAGES}"
prof_py pi_harness.profiling.collect "${args[@]}"
