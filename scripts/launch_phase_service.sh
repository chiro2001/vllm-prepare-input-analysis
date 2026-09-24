#!/usr/bin/env bash
# Bring up a vLLM 0.26.0 + vllm-ascend 0.26.0rc1 service on chip3 with
# thread-level phase timing armed, run a workload, stop, and collect.
#
# Usage:
#   launch_phase_service.sh --run-id ID [options]
#
#   --run-id ID            required, filesystem-safe; runs/<ID>/ is the output
#   --model-key KEY        qwen35-08b (default) | qwen35-2b | <absolute path>
#   --port P               default: first free port >= 18100
#   --chip N               default 3  (chip3 => /dev/davinci3, cpus 120-159)
#   --tp N                 default 1
#   --backend B            uni (default) | mp
#   --cudagraph-mode M     FULL_DECODE_ONLY (default) | NONE
#   --max-model-len N      default 2048
#   --max-num-seqs N       default 8
#   --max-num-batched-tokens N   default 2048
#   --gpu-mem F            default 0.80
#   --requests N           default 1
#   --concurrency N        default 1
#   --prompt-tokens N      default 128
#   --max-tokens N         default 64
#   --warmup-requests N    default 2; identical requests issued *before*
#                          /start_profile so that torch.compile / NPU graph
#                          capture for the request shapes lands outside the
#                          measurement window (measured: ~23 s of it)
#   --async-scheduling on|off   default on (vLLM 0.26 default on this build)
#   --ready-timeout S      default 1800
#   --phase-timing on|off  default on (arms VLLM_LITE_PROFILER_LOG_PATH)
#   --pystack-interval-us N
#                          default 1000 (= upstream VLLM_PYSTACK_INTERVAL_US).
#                          The Python stack sampler is started by
#                          /start_profile unconditionally and formats a stack for
#                          every *Python* thread of the process, so at 1 ms it
#                          takes GIL time away from the engine thread.  0 means
#                          "never sample" (the sampler has no off switch, so the
#                          interval is pushed past the run length).
#   --cpu-pinning on|off   default on (run pin_worker_cpus.sh after ready)
#   --keep                 leave the container running (for interactive work)
#   --serve-only           bring the service up, write run_manifest.json, and
#                          leave it running *without* running a workload.  The
#                          point is to let another stage (perf / libkperfx /
#                          msprof) attach to the engine-core TID from
#                          runs/<id>/run_manifest.json and then drive its own
#                          requests against --base-url.  Implies --keep.
#   --dry-run              print the docker command and exit
#   --extra-serve-args "..."   appended verbatim to `vllm serve`
#
# Opt-in docker hook (introduced for the prepare_input sub-scope experiment,
# default OFF => zero behaviour change):
#   PREPARE_INPUT_EXTRA_DOCKER_ARGS="..."
#       word-split and inserted into `docker run` immediately before the image
#       tag, i.e. in the option position.  Used to bind-mount the instrumented
#       runner and to set PI_SUBSCOPE/PI_SUBSCOPE_LOG.  See
#       scripts/launch_subscope_service.sh.  Unset (the default) reproduces the
#       previous command line byte for byte.
#
# The script never touches NPUs other than --chip, never changes host state
# (no IRQ affinity, no systemd), and records everything under
# $PROJECT_ROOT/runs/<run-id>/.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PREPARE_INPUT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

IMAGE_TAG=${PHASE_IMAGE:-local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler}
DEFAULT_MODEL_ROOT=${MODEL_ROOT:-/home/REMOTE_USER/models}

RUN_ID=""
MODEL_KEY=qwen35-08b
PORT=""
CHIP=3
TP=1
BACKEND=uni
CUDAGRAPH_MODE=FULL_DECODE_ONLY
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=2048
GPU_MEM=0.80
REQUESTS=1
CONCURRENCY=1
PROMPT_TOKENS=128
MAX_TOKENS=64
WARMUP_REQUESTS=2
ASYNC_SCHEDULING=on
READY_TIMEOUT=1800
PHASE_TIMING=on
PYSTACK_INTERVAL_US=1000
CPU_PINNING=on
KEEP=0
SERVE_ONLY=0
DRY_RUN=0
EXTRA_ARGS=""
DTYPE=bfloat16
SEED=1024

die() { printf 'launch_phase_service: ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

usage() { sed -n '2,34p' "$0"; exit 0; }

while (($#)); do
    case "$1" in
        --run-id) RUN_ID=$2; shift 2 ;;
        --model-key) MODEL_KEY=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --chip) CHIP=$2; shift 2 ;;
        --tp) TP=$2; shift 2 ;;
        --backend) BACKEND=$2; shift 2 ;;
        --cudagraph-mode) CUDAGRAPH_MODE=$2; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS=$2; shift 2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS=$2; shift 2 ;;
        --gpu-mem) GPU_MEM=$2; shift 2 ;;
        --requests) REQUESTS=$2; shift 2 ;;
        --concurrency) CONCURRENCY=$2; shift 2 ;;
        --prompt-tokens) PROMPT_TOKENS=$2; shift 2 ;;
        --max-tokens) MAX_TOKENS=$2; shift 2 ;;
        --warmup-requests) WARMUP_REQUESTS=$2; shift 2 ;;
        --async-scheduling) ASYNC_SCHEDULING=$2; shift 2 ;;
        --ready-timeout) READY_TIMEOUT=$2; shift 2 ;;
        --phase-timing) PHASE_TIMING=$2; shift 2 ;;
        --pystack-interval-us) PYSTACK_INTERVAL_US=$2; shift 2 ;;
        --cpu-pinning) CPU_PINNING=$2; shift 2 ;;
        --dtype) DTYPE=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --image) IMAGE_TAG=$2; shift 2 ;;
        --extra-serve-args) EXTRA_ARGS=$2; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --serve-only) SERVE_ONLY=1; KEEP=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n $RUN_ID ]] || die "--run-id is required"
[[ $RUN_ID =~ ^[A-Za-z0-9_.-]+$ ]] || die "--run-id must be filesystem safe"
[[ $CHIP =~ ^[0-9]+$ ]] || die "--chip must be a number"
[[ $PHASE_TIMING == on || $PHASE_TIMING == off ]] || die "--phase-timing must be on/off"
[[ $CPU_PINNING == on || $CPU_PINNING == off ]] || die "--cpu-pinning must be on/off"
[[ $ASYNC_SCHEDULING == on || $ASYNC_SCHEDULING == off ]] || die "--async-scheduling must be on/off"
[[ $PYSTACK_INTERVAL_US =~ ^[0-9]+$ ]] || die "--pystack-interval-us must be a non-negative integer"

case "$MODEL_KEY" in
    qwen35-08b) MODEL_DIR=$DEFAULT_MODEL_ROOT/Qwen3.5-0.8B; SERVED_NAME=qwen35-08b ;;
    qwen35-2b) MODEL_DIR=$DEFAULT_MODEL_ROOT/Qwen3.5-2B; SERVED_NAME=qwen35-2b ;;
    /*) MODEL_DIR=$MODEL_KEY; SERVED_NAME=$(basename "$MODEL_KEY" | tr 'A-Z' 'a-z') ;;
    *) die "unknown --model-key $MODEL_KEY" ;;
esac
[[ -d $MODEL_DIR ]] || die "model dir not found: $MODEL_DIR"

FIRST=$((40 * CHIP))
CPUSET="$FIRST-$((FIRST + 39))"
MEMS=$((CHIP / 2))
CARD=$((CHIP / 2))
SUBCHIP=$((CHIP % 2))
MAIN_CPUS="$((FIRST + 2))-$((FIRST + 37))"
ACL_CPU=$((FIRST + 38))
REL_CPU=$((FIRST + 39))
CLIENT_CPUS="$FIRST,$((FIRST + 1))"
CONTAINER="pi-phase-chip${CHIP}-${RUN_ID}"
CONTAINER_MODEL="/models/$(basename "$MODEL_DIR")"

RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
LITE_DIR="$RUN_DIR/lite-profiler"
LOG_PATH=/runmeta/lite-profiler/lite.log
HOST_LITE_LOG="$LITE_DIR/lite.log"

pick_port() {
    local candidate=$1
    while ((candidate < 19000)); do
        if ! ss -ltn "( sport = :$candidate )" 2>/dev/null | tail -n +2 | grep -q .; then
            printf '%s' "$candidate"
            return 0
        fi
        candidate=$((candidate + 1))
    done
    die "no free port found in 18100-18999"
}
[[ -n $PORT ]] || PORT=$(pick_port 18100)
[[ $PORT =~ ^[0-9]+$ ]] || die "--port must be a number"
((PORT >= 18100)) || die "port below the reserved 18100 range"
BASE_URL="http://127.0.0.1:$PORT"

mkdir -p "$RUN_DIR/commands" "$RUN_DIR/host" "$RUN_DIR/requests" "$RUN_DIR/mappings"
[[ $PHASE_TIMING == on ]] && mkdir -p "$LITE_DIR"
cp -f "$SCRIPT_DIR/container_launch.sh" "$RUN_DIR/commands/launch.sh"
chmod +x "$RUN_DIR/commands/launch.sh"

# ---------------------------------------------------------------- docker args
DOCKER_ARGS=(
    sudo -n docker run -d
    --name "$CONTAINER"
    --label "pi.project=prepare-input-phase"
    --label "pi.run_id=$RUN_ID"
    --label "pi.chip=$CHIP"
    --privileged
    --network host
    --shm-size=64g
    --ulimit memlock=-1:-1
    --cpuset-cpus="$CPUSET"
    --cpuset-mems="$MEMS"
    --device "/dev/davinci$CHIP"
    --device /dev/davinci_manager
    --device /dev/devmm_svm
    --device /dev/hisi_hdc
    -v /etc/hccn.conf:/etc/hccn.conf:ro
    -v /usr/local/dcmi:/usr/local/dcmi:ro
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro
    -v "$MODEL_DIR:$CONTAINER_MODEL:ro"
    -v "$RUN_DIR:/runmeta:rw"
    -e "ASCEND_RT_VISIBLE_DEVICES=$CHIP"
    -e MSMONITOR_USE_DAEMON=0
    -e TASK_QUEUE_ENABLE=1
    -e OMP_NUM_THREADS=1
    -e OMP_PROC_BIND=false
    -e PYTHONUNBUFFERED=1
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    -e HCCL_BUFFSIZE=512
    -e VLLM_USE_MODELSCOPE=False
    -e HF_HUB_OFFLINE=1
    -e TRANSFORMERS_OFFLINE=1
)
if [[ $PHASE_TIMING == on ]]; then
    DOCKER_ARGS+=(-e "VLLM_LITE_PROFILER_LOG_PATH=$LOG_PATH")
    if ((PYSTACK_INTERVAL_US == 0)); then
        DOCKER_ARGS+=(-e "VLLM_PYSTACK_INTERVAL_US=4294967295")
    else
        DOCKER_ARGS+=(-e "VLLM_PYSTACK_INTERVAL_US=$PYSTACK_INTERVAL_US")
    fi
    # Upstream default is /tmp/pystack.csv *inside* the container, which is
    # destroyed with the container.  Point it at the run directory instead.
    DOCKER_ARGS+=(-e "VLLM_PYSTACK_LOG=/runmeta/lite-profiler/pystack.csv")
fi
# Everything from the image tag onwards is the container *command* (the
# `vllm serve ...` line).  Keeping it in its own array is what lets the opt-in
# hook below insert `docker run` options in the correct position.
DOCKER_CMD_ARGS=(
    "$IMAGE_TAG"
    /runmeta/commands/launch.sh
    serve "$CONTAINER_MODEL"
    --host 127.0.0.1
    --port "$PORT"
    --data-parallel-size 1
    --tensor-parallel-size "$TP"
    --distributed-executor-backend "$BACKEND"
    --seed "$SEED"
    --served-model-name "$SERVED_NAME"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --trust-remote-code
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEM"
    --no-enable-prefix-caching
    --compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}"
    --additional-config '{"enable_cpu_binding":false}'
)
if [[ -n $EXTRA_ARGS ]]; then
    # shellcheck disable=SC2206 - deliberate word splitting of a user string
    DOCKER_CMD_ARGS+=($EXTRA_ARGS)
fi
if [[ $ASYNC_SCHEDULING == off ]]; then
    DOCKER_CMD_ARGS+=(--no-async-scheduling)
fi

# ------------------------------------------------- opt-in extra docker flags
# (prepare_input sub-scope experiment; empty by default => no change)
# NOTE: word splitting is deliberate: the hook is a flat string of docker
# *options* (-v/-e/...) and they must land before the image tag, never after it
# (an option placed after the image is parsed by the container command, which
# surfaced as `vllm: error: unrecognized arguments: -v ...`).
PREPARE_INPUT_EXTRA_DOCKER_ARGS=${PREPARE_INPUT_EXTRA_DOCKER_ARGS:-}
if [[ -n $PREPARE_INPUT_EXTRA_DOCKER_ARGS ]]; then
    # shellcheck disable=SC2206
    DOCKER_ARGS+=($PREPARE_INPUT_EXTRA_DOCKER_ARGS)
fi

DOCKER_ARGS+=("${DOCKER_CMD_ARGS[@]}")

if ((DRY_RUN)); then
    printf '%q ' "${DOCKER_ARGS[@]}"
    printf '\n'
    exit 0
fi

# ----------------------------------------------------------------- preflight
docker ps --format '{{.Names}}' >"$RUN_DIR/host/docker_ps.before.txt"
if sudo -n docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    die "container $CONTAINER already exists"
fi
if ss -ltn "( sport = :$PORT )" 2>/dev/null | tail -n +2 | grep -q .; then
    die "port $PORT is already in use"
fi
sudo -n npu-smi info -t proc-mem -i "$CARD" -c "$SUBCHIP" \
    >"$RUN_DIR/host/npu_proc_mem.before.txt" 2>&1 || true
if grep -q "Process id:" "$RUN_DIR/host/npu_proc_mem.before.txt"; then
    die "chip $CHIP already has an NPU process: $(tr '\n' ' ' <"$RUN_DIR/host/npu_proc_mem.before.txt")"
fi
sudo -n docker image inspect "$IMAGE_TAG" >"$RUN_DIR/host/image.inspect.json"
IMAGE_ID=$(sudo -n docker image inspect "$IMAGE_TAG" --format '{{.Id}}')
sudo -n docker image inspect "$IMAGE_TAG" \
    --format 'entrypoint={{json .Config.Entrypoint}} cmd={{json .Config.Cmd}}' \
    >"$RUN_DIR/host/image.entrypoint.txt"
printf '%q ' "${DOCKER_ARGS[@]}" >"$RUN_DIR/commands/docker_run.txt"
printf '\n' >>"$RUN_DIR/commands/docker_run.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/host/date.before.txt"
T0=$(date +%s)

# ------------------------------------------------------------------ manifest
PI_EXTRA_ARGS="$PREPARE_INPUT_EXTRA_DOCKER_ARGS" \
python3 - "$RUN_DIR" "$RUN_ID" "$IMAGE_TAG" "$IMAGE_ID" "$MODEL_DIR" "$SERVED_NAME" \
    "$CHIP" "$CPUSET" "$MEMS" "$MAIN_CPUS" "$ACL_CPU" "$REL_CPU" "$PORT" "$BACKEND" \
    "$TP" "$CUDAGRAPH_MODE" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" \
    "$GPU_MEM" "$REQUESTS" "$CONCURRENCY" "$PROMPT_TOKENS" "$MAX_TOKENS" \
    "$PHASE_TIMING" "$CPU_PINNING" "$DTYPE" "$SEED" "$CONTAINER" \
    "$WARMUP_REQUESTS" "$ASYNC_SCHEDULING" "$PYSTACK_INTERVAL_US" <<'PY'
import hashlib
import json
import os
import pathlib
import subprocess
import sys

(run_dir, run_id, image, image_id, model_dir, served_name, chip, cpuset, mems,
 main_cpus, acl, rel, port, backend, tp, cudagraph, max_model_len, max_num_seqs,
 max_batched, gpu_mem, requests, concurrency, prompt_tokens, max_tokens,
 phase_timing, cpu_pinning, dtype, seed, container, warmup_requests,
 async_scheduling, pystack_interval_us) = sys.argv[1:34]

run_dir = pathlib.Path(run_dir)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rev(path: str) -> dict:
    """Best-effort revision fingerprint for a model directory."""
    p = pathlib.Path(path)
    out = {"path": path, "exists": p.is_dir()}
    files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else []
    out["n_files"] = len(files)
    out["total_bytes"] = sum(x.stat().st_size for x in files)
    for name in ("config.json", "model.safetensors.index.json"):
        cand = p / name
        if cand.is_file():
            out[f"sha256:{name}"] = sha256(cand)
    return out


scripts_dir = pathlib.Path(__file__).resolve().parent if "__file__" in dir() else None
manifest = {
    "run_id": run_id,
    "kind": "service-bringup",
    "stage": "phase-timing",
    "created_utc": subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
    ).stdout.strip(),
    "container": container,
    "image": {"ref": image, "id": image_id},
    "model": rev(model_dir) | {"served_name": served_name},
    "hardware": {
        "host": "a3-22",
        "chip": int(chip),
        "device": f"/dev/davinci{chip}",
        "cpuset": cpuset,
        "cpuset_mems": int(mems),
        "worker_main_cpus": main_cpus,
        "acl_cpu": int(acl),
        "release_cpu": int(rel),
    },
    "server": {
        "backend": backend,
        "tensor_parallel_size": int(tp),
        "cudagraph_mode": cudagraph,
        "max_model_len": int(max_model_len),
        "max_num_seqs": int(max_num_seqs),
        "max_num_batched_tokens": int(max_batched),
        "gpu_memory_utilization": float(gpu_mem),
        "dtype": dtype,
        "seed": int(seed),
        "port": int(port),
        "enable_cpu_binding": False,
        "cpu_pinning": cpu_pinning,
        "prefix_caching": False,
        "async_scheduling": async_scheduling,
        "async_scheduling_evidence": "server log line: 'Asynchronous scheduling is "
                                     "%s.' (vllm/config/vllm.py)",
    },
    "workload": {
        "requests": int(requests),
        "concurrency": int(concurrency),
        "prompt_tokens_target": int(prompt_tokens),
        "max_tokens": int(max_tokens),
        "warmup_requests_before_profile_window": int(warmup_requests),
        "streaming": True,
        "temperature": 0.0,
    },
    "instrumentation": {
        "phase_timing": phase_timing,
        "extra_docker_args": os.environ.get("PI_EXTRA_ARGS", ""),
        "mode": "liteprofiler-v2 (env armed, no --profiler-config)",
        "log_path_in_container": "/runmeta/lite-profiler/lite.log",
        "log_path_on_host": str(run_dir / "lite-profiler" / "lite.log"),
        "activate": "POST /start_profile ; POST /stop_profile",
        "pystack": {
            "log_path_on_host": str(run_dir / "lite-profiler" / "pystack.csv"),
            "interval_us_requested": int(pystack_interval_us),
            "interval_us_effective": 4294967295 if int(pystack_interval_us) == 0
                                     else int(pystack_interval_us),
            "note": "started unconditionally by /start_profile; samples every Python "
                    "thread of the process and therefore competes for the GIL",
        },
    },
    "scripts_sha256": {},
    "env": {
        "ASCEND_RT_VISIBLE_DEVICES": chip,
        "OMP_NUM_THREADS": "1",
        "TASK_QUEUE_ENABLE": "1",
        "MSMONITOR_USE_DAEMON": "0",
    },
}
root = run_dir.parent.parent
for candidate in (
    root / "scripts" / "launch_phase_service.sh",
    root / "scripts" / "container_launch.sh",
    root / "scripts" / "phase_smoke_client.py",
    root / "scripts" / "pin_worker_cpus.sh",
    root / "scripts" / "parse_phase_timing.py",
    root / "scripts" / "chip3_lock.sh",
):
    manifest["scripts_sha256"][candidate.name] = (
        sha256(candidate) if candidate.is_file() else None
    )

(run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"wrote {run_dir / 'manifest.json'}")
PY

# ------------------------------------------------------------------- startup
HOST_CMD="${DOCKER_ARGS[*]}"
note "creating $CONTAINER on chip $CHIP (cpuset $CPUSET/mems $MEMS, port $PORT, backend $BACKEND, tp $TP)"
"${DOCKER_ARGS[@]}" >"$RUN_DIR/host/docker_run.stdout.txt"
CONTAINER_ID=$(sudo -n docker inspect -f '{{.Id}}' "$CONTAINER")
printf '%s\n' "$CONTAINER_ID" >"$RUN_DIR/host/container_id.txt"
sudo -n docker inspect "$CONTAINER" >"$RUN_DIR/host/docker_inspect.created.json"

cleanup() {
    local rc=$?
    if ((KEEP)); then
        note "keeping container $CONTAINER (--keep)"
        return $rc
    fi
    if sudo -n docker container inspect "$CONTAINER" >/dev/null 2>&1; then
        sudo -n docker logs --timestamps "$CONTAINER" >"$RUN_DIR/host/docker-logs.txt" 2>&1 || true
        note "stopping $CONTAINER"
        timeout 180 sudo -n docker stop --time 60 "$CONTAINER" >"$RUN_DIR/host/docker_stop.txt" 2>&1 || true
        timeout 180 sudo -n docker rm -f "$CONTAINER" >"$RUN_DIR/host/docker_rm.txt" 2>&1 || true
    fi
    date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/host/date.after.txt"
    docker ps --format '{{.Names}}' >"$RUN_DIR/host/docker_ps.after.txt"
    sudo -n npu-smi info -t proc-mem -i "$CARD" -c "$SUBCHIP" \
        >"$RUN_DIR/host/npu_proc_mem.after.txt" 2>&1 || true
    return $rc
}
trap 'cleanup >/dev/null 2>&1 || true' EXIT

READY=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE_URL/health" || true)
    if [[ $code == 200 ]]; then READY=1; break; fi
    if ! sudo -n docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true; then
        break
    fi
    sleep 1
done
if ((READY == 0)); then
    sudo -n docker logs --timestamps "$CONTAINER" >"$RUN_DIR/host/docker-logs.failed.txt" 2>&1 || true
    cleanup
    echo "PHASE_RESULT=FAIL reason=service-not-ready run_dir=$RUN_DIR" >&2
    exit 1
fi
READY_EPOCH=$(date +%s)
note "service ready after $((READY_EPOCH - T0))s"

curl -s "$BASE_URL/v1/models" >"$RUN_DIR/requests/models.ready.json"
sudo -n docker inspect "$CONTAINER" >"$RUN_DIR/host/docker_inspect.ready.json"

# ------------------------------------------------------- worker identification
sudo -n npu-smi info -t proc-mem -i "$CARD" -c "$SUBCHIP" \
    >"$RUN_DIR/mappings/npu_proc_mem.ready.txt" 2>&1 || true
sleep 2
sudo -n docker exec "$CONTAINER" bash -lc \
    'ps -eLo pid,tid,comm,args --no-headers | head -400' \
    >"$RUN_DIR/mappings/container_threads.txt" 2>/dev/null || true
APIPID=$(sudo -n docker inspect -f '{{.State.Pid}}' "$CONTAINER")
printf 'container_main_host_pid=%s\n' "$APIPID" >"$RUN_DIR/mappings/container_main_pid.txt"

if [[ $CPU_PINNING == on ]]; then
    note "pinning worker cpus (main=$MAIN_CPUS acl=$ACL_CPU rel=$REL_CPU)"
    "$SCRIPT_DIR/pin_worker_cpus.sh" --chip "$CHIP" --container "$CONTAINER" \
        >"$RUN_DIR/mappings/cpu_pinning.txt" 2>&1 || \
        note "WARNING: cpu pinning failed, see mappings/cpu_pinning.txt"
    "$SCRIPT_DIR/pin_worker_cpus.sh" --chip "$CHIP" --container "$CONTAINER" --verify \
        >>"$RUN_DIR/mappings/cpu_pinning.txt" 2>&1 || true
fi

# ------------------------------------------------------------------ workload
run_workload() {
    # $1 outdir  $2 tag  $3 stdout  $4 stderr
    taskset -c "$CLIENT_CPUS" python3 "$SCRIPT_DIR/phase_smoke_client.py" \
        --base-url "$BASE_URL" --model "$SERVED_NAME" --outdir "$1" \
        --requests "$REQUESTS" --concurrency "$CONCURRENCY" \
        --prompt-tokens "$PROMPT_TOKENS" --max-tokens "$MAX_TOKENS" --seed "$SEED" \
        --tag "$2" >"$3" 2>"$4"
}

if ((SERVE_ONLY)); then
    # ---------------------------------------------------------- serve-only
    # Everything above (readiness, worker identification, CPU pinning) has
    # already happened, which is exactly what a profiler needs.  Write the
    # hand-off manifest while the engine is live and warm, then return with
    # the container still running so perf / libkperfx can attach to the TID
    # recorded in runs/<run-id>/run_manifest.json before any measured traffic
    # is generated.
    note "serve-only: skipping warmup, profile window and workload"
    printf 'serve_only=1\nbase_url=%s\ncontainer=%s\nport=%s\n' \
        "$BASE_URL" "$CONTAINER" "$PORT" >"$RUN_DIR/host/serve_only.txt"
    python3 "$SCRIPT_DIR/run_manifest.py" --run-id "$RUN_ID" \
        --project-root "$PROJECT_ROOT" \
        >"$RUN_DIR/host/run_manifest.log" 2>&1 || \
        note "WARNING: run_manifest.py failed, see host/run_manifest.log"
    cat "$RUN_DIR/host/run_manifest.log" >&2 || true
    echo "PHASE_RESULT=SERVE_ONLY run_id=$RUN_ID run_dir=$RUN_DIR port=$PORT chip=$CHIP"
    exit 0
fi

if ((WARMUP_REQUESTS > 0)); then
    # Same shapes as the measured workload, so the first measured step does not
    # carry torch.compile / NPU graph capture.  Measured cost of that capture on
    # 0.8B + 128-token prompt: ~23 s, all of it inside `forward`.
    note "warmup: $WARMUP_REQUESTS request(s) with the measured shape, before /start_profile"
    mkdir -p "$RUN_DIR/requests/warmup"
    set +e
    run_workload "$RUN_DIR/requests/warmup" "${RUN_ID}-warmup" \
        "$RUN_DIR/requests/warmup/client.stdout.txt" \
        "$RUN_DIR/requests/warmup/client.stderr.txt"
    WARMUP_RC=$?
    set -e
    printf 'warmup_client_rc=%s\n' "$WARMUP_RC" >"$RUN_DIR/requests/warmup/client.rc.txt"
    note "warmup finished rc=$WARMUP_RC"
fi

# ------------------------------------------------------------- profile window
if [[ $PHASE_TIMING == on ]]; then
    note "POST /start_profile"
    curl -s --max-time 300 -X POST "$BASE_URL/start_profile" -w '\nhttp_code=%{http_code}\n' \
        >"$RUN_DIR/host/start_profile.txt" || true
    sleep 2
else
    # With no profiler configured and no LiteProfiler armed, EngineCore.profile()
    # falls through to model_executor.profile(), and the worker raises
    # "Profiler is not enabled".  So the window simply is not opened.
    note "phase timing OFF: skipping /start_profile (would raise 'Profiler is not enabled')"
fi

note "workload: requests=$REQUESTS concurrency=$CONCURRENCY prompt~$PROMPT_TOKENS max_tokens=$MAX_TOKENS"
set +e
run_workload "$RUN_DIR/requests" "$RUN_ID" \
    "$RUN_DIR/requests/client.stdout.txt" "$RUN_DIR/requests/client.stderr.txt"
CLIENT_RC=$?
set -e
printf 'client_rc=%s\n' "$CLIENT_RC" >"$RUN_DIR/requests/client.rc.txt"
note "workload finished rc=$CLIENT_RC"

if [[ $PHASE_TIMING == on ]]; then
    note "POST /stop_profile"
    curl -s --max-time 300 -X POST "$BASE_URL/stop_profile" -w '\nhttp_code=%{http_code}\n' \
        >"$RUN_DIR/host/stop_profile.txt" || true
    sleep 2
fi

# ---------------------------------------------------------------- collection
sudo -n docker logs --timestamps "$CONTAINER" >"$RUN_DIR/host/docker-logs.txt" 2>&1 || true
if [[ -f $HOST_LITE_LOG ]]; then
    wc -l <"$HOST_LITE_LOG" >"$RUN_DIR/lite-profiler/lite.log.lines.txt"
    sha256sum "$HOST_LITE_LOG" >"$RUN_DIR/lite-profiler/lite.log.sha256"
    head -5 "$HOST_LITE_LOG" >"$RUN_DIR/lite-profiler/lite.log.head.txt"
    python3 "$SCRIPT_DIR/parse_phase_timing.py" "$HOST_LITE_LOG" \
        --json "$RUN_DIR/lite-profiler/phase_timing.json" \
        --csv "$LITE_DIR/phase_timing.csv" \
        >"$RUN_DIR/lite-profiler/phase_timing.txt" 2>"$RUN_DIR/lite-profiler/phase_timing.stderr.txt" \
        || note "WARNING: phase timing parse failed"
else
    note "WARNING: $HOST_LITE_LOG not found"
fi
PYSTACK_CSV="$LITE_DIR/pystack.csv"
if [[ -f $PYSTACK_CSV ]]; then
    wc -l <"$PYSTACK_CSV" >"$RUN_DIR/lite-profiler/pystack.rows.txt"
    ls -l "$PYSTACK_CSV" | awk '{print $5}' >"$RUN_DIR/lite-profiler/pystack.bytes.txt"
    sha256sum "$PYSTACK_CSV" >"$RUN_DIR/lite-profiler/pystack.sha256"
    note "pystack: $(cat "$RUN_DIR/lite-profiler/pystack.rows.txt") rows, \
$(cat "$RUN_DIR/lite-profiler/pystack.bytes.txt") bytes"
else
    note "pystack.csv absent (sampler never fired, or phase timing was off)"
fi

sudo -n docker inspect "$CONTAINER" >"$RUN_DIR/host/docker_inspect.final.json"

STOP_T0=$(date +%s)
cleanup
STOP_T1=$(date +%s)

python3 - "$RUN_DIR" "$T0" "$READY_EPOCH" "$STOP_T1" "$CLIENT_RC" "$CONTAINER_ID" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
t0, ready, stop, client_rc, cid = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                                   int(sys.argv[5]), sys.argv[6])
manifest_path = run_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest["timing"] = {
    "container_create_epoch": t0,
    "ready_epoch": ready,
    "finished_epoch": stop,
    "startup_s": ready - t0,
    "total_s": stop - t0,
}
manifest["result"] = {"client_rc": client_rc, "container_id": cid}
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"manifest updated: {manifest_path}")
PY

(cd "$RUN_DIR" && find . -type f ! -name 'artifacts.sha256' -print0 | sort -z \
    | xargs -0 -r sha256sum) >"$RUN_DIR/artifacts.sha256"

# Machine-readable hand-off for anyone who has to profile this run (perf,
# libkperfx, msprof).  Non-fatal: the run itself succeeded either way.
python3 "$SCRIPT_DIR/run_manifest.py" --run-id "$RUN_ID" \
    --project-root "$PROJECT_ROOT" >"$RUN_DIR/host/run_manifest.log" 2>&1 || \
    echo "WARNING: run_manifest.py failed, see $RUN_DIR/host/run_manifest.log" >&2

echo "PHASE_RESULT=DONE run_id=$RUN_ID run_dir=$RUN_DIR port=$PORT chip=$CHIP client_rc=$CLIENT_RC"
