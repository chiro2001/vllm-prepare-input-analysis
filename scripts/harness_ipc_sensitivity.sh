#!/usr/bin/env bash
# 在 a3-22 上测：**引擎启动参数（决定持久 buffer 形状）如何改变 IPC**。
#
# 这是"为什么必须匹配真机口径"的定量证据：同一个 prepare_input 代码，
# 只换 preset，工作集从 64KB(L1/L2) 涨到 67MB(DRAM)，IPC 与 stall 分布都会变。
#
# 用法（在 a3-22 上，需要 sudo -n 读 PMU）：
#   bash scripts/harness_ipc_sensitivity.sh [--seconds 20] [--out data/harness/ipc_sensitivity.csv]
set -euo pipefail

REPO="${REPO:-$HOME/projects/vllm/prepare-input-phase}"
SECONDS_PER_RUN="${PI_IPC_SECONDS:-20}"
OUT="${PI_IPC_OUT:-$REPO/data/harness/ipc_sensitivity.csv}"
IMG="${PI_IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler}"
CPUS="${PI_CPUS:-200-215}"

source "$REPO/harness/scripts/pi_env.sh"
pi_env_selfcheck

EVENTS="cycles,instructions,branches,branch-misses,L1-dcache-load-misses,LLC-load-misses"

mkdir -p "$(dirname "$OUT")"
if [[ ! -f $OUT ]]; then
  echo "preset,batch,isl,max_model_len,max_num_reqs,max_num_batched_tokens,batches_per_step,cycles,instructions,ipc,branch_miss_pct,L1d_miss_pct,note" > "$OUT"
fi

# preset -> extra args
run_one() {
  local preset="$1"; shift
  local extra="$*"
  local tag="ipc_${preset}_$(date +%H%M%S)"

  echo "=== preset=$preset extra='${extra}' ==="
  # 容器 STOP 门控：先起进程并停住，perf 挂上后再放行 —— 短负载也不会丢头
  local cid
  cid=$(docker run -d --rm --name "$tag" --network none --cpuset-cpus "$CPUS" \
      --user "$(id -u):$(id -g)" "${PI_DOCKER_ENVS[@]}" \
      -v "$REPO:/work" -v "${PI_MODELS_DIR}:/models:ro" -w /work "$IMG" \
      # 必须用 --steady-seconds（按墙钟跑）：用 --steps 的话负载跑完就退出，
      # 而 realmachine 预设只有 1 个请求、~200 步就结束 —— 这正是第一次失败的原因。
      bash -c "kill -STOP \$\$; exec python3 -m pi_harness.runner._replay_main \
        --preset $preset $extra --steady-seconds 600 --no-timing")
  sleep 3
  local pid
  pid=$(docker inspect -f '{{.State.Pid}}' "$tag")
  # 放行（容器 init 被 STOP，需要 CONT）
  kill -CONT "$pid" 2>/dev/null || true
  # 等它真的进入稳态
  sleep 40
  if [[ ! -d /proc/$pid ]]; then
    echo "  ! process exited early; see: docker logs $tag" >&2
    docker logs "$tag" 2>&1 | tail -5 >&2
    docker rm -f "$tag" >/dev/null 2>&1 || true
    return 1
  fi
  local stat
  stat=$(sudo -n perf stat -p "$pid" -e "$EVENTS" --timeout "$((SECONDS_PER_RUN * 1000))" 2>&1 || true)
  docker rm -f "$tag" >/dev/null 2>&1 || true

  local cyc inst br brm l1d llc
  cyc=$(echo "$stat"  | awk '/cycles/        {gsub(/,/,"",$1); print $1; exit}')
  inst=$(echo "$stat" | awk '/instructions/  {gsub(/,/,"",$1); print $1; exit}')
  br=$(echo "$stat"   | awk '/branches/      {gsub(/,/,"",$1); print $1; exit}')
  brm=$(echo "$stat"  | awk '/branch-misses/ {gsub(/,/,"",$1); print $1; exit}')
  l1d=$(echo "$stat"  | awk '/L1-dcache-load-misses/ {gsub(/,/,"",$1); print $1; exit}')
  llc=$(echo "$stat"  | awk '/LLC-load-misses/       {gsub(/,/,"",$1); print $1; exit}')
  local ipc bmp l1p
  ipc=$(python3 -c "print(f'{$inst/$cyc:.4f}' if $cyc else 'nan')")
  bmp=$(python3 -c "print(f'{$brm/$br*100:.3f}' if $br else 'nan')")
  l1p=$(python3 -c "print(f'{$l1d/$cyc*1000:.3f}' if $cyc else 'nan')")

  echo "  cycles=$cyc instructions=$inst ipc=$ipc branch_miss=${bmp}% L1d_miss/kc=$l1p LLC=$llc"
  echo "$preset,-,-,-,-,-,-,$cyc,$inst,$ipc,$bmp,$l1p,\"$extra\"" >> "$OUT"
  # 完整原始输出留档
  echo "$stat" > "$REPO/data/harness/ipc_${preset}_perfstat.txt"
}

run_one realmachine ""
run_one stress ""
run_one modelmax ""

echo
echo "wrote $OUT"
column -s, -t < "$OUT" 2>/dev/null || cat "$OUT"
