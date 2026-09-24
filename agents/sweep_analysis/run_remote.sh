#!/usr/bin/env bash
# sweep_analysis 补充测量驱动（无卡 harness）。
#
# 唯一合法入口：
#   ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh \
#       "cd /work/harness && PI_MODEL_TMP=/tmp/pi_models python -m pi_harness.runner.cli <args>"'
#
# 用法：
#   bash agents/sweep_analysis/run_remote.sh detail   # 每配置 per-step/substep/summary dump（1 rep）
#   bash agents/sweep_analysis/run_remote.sh rep3     # 同 sweep 网格的 --repeat 3 复测（长表 CSV）
#   bash agents/sweep_analysis/run_remote.sh mode     # cpu_fallback / noop / inject:15 三模式对照
#   bash agents/sweep_analysis/run_remote.sh isl_b1   # batch=1 的 ISL 对照（隔离 chunk 摊平效应）
#
# 所有产物写入 a3-22:data/harness/sweep_*（本 agent 的写入范围）。
set -u

PHASE="${1:-detail}"
REMOTE_ROOT='~/projects/vllm/prepare-input-phase'
OUTDIR='/work/data/harness'
COMMON='--osl 128 --has-gdn --max-num-reqs 64 --max-num-batched-tokens 16384'
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

run_one() {  # $1 = cli args, $2 = tag
  local args="$1" tag="$2"
  local t0 t1 logf
  logf="$LOGDIR/$tag.log"
  t0=$(date +%s)
  printf '[run] %s :: %s\n' "$tag" "$args"
  printf '[run] %s :: %s\n' "$(date -Is)" "$args" >>"$logf"
  ssh a3-22 "cd $REMOTE_ROOT && bash harness/scripts/pi-docker.sh 'cd /work/harness && PI_MODEL_TMP=/tmp/pi_models python -m pi_harness.runner.cli $args --out $OUTDIR --tag $tag'" \
    >>"$logf" 2>&1 \
    || { echo "[run] FAILED $tag"; tail -5 "$logf"; return 1; }
  t1=$(date +%s)
  grep -E '^\[cli\] wrote' "$logf" | tail -3 | sed 's/^/  /'
  printf '[run] ok %s (%ss)\n' "$tag" "$((t1 - t0))"
}

# 基准点 = batch 16 / isl 1024 / osl 128 / block 128 / chunk 16384 / spec 0 / prefix 0
# （同时也是 isl=1024 / batch=16 / blocksize=128 / spec_k=0 / prefix=0.0 的取值点）
BASE='--batch 16 --isl 1024 --steps 80'

# 裁剪后的 detail 网格：每个维度取 3-4 个最有信息量的点 + block_size 全覆盖 + 基准点。
# 每次 CLI 调用有 ~40s 固定开销（import vllm + 按 max_model_len 分配大 buffer），
# 因此不做全网格 per-step dump；其余取值点用 triton 校准模型外推（见 REPORT §3）。
detail() {
  run_one "$BASE $COMMON" "sweep_d_base"
  for v in 512 2048 8192; do
    run_one "--batch 16 --isl $v --steps 80 $COMMON" "sweep_d_isl_$v"
  done
  for v in 1 2 8 64; do
    run_one "--batch $v --isl 1024 --steps 80 $COMMON" "sweep_d_batch_$v"
  done
  for v in 128 1024 8192; do
    run_one "--batch 16 --isl 1024 --chunk-size $v --steps 80 $COMMON" "sweep_d_chunk_$v"
  done
  for v in 32 64 256; do
    run_one "--batch 16 --isl 1024 --block-size $v --steps 80 $COMMON" "sweep_d_blocksize_$v"
  done
  for v in 1 3; do
    run_one "--batch 16 --isl 1024 --spec-k $v --steps 80 $COMMON" "sweep_d_spec_$v"
  done
  for v in 0.5 1.0; do
    run_one "--batch 16 --isl 1024 --prefix-hit-ratio $v --steps 80 $COMMON" "sweep_d_prefix_$v"
  done
}

rep3() {
  local dim v
  for dim in isl batch chunk_size block_size spec_k prefix_hit_ratio; do
    case "$dim" in
      isl) vals='128 512 1024 2048 4096 8192'; extra='' ;;
      batch) vals='1 2 4 8 16 32 64'; extra='--isl 1024' ;;
      chunk_size) vals='128 256 512 1024 2048 8192'; extra='--isl 1024' ;;
      block_size) vals='32 64 128 256'; extra='--isl 1024' ;;
      spec_k) vals='0 1 2 3'; extra='--isl 1024' ;;
      prefix_hit_ratio) vals='0.0 0.25 0.5 0.75 1.0'; extra='--isl 1024' ;;
    esac
    case "$dim" in
      isl) run_one "--sweep isl --batch 16 --steps 80 --repeat 3 $COMMON" "sweep_r3" ;;
      batch) run_one "--sweep batch --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3" ;;
      chunk_size) run_one "--sweep chunk_size --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3" ;;
      block_size) run_one "--sweep block_size --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3" ;;
      spec_k) run_one "--sweep spec_k --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3" ;;
      prefix_hit_ratio) run_one "--sweep prefix_hit_ratio --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3" ;;
    esac
  done
}

mode() {
  run_one "--batch 16 --isl 1024 --steps 200 --slot-mapping-mode cpu_fallback $COMMON" "sweep_mode_cpu"
  run_one "--batch 16 --isl 1024 --steps 200 --slot-mapping-mode noop $COMMON" "sweep_mode_noop"
  run_one "--batch 16 --isl 1024 --steps 200 --slot-mapping-mode inject:15 $COMMON" "sweep_mode_inj15"
}

isl_b1() {
  local v
  for v in 512 2048 8192; do
    run_one "--batch 1 --isl $v --steps 80 $COMMON" "sweep_d_islb1_$v"
  done
}

push_code() {
  rsync -a --include='*/' --include='*.py' --exclude='*' \
    "$HERE/" a3-22:projects/vllm/prepare-input-phase/agents/sweep_analysis/
}

# grid：分析侧网格驱动（含 triton 列 / substep / meta），一个进程跑完一个维度
grid() {
  local dims="${1:-isl,batch,chunk_size,block_size,spec_k,prefix_hit_ratio}"
  local extra="${2:-}"
  local cpus="${3:-}"
  local cpusarg=""
  [[ -n "$cpus" ]] && cpusarg="--cpus $cpus"
  push_code
  ssh a3-22 "cd $REMOTE_ROOT && bash harness/scripts/pi-docker.sh $cpusarg \
    'cd /work/harness && PI_MODEL_TMP=/tmp/pi_models PYTHONPATH=/work/harness:/work/agents/sweep_analysis \
     python /work/agents/sweep_analysis/sweep_grid.py --dims $dims --steps 80 --repeat 1 $extra \
     --out /work/data/harness'" 2>&1 | tee -a "$LOGDIR/grid.txt" | grep -E '^\[grid\]'
}

# probe：子步骤归因探针（wrap / ccall）
probe() {
  local mode="${1:-wrap}" label="${2:-base}" extra="${3:---batch 16 --isl 1024}"
  push_code
  ssh a3-22 "cd $REMOTE_ROOT && bash harness/scripts/pi-docker.sh \
    'cd /work/harness && PI_MODEL_TMP=/tmp/pi_models PYTHONPATH=/work/harness:/work/agents/sweep_analysis \
     python /work/agents/sweep_analysis/probe_substeps.py --mode $mode --label $label $extra \
     --steps 40 --out /work/data/harness'" 2>&1 | tee "$LOGDIR/probe_$label.txt" | grep -E '^\[probe\]|^   '
}

noise_gate() {
  local when="$1"
  # 注意：--json 必须是 **宿主机** 路径（gate 在宿主机上跑，不是容器里）
  ssh a3-22 "cd $REMOTE_ROOT && bash scripts/hostnoise_gate.sh --cpus 200-215 --sample-s 3 \
      --json data/harness/sweep_noise_${when}.json" 2>&1 | tee "$LOGDIR/noise_$when.txt"
}

# phase2：repeat-3 复测（跨轮中位数 + min/max 离散度）+ 噪声基线 + 三模式对照
phase2() {
  noise_gate pre
  for dim in isl batch chunk_size block_size spec_k prefix_hit_ratio; do
    case "$dim" in
      isl)             run_one "--sweep isl --batch 16 --steps 80 --repeat 3 $COMMON" "sweep_r3_isl" ;;
      batch)           run_one "--sweep batch --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3_batch" ;;
      chunk_size)      run_one "--sweep chunk_size --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3_chunk" ;;
      block_size)      run_one "--sweep block_size --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3_block" ;;
      spec_k)          run_one "--sweep spec_k --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3_spec" ;;
      prefix_hit_ratio) run_one "--sweep prefix_hit_ratio --isl 1024 --steps 80 --repeat 3 $COMMON" "sweep_r3_prefix" ;;
    esac
  done
  noise_gate post
  mode
  noise_gate final
}

# phase3：主力网格 + warmup0 对照 + 安静核复现 + 交错三模式 + 子步骤探针
phase3() {
  # a) 六个维度全网格（warmup=3，与 harness sweep 网格一致），逐步池化
  grid "isl,batch,chunk_size,block_size,spec_k,prefix_hit_ratio" \
       "--dump-steps --pool-name sweep_steps_pool.csv --tag sweep_grid" ""
  # b) warmup=0：把 prefill 步留在测量窗口内（ISL / chunk 的摊平效应）
  grid "isl,chunk_size" \
       "--warmup 0 --dump-steps --pool-name sweep_steps_pool_w0.csv --tag sweep_w0" ""
  # c) 同样的维度在 hostnoise gate 推荐的"安静核"上复现
  grid "isl,batch,chunk_size" \
       "--dump-steps --pool-name sweep_steps_pool_quiet.csv --tag sweep_gridq" "201-207,209,211-213,215"
  # d) 三模式交错 2 轮（避免整段噪声漂移污染某一个模式）
  interleaved_modes 2
  # e) 子步骤归因
  probe wrap base "--batch 16 --isl 1024"
  probe ccall base "--batch 16 --isl 1024"
}

interleaved_modes() {
  local rounds="${1:-2}" r
  for r in $(seq 1 "$rounds"); do
    run_one "--batch 16 --isl 1024 --steps 200 --slot-mapping-mode noop $COMMON" "sweep_mode_r${r}_noop"
    run_one "--batch 16 --isl 1024 --steps 200 --slot-mapping-mode cpu_fallback $COMMON" "sweep_mode_r${r}_cpu"
    run_one "--batch 16 --isl 1024 --steps 200 --slot-mapping-mode inject:15 $COMMON" "sweep_mode_r${r}_inj15"
  done
}

# phase5：补 chunk 网格 + warmup0 + 安静核 + 交错三模式（分析侧驱动，绕开 CLI 回归）
phase5() {
  push_code
  # chunk_size：修掉 sweep_grid.py 里 RunnerConfig(chunk_size=...) 的写法后重跑
  grid "chunk_size" "--dump-steps --pool-name sweep_steps_pool.csv --tag sweep_grid" "" 
  grid "chunk_size" "--warmup 0 --dump-steps --pool-name sweep_steps_pool_w0.csv --tag sweep_w0" ""
  grid "isl,batch" "--warmup 0 --dump-steps --pool-name sweep_steps_pool_w0.csv --tag sweep_w0" ""
  grid "chunk_size" "--dump-steps --pool-name sweep_steps_pool_quiet.csv --tag sweep_gridq" \
       "201-207,209,211-213,215"
  # 同一网格在不同时间窗再跑 2 轮（跨轮离散度）
  grid "isl,batch,block_size,spec_k,prefix_hit_ratio" \
       "--dump-steps --pool-name sweep_steps_pool.csv --tag sweep_grid" ""
  grid "isl,batch,block_size,spec_k,prefix_hit_ratio" \
       "--dump-steps --pool-name sweep_steps_pool.csv --tag sweep_grid" ""
  # 交错三模式
  ssh a3-22 "cd $REMOTE_ROOT && bash harness/scripts/pi-docker.sh \
    'cd /work/harness && PI_MODEL_TMP=/tmp/pi_models PYTHONPATH=/work/harness:/work/agents/sweep_analysis \
     python /work/agents/sweep_analysis/mode_compare.py --rounds 3 --steps 200 \
     --out /work/data/harness --tag sweep_modecmp'" 2>&1 \
    | tee "$LOGDIR/modecmp.txt" | grep -E '^\[mode\]|^   '
}

# pgrid：preset 感知扫描（分析侧驱动，绕开 CLI 回归）
#   pgrid <preset> <dims> <extra-args> [cpus]
pgrid() {
  local preset="${1:-realmachine}" dims="${2:-batch,isl}" extra="${3:-}" cpus="${4:-}"
  local cpusarg=""
  [[ -n "$cpus" ]] && cpusarg="--cpus $cpus"
  push_code
  ssh a3-22 "cd $REMOTE_ROOT && bash harness/scripts/pi-docker.sh $cpusarg \
    'cd /work/harness && PI_MODEL_TMP=/tmp/pi_models PYTHONPATH=/work/harness:/work/agents/sweep_analysis \
     python /work/agents/sweep_analysis/preset_grid.py --preset $preset --dims $dims $extra \
     --out /work/data/harness'" 2>&1 | tee -a "$LOGDIR/pgrid.txt" | grep -E '^\[pgrid\]'
}

# phase-rm：真机口径（第一优先）
phase_rm() {
  # 1) 真机口径 batch 扫描（max_num_reqs=8 自然把 batch 卡在 8；这里只跑 1,2,4,8）
  pgrid realmachine "batch" "--values 1,2,4,8 --steps 200 --repeat 3 --dump-steps --tag rm" ""
  # 2) 真机口径 isl 扫描（warmup=3：窗口是 decode；ISL 只改 token_ids_cpu 宽度）
  pgrid realmachine "isl" "--values 64,128,256,512,1024 --steps 200 --repeat 3 --dump-steps --tag rm" ""
  # 2b) 同网格 warmup=0：把 prefill 步留在窗口内，才看得到 ISL 对 chunked prefill 的作用
  pgrid realmachine "isl" "--values 128,512,1024,2048 --warmup 0 --steps 200 --repeat 3 --dump-steps --tag rmw0" ""
  # 3) 真机口径但放开 max_num_reqs=64（保留 MML=2048 这个关键值）看 batch 上限
  pgrid realmachine "batch" "--values 1,8,16,32,64 --max-num-reqs 64 --steps 200 --repeat 3 --dump-steps --tag rm64" ""
  # 4) 真机口径 (noop) steady-seconds 长跑，取 ~27k 步复现基线
  pgrid realmachine "batch" "--values 1 --steps 40000 --steady-seconds 12 --repeat 1 --dump-steps --tag rmsteady" ""
}

# phase-noise：realmachine 口径 noop vs cpu_fallback 占比对照
phase_noisemode() {
  pgrid realmachine "batch" "--values 1 --steps 200 --repeat 3 --slot-mapping-mode cpu_fallback --tag rmcpu" ""
  pgrid realmachine "batch" "--values 1 --steps 200 --repeat 3 --slot-mapping-mode noop --tag rmnoop" ""
  pgrid realmachine "batch" "--values 1 --steps 200 --repeat 3 --slot-mapping-mode inject:15 --tag rminj" ""
  # 关键：三个模式**交错**跑（消除整段窗口漂移），realmachine 口径
  push_code
  ssh a3-22 "cd $REMOTE_ROOT && bash harness/scripts/pi-docker.sh \
    'cd /work/harness && PI_MODEL_TMP=/tmp/pi_models PYTHONPATH=/work/harness:/work/agents/sweep_analysis \
     python /work/agents/sweep_analysis/mode_compare.py --preset realmachine --rounds 3 --steps 200 \
     --batch 1 --isl 128 --osl 64 --out /work/data/harness --tag sweep_modecmp_rm'" 2>&1 \
    | tee "$LOGDIR/modecmp_rm.txt" | grep -E '^\[mode\]|^   '
}

# phase-stress：stress 口径重跑原来 6 个维度
phase_stress() {
  pgrid stress "isl,batch,chunk_size,block_size,spec_k,prefix_hit_ratio" \
        "--steps 80 --repeat 1 --dump-steps --tag stress" ""
}

case "$PHASE" in
  detail) detail ;;
  base) run_one "$BASE $COMMON" "sweep_d_base" ;;
  phase2) phase2 ;;
  noise) noise_gate "${2:-manual}" ;;
  rep3) rep3 ;;
  mode) mode ;;
  isl_b1) isl_b1 ;;
  grid) grid "${2:-isl,batch,chunk_size,block_size,spec_k,prefix_hit_ratio}" "${3:-}" ;;
  probe) probe "${2:-wrap}" "${3:-base}" "${4:---batch 16 --isl 1024}" ;;
  phase3) phase3 ;;
  phase5) phase5 ;;
  pgrid) pgrid "${2:-realmachine}" "${3:-batch,isl}" "${4:-}" "${5:-}" ;;
  phase_rm) phase_rm ;;
  phase_noisemode) phase_noisemode ;;
  phase_stress) phase_stress ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac
echo "[run] phase $PHASE done"
