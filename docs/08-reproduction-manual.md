# 完整复现手册

> 从零把本项目的全部结果跑出来。所有命令都在 a3-22（Kunpeng 920B + Ascend A3）上验证过。
> 目录约定：`R=~/projects/vllm/prepare-input-phase`（a3-22），
> 本地同构目录 `/home/chiro/projects/vllm/preparing-input-phase`。

## 0. 环境与前置

| 项 | 值 |
|---|---|
| 硬件 | Kunpeng 920B：4 socket × 80 core × 2 SMT = 640 逻辑核；8 NUMA node；16 Ascend 910 die |
| 本次用卡 | **chip3** = `/dev/davinci3` = `npu-smi -i 1 -c 1` |
| CPU 切片 | `120-159`（NUMA node 1）：worker `122-157`、acl `158`、release `159` |
| 权限 | `sudo -n` 免密；可跑 privileged docker；`perf_event_paranoid=2` |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler`（基线）<br>`local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler`（+LiteProfiler patch） |
| 软件 | vLLM 0.26.0 `568afb3a1`；vllm-ascend 0.26.0rc1 `f2f74a16c`；python 3.12.13；CANN 9.1.0 |

**常用坑（省你几小时）**

1. `npu-smi info -t proc-mem -i 3` **不是** chip3——那是 NPU3（phy 6/7）。chip3 必须 `-i 1 -c 1`。
2. `docker inspect -f {{.State.Pid}}` 返回的是 APIServer（pid 1），**不是** EngineCore。
   定位 EngineCore 用 `npu-smi info -t proc-mem -i 1 -c 1`（uni executor 下它报告的宿主 PID
   就是 EngineCore 主线程 = 主 TID）。
3. `perf -t` / libkperfx 的 target 必须是**忙线程的宿主 TID**；传进程组长会得到 `<not counted>`。
4. 容器内 uid 无 passwd 条目会让 `torch.inductor` 崩（`getpwuid(): uid not found`）。
   起容器时补 `-e USER/-e LOGNAME/-e TORCHINDUCTOR_CACHE_DIR/-e TRITON_CACHE_DIR`。
5. **CPU `120-159` 不独占**（宿主机有别人的进程 affinity 覆盖全核）。所有 wall 数字必须
   配 `hostnoise_gate.sh` 快照，PMU 一律 per-thread。
6. `POST /start_profile` 会**无条件**启动 1 ms 的 pystack 采样器（+20% prepare_input / +60% ITL）。
   做占比测量必须显式设 `--pystack-interval-us 0`。
7. async scheduling 下每轮 engine 循环写**两行** `Step:Model`（dispatch + batch_queue wait）。
   算占比必须用 `--step-scope Step:Schedule --window-mode next`，或直接数 `prepare input` 行。

## 1. 拿锁（chip3 串行）

```bash
ssh a3-22
R=~/projects/vllm/prepare-input-phase
bash $R/scripts/chip3_lock.sh acquire --purpose "why"   # mkdir 语义
bash $R/scripts/chip3_lock.sh status
bash $R/scripts/hostnoise_gate.sh --out $R/data/hostnoise-$(date -u +%FT%H%M%SZ).json
# ... 干活 ...
bash $R/scripts/chip3_lock.sh release
```

## 2. 真机服务 + phase 计时

```bash
# 2.1 基线臂（不插桩），跑一轮 workload 后销毁容器
bash $R/scripts/launch_subscope_service.sh \
  --mode baseline --pystack-interval-us 0 \
  --run-id demo-baseline -- \
  --model /home/REMOTE_USER/models/Qwen3.5-0.8B --served-name qwen35-08b \
  --max-num-seqs 1 --max-model-len 2048 --max-num-batched-tokens 2048 \
  --cudagraph-mode FULL_DECODE_ONLY \
  --isel 128 --osel 1024 --concurrency 1

# 2.2 只起服务、不跑 workload（给 perf/libkperfx 用）
bash $R/scripts/launch_subscope_service.sh --mode baseline --serve-only \
  --run-id demo-prof -- ...同上...
cat $R/runs/demo-prof/run_manifest.json     # ← engine_core.pid/.tid 在这里

# 2.3 解析 phase 计时
python3 $R/scripts/measure/analyze_lite.py \
  $R/runs/demo-baseline/lite-profiler/lite.log \
  --step-scope Step:Schedule --window-mode next \
  --out $R/data/measure/demo-baseline
```

关键环境变量（启动器已设好，手工起服务时别漏）：
`MSMONITOR_USE_DAEMON=0`、`TASK_QUEUE_ENABLE=1`、`OMP_NUM_THREADS=1`、
`ASCEND_RT_VISIBLE_DEVICES=3`、`VLLM_LITE_PROFILER_LOG_PATH=/runmeta/lite-profiler/lite.log`。

## 3. 子阶段插桩（回答"prepare_input 里谁最贵"）

```bash
# 3.1 应用插桩（把 patched 文件用 -v 挂进容器，不重建镜像）
bash $R/instrument/apply.sh          # 校验 sha256 + AST 等价性
bash $R/instrument/verify_patch.py   # 应为 PASS

# 3.2 跑（PI_SUBSCOPE=on 打开探针）
bash $R/scripts/launch_subscope_service.sh --mode on --pid-host on \
  --run-id demo-subscope -- ...上面的服务参数...

# 3.3 解析
python3 $R/scripts/parse_subscope.py $R/runs/demo-subscope/pi-subscope/raw.csv \
  --probe-floor-us 2.2 --out $R/data/subscope/demo-subscope

# 3.4 还原
bash $R/instrument/revert.sh
```

**读数纪律**（血的教训）：

* `pi: am.builder_build@<ClassName>` 才能区分是哪个 builder；只看 `pi: am.builder_build`
  会把 3 次 GDN 调用和 1 次 full-attn 调用混在一起，得出"每次 300 µs"的错误结论。
* 探针自身有 **2.2 µs/个** 的地板；p50 低于地板的子项不可引用
  （`parse_subscope.py` 会自动列进 `meta.json.scopes_below_probe_floor`）。
* `pi_note` 的诊断值必须用 `lazy(fn)` 延迟求值，否则探针假设与实际对象不符时会
  **把引擎跑挂**（`AttributeError: 'AscendGDNAttentionMetadataBuilder' object has no attribute 'layer_names'`）。

## 4. 火焰图（flamegraph-rs）

```bash
TID=$(python3 -c "import json;print(json.load(open('$R/runs/demo-prof/run_manifest.json'))['engine_core']['tid'])")

# 4.1 采集（dwarf 是因为需要 python/numpy 栈；只采该 TID）
bash $R/scripts/flamegraph.sh record --tid $TID --seconds 20 \
  --out $R/data/profiles/demo/flame.data

# 4.2 出图（tools/bin 下是 musl 静态的 flamegraph/inferno-*）
bash $R/scripts/flamegraph.sh render \
  --data $R/data/profiles/demo/flame.data \
  --out  $R/figures/05-flame-demo.svg

# 4.3 热点表 + children 分解
sudo perf report -i $R/data/profiles/demo/flame.data --stdio --no-children -g none \
  | head -40 > $R/data/profiles/demo/hotspots.txt
sudo perf report -i $R/data/profiles/demo/flame.data --stdio --children \
  | grep -A 30 'gdn_attn_builder' > $R/data/profiles/demo/gdn.children.txt
```

采样率固定 `-F 999`（a3-22 `max_sample_rate=100000`，不会节流；跨机可比）。

## 5. topdown / IPC（libkperfx）

```bash
cd $R/tools/libkperfx
TID=...   # 同上，必须是忙线程的宿主 TID

# 5.1 单组快速看（3 s 窗口）
sudo KPERFX_TARGET=$TID ./bin/kperfx run --preset topdown --window 3 --json ../../data/profiles/demo/kperfx-group.json

# 5.2 9 组全覆盖（一次完整扫描 ≈ 9×window + 1.4 s）
python3 $R/scripts/measure/pmu_tid_sweep.py --tid $TID --groups 1-9 --window 3 \
  --out $R/data/profiles/demo/
```

**必须引用置信度**：920B 单组上限 **8** 个计数器，`confidence = min(1, 8/n)`。
`confidence < 0.99` 必须重跑或标注。另外本机 `memstall_l3miss` 与 `dram_*` 恒为 0，
所以 `mem_l3_dram_bound` 实际承载 "L2 以下全部 stall"，**不要写成 "DRAM bound = 0%"**。

## 6. 无卡复现（不需要任何 NPU 设备）

```bash
# 6.1 必须用 realmachine preset（逐字对齐真机启动参数，否则 IPC 1.553 vs 0.949）
bash $R/harness/scripts/pi-docker.sh --preset realmachine \
  --batch 1 --isl 128 --osl 64 --steady-seconds 12

# 6.2 参数扫描
bash $R/harness/scripts/pi-docker.sh --preset realmachine --sweep batch:1,8,32,64,128

# 6.3 看子步骤分解
cat $R/data/harness/substep/tmp_bs16b/summary.json
```

harness 绑定 CPU `200-215`（**绝不能用 120-159**），`--network none`，不挂 `/dev/davinci*`。
已知偏差在 `docs/06-synthetic-load.md` §5 的 D1–D10 表里逐条登记。

## 7. 打包上传

```bash
bash $R/scripts/package_and_upload.sh --tag $(date -u +%Y%m%dT%H%M%SZ)
# → cos://COS_BUCKET/share/prepare-input-cpu-analysis-<tag>.tar.gz
# → 自动登记到 http://LINKS_IP:18080/
```

原始 `perf.data` **不进包**（体积原因），只留在 a3-22 `data/profiles/`。

## 8. 复现 checklist

- [ ] chip3 空闲（`npu-smi -i 1 -c 1` = `No process in device`）
- [ ] hostnoise 快照已存
- [ ] 镜像 digest 已记入 manifest
- [ ] `--pystack-interval-us 0`（占比测量）
- [ ] 占比分母口径已声明（`Step:Schedule` 还是 `Step:Model`）
- [ ] PMU `confidence >= 0.99`
- [ ] 插桩臂与基线臂都跑了（用于扣探针代价）
- [ ] 只挂 `/dev/davinci3`，退出后确认 chip3 回到空闲、端口释放
