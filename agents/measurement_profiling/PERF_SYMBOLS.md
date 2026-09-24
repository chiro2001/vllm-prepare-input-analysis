# perf 采集与符号化 —— 结论与操作手册

> 交付物：`scripts/measure/perf_capture.sh` + 本文。自测证据：
> `data/profiles/perf-selftest-20260923T180711Z/`（4 组采集，含 manifest / hotspots /
> srcline / instr / annotate / flame.svg；`perf.data` 只留在 a3-22，未上传）。
> 全部结论均在 a3-22 实测，命令可复制重跑。

## 结论速览（三行）

1. **真机主采集：`--call-graph fp`**，目标进程带 `PYTHONPERFSUPPORT=1`，容器用
   `--pid=host`（或 `--perf-map`）。15 s @999 Hz 只有 **5.0 MB**，C 层与 **Python 层**
   帧全都拿到（`py::main:/work/x.py` 这种帧，样本覆盖率 100%）。
2. **`--call-graph dwarf` 不要当默认**：同样 15 s @999 Hz 是 **128.7 MB**（约 fp 的
   **25.7×**，约 8.6 MB/s），且 **一条 Python 帧都没有**（0%）；只在 ≤60 s 的单点深挖
   时用，60 s 约 515 MB、170 s 就会越过 1.5 GB 红线。
3. **`py-spy` 只能当 Python 层旁证**：`--native` 在本镜像几乎不可用（211 成功 / 783 失败
   = 21%），去掉 `--native` 则 994 成功 / 0 失败（100%），但没有 C 层；`--symfs` 与本
   问题无关（叶子符号本来就 98.6% 已解析）。

---

## 1. 环境事实（决定了整个方案）

| 项 | 值 |
|---|---|
| 宿主机 | a3-22 / host22，Kunpeng 920B，640 核，openEuler 24.03，kernel 6.6.0-159.4.3.154 |
| perf | 6.6.0-159.4.13.167.oe2403sp4.aarch64（`perf_event_paranoid=2`，必须 `sudo -n`）|
| 镜像 | `local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler`，id `sha256:b2c6ff72ff93` |
| 镜像里的 python | **3.12.13**（`/usr/local/python3.12.13/bin/python3`，`vllm` 的 shebang 就指向它）|
| 宿主 python | 3.11.6（`/usr/lib64/libpython3.11.so.1.0`）——历史报告里那个 libpython3.11 是**宿主**的 |

两个容易踩的坑，先记住：

* **Python 3.12 的 perf trampoline**：`PYTHONPERFSUPPORT=1`（等价 `python3 -X perf`）
  会让 CPython 把每个 Python 帧写进 `/tmp/perf-<pid>.map`，perf 于是能打印
  `py::<函数名>:<文件>`。实测 `sys.is_stack_trampoline_active()` = True；
  `PYTHONPERFSUPPORT=0` 是关闭，未设置也是关闭。
* **容器 PID 命名空间会错名字**：map 文件名用的是进程自己的 `getpid()`，
  而宿主 perf 用**宿主 TID** 去找。自测里容器内 python 是 PID 1，map 就写成
  `/tmp/perf-1.map`，而 perf 找 `/tmp/perf-175935.map`。两种修法都实测过（见第 4 节）。

## 2. 实测矩阵（a3-22，CPU 200-215，单线程 CPU-bound 负载）

负载：`selftest_load.py`（纯 python dict/属性/字符串 + numpy + torch，单线程，
`LOAD_SECONDS` 控制时长），跑在 liteprofiler 镜像里；宿主 root 用
`perf record -t <host TID>` 采它 —— 这正是真机采集形态（宿主 perf 采容器内进程）。

| 目录 | 窗口 | call-graph | trampoline | perf.data | 样本 | `[unknown]`(self) | Python 帧样本覆盖 | 平均栈深 |
|---|---|---|---|---|---|---|---|---|
| `fp/` | 15 s | `fp` | 开 | **5,003,968 B (4.8 MiB)** | 15 028 | 1.36 % | **100.0 %** | 29.2 帧 |
| `dwarf/` | 15 s | `dwarf,8192` | 开 | **128,701,316 B (122.7 MiB)** | 14 991 | 1.24 % | **0.0 %** | 7.2 帧 |
| `fp_notramp/` | 8 s | `fp` | **关** | 2,228,300 B | 7 972 | 1.47 % | **0.0 %** | 19.2 帧 |
| `fp_nons/` | 8 s | `fp` | 开 | 2,896,380 B | 8 046 | 1.34 % | **100.0 %** | 29.2 帧 |

（`fp_nons` = 容器**没有** `--pid=host`，靠 `--perf-map /tmp/perf-1.map` 把 map 链到
`/tmp/perf-<host TID>.map`。）

### 2.1 怎么读这张表

* **`[unknown]`(self) 这个指标骗人**。它衡量的是采样点的叶子符号，而解释器样本的叶子
  永远落在 `libpython3.12.so`（有 symtab），所以四种配置都是 ~99% resolved。真正丢掉的
  是一整条**调用链**。要看的是 `symbol_quality.python_frames.samples_with_py_frame_pct`：
  **100% vs 0%**。两者差的就是「火焰图里能不能看到 Python 函数」。
* fp+trampoline 的典型栈（43 帧全解析，取自 `fp/folded.txt`）：

  ```
  python3;_start;__libc_start_main;[libc.so.6];Py_BytesMain;Py_RunMain;
  ...;py_trampoline_evaluator;py::<module>:/work/selftest_load.py;
  _PyEval_EvalFrameDefault;...;py::build:/work/selftest_load.py;
  PyObject_Vectorcall;PyObject_Malloc 2895339
  ```

* **dwarf 拿到的是原生栈**：它把 numpy/torch/openblas 的 C 栈挖得更深（能看到
  `PyArray_GetCastingImpl`、`npy_get_floatstatus_barrier`），但遇到 trampoline 帧就停，
  所以 **Python 函数一个都不出现**。想同时要 Python 帧 + 更深 C 栈，只能跑两遍。
* `folded.txt` 的权重是**周期数**（inferno-collapse-perf 用 perf script 里的 period），
  不是样本条数：15 s 的权重总和约 43 G cycles。做「样本占比」统计要么用 `hotspots.csv`
  的 `samples` 列，要么按本文方法直接数 perf script 的 sample block。

### 2.2 体积外推（关键约束）

| 窗口 | fp | dwarf,8192 |
|---|---|---|
| 15 s | 5.0 MB | 128.7 MB |
| 60 s | 约 20 MB | 约 515 MB |
| 300 s | 约 100 MB | **约 2.6 GB（超 1.5 GB 红线）** |

dwarf 大约在 **170 s** 处越过 1.5 GB，所以 `perf_capture.sh` 把 **fp 设为默认**；
要 dwarf 就自己显式传，并且窗口 ≤ 60 s。

## 3. 热点层（`fp/` 自测结果节选）

`hotspots.csv`（top 10，self overhead）：

```
rank,dso,symbol,self_pct,samples
1,libopenblas-24fd393f.so.0,sgemm_kernel_ARMV8SVE,17.93,2696
2,libpython3.12.so.1.0,gc_collect_main,12.48,1865
3,libpython3.12.so.1.0,_PyEval_EvalFrameDefault,9.62,1446
4,libpython3.12.so.1.0,visit_reachable,5.48,815
5,libpython3.12.so.1.0,visit_decref,4.95,745
6,libpython3.12.so.1.0,_PyObject_Malloc,4.40,662
7,libpython3.12.so.1.0,_PyObject_Free,2.19,330
8,libpython3.12.so.1.0,dictiter_iternextitem,1.93,290
9,libpython3.12.so.1.0,tupledealloc,1.48,223
10,libpython3.12.so.1.0,PyFloat_FromDouble,1.37,206
```

**fp 与 dwarf 的 self 热点一致**（前 6 名同序、占比差 ≤ 4.4 pp），说明「用 fp 拿热点排序」
这件事本身没有偏差，差别只在调用链。

`hotspots_srcline.csv`（源码行，top 4）与 `hotspots_instr.csv`（热点指令，top 4）：

```
1,pycore_gc.h:60,54.15,gc_collect_main
2,pycore_gc.h:59,31.32,gc_collect_main
3,object.h:646,16.01,_PyEval_EvalFrameDefault
4,generated_cases.c.h:4650,5.80,_PyEval_EvalFrameDefault

1,gc_collect_main,2ebc0c,"ldr     x4, [x3]",31.32
2,gc_collect_main,2ec698,"ldr     x0, [x0]",24.76
3,gc_collect_main,2ebdb0,"ldr     x22, [x22]",16.89
4,gc_collect_main,2ec658,"ldr     x22, [x22]",11.96
```

## 4. annotate 层可用性（重点回答）

结论：**能注解，但能不能给出 `file:line` 完全取决于该 .so 有没有 debug info。**

| 对象 | 结果 |
|---|---|
| 镜像 `libpython3.12.so.1.0`（3.12.13） | 有 `.debug_info/.debug_line`，`perf annotate -l` 给出 `Sorted summary`（`object.h:646`、`pycore_gc.h:60` 等）和每行指令的 `// file:line` |
| `libopenblas-24fd393f.so.0` | `Sorted summary` 是空的，只剩地址；进 `annotate.no_debug_info`，CSV 里留空 |
| **宿主 `libpython3.11.so.1.0`（历史报告里的那个）** | **stripped**：`perf annotate -l --symbol _PyEval_EvalFrameDefault` 只输出 `libpython3.11.so.1.0[1647d8]` 这类纯地址，**没有源码行** |

所以：**历史报告只有 C 层符号，根因是当时采的是宿主 python 3.11（stripped），
不是 perf 参数不对。** 真机要采的是镜像里 3.12.13 的进程，注解与源码行都能拿到。

`perf_capture.sh` 的 annotate 行为（已实测）：

* 对 `hotspots.csv` 前 3 个「可注解」符号各跑一次
  `sudo -n perf annotate -i perf.data --stdio -l --symbol <sym>`；
  候选会剔除 `[JIT] tid N`（perf map，objdump 打不开，会报 `Couldn't annotate py::...`）
  与 `[kernel.kallsyms]`，并清掉 `[.]`/`[k]`/`+0x` 后缀。
* 实测三连：`sgemm_kernel_ARMV8SVE`（无源码）、`gc_collect_main`（有）、
  `_PyEval_EvalFrameDefault`（有）；manifest 记为
  `annotate = {"ok": true, "symbols": [3 个], "no_debug_info": ["sgemm_kernel_ARMV8SVE"]}`。
* `_PyEval_EvalFrameDefault` 与 `gc_collect_main` 之所以有源码行，是因为镜像里的
  libpython 编译时带了 `-g`（`.debug_line` 保留）；源码文本本身不在镜像里，
  所以 perf 只给 `file:line`，不给源码正文 —— 这正好是 CSV 需要的字段。

### 4.1 `--symfs` 为什么不用

`--symfs` 只在「DSO 路径找不到」时有用。这里叶子符号解析率已经是 98.6%，缺的不是符号根，
而是 Python 帧 —— 那是 **map 文件 + 调用链**的问题，`--symfs` 不解决。
（4 组数据都不传 `--symfs`，解析率 98.5~98.8%。）

## 5. py-spy 实测（备选旁证）

宿主 `~/.local/bin/py-spy 0.4.2`（aarch64 wheel），
`-v ~/.local/bin/py-spy:/opt/pyspy/py-spy:ro` 挂进容器后在容器内 `docker exec` 采同一进程：

| 模式 | 结果 |
|---|---|
| `py-spy record --native` | `Samples: 211 / Errors: 783`（**21% 成功**，原生栈基本采不动）|
| `py-spy record`（不带 `--native`） | `Samples: 994 / Errors: 0`（**100% 成功**，但只有 Python 帧）|

结论：py-spy 可以作为「Python 层火焰图」的旁证（`--format flamegraph`，产物
`pyspy-nonative.svg`），但不能替代 perf —— 它没有 C 层、没有指令级、没有 topdown。
注意 py-spy 必须**在容器内**跑；宿主直接跑会 `Failed to get process executable name`，
因为 `/proc/<pid>/exe` 指向容器里才有的 python。

## 6. 真机采集形态与可复制命令

形态就一句：**宿主 root 的 perf 采「容器内进程的宿主 TID」**。

### 6.1 启动服务容器（推荐 `--pid=host`）

```bash
IMG=local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler
docker run -d --name pi-phase-chip3-<run> --pid=host --privileged \
  --cpuset-cpus=120-159 -e PYTHONPERFSUPPORT=1 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 <其余原有参数> $IMG "<原启动命令>"
```

`--pid=host` 让进程的 `getpid()` 就是宿主 PID，map 名字天然对得上，
`/tmp` 也不用共享。`PYTHONPERFSUPPORT=1` 对 vLLM 无副作用（见第 7 节）。

### 6.2 采集

```bash
cd ~/projects/vllm/prepare-input-phase
HOST_TID=<engine-core 主线程的宿主 TID，见下>
sudo -n ls /tmp/perf-$HOST_TID.map     # 必须存在，否则后面没有 py:: 帧

bash scripts/measure/perf_capture.sh \
  --tid "$HOST_TID" --seconds 15 --freq 999 \
  --call-graph fp --name "prepare-input-c64-isl128" \
  --outdir data/profiles/pi-<run>/<point>
```

engine-core 主线程 TID 的找法：

```bash
docker top <container> -H | head            # 先看线程清单
# 或者挑 utime+stime 增长最快的线程：
for t in /proc/<pid>/task/*; do awk '{print $14+$15, FILENAME}' $t/stat; done | sort -nr | head
```

### 6.3 容器不能 `--pid=host` 时的退路（已实测）

```bash
NNS=$(docker exec <container> bash -c 'for p in /proc/[0-9]*; do \
  [ "$(cat $p/comm 2>/dev/null)" = python3 ] && basename $p; done' | head -1)
bash scripts/measure/perf_capture.sh --tid "$HOST_TID" --seconds 8 \
  --call-graph fp --perf-map /tmp/perf-$NNS.map --name ... --outdir ...
```

脚本会用 `sudo` 把 `/tmp/perf-$NNS.map` 链到 `/tmp/perf-$HOST_TID.map`
（map 是 `root:600`，普通用户判读会误判成不可读，所以判读本身也走 sudo）。
**前提是容器要 `-v /tmp:/tmp`**，否则宿主看不到 map。另外 `/tmp/perf-1.map`
这个名字会被**每一个**容器复用，有串号风险 —— 所以这只当退路，首选仍是 `--pid=host`。

## 7. 遗留风险

1. **trampoline 对被测负载的扰动**：`PYTHONPERFSUPPORT=1` 会让每次 Python 调用多走一层
   trampoline（CPython 官方承认有开销）。**实测（`trampoline-overhead.txt`，同一个单线程
   纯解释器负载，开/关各一轮配对比）：20 s 窗口 1479 → 1370 次迭代、15 s 窗口 1120 → 1036 次，
   即 trampoline 让这个**纯 Python 解释器密集型**负载慢 ~7.4%**（上界；prepare_input 里
   tensor 调用占比越高，相对开销越小）。做**耗时/占比**结论时要用不开 trampoline 的对照组，
   或把这一项算进偏差；只做**热点归因**时可以忽略。
2. **map 与数据同生共死**：`perf.data` 里**不内嵌**符号表（实测 `strings perf.data |
   grep -c 'py::'` = 0），符号是 report/script 阶段从 `/tmp/perf-<tid>.map` 现读的。
   所以采集完要**趁容器还活着**跑完 report/folded/annotate；容器一停、map 一删，同一个
   `perf.data` 就再也印不出 `py::` 帧（实测：把 map 移走，`py::` 帧数 26354 → 0）。
   自测的两个 map 已归档在证据目录里（`perf-map-174359.map` = `--pid=host` 情形，
   `perf-map-1.map` = 容器命名空间 PID 1 情形），要重新符号化就：

   ```bash
   sudo -n cp perf-map-<tid>.map /tmp/perf-<tid>.map    # tid = manifest 里的 tid
   sudo -n perf script -i fp/perf.data | grep -c 'py::'
   ```
3. **`[unknown]` 底帧**：几乎每条栈底部都有一帧 `[unknown] (libc.so.6)`
   （`__libc_start_call_main` 无符号），所以 `samples_with_unknown_frame_pct` 恒等于
   约 100%，别看这个数，看 `unknown_frame_line_pct`（3.4~5.3%）。
4. **fp 对无帧指针的 C++ 库会截断**：`libtorch_cpu.so` 里没开帧指针的函数会让栈到此为止；
   需要更深的 C++ 栈时再用 dwarf 单独补一次。
5. **`perf.data` 归 root 所有（0644）**：这是故意的 —— perf 拒绝读「既非自己、也非 root」
   的 perf.data；root:0644 则 root 和任何用户都能读。目录归用户，删除/打包不受影响。

## 8. 复现清单

```bash
# 0) 单线程 CPU-bound 负载（自测证据里同款）
docker run -d --name picsel-host --pid=host --privileged --cpuset-cpus=200-215 \
  -v $PWD/data/profiles/<dir>:/work -v /tmp:/tmp -e PYTHONPERFSUPPORT=1 \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e LOAD_SECONDS=900 <IMG> -c "cd /work && exec python3 selftest_load.py > /work/load.log 2>&1"

# 1) fp + trampoline（主配置）
bash scripts/measure/perf_capture.sh --tid <hosttid> --seconds 15 --call-graph fp \
  --name selftest-fp-trampoline --outdir data/profiles/<dir>/fp

# 2) dwarf（对照，注意体积）
bash scripts/measure/perf_capture.sh --tid <hosttid> --seconds 15 --call-graph dwarf \
  --name selftest-dwarf-trampoline --outdir data/profiles/<dir>/dwarf
```

`manifest.json` 关键字段（真实取自 `fp/manifest.json`）：

```json
{"cmd": ["..."], "perf_record_cmd": ["sudo","-n","perf","record","..."],
 "tid": 174359, "seconds": 15, "freq": 999, "call_graph": "fp",
 "perf_version": "perf version 6.6.0-159.4.13.167.oe2403sp4.aarch64",
 "kernel_version": "6.6.0-159.4.3.154.oe2403sp4.aarch64",
 "sample_count": 15028, "lost_samples": 0,
 "symbol_quality": {"resolved_pct": 98.64, "unknown_pct": 1.36,
   "top_dso": ["libpython3.12.so.1.0", "libopenblas-24fd393f.so.0", "..."],
   "dso_self_pct": {"libpython3.12.so.1.0": 63.88, "libopenblas-24fd393f.so.0": 28.37},
   "python_frames": {"samples_with_py_frame_pct": 100.0,
                     "unknown_frame_line_pct": 3.48, "mean_frames_per_sample": 29.16}},
 "annotate": {"symbols": ["sgemm_kernel_ARMV8SVE","gc_collect_main","_PyEval_EvalFrameDefault"],
              "ok": true, "no_debug_info": ["sgemm_kernel_ARMV8SVE"]},
 "started_utc": "2026-09-23T18:35:44Z", "finished_utc": "2026-09-23T18:36:02Z",
 "notes": ["..."]}
```
