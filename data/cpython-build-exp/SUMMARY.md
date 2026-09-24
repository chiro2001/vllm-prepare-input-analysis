# cpython-build-exp 数据摘要（机读口径的总表）

> 主文档：[`docs/10-cpython-directions/06-build-config-experiment.md`](../../docs/10-cpython-directions/06-build-config-experiment.md)
> 交接报告：[`agents/cpython_build_exp/REPORT.md`](../../agents/cpython_build_exp/REPORT.md)

## 0. 身份与口径

| 项 | 值 |
|---|---|
| 机器 | a3-22 = host22，Kunpeng 920B，640 逻辑核 / 8 NUMA，kernel 6.6.0-159.4.3.154.oe2403sp4 |
| 镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler` |
| image id | `sha256:dc9a31b8330d399ad8e91dabaca25798c3d838c36851897bf9a1f77f793072ec` |
| repo digest | `sha256:24ae7427b6cad5ee29e0665e6f69a4d51c9f6178035e38f2ed161bb3d29fe81c` |
| 构建编译器 | 容器内 gcc 12.3.1（openEuler 12.3.1-111.oe2403sp3），binutils 2.41，make 4.4.1 |
| 源码 | Python-3.12.13.tgz，`sha256=0816c4761c97ecdb3f50a3924de0a93fd78cb63ee8e6c04201ddfaedca500b0b` |
| 绑核 | 构建 200-239（C 臂后期 216-239）；基准/PMU `taskset -c 200-203`；harness 200-215 |
| 环境 | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0` |

## 1. 臂定义与构建

| arm | configure 追加参数（共同 `--enable-shared`） | `USE_COMPUTED_GOTOS` | configure | make -j16 | libpython 体积 | `.text` |
|---|---|---:|---:|---:|---:|---:|
| A | `--without-computed-gotos` | 0 | 17 s | 27 s | 30,812,384 B | 0x24d30c |
| B | `--with-computed-gotos` | 1 | 17 s | 27 s | 30,689,296 B | 0x24dfcc |
| C | `--enable-optimizations --with-lto --with-computed-gotos` | 1 | 44 s | 813 s | 30,209,248 B | 0x23124c |
| 镜像 stock | 安装面 `pyconfig.h` 声称 undef（**与实际 .so 不符**） | （声称 0） | — | — | 30,689,272 B | 0x24dfcc |

## 2. 分派路径判定（决定性）

| 判据 | 镜像 stock | A | B | C |
|---|---|---|---|---|
| `opcode_targets` 符号（256×8 B） | **有**（0x800 @ `.data.rel.ro`） | 无 | 有（0x800） | 有 |
| `_PyEval_EvalFrameDefault` 内 `br xN` | **253** | **1** | 253 | **217** |
| 该函数体积 / 指令数 | **46268 B / 11567** | 44072 B / 11018 | 46268 B / 11567 | — |
| 分派指令形态 | `ldr x0,[x1,w8,sxtw#3]; br x0` | `adr/add(...,sxth#2)/br x0` | 同镜像 | — |

## 3. `pi_sim_bench`（本负载形状，ns/step）

主口径 11 轮交错（`data/bench-pi-clean/`）；括号内为 7 轮复现（`data/bench-pi/`）。

| arm | 中位 | 最小 | 相对 A | 相对 B |
|---|---:|---:|---:|---:|
| A | 256266（263673） | 252803 | 1.0000× | 1.0840× |
| B | 236414（251374） | 234275 | 0.9225×（−7.8%） | 1.0000× |
| C | **182984（192233）** | **181791** | **0.7140×（−28.6%）** | **0.7740×（−22.6%）** |
| 镜像 stock | 242222 | 241387 | 0.9404× | 1.0286× |

> 注：`data/bench-pi-stock/`、`data/bench-dispatch-stock/` 里是**单独会话**跑的镜像对照，
> 只用于复核；**结论一律引用上面 `data/bench-*-mixed/` 的同会话交错数据**，
> 因为本机会话间漂移可达 3–5%。

`work_digest` 三臂 + 镜像**全部 = 77232000**（工作量相同）。

### 3.1 同会话逐轮交错：镜像 vs 三臂（`data/bench-*-mixed/`，7 轮中位）

> 镜像必须与各臂在**同一会话**里交错跑；本机会话间漂移可达 3–5%。

| 内核 | A（switch） | B（CG） | C（PGO+LTO） | 镜像 stock | 镜像/A | 镜像/B | B/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| `int_loop` | 112.61 | 100.89 | 86.86 | 107.64 | −4.4% | +6.7% | 0.896 |
| `attr_loop` | 66.80 | 62.30 | 58.54 | 62.40 | −6.6% | +0.2% | 0.933 |
| `call_loop` | 125.99 | 113.02 | 107.40 | 112.67 | −10.6% | −0.3% | 0.897 |
| `branch_loop` | 101.02 | 90.25 | 67.04 | 89.61 | −11.3% | −0.7% | 0.893 |
| `global_loop` | 63.31 | 58.17 | 49.66 | 58.05 | −8.3% | −0.2% | 0.919 |
| `pi_sim_bench`（ns/step） | 257573 | 235489 | 183034 | 242222 | −6.0% | +2.9% | 0.914 |

⇒ 镜像落在 computed gotos 一侧（5 个分派核里 4 个与 B 臂相差 ≤0.7%），不是 switch 一侧。

## 4. `perf stat`（`pi_sim_bench --steps 4000 --rounds 5`，`data/perf-clean/`）

| 计数 | A | B | C | B/A | C/B |
|---|---:|---:|---:|---:|---:|
| cycles | 15,153,701,127 | 14,145,882,614 | 10,874,288,079 | 0.9335 | 0.7687 |
| instructions | 35,904,059,214 | 34,062,602,480 | 29,908,817,262 | 0.9487 | 0.8781 |
| branches | 7,849,095,216 | 7,152,758,930 | 5,826,018,326 | 0.9113 | 0.8145 |
| branch-misses | 92,413,390 | 81,140,060 | 63,921,788 | 0.8780 | 0.7878 |
| L1I load-misses | 495,328,741 | 490,760,814 | 334,154,269 | 0.9908 | 0.6809 |
| iTLB load-misses | 99,177,696 | 94,796,845 | 34,521,681 | 0.9558 | 0.3642 |
| **IPC** | 2.3693 | 2.4080 | **2.7504** | +1.6% | **+14.2%** |
| branch-miss 率 | 1.1774% | 1.1344% | 1.0972% | −0.043 pp | −0.037 pp |
| L1I miss 率 | 7.2390% | 8.0824% | 7.5167% | +0.84 pp | −0.57 pp |
| iTLB miss 率 | 1.4494% | 1.5612% | 0.7765% | +0.11 pp | −0.78 pp |

## 5. 真实 `prepare_input` harness（`--preset realmachine --batch 1 --isl 128 --steps 200`）

| arm | n | `prepare_inputs_us` p50 中位 | p50 最小 | mean 中位 | `update_states_us` p50 | 相对 stock |
|---|---:|---:|---:|---:|---:|---:|
| stock | 5 | **464.7** | 458.9 | 469.7 | 18.7 | 1.0000× |
| A | 2 | 464.8 | 457.5 | 473.2 | 19.4 | +0.01% |
| B | 2 | 451.6 | 450.4 | 458.7 | 18.2 | −2.83% |
| **C** | **3** | **392.3** | **388.3** | **415.6** | **15.6** | **−15.59%** |

## 6. 镜像原版产物哈希（二进制本体留在 a3-22 `exp-cpython/refimg/`，未进交付包）

```
de439dc24ea47b35d5e50ee269c7421b5f0eb5d4d129a9ab7ef1210d487c3c88  image-libpython3.12.so.1.0
68f930a2e58929231d6622e98fe77ac25e0b0bc84b7a082220688dd5badc8114  image-python3.12
```
