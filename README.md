# vLLM 0.26.0 `prepare_input` 阶段 CPU 侧负载分析

> 完整交付包。**阅读入口：[`docs/00-INDEX.md`](docs/00-INDEX.md)**（一分钟结论 + 文档地图）。

## 这是什么

对 vLLM 0.26.0（+ vllm-ascend 0.26.0rc1）在 Kunpeng 920B + Ascend A3 上
`prepare_input` 阶段的 CPU 侧负载做的系统分析，回答四个问题：

1. **何时成为瓶颈** —— 占比曲线与临界判据（判据公式 + 真机实测）
2. **负载长什么样** —— topdown / IPC / 热点函数 / 热点源码行 / 热点指令 / 火焰图
3. **谁最贵** —— 逐子步骤分解到具体函数与调用点
4. **怎么在没有卡的时候复现** —— 带保真度判据的无卡负载构造

## 一分钟结论

| # | 结论 |
|---|---|
| 1 | 低并发 decode 下 `prepare input` 占单步 **55.3%**（B=1，0.8B/TP1/graph，pystack off）；B=64 时降到 38.7% ⇒ **低并发才是最该优化的区域** |
| 2 | 真机 topdown `frontend_bound 66.01%` / `retiring 12.85%` / `IPC 0.771`，同步调用合计仅 **0.27%** ⇒ **不是等卡，是 CPython 逐条派发** |
| 3 | 头号热点 = `AscendGDNAttentionMetadataBuilder.build`，**3 次 × 303 µs = 908 µs（29.3%）** |
| 4 | 一条可落地优化：`compute_num_computed_tokens` 的缓存在 `.replace()` 后被击穿，**预期省 ~130 µs/步** |
| 5 | 无卡 harness 达标：热点 **top-20 重合 80%**、top-1 同符号、扁平度同形 |
| 6 | 历史"占 59.6%"含 **+24% 的 pystack 采样器污染**，引用前必读 `docs/09` |

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 10 篇正文文档（共 4 264 行），入口 `00-INDEX.md` |
| `figures/` | 43 张图（SVG + PNG），含 3 张火焰图 |
| `data/` | 结构化数据：静态分解 / 历史基线 / 真机测量 / 子阶段 / profiling / harness |
| `harness/` | **无卡复现工具**（真实代码路径 + 三层 shim），含 README 与自验收脚本 |
| `instrument/` | 子阶段插桩（`-v` 挂载、AST 等价性已证明） |
| `scripts/` | 采集 / 解析 / 绘图 / 打包脚本，均可 `--help` |
| `plan/` | 任务规划、协作约定、实验矩阵 |
| `agents/*/REPORT.md` | 各子任务的交接报告（含失败路径与踩坑记录） |
| `refs/` | 关键源码快照（用于行号引用） |

## 复现

见 [`docs/08-reproduction-manual.md`](docs/08-reproduction-manual.md)。
一句话版：

```bash
# 真机（chip3 = /dev/davinci3, CPU 120-159）
bash scripts/launch_subscope_service.sh --mode baseline --pystack-interval-us 0 -- ...

# 无卡（不需要任何 NPU 设备）
bash harness/scripts/pi-docker.sh --preset realmachine --batch 1 --isl 128
```

## 口径纪律（引用任何数字前必读）

1. **占比分母**：async 下每轮 engine 循环写两行 `Step:Model`，用它当分母会得到 69.5%，
   正确值 55.3%。一律用 `--step-scope Step:Schedule --window-mode next`。
2. **pystack**：占比测量必须 `--pystack-interval-us 0`，否则 +24%。
3. **探针代价**：49 个探针有 2.2 µs/个的地板（合计 +9.5%），引用子 scope 绝对值时必须扣除。
4. **PMU**：`memstall_l3miss`/`dram_*` 在 920B 上恒为 0，不能写成 "DRAM bound = 0%"；
   所有 topdown/IPC 必须带 `min_confidence`。
5. **harness**：`--preset` 必须逐字匹配真机启动参数，否则 IPC 会偏 64%。

## 原始数据

原始 `perf.data`（419 MiB + 245 MiB）与符号目录**未进交付包**（体积原因），
留在 `a3-22:~/projects/vllm/prepare-input-phase/data/profiles/`；
重绘火焰图需要它们或对应的 `--symfs` 目录，配方见 `docs/05-hotspots.md`。

## 关于本仓库的标识符

本仓库是**净化版**：内部账号/员工号、内网地址、主机名、对象存储桶名等已替换为
占位符（如 `REMOTE_USER`、`A3_22_IP`、`COS_BUCKET`），**技术内容未改动**。

完整替换表、保留项判断依据、以及一键还原/重做的方法见
**[`docs/SANITIZATION.md`](docs/SANITIZATION.md)**。
工具：`scripts/sanitize_for_publish.sh`（真实值外置在 `.sanitize-map.tsv`，不进仓库）。

未净化的原始工作树另有两处：a3-22 `~/projects/vllm/prepare-input-phase/`，
以及内部分发的 COS 交付包。
