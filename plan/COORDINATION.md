# prepare_input CPU 侧分析 —— 协作约定

> 建立：2026-09-24（Asia/Shanghai）。所有 agent 动手前先读本文件。

## 1. 目标（一句话）

说明 vLLM 0.26.0（vllm-ascend 0.26.0rc1）里 `prepare_input` 阶段**何时**成为瓶颈、
CPU 侧负载长什么样，并给出**可无卡复现**的负载构造 + 完整 profiling 证据链。

## 2. 唯一写入范围（硬约束，违反即回滚）

| 位置 | 路径 |
|---|---|
| 本地（开发机） | `/home/chiro/projects/vllm/preparing-input-phase` |
| 远端 a3-22 | `/home/REMOTE_USER/projects/vllm/prepare-input-phase` |

只读参考（可以读、**不要写**）：

- `a3-22:/home/REMOTE_USER/projects/vllm/HIST_PROJECT`（历史工具链、DevKit、liteprofiler）
- `a3-21:/home/REMOTE_USER/projects/profiling/libkperfx`（920B PMU 库）
- `~/projects/vllm/HIST_PROJECT/vllm-ascend`（本地 vllm-ascend 0.26.0rc1 源码）
- `refs/vllm`（本地解出的 vLLM 0.26.0 源码，与运行镜像一致）

禁止：动别人的容器（`dsv41-*`、`cann910`）、别人目录、宿主机系统配置（irqbalance /
CPU governor / 内核参数）；`sysctl`、`npu-smi set` 之类一律不做。

## 3. 目标硬件与资源

| 项 | 值 |
|---|---|
| 主机 | a3-22（Kunpeng 920B，640 逻辑核 / 4 socket×80 core / 8 NUMA） |
| NPU | **chip3 = `/dev/davinci3`**（physical chip 3，NPU1 chip1），量测前后必须 `No process in device` |
| CPU 切片 | `120-159`（NUMA node 1）；worker `122-157`，acl `158`，release `159` |
| 其他 NPU | chip0/2/4-15 全部**不许碰**（有他人任务） |
| 特权 | `sudo -n` 免密；可跑 privileged docker；PMU 采集必须 root |

**串行化**：任何占 NPU / 跑 benchmark / 跑 perf 的动作，先拿锁
`a3-22:~/projects/vllm/prepare-input-phase/locks/chip3.lock`（`flock` 语义：
先 `mkdir locks/chip3.holder` 成功者持有，收工 `rmdir`）。写文件
`locks/chip3.lease.json` 记录：owner、目的、PID、开始时间。

## 4. 固定身份（不要把不同版本混池）

| 组件 | 值 |
|---|---|
| vLLM | 0.26.0，commit `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vllm-ascend | 0.26.0rc1，commit `f2f74a16c3c50a76f4349d807918e83edec1e35c` |
| 基础镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:v0.26.0rc1-a3-openeuler`（digest 以 `docker inspect` 为准，写进每轮 manifest） |
| 插桩镜像 | `local/vllm-ascend-liteprofiler:v0.26.0rc1-openeuler`（LiteProfiler patch，用于 phase 计时；需核对 patch 与基线一致） |
| 模型 | 首选 `Qwen3.5-0.8B`（BF16，TP1，快）；对照 `Qwen3.5-2B` / `Qwen3.5-27B-w8a8-mtp` |

## 5. 目录约定

```
docs/        最终交付文档（中文，00-INDEX.md 为入口）
figures/     图（SVG/PNG），文件名 = 文档章节号 + 语义
data/        结构化数据：CSV/JSON/汇总表（原始 perf.data 只在 a3-22 留档，不上传）
scripts/     可复现脚本（采集 + 分析）
agents/<id>/ 每个 agent 自己的原始笔记与中间产物（正式结论要回填 docs/）
plan/        计划、协调、交接
refs/        源码快照（只读）
```

## 6. 数据口径

- 每轮实验写 manifest：镜像 digest、模型 revision、workload 参数、绑核、时间戳、脚本哈希。
- 每个"占比"数字必须写清分子分母：`prepare_input` 占 **engine-core 单步 wall** 还是
  **端到端吞吐**；两者不可混用。
- topdown / IPC 数字必须带 `time_running/time_enabled`（libkperfx multiplex 置信度）。
- 所有 perf/flamegraph 采集与 msprof/DevKit 不共存（互相污染），一次只跑一种。

## 7. 关键待答问题（交付必须回答）

1. `prepare_input` 的**边界**到底包含哪些子步骤，各步 CPU 复杂度是多少？
2. 它在 prefill / decode / chunked-prefill / 高并发 / MTP 下的占比曲线？
3. 何时成为瓶颈（相对 NPU 计算时间的临界条件）？
4. 无卡模拟负载怎么构造，凭什么说和真机一致（topdown/IPC/热点函数）？
5. 热点函数 / 热点源码行 / 热点指令分别是什么，有哪些可优化点？
