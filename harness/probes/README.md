# `harness/probes/` — probe_devdeps 的无卡设备依赖探针

> owner：`probe_devdeps`（`agents/probe_devdeps/REPORT.md` 是本目录的总报告）。
> 唯一运行入口：`ssh a3-22 'cd ~/projects/vllm/prepare-input-phase && bash harness/scripts/pi-docker.sh "<cmd>"'`

## 脚本

| 脚本 | 作用 | 典型命令（容器内路径 `/work`） |
|---|---|---|
| `import_gradient.py` | A：导入梯度 16 步（每步独立子进程，完整栈落盘） | `python harness/probes/import_gradient.py [--list] [--only A4c A8] [--set-env K=V] [--suffix _x]` |
| `scope_prepare_input.py` | E：`prepare input` scope 边界（AST），写 `devdeps_scope.json` + `devdeps_scope_boundary.md` | `python harness/probes/scope_prepare_input.py [--depth 3] [--markdown]` |
| `smoke_shim.py` | D：shim 冒烟 17 用例 | `python harness/probes/smoke_shim.py [--list] [--only C5 C6]` |
| `probe_dynamic_hits.py` | C：`torch.npu.*`/`torch.accelerator.*`/`torch_npu.*` 计数代理 + triton_cpu 接入 | `python harness/probes/probe_dynamic_hits.py [--no-patch] [--no-triton]` |
| `shim_patch_proposal.py` | **参考实现**：device 层重映射三层 patch + 13 用例自测 | `python harness/probes/shim_patch_proposal.py --selftest [--dump]` |
| `gen_devdeps_tables.py` | 生成 `devdeps_apilist.{md,json}` / `devdeps_matrix.{md,json}` / `devdeps_manifest.json` | `python harness/probes/gen_devdeps_tables.py` |
| `collect_manifest.py` | 容器内 manifest（版本 / 环境 / sha256） | `python harness/probes/collect_manifest.py` |
| `collect_host_manifest.sh` | 宿主机 manifest（镜像 digest / docker run 模板） | `bash harness/probes/collect_host_manifest.sh` |
| `push.sh` / `pull_data.sh` | 同步 helper（默认**不碰** shim；见下） | 见下 |

## 同步

```bash
bash agents/replay_harness/harness/probes/push.sh push              # 只推 probes/
bash agents/replay_harness/harness/probes/push.sh push --with-report # 连 agents/probe_devdeps/REPORT.md
bash agents/replay_harness/harness/probes/pull_data.sh               # 只拉 devdeps_* 小文件
bash agents/replay_harness/harness/probes/pull_data.sh --stacks      # 连 devdeps_stacks/ 一起拉
```

**不要用仓库根的 `harness/scripts/sync.sh`**：它在本仓库布局下会把 `LOCAL_ROOT` 解析成
`<repo>/agents`，导致 rsync 报 `link_stat .../agents/harness failed`。本目录的 `push.sh` 用显式相对路径规避。

## 目录归属提醒

* `harness/probes/**`、`data/harness/devdeps_*`、`agents/probe_devdeps/**` → **probe_devdeps**
* `harness/pi_harness/shim/**` → **replay_harness**（2026-09-24 分工调整后归属变更）。
  本仓库里 `agents/replay_harness/harness/pi_harness/shim/__init__.py` 是**远端的只读镜像**
  （sha256 与 a3-22 一致，见 REPORT 第 3 节），仅供本地引用；
  `push.sh` 默认**不推送**它，避免用过期副本覆盖 replay_harness 的版本。
  需要显式推送时用 `push.sh push --with-shim`。
* `harness/pi_harness/runner/**` → **replay_harness**（本目录只**读**它的 `triton_cpu.py`）。

## 一次性跑全（复现）

见 `agents/probe_devdeps/REPORT.md` 第 8.2 节。
