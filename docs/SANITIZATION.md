# 发布净化说明（GitHub 同步口径）

> 本项目在同步到 GitHub 之前做了一次**机械化净化**：把能定位到具体人、
> 或能用于访问内部系统的标识替换为占位符。**技术内容一字未改。**
>
> 时间：2026-09-24 · 脚本：`scripts/sanitize_for_publish.sh` · 可逆

---

## 1. 为什么做

本地工作树里包含运行痕迹（路径、日志、manifest），其中有几类**不适合进公开渠道**的信息：

- 内部员工号与其它协作者的用户名（出现在 `ssh` 路径、`/home/<user>/` 路径、进程清单里）
- 内部主机名 / 内网 IP / SSH 跳板与链接服务地址
- 对象存储（COS）桶名与上传日志
- 一个内部分析项目的目录名

**注意：没有发现任何密钥、token、证书或密码**——这一点在净化前专门扫过
（COS secretid/secretkey、GitHub token、私钥、`password=`/`api_key=` 等模式**全部为空**）。
净化针对的是"身份与基础设施标识"，不是凭据。

## 2. 替换表（全部机械替换，可逆）

| 原值类别 | 占位符 | 命中次数 |
|---|---|---:|
| 内部员工号（出现在 `/home/<id>/` 等路径） | `REMOTE_USER` | 1565 |
| 另一位协作者的用户名 | `OTHER_USER` | 429 |
| 内部分析项目目录名 | `HIST_PROJECT` | 701 |
| 内网 IP（目标机） | `A3_22_IP` | 2 |
| 链接服务公网 IP | `LINKS_IP` | 2 |
| 链接服务域名 | `LINKS_HOST` | 4 |
| 链接服务用户名 | `LINKS_USER@LINKS_HOST` | 3 |
| COS 桶名（短名 / 全名） | `COS_BUCKET` / `COS_BUCKET_FQ` | 6 / 2 |
| COS 端点 | `COS_ENDPOINT` | 2 |
| 发布者邮箱 | `PUBLISHER_EMAIL` | 1 |

替换在**文本文件**上执行；二进制文件不替换，改为在 `.gitignore` 里排除（见 §3）。

## 3. 整体排除的内容（`.gitignore`）

| 路径 | 原因 |
|---|---|
| `refs/` | upstream vLLM 源码（Apache-2.0，39 MB）。**不属于本项目产出**；需要时按 `docs/00-INDEX.md` 记录的 commit `568afb3a1` 自行 clone |
| `agents/env_toolchain/staging/` | 第三方 PMU 工具（libkperfx）的副本与编译产物；产物内的调试路径会带出内部用户名 |
| `coscli.log`、`coscli_output/` | 对象存储上传日志（含桶名与目标路径） |
| `**/.git/` | 嵌套仓库会携带提交者邮箱等元数据 |
| `**/perf.script.gz`、`perf.data*` | 原始 perf 采样（含采集时的宿主路径）。按项目约定**原始采样只在 a3-22 留档** |
| `__pycache__/`、`*.pyc`、`.venv/`、`.mypy_cache/` | 可重建的缓存 |

> 净化后复核：上表 §2 的全部原值在工作树中**命中数为 0**（唯一例外在已排除的文件里）。

## 4. 保留的标识（有意保留）

以下信息**未被替换**，判断依据是"不指向具体自然人、也不授予访问权"：

| 类别 | 例子 | 保留理由 |
|---|---|---|
| 硬件与软件名 | Kunpeng 920B、Ascend 910、CANN 9.1.0、vLLM 0.26.0 | 公开产品名，是技术结论的必要上下文 |
| 公共镜像仓库与站点 | `quay.nju.edu.cn`、`hiascend.com`、`gitcode.com`、`github.com` | 公开可访问 |
| 目标机别名 | `a3-22`、`a3-21` | 仅是节点代号；在已去掉用户名与 IP 后不具备定位性 |
| 公开的 issue/PR 编号 | vLLM #52297、vllm-ascend #16246 等 | 公开信息 |

**如果你的口径更严**（例如 `a3-22` 也要隐去），改 `scripts/sanitize_for_publish.sh`
的 `MAP` 数组再跑一次 `--apply` 即可，无需手工改文件。

## 5. 如何还原 / 如何重做

```bash
# 首次使用：从模板建映射表并填入真实值
cp scripts/sanitize-map.example.tsv .sanitize-map.tsv
$EDITOR .sanitize-map.tsv

# 查看当前命中情况（不修改文件）
bash scripts/sanitize_for_publish.sh --check

# 把占位符换回原值（需要原始标识时）
bash scripts/sanitize_for_publish.sh --revert

# 重新净化
bash scripts/sanitize_for_publish.sh --apply
```

**真实值放在哪里？** 在 `.sanitize-map.tsv`（仓库根目录），**该文件被 `.gitignore` 排除**，
不进仓库；仓库里只有模板 `scripts/sanitize-map.example.tsv`。
所以 `sanitize_for_publish.sh` 本身可以公开——它只是读表干活，不含任何真实标识。

> **这个设计是踩坑换来的**：最早的版本把"原值 → 占位符"表直接写在脚本里，
> 结果 `--apply` 会**把脚本自己也替换掉**（映射项 `<原值>|REMOTE_USER` 变成
> `REMOTE_USER|REMOTE_USER`），导致 `--revert` 失效、且原值反而随脚本留在仓库里。
> 现在的版本显式跳过脚本自身，并把真实值外置。

> ⚠️ `--revert` 能还原§2 的字符串替换，但**不会**把 `.gitignore` 排除的文件找回来。
> 完整未净化的工作树另有两份副本：
> - a3-22 `~/projects/vllm/prepare-input-phase/`（原始）
> - COS 交付包 `prepare-input-cpu-analysis-*.tar.gz`（原始，未净化）

## 6. 与 COS 交付包的关系

| 渠道 | 内容 |
|---|---|
| **COS 交付包**（`prepare-input-cpu-analysis-*.tar.gz`） | **未净化**，含真实路径与主机名。面向内部同事 |
| **GitHub 仓库**（本仓库） | **已净化**。面向更广的读者 |

两者的技术结论、数据、图、脚本**完全一致**，差别仅在标识符。
