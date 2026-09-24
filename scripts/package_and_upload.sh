#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# package_and_upload.sh —— 打包 prepare_input 交付物 → COS → links-server
#
#   bash scripts/package_and_upload.sh [--tag TAG] [--dry-run] [--no-src]
#
# 步骤：
#   1) 在本地把 docs/ figures/ data/ scripts/ plan/ agents/*/REPORT.md 打成一个 tar.gz
#   2) 生成 MANIFEST.txt（文件清单 + sha256 + 大小）
#   3) coscli 上传到 cos://COS_BUCKET/share/
#   4) 设 public-read ACL 并自检外网可达
#   5) 追加 www 版本（含图，供浏览器直接看）到 links-server
#
# 依赖：本地有 coscli（~/.local/bin/coscli）且 ~/.cos.yaml 已配置；
#       能 ssh LINKS_USER@X_LINKS_HOST。
# 注意：原始 perf.data 不上传（体积原因），只在 a3-22 留档。
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=${ROOT:-/home/chiro/projects/vllm/preparing-input-phase}
TAG=${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
DRY=0
WITH_SRC=1
while (($#)); do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --no-src) WITH_SRC=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SHARE=${SHARE:-share}
BUCKET=${BUCKET:-COS_BUCKET}
BUCKET_FQ=${BUCKET_FQ:-COS_BUCKET_FQ.COS_ENDPOINT}
LINKS_HOST=${LINKS_HOST:-LINKS_USER@X_LINKS_HOST}
NAME="prepare-input-cpu-analysis-$TAG"

cd "$ROOT"

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/pinput-pkg.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME"

echo "[pkg] staging in $STAGE/$NAME"

# ---- 1. 内容（用 tar 带排除规则，避免把大体积派生物塞进包里）----
# 排除规则：
#   *.perf.data / perf.data*     原始 perf 采样（约定只在 a3-22 留档）
#   */.conda / .conda            本地 conda 环境（GB 级）
#   *.tmp / .*.txt.?*            工具中断残留的临时文件
#   任何 >40M 的单个文件         体积红线（打包后会重新统计）
EXCLUDES=(
  --exclude=perf.data --exclude='perf.data.*' --exclude='*.perf.data'
  --exclude=.conda --exclude='*/.conda' --exclude='.venv'
  --exclude='.*.txt.??????' --exclude='*.tmp'
  --exclude='refs/vllm-ascend' --exclude='refs/vllm/tests'
  --exclude='refs/vllm/docs' --exclude='refs/vllm/csrc'
  --exclude='refs/vllm/rust' --exclude='refs/vllm/examples'
)
for d in docs figures data scripts plan; do
  if [ -d "$d" ]; then
    mkdir -p "$STAGE/$NAME"
    tar cf - "${EXCLUDES[@]}" "$d" | ( cd "$STAGE/$NAME" && tar xf - )
  fi
done

# 超大盘点（只报告，不自动删；超过 40M 的文件会让包变得难下载）
echo "[pkg] 大于 40M 的文件（建议人工确认是否必要）："
find "$STAGE/$NAME" -type f -size +40M -printf '  %s  %p\n' 2>/dev/null | sort -rn | head -10 || true

# agents：只带 REPORT.md / *.md / *.json / *.csv（不带原始 perf.data 与临时物）
mkdir -p "$STAGE/$NAME/agents"
for d in agents/*/; do
  a=$(basename "$d")
  mkdir -p "$STAGE/$NAME/agents/$a"
  find "$d" -maxdepth 2 -type f \
    \( -name "*.md" -o -name "*.json" -o -name "*.csv" -o -name "*.tsv" \
       -o -name "*.txt" -o -name "*.log" -o -name "*.svg" \) \
    -size -8M -exec cp -f {} "$STAGE/$NAME/agents/$a/" \; 2>/dev/null || true
done

# refs：可选，只带关键源码文件清单（不带整棵树）
if [ "$WITH_SRC" = "1" ] && [ -d refs ]; then
  mkdir -p "$STAGE/$NAME/refs"
  for f in refs/vllm/vllm/v1/worker/gpu_model_runner.py \
           refs/vllm/vllm/v1/worker/gpu_input_batch.py \
           refs/vllm/vllm/v1/worker/block_table.py \
           refs/vllm/vllm/v1/core/sched/output.py; do
    [ -f "$f" ] && cp -f "$f" "$STAGE/$NAME/refs/$(echo "$f" | tr '/' '_')"
  done
  for f in refs/vllm-ascend/vllm_ascend/worker/model_runner_v1.py; do
    [ -f "$f" ] && cp -f "$f" "$STAGE/$NAME/refs/vllm-ascend_worker_model_runner_v1.py"
  done
fi

[ -f README.md ] && cp -f README.md "$STAGE/$NAME/" || true

# ---- 2. MANIFEST ----
( cd "$STAGE/$NAME" && find . -type f ! -name MANIFEST.txt -print0 \
    | sort -z | xargs -0 sha256sum > MANIFEST.txt )
{
  echo "# $NAME"
  echo "# generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# files: $(wc -l < "$STAGE/$NAME/MANIFEST.txt")"
  echo "# size: $(du -sh "$STAGE/$NAME" | cut -f1)"
} >> "$STAGE/$NAME/MANIFEST.txt"

TARBALL="$STAGE/$NAME.tar.gz"
( cd "$STAGE" && tar czf "$NAME.tar.gz" "$NAME" )
SHA=$(sha256sum "$TARBALL" | awk '{print $1}')
echo "$SHA  $(basename "$TARBALL")" > "$TARBALL.sha256"
SIZE_H=$(numfmt --to=iec --suffix=B "$(stat -c %s "$TARBALL")")
echo "[pkg] $TARBALL  $SIZE_H  sha256=$SHA"

if [ "$DRY" = "1" ]; then
  echo "[pkg] dry-run：保留 $TARBALL"
  echo "MANIFEST_HEAD:"; head -5 "$STAGE/$NAME/MANIFEST.txt"
  exit 0
fi

# ---- 3. 上传 ----
BASE=$(basename "$TARBALL")
KEY="$SHARE/$BASE"
echo "[coscli] uploading $BASE -> cos://$BUCKET/$KEY"
coscli cp "$TARBALL" "cos://$BUCKET/$KEY" 2>&1 | tail -2
coscli cp "$TARBALL.sha256" "cos://$BUCKET/$KEY.sha256" 2>&1 | tail -2 || true

for k in "$KEY" "$KEY.sha256"; do
  coscli object-acl --method put "cos://$BUCKET/$k" --acl public-read >/dev/null 2>&1 \
    || echo "[coscli] WARN: ACL set failed for $k" >&2
done

URL="https://$BUCKET_FQ/$KEY"
CODE=$(curl -s -o /dev/null -m 30 -w "%{http_code}" -r 0-0 "$URL" || echo 000)
echo "[check] external reachability: HTTP $CODE  $URL"
case "$CODE" in 200|206) ;; *) echo "[check] WARN: expected 200/206" >&2 ;; esac

# ---- 4. 登记 links-server ----
ssh -o BatchMode=yes "$LINKS_HOST" \
  "bash ~/links-server/link-add.sh $(printf '%q' "$NAME.tar.gz") $(printf '%q' "$URL") $(printf '%q' "$SIZE_H") $(printf '%q' "vLLM prepare_input CPU 侧分析交付包（文档+图+profiling 数据）")"

echo "[done] $URL"
