#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sanitize_for_publish.sh —— 发布前净化：把内部标识替换为可读占位符
#
#   bash scripts/sanitize_for_publish.sh [--check] [--revert]
#
# 目的：把本项目同步到 GitHub 之前，去掉**能定位到具体人或授予访问权**的标识，
#       同时保留技术内容与可复现性。替换是可逆的（见 docs/SANITIZATION.md）。
#
#   --check   只报告命中次数，不修改文件（默认行为，安全）
#   --apply   实际执行替换
#   --revert  把占位符换回原值
#
# 不处理的路径：refs/（upstream 源码，已被 .gitignore 排除）、coscli*（同上）
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODE=check
case "${1:-}" in
  --apply)  MODE=apply ;;
  --check|"") MODE=check ;;
  --revert) MODE=revert ;;
  *) echo "usage: $0 [--check|--apply|--revert]" >&2; exit 2 ;;
esac

# 只在文本文件上操作；跳过被排除的目录。
FIND_EXPR=(
  -type f
  -not -path "./refs/*"
  -not -path "./coscli_output/*"
  -not -name "coscli.log"
  -not -path "*/.git/*"
  -not -path "*/__pycache__/*"
  -not -name "*.pyc"
)

# 替换表：原值 → 占位符（顺序重要：长/具体的在前）
MAP=(
  "PUBLISHER_EMAIL|PUBLISHER_EMAIL"
  "LINKS_USER@LINKS_HOST|LINKS_USER@LINKS_HOST"
  "COS_BUCKET_FQ|COS_BUCKET_FQ"
  "COS_ENDPOINT|COS_ENDPOINT"
  "A3_22_IP|A3_22_IP"
  "LINKS_IP|LINKS_IP"
  "COS_BUCKET|COS_BUCKET"
  "LINKS_HOST|LINKS_HOST"
  "REMOTE_USER|REMOTE_USER"
  "HIST_PROJECT|HIST_PROJECT"
)
# 单独处理：OTHER_USER 需词边界，避免误伤哈希串
YXT_FROM='\byxt\b'
YXT_TO='OTHER_USER'

count_hits() {   # $1 = 字面量
  local lit=$1 n=0
  while IFS= read -r -d '' f; do
    c=$(grep -F -c -- "$lit" "$f" 2>/dev/null || true)
    [ -n "$c" ] && n=$((n + c))
  done < <(find . "${FIND_EXPR[@]}" -print0)
  echo "$n"
}

apply_edit() {   # $1 = sed 表达式（在文本文件上就地执行）
  local expr=$1 changed=0
  while IFS= read -r -d '' f; do
    # 跳过二进制
    if grep -qI . "$f" 2>/dev/null; then
      if sed -i "$expr" "$f" 2>/dev/null; then changed=$((changed + 1)); fi
    fi
  done < <(find . "${FIND_EXPR[@]}" -print0)
  echo "$changed"
}

echo "== 净化模式: $MODE =="
echo
printf '%-34s %s\n' "原值" "出现次数"
printf '%-34s %s\n' "----------------------------------" "--------"
for pair in "${MAP[@]}"; do
  from=${pair%%|*}
  printf '%-34s %s\n' "$from" "$(count_hits "$from")"
done
printf '%-34s %s\n' "OTHER_USER (词边界)" "$(count_hits OTHER_USER)"

if [ "$MODE" = check ]; then
  echo
  echo "（仅检查，未修改任何文件。执行：bash $0 --apply）"
  exit 0
fi

if [ "$MODE" = revert ]; then
  echo
  echo "== 还原 =="
  for ((i=${#MAP[@]}-1; i>=0; i--)); do
    from=${MAP[$i]%%|*}; to=${MAP[$i]##*|}
    n=$(apply_edit "s|${to}|${from}|g")
    echo "  ${to} -> ${from}  (扫描 ${n} 个文本文件)"
  done
  n=$(apply_edit "s/${YXT_TO}/OTHER_USER/g")
  echo "  ${YXT_TO} -> OTHER_USER  (扫描 ${n} 个文本文件)"
  echo
  echo "还原完成。建议复核：bash $0 --check"
  exit 0
fi

echo
echo "== 执行替换 =="
for pair in "${MAP[@]}"; do
  from=${pair%%|*}; to=${pair##*|}
  n=$(apply_edit "s|${from}|${to}|g")
  echo "  ${from} -> ${to}"
done
n=$(apply_edit "s/${YXT_FROM}/${YXT_TO}/g")
echo "  OTHER_USER -> ${YXT_TO}（词边界）"

echo
echo "== 复核（应全为 0）=="
for pair in "${MAP[@]}"; do
  from=${pair%%|*}
  printf '%-34s %s\n' "$from" "$(count_hits "$from")"
done
printf '%-34s %s\n' "OTHER_USER" "$(count_hits OTHER_USER)"

echo
echo "净化完成。原始版本（未净化）保存在 a3-22 与 COS 交付包中；"
echo "如需还原：bash $0 --revert"
