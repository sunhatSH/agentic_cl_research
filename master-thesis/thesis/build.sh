#!/usr/bin/env bash
# 论文构建脚本 —— 在 master-thesis/ 下可复现编译。
# 用法：  bash thesis/build.sh         # 编译
#         bash thesis/build.sh clean   # 清理产物
# 产物：  thesis/最终稿/main.pdf
set -euo pipefail

# 切到本脚本所在目录的上一级（master-thesis/），再进 thesis/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"                     # now in master-thesis/thesis/

# ructhesis.cls / 字体 / figures 在上一级（master-thesis/），加进搜索路径
export TEXINPUTS="..:${TEXINPUTS:-}"

if [[ "${1:-}" == "clean" ]]; then
  latexmk -C -output-directory=最终稿 main.tex
  rm -f 最终稿/*.bcf* 最终稿/*.bbl* 最终稿/ref.bib
  echo "cleaned."
  exit 0
fi

# latexmk 自动跑 xelatex → biber → xelatex ×2（biber 在 thesis/ 目录解析 latex/ref.bib）
# 注意：显式传 ./main.tex，避免 TEXINPUTS 含 ".." 时 latexmk 误解析到上级模板的 main.tex
latexmk -xelatex \
  -interaction=nonstopmode \
  -file-line-error \
  -output-directory=最终稿 \
  ./main.tex

echo
echo "=== 产物 ==="
ls -la 最终稿/main.pdf
echo
echo "=== 错误检查 ==="
grep -E "^! " 最终稿/main.log | sort -u || echo "（无致命错误）"
echo
echo "=== 未定义引用 ==="
grep -cE "Citation .* undefined|Reference .* undefined" 最终稿/main.log || true
