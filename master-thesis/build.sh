#!/usr/bin/env bash
# 论文构建脚本 —— 在 master-thesis/ 下可复现编译。
# 用法：  bash build.sh         # 编译
#         bash build.sh clean   # 清理产物
# 产物：  最终稿/main.pdf
set -euo pipefail

# 切到本脚本所在目录（master-thesis/）——ructhesis.cls / 字体 / figures / latex 均在此层
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"                     # now in master-thesis/

# 中文字体按文件名寻址（simsun.ttc 等在本目录），交给 xelatex 的字体搜索路径
export OSFONTDIR="${OSFONTDIR:-$HOME/.fonts}"

if [[ "${1:-}" == "clean" ]]; then
  latexmk -C -output-directory=最终稿 main.tex
  rm -f 最终稿/*.bcf* 最终稿/*.bbl* 最终稿/ref.bib
  echo "cleaned."
  exit 0
fi

# latexmk 自动跑 xelatex → biber → xelatex ×2（biber 解析 latex/ref.bib）
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
