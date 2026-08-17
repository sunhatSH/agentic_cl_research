#!/usr/bin/env bash
# GDN kernel grid 探测脚本 —— 定位 fla chunk_gated_delta_rule_fwd_kernel_h_blockdim64
# 在 seqlen=65536 时 grid 越界（r0 GDN 崩溃 "Triton Error [CUDA]: invalid argument"）。
#
# 用法（集群任意有 conda 环境的节点，不需要 GPU）：
#   bash /mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research/scripts/analysis/probe_fla_gdn.sh
# 跑完把屏幕输出（或 logs/fla_gdn_probe.log）贴回给 Claude。
#
# 纯读源码 + pip 查版本，不 import fla、不建 CUDA context、不 launch kernel → 无 GPU 也能跑。
set -uo pipefail

OUT=/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research/logs/fla_gdn_probe.log
mkdir -p "$(dirname "$OUT")"

# fla 安装目录：优先 pip 查（不 import），失败再退固定路径。
F=$(pip show flash-linear-attention 2>/dev/null | awk -F': ' '/^Location/{print $2"/fla"}')
if [ -z "${F:-}" ] || [ ! -d "$F" ]; then
  F=/opt/conda/lib/python3.11/site-packages/fla
fi

{
  echo "===== 1. fla version + location ====="
  pip show flash-linear-attention 2>/dev/null | grep -iE "^(Name|Version|Location)"
  echo "resolved fla dir: $F"
  echo "dir exists: $([ -d "$F" ] && echo yes || echo NO)"
  echo

  CDH="$F/ops/common/chunk_delta_h.py"
  echo "===== 2. chunk_delta_h.py 620-700 (含崩溃点 692 的 grid= 公式) ====="
  if [ -f "$CDH" ]; then sed -n '620,700p' "$CDH"; else echo "MISSING: $CDH"; fi
  echo

  echo "===== 3. chunk_delta_h.py grid/BT/cdiv/NT/kernel 关键行 ====="
  if [ -f "$CDH" ]; then
    grep -nE "grid|BT|BS|BK|BV|chunk_size|cdiv|\bNT\b|def .*fwd_h|blockdim64" "$CDH"
  fi
  echo

  CK="$F/ops/gated_delta_rule/chunk.py"
  echo "===== 4. gated_delta_rule/chunk.py BT/chunk_size 常量 ====="
  if [ -f "$CK" ]; then
    grep -nE "BT|chunk_size|= 64|= 32|cdiv" "$CK" | head -30
  else
    echo "MISSING: $CK"
  fi
} > "$OUT" 2>&1

echo "done -> $OUT"
echo "---- 内容如下 ----"
cat "$OUT"
