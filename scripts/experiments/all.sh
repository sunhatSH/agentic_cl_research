#!/usr/bin/env bash
# 汇总入口：按顺序跑 scripts/experiments/ 下的【所有】单实验脚本(每个 = 训练+评测)。
# 当前 = b1(baseline) + r4(完整方案)。新增实验见 README.md。
#   启动: bash scripts/experiments/all.sh
#         NNODES=4 bash scripts/experiments/all.sh   # 32 卡
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 实验顺序(按此列表逐个训练+评测)。加实验就往这里追加脚本名。
EXPERIMENTS=(
    b1.sh
    r4.sh
)

rc=0
for e in "${EXPERIMENTS[@]}"; do
    echo "################ [all] 实验 $e ################"
    bash "$SCRIPT_DIR/$e" "$@" || { echo "[all] 实验 $e 失败(继续后续)" >&2; rc=1; }
done
exit $rc
