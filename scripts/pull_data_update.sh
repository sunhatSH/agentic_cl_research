#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 马来集群侧:从 AOSS 桶拉取最新训练数据(2026-08-28 上传的 3200×2 数据集 + taskspecs_w3)
#
# 在【马来集群】上跑(仓库根 /mnt/afs/sunhao4/workspace/agentic_cl_research)。
# 桶与上传侧同(data_transfer_hub,endpoint cn-tj-01),马来从它同步。
#
# 拉什么:
#   1. datasets/train_cl.parquet + train_cl.jsonl  — 6400 行新数据集(coding3200+office3200)
#   2. taskspecs_w3.tar.gz (8.3G) → 解压成 datasources/taskspecs_w3/  — 51909 任务 files+answer_key
#      (马来侧之前缺 taskspecs_w3,只有旧 taskspecs/)
#
# 用法(马来集群):
#   bash scripts/pull_data_update.sh            # 拉 + 解压
#   bash scripts/pull_data_update.sh --dry-run  # 只列不拉
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
RCLONE="${RCLONE_BIN:-rclone}"          # 马来侧 rclone 路径(不同则 export RCLONE_BIN=)
CONF=/tmp/aoss_rclone.conf
DEST="aoss:data_transfer_hub/sunhao4/workspace/agentic_cl_research"
DRY="${1:-}"

# ── rclone config(与上传侧一致;凭证从 ~/.aws/credentials 读)──
if [ ! -f "$CONF" ]; then
  AK=$(awk -F'= ' '/aws_access_key_id/{print $2}' ~/.aws/credentials)
  SK=$(awk -F'= ' '/aws_secret_access_key/{print $2}' ~/.aws/credentials)
  cat > "$CONF" <<EOF
[aoss]
type = s3
provider = Other
access_key_id = $AK
secret_access_key = $SK
endpoint = http://aoss-internal.cn-tj-01.sensecoreapi-oss.cn
EOF
  chmod 600 "$CONF"
fi
R="$RCLONE --config $CONF"

echo "=== 马来侧拉取开始 $(date) ==="

# ── 1. 训练集 parquet/jsonl(小,~19M)──
echo ">>> [1/3] 拉 datasets/train_cl.*"
if [ "$DRY" != "--dry-run" ]; then
  $R copy "$DEST/datasets/" datasets/ \
    --include "train_cl.parquet" --include "train_cl.jsonl" \
    --transfers 4 --checkers 8 --progress 2>&1 | tail -3
fi

# ── 2. taskspecs_w3.tar.gz(8.3G)──
echo ">>> [2/3] 拉 taskspecs_w3.tar.gz (8.3G)"
if [ "$DRY" != "--dry-run" ]; then
  $R copy "$DEST/taskspecs_w3.tar.gz" ./ --transfers 1 --progress 2>&1 | tail -3
fi

# ── 3. 解压 taskspecs_w3 → datasources/ ──
echo ">>> [3/4] 解压 taskspecs_w3.tar.gz → datasources/taskspecs_w3/"
if [ "$DRY" != "--dry-run" ]; then
  tar xzf taskspecs_w3.tar.gz -C datasources/
  echo "解压后子目录数: $(ls datasources/taskspecs_w3 | wc -l) (应 51909)"
fi

# ── 4. 同步更新的【代码文件】(马来非 git 仓库,代码也走桶)──
#    GT 加载修复(prompts.py F14/F15) + batch 统一 32(base + 实验 config)
echo ">>> [4/4] 同步代码文件(prompts.py + batch config)"
if [ "$DRY" != "--dry-run" ]; then
  $R copy "$DEST/agents/prompts.py" agents/ 2>&1 | tail -1
  $R copy "$DEST/configs/base.yaml" configs/ 2>&1 | tail -1
  $R copy "$DEST/configs/exp1/cl2r_base.yaml" configs/exp1/ 2>&1 | tail -1
  $R copy "$DEST/configs/run/" configs/run/ \
    --include "b1_9b_16gpu.yaml" --include "k2_9b_16gpu.yaml" --include "r0-25k_9b_16gpu.yaml" 2>&1 | tail -1
  echo "  代码同步完成; 校验 batch: $(grep -m1 train_batch_size configs/base.yaml)"
fi

echo "=== 完成 $(date) ==="
echo "校验:"
echo "  python3 -c \"import pyarrow.parquet as pq; print(pq.read_table('datasets/train_cl.parquet').num_rows)\"  # 6400"
echo "  ls datasources/taskspecs_w3 | wc -l   # 51909"
echo "  grep train_batch_size configs/base.yaml   # 32"
