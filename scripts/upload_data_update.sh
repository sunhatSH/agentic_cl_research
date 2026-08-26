#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 增量迁移：把修 bug 后的新数据传到 AOSS，更新马来集群
#
# 传什么（只传新增/更新的，不重复传已有 tar）：
#   1. review_ws.tar.gz      — F9 新增 review 任务 workspace 快照（336M→压缩95M）
#   2. datasources/labeled/  — 8-22 新增的一堆 jsonl（difficulty/correspondence 等）
#   3. datasets/train_cl.* + train_exp2.* — 修 bug 后重造的训练集（小，直接传）
#
# 不传（AOSS 已有，8-21 push 完成）：
#   - taskspecs_w3.tar / generated_tasks_hermes.tar（28G each，源 7-22 后没变）
#   - train.parquet / train_aligned.parquet（旧实验，未更新）
#
# 用法：
#   bash scripts/upload_data_update.sh          # 压缩 + 传
#   bash scripts/upload_data_update.sh --dry-run  # 只列清单不传
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
R="/mnt/afs_toolcall/sunhao4/bin/rclone --config /tmp/aoss_rclone.conf"
DEST="aoss:data_transfer_hub/sunhao4/workspace/agentic_cl_research"
DRY="${1:-}"

echo "=== 增量迁移开始 ==="

# ── 0. 确保 rclone config 在 ──
if [ ! -f /tmp/aoss_rclone.conf ]; then
  AK=$(awk -F'= ' '/aws_access_key_id/{print $2}' /mnt/afs_toolcall/sunhao4/.aws/credentials)
  SK=$(awk -F'= ' '/aws_secret_access_key/{print $2}' /mnt/afs_toolcall/sunhao4/.aws/credentials)
  cat > /tmp/aoss_rclone.conf <<EOF
[aoss]
type = s3
provider = Other
access_key_id = $AK
secret_access_key = $SK
endpoint = http://aoss-internal.cn-tj-01.sensecoreapi-oss.cn
EOF
  chmod 600 /tmp/aoss_rclone.conf
fi

# ── 1. 压缩 review_ws（大目录，336M→95M）──
echo ">>> [1/4] 压缩 review_ws"
if [ "$DRY" != "--dry-run" ]; then
  tar czf /tmp/review_ws.tar.gz datasources/review_ws/
  ls -lh /tmp/review_ws.tar.gz
fi

# ── 2. 传 review_ws.tar.gz ──
echo ">>> [2/4] 传 review_ws.tar.gz"
if [ "$DRY" != "--dry-run" ]; then
  $R copy /tmp/review_ws.tar.gz "$DEST/review_ws.tar.gz" --progress 2>&1 | tail -2
fi

# ── 3. 传 datasources/labeled（新增 jsonl，小，直接传）──
echo ">>> [3/4] 传 datasources/labeled"
if [ "$DRY" != "--dry-run" ]; then
  $R copy datasources/labeled/ "$DEST/datasources/labeled/" \
    --transfers 8 --checkers 16 --progress 2>&1 | tail -3
fi

# ── 4. 传重造的训练集（小，直接传；rclone 用 --include 过滤）──
echo ">>> [4/4] 传训练集 parquet/jsonl"
if [ "$DRY" != "--dry-run" ]; then
  $R copy datasets/ "$DEST/datasets/" \
    --include "train_cl.parquet" --include "train_cl.jsonl" \
    --include "train_exp2.parquet" --include "train_exp2.jsonl" \
    --transfers 8 --checkers 16 --progress 2>&1 | tail -3
fi

echo ""
echo "=== 增量迁移完成 ==="
echo "马来侧解压 review_ws："
echo "  cd /mnt/afs/sunhao4/workspace/agentic_cl_research && tar xzf review_ws.tar.gz"
