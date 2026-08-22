#!/usr/bin/env bash
# 数据 Pipeline：taskspec → 训练数据（7 步串起来）
# 见 doc/训练与推理流程.md §6.5 / paper/drafts/Paper_Method_draft_CN.md §4.6
#
# 7 步：
#   1. build_fs_seeds        : taskspec files/ → 沙箱 seed 镜像
#   2. label_buckets         : taskspec → 9 桶能力标签（--write 回写 manifest）
#   3. taskspec_to_queries   : taskspec → queries.jsonl（新增脚本）
#   4. collect_cold          : queries + fs-seeds → trajectory（需集群+沙箱+GPU）
#   5. warmup_buffer         : trajectory → buffer.sqlite 预热
#   6. trajectory_to_parquet : trajectory → train/val.parquet（新增脚本）
#   7. (填 cluster.yaml，训练用 train_Ngpu.sh)
#
# 用法：
#   bash scripts/pipeline/run_data_pipeline.sh --offline      # Step 1-3,6（离线，本机/开发机，无 GPU）
#   bash scripts/pipeline/run_data_pipeline.sh --collect      # Step 4-5（需集群+沙箱+GPU）
#   bash scripts/pipeline/run_data_pipeline.sh --all          # 全跑（1-6）
#   bash scripts/pipeline/run_data_pipeline.sh --only 3       # 只跑 Step 3
#   bash scripts/pipeline/run_data_pipeline.sh --limit 50     # 只处理前 50 个 task（调试）
#
# 环境变量（可选，覆盖默认路径）：
#   TASKSPECS_DIR   默认 datasources/taskspecs
#   QUERIES_FILE    默认 datasets/queries.jsonl
#   ROLLOUT_DIR     默认 datasources/mock/rollouts
#   BUFFER_FILE     默认 logs/cold/buffer.sqlite
#   WARMUP_FILE     默认 buffer_dumps/warmup.sqlite
#   DATASETS_DIR    默认 datasets
#   COLLECT_BACKEND 默认 e2b（collect_cold 的沙箱后端）
#   COLLECT_ACTOR_BASE / COLLECT_ACTOR_MODEL  默认本地 vllm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

# ---- 路径（与 doc/训练与推理流程.md §6.5 表对齐）--------------------------
TASKSPECS_DIR="${TASKSPECS_DIR:-$ROOT_DIR/datasources/taskspecs}"
FS_SEEDS_DIR="$ROOT_DIR/docker/sandbox/fs-seeds"
MANIFEST="$FS_SEEDS_DIR/manifest.json"
QUERIES_FILE="${QUERIES_FILE:-$ROOT_DIR/datasets/queries.jsonl}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$ROOT_DIR/datasources/mock/rollouts}"
BUFFER_FILE="${BUFFER_FILE:-$ROOT_DIR/logs/cold/buffer.sqlite}"
WARMUP_FILE="${WARMUP_FILE:-$ROOT_DIR/buffer_dumps/warmup.sqlite}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT_DIR/datasets}"
COLLECT_BACKEND="${COLLECT_BACKEND:-e2b}"
COLLECT_ACTOR_BASE="${COLLECT_ACTOR_BASE:-http://127.0.0.1:8000/v1}"
COLLECT_ACTOR_MODEL="${COLLECT_ACTOR_MODEL:-cold-actor}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"

# ---- 参数解析 ------------------------------------------------------------
DO_OFFLINE=0; DO_COLLECT=0; ONLY=""; LIMIT=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --offline)  DO_OFFLINE=1; shift ;;
        --collect)  DO_COLLECT=1; shift ;;
        --all)      DO_OFFLINE=1; DO_COLLECT=1; shift ;;
        --only)     ONLY="$2"; shift 2 ;;
        --limit)    LIMIT="--limit $2"; shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if [[ $DO_OFFLINE -eq 0 && $DO_COLLECT -eq 0 && -z "$ONLY" ]]; then
    echo "用法: bash scripts/pipeline/run_data_pipeline.sh --offline|--collect|--all|--only <N>" >&2
    echo "  --offline  Step 1,2,3,6（离线预处理，无 GPU）" >&2
    echo "  --collect  Step 4,5（冷启动采集，需集群+沙箱+GPU）" >&2
    echo "  --all      Step 1-6 全跑" >&2
    echo "  --only 3   只跑指定 Step" >&2
    exit 2
fi

run_step() {
    local n="$1"; shift
    if [[ -z "$ONLY" || "$ONLY" == "$n" ]]; then
        echo ""
        echo "================ Step $n ================"
        "$@"
    fi
}

# ---- Step 1: build_fs_seeds（taskspec files/ → 沙箱 seed）-----------------
run_step 1 python "$ROOT_DIR/scripts/data/build_fs_seeds.py" $LIMIT

# ---- Step 2: label_buckets（taskspec → 9 桶标签，--write 回写 manifest）--
run_step 2 python "$ROOT_DIR/scripts/data/label_buckets.py" --write

# ---- Step 3: taskspec_to_queries（taskspec → queries.jsonl）---------------
run_step 3 python "$ROOT_DIR/scripts/data/taskspec_to_queries.py" \
    --taskspecs "$TASKSPECS_DIR" \
    --output "$QUERIES_FILE" $LIMIT

# ---- Step 4: collect_cold（queries + fs-seeds → trajectory，需集群）------
if [[ $DO_COLLECT -eq 1 || "$ONLY" == "4" ]]; then
    run_step 4 python "$ROOT_DIR/scripts/collect/collect_cold.py" \
        --queries "$QUERIES_FILE" \
        --backend "$COLLECT_BACKEND" \
        --actor-base "$COLLECT_ACTOR_BASE" \
        --actor-model "$COLLECT_ACTOR_MODEL" \
        --out "$BUFFER_FILE" \
        "${EXTRA_ARGS[@]}"
fi

# ---- Step 5: warmup_buffer（trajectory → buffer.sqlite 预热）--------------
if [[ $DO_COLLECT -eq 1 || "$ONLY" == "5" ]]; then
    run_step 5 python "$ROOT_DIR/scripts/warmup_buffer.py" \
        --in-dir "$ROLLOUT_DIR" \
        --out "$WARMUP_FILE"
fi

# ---- Step 6: trajectory_to_parquet（trajectory → train/val.parquet）-------
run_step 6 python "$ROOT_DIR/scripts/data/trajectory_to_parquet.py" \
    --input "$ROLLOUT_DIR"/cold/*.jsonl \
    --out-dir "$DATASETS_DIR" \
    --val-fraction "$VAL_FRACTION"

echo ""
echo "================ Pipeline 完成 ================"
echo "Step 7（手动）: 在 configs/cluster.yaml 填 data.train_files / val_files 指向："
echo "  $DATASETS_DIR/train.parquet"
echo "  $DATASETS_DIR/val.parquet"
echo "然后: bash scripts/train.sh 8gpu（或 16gpu/32gpu/64gpu）"
