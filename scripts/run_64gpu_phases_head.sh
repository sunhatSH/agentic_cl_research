#!/usr/bin/env bash
# 64 卡多阶段连续训练主控脚本（head 节点运行）。
# 按实验路线依次启动：B1 → K1~K5 → R4 → C1~C4 → S1~S2。
# 每实验自动保留独立 ckpt + 日志 + rollout/val 数据 + 图表。
#
# 用法（head 节点）：
#   bash scripts/run_64gpu_phases_head.sh
# 每个 worker 节点也需跑本脚本（自动切 worker 模式）：
#   bash scripts/run_64gpu_phases_head.sh --worker --master-addr <HEAD_IP>
#
# 也可只跑单个 phase/实验：
#   bash scripts/run_64gpu_phases_head.sh --only phase2    # Phase 2 全实验
#   bash scripts/run_64gpu_phases_head.sh --only r4         # 单实验 R4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 参数 ──
ONLY=""
WORKER_MODE=0
MASTER_ADDR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2;;
    --worker) WORKER_MODE=1; MASTER_ADDR="${2:-}"; shift 1;;
    *) echo "[run_phases] 未知参数: $1" >&2; exit 2;;
  esac
done

# ── worker 模式：保持连接等 head ──
if [ "$WORKER_MODE" = "1" ]; then
  echo "[run_phases] worker 模式，连 $MASTER_ADDR"
  while true; do
    # 每轮重连 head 的 ray cluster，直到 head 发完所有实验
    ray start --address "$MASTER_ADDR:6379" 2>/dev/null || true
    # 等 head 的 train_cl.sh 完成当前实验后 ray stop，worker 自动退出
    sleep 10
    if ! pgrep -f "ray.*start.*$MASTER_ADDR" >/dev/null 2>&1; then
      ray stop --force 2>/dev/null || true
    fi
    ray stop --force 2>/dev/null || true
    sleep 5
  done
  exit 0
fi

# ── 实验路线（按设计文档 Phase 1→2→3→4→5，共 21 实验）─
# 每项: "phase_config 实验名 --buckets?"
# Phase 1: B1 遗忘基线（无 CL）
# Phase 2: KL 单独验证 K1-K5, K2-R
# Phase 3: R0(10k/25k), R3, R4, R4-w, R4-K（replay buffer 消融）
# Phase 4: C1-C4（curriculum / 桶序）
# Phase 5: S1, S2（最终 scale-up）
declare -a EXPERIMENTS=(
  "phase1/b1:B1"
  "phase2/k1:K1"
  "phase2/k2:K2"
  "phase2/k3:K3"
  "phase2/k4:K4"
  "phase2/k5:K5"
  "phase2/k2-r:K2-R"
  "phase3/r0-10k:R0-10K"
  "phase3/r0-25k:R0-25K"
  "phase3/r3:R3"
  "phase3/r4:R4"
  "phase3/r4-w:R4-W"
  "phase3/r4-k:R4-K"
  "phase4/c1:C1"
  "phase4/c2:C2"
  "phase4/c3:C3"
  "phase4/c4:C4"
  "phase5/s1:S1"
  "phase5/s2:S2"
)

# 是否按桶顺序训（B1/R 系列可能有 buckets 版）
# CL 实验(R4/C 系列)需要 --buckets，B1 baseline 不需要
# 简化：从实验名判断 —— 不含 "B" 的 CL 实验加 --buckets
BUCKETS_FLAGS=(
  "B1" "K1" "K2" "K3" "K4" "K5" "K2-R"
  "R0-10K" "R0-25K" "R3" "R4" "R4-W" "R4-K"
  "C1" "C2" "C3" "C4"
  "S1" "S2"
)

TRAIN_SCRIPT="$SCRIPT_DIR/train_64gpu.sh"
TOTAL=${#EXPERIMENTS[@]}

echo "================================================================"
echo "[run_phases] 64 卡多阶段连续训练"
echo "[run_phases] 共 $TOTAL 个实验"
echo "[run_phases] 每实验独立保留:"
echo "  ckpts/<exp>/                      模型 checkpoint"
echo "  logs/experiments/<exp>/train.log  训练日志"
echo "  logs/experiments/<exp>/rollout/    rollout 生成轨迹"
echo "  logs/experiments/<exp>/val/        验证生成文本"
echo "================================================================"

FINISHED=0
SKIPPED=0
FAILED=0

for i in "${!EXPERIMENTS[@]}"; do
  entry="${EXPERIMENTS[$i]}"
  phase_config="${entry%%:*}"
  exp="${entry##*:}"
  config="$ROOT_DIR/configs/$phase_config.yaml"

  if [[ -n "$ONLY" && "$exp" != "$ONLY" ]]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # 检查已完成的实验（ckpt 存在 = 跳过）
  if ls -d "$ROOT_DIR/ckpts/$exp"/global_step_* >/dev/null 2>&1; then
    echo "[run_phases] $((i+1))/$TOTAL  [$exp] ⏭ 已存 ckpt，跳过（加 FORCE=1 重跑）"
    SKIPPED=$((SKIPPED + 1))
    [ "${FORCE:-0}" != "1" ] && continue
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "[run_phases] $((i+1))/$TOTAL  ▶ [$exp]  $config"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # B1 不加 --buckets（baseline 单次非桶序），CL 实验加 --buckets
  _buckets=""
  if [[ "$exp" != "B1" ]] && [[ "$exp" != K* ]] && [[ "$exp" != "K2-R" ]]; then
    _buckets="--buckets"
  fi

  if bash "$TRAIN_SCRIPT" --config "$config" $_buckets; then
    FINISHED=$((FINISHED + 1))
    echo "[run_phases] ✓ $exp 成功"
  else
    FAILED=$((FAILED + 1))
    echo "[run_phases] ✗ $exp 失败（exit=$?），继续下一个" >&2
    # 不 abort：一个崩了继续跑后面的
  fi
done

echo ""
echo "================================================================"
echo "[run_phases] 完成: $FINISHED  |  跳过: $SKIPPED  |  失败: $FAILED"
echo "================================================================"
for entry in "${EXPERIMENTS[@]}"; do
  exp="${entry##*:}"
  ckpt=$(ls -d "$ROOT_DIR/ckpts/$exp"/global_step_* 2>/dev/null | wc -l)
  echo "  $exp  ckpt=$ckpt  | 日志: logs/experiments/$exp/"
done
