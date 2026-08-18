#!/usr/bin/env bash
# r0 4 卡调试启动（单机 4 GPU）——复现 16 卡崩溃用。
#
# 保持 r0-25k_9b_16gpu.yaml 的机制层不变（fsdp2 + remove_padding + fused triton
# + SP=4 + custom_language_model），只把 nnodes/gpus 压到 1×4：
#   SP=4 → DP=4÷4=1；train_batch=32×n8=256 % DP1=0；ppo_mini=32 % DP1=0。
#
# ★ 独立实验名（防和 16 卡抢 log）：4 卡和 16 卡共享同一 config 的 experiment_name
#   → 会写同一个 logs/experiments/<exp>/train.log，4 卡启动时把 16 卡活日志 mv 进
#   archive/ 并 tee 覆盖。故这里把实验名 _16gpu 换成 _4gpu（无 _16gpu 则追加），并
#   同时 override 两处：
#     · --exp-name <name>              → 控制 shell 层 _exp（train.log / ckpts /
#       logs/metrics / rollouts/training），见 _train_impl.sh 的 _exp_name/_LOGDIR。
#     · trainer.experiment_name=<name> → 控制 verl FileLogger/SWANLAB + cl_replay_hook_v1
#       的 buffer_stats/buffer_dumps/rollouts 路径（它读 config 的 trainer.experiment_name）。
#   两处都要改，否则 train.log 和 buffer_dumps/rollouts 会一半进 16 卡目录、一半进 4 卡目录。
#   实验名自动在 logs/experiments/<name>/ 下新开一个目录（_train_impl.sh mkdir -p）。
#
# 用法：
#   bash scripts/train_4gpu.sh                                   # 用 r0-25k_9b_16gpu.yaml
#   bash scripts/train_4gpu.sh configs/run/r0-25k_9b_16gpu.yaml trainer.total_training_steps=2
#   EXP_NAME=qwen35_9b_r0-25k_4gpu bash scripts/train_4gpu.sh    # 手动指定实验名
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
IMPL="$DIR/_train_impl.sh"

CONFIG="${1:-configs/run/r0-25k_9b_16gpu.yaml}"

# 4 卡实验名：优先 EXP_NAME，否则从 config 的 experiment_name 派生（_16gpu → _4gpu）。
_exp="${EXP_NAME:-}"
_base=""
if [ -z "$_exp" ]; then
  _base="$(grep -E '^[[:space:]]*experiment_name:' "$ROOT/$CONFIG" 2>/dev/null | head -1 \
    | sed -E 's/.*experiment_name:[[:space:]]*//;s/[[:space:]"'"'"']*//g')" || true
  _base="${_base:-$(basename "$CONFIG" .yaml)}"
  _exp="${_base//_16gpu/_4gpu}"
  case "$_exp" in
    *_4gpu) : ;;                 # 已含 _4gpu（r0-25k_16gpu → r0-25k_4gpu）
    *) _exp="${_exp}_4gpu" ;;    # 无 _16gpu 时兜底追加
  esac
fi

if [ -n "$_base" ]; then
  echo "[train_4gpu] 实验名 = $_exp（16 卡同 config 名 = $_base，已隔离）"
else
  echo "[train_4gpu] 实验名 = $_exp"
fi

exec bash "$IMPL" \
  --nnodes 1 --gpus-per-node 4 \
  --rollout-tp 2 --ulysses-sp 4 \
  --train-batch 32 --ppo-mini 32 \
  --gpu-mem-util 0.75 \
  --exp-name "$_exp" \
  --config "$ROOT/$CONFIG" \
  "trainer.experiment_name=$_exp" \
  "trainer.nnodes=1" "trainer.n_gpus_per_node=4" \
  "${@:2}"
