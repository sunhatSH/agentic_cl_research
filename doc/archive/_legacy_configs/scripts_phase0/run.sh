#!/usr/bin/env bash
# Phase 0 — 冷启动数据来源消融 (P0-A..E)
# 见 doc/Plan_冷启动数据来源消融.md
#
# 三阶段判定链：
#   Stage 1  从两来源目录 (27B + gpt5) 按配比生成 5 个 buffer (warmup_buffer.py --ratio-27b)
#   Stage 2  轨1 冷启动自身指标 gate (无 GPU)：桶配额/tool-call/judge/多样性
#   Stage 3  轨2 下游短RL 验收 (占 GPU)：仅通过 Stage 2 的臂，固定新任务+同种子
#
# 用法：
#   bash scripts/phase0/run.sh --collect       # 只做 Stage 1 (建 5 buffer)
#   bash scripts/phase0/run.sh --gate          # 只做 Stage 2 (轨1 gate，无 GPU)
#   bash scripts/phase0/run.sh --train         # 只做 Stage 3 (短RL，需 GPU)
#   bash scripts/phase0/run.sh --all           # 全跑
#   bash scripts/phase0/run.sh --only p0-c ... # 单臂
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_DIR="$ROOT_DIR/configs/phase0"
DUMP_DIR="${DUMP_DIR:-$ROOT_DIR/data/mock/buffer_dumps}"
ROLLOUT_DIR="${ROLLOUT_DIR:-$ROOT_DIR/data/mock/rollouts}"   # {local,remote}/rollouts_*.jsonl
MIX_SEED="${MIX_SEED:-0}"

# 臂定义：name  ratio_27b  buffer_file
ARMS=(
    "p0-a 1.0 p0-a_27b100.sqlite"
    "p0-b 0.0 p0-b_gpt5100.sqlite"
    "p0-c 0.5 p0-c_mix5050.sqlite"
    "p0-d 0.7 p0-d_mix7030.sqlite"
    "p0-e 0.3 p0-e_mix3070.sqlite"
)

DO_COLLECT=0; DO_GATE=0; DO_TRAIN=0; ONLY=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --collect) DO_COLLECT=1; shift ;;
        --gate)    DO_GATE=1; shift ;;
        --train)   DO_TRAIN=1; shift ;;
        --all)     DO_COLLECT=1; DO_GATE=1; DO_TRAIN=1; shift ;;
        --only)    ONLY="$2"; shift 2 ;;
        *)         EXTRA_ARGS+=("$1"); shift ;;
    esac
done
if [[ $DO_COLLECT -eq 0 && $DO_GATE -eq 0 && $DO_TRAIN -eq 0 ]]; then
    echo "nothing to do; pass --collect / --gate / --train / --all" >&2
    exit 2
fi

# ---- Stage 1: build the 5 ratio-mixed buffers -------------------------------
if [[ $DO_COLLECT -eq 1 ]]; then
    echo "### Phase 0 Stage 1: build ratio-mixed buffers from $ROLLOUT_DIR"
    for arm in "${ARMS[@]}"; do
        read -r name ratio bfile <<< "$arm"
        [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
        echo "=== [$name] ratio-27b=$ratio -> $bfile ==="
        python "$ROOT_DIR/scripts/warmup_buffer.py" \
            --in-dir "$ROLLOUT_DIR" \
            --out "$DUMP_DIR/$bfile" \
            --ratio-27b "$ratio" \
            --mix-seed "$MIX_SEED"
    done
fi

# ---- Stage 2: 轨1 冷启动自身指标 gate (无 GPU) ------------------------------
if [[ $DO_GATE -eq 1 ]]; then
    echo "### Phase 0 Stage 2: cold-start self metrics gate (no GPU)"
    for arm in "${ARMS[@]}"; do
        read -r name ratio bfile <<< "$arm"
        [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
        echo "=== gate [$name] ($bfile) ==="
        python "$ROOT_DIR/scripts/phase0/gate_coldstart.py" \
            --buffer "$DUMP_DIR/$bfile" \
            --manifest "$DUMP_DIR/${bfile%.sqlite}.manifest.json" \
            "${EXTRA_ARGS[@]}"
    done
fi

# ---- Stage 3: 轨2 下游短RL 验收 (需 GPU) ------------------------------------
if [[ $DO_TRAIN -eq 1 ]]; then
    echo "### Phase 0 Stage 3: downstream short-RL acceptance (GPU)"
    for arm in "${ARMS[@]}"; do
        read -r name ratio bfile <<< "$arm"
        [[ -n "$ONLY" && "$name" != "$ONLY" ]] && continue
        if [[ ! -f "$DUMP_DIR/$bfile" ]]; then
            echo "!! [$name] buffer $bfile missing; run --collect first" >&2
            exit 3
        fi
        echo "=== short-RL [$name] ==="
        bash "$ROOT_DIR/scripts/train.sh" "$CONFIG_DIR/${name}.yaml" "${EXTRA_ARGS[@]}"
    done
fi
