#!/usr/bin/env bash
# 冷启动数据流水线：按桶 floor 补采循环 → warmup 灌桶 → parquet，一条脚本串完，tmux 后台跑。
#
# 五段（补采循环直到每桶「过质检的干净数」≥ floor，parquet 后置因为它不淘汰数据）：
#   1. 构造：build_topup_queries.py → 按桶分块 queries + finance/safety 从训练集 borrow 去重
#   2. 补采循环：{ run_cold_start(incremental,幂等) → qc_cold_start.sh(规则+LLM+purify) →
#                 按桶统计干净数 }，不够就提高 num-queries 再来一轮；最多 N 轮 / 候选耗尽即停
#   3. warmup：warmup_buffer.py 把过质检干净集按桶灌进 buffer_dumps/warmup.sqlite（冷启动最终产物）
#   4. parquet：trajectory_to_parquet.py 只转过质检的干净集（最后一步，不淘汰数据）
#
# 用法：
#   # 冒烟（少量 + 到质检为止，不 warmup/parquet）
#   bash scripts/cold_start_pipeline.sh --smoke --topup-step 4 --max-rounds 1 --no-parquet
#   # 正式（按桶达 floor，dsv4，tmux 后台）
#   tmux new -d -s cold "bash scripts/cold_start_pipeline.sh --max-concurrent 32"
#
# 参数（都有默认）：
#   --cold PATH         cold queries（默认 datasets/queries_cold.jsonl，只 seed 已打标）
#   --borrow-from PATH  借用训练集（默认 datasets/queries_train.jsonl；--no-borrow 禁用）
#   --floors CSV        9 桶 floor（默认读 configs/base.yaml 的 bucket_floors）
#   --overshoot F       每桶候选 = floor×F 预留质检淘汰（默认 1.4）
#   --max-rounds N      补采循环上限轮数（默认 5）
#   --topup-step N      每轮 num-queries 增量（默认 512）
#   --start-num N       首轮 num-queries（默认 = 各 floor×overshoot 之和的估计）
#   --max-concurrent N  沙箱并发（默认 32；两项目同跑调低）
#   --actor-model M     默认 deepseek-v4-pro
#   --out DIR           采集输出根（默认 agentic_cl_rollouts/real）
#   --tag NAME          输出子目录 tag（默认 model+时间）
#   --smoke             mock judge + 写 smoke/ 目录
#   --no-borrow         禁用训练集 borrow（finance/safety 缺就缺，记 warning）
#   --no-warmup         跳过 warmup 灌桶
#   --no-parquet        跳过 parquet（到 warmup 为止）
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

COLD="datasets/queries_cold.jsonl"
BORROW_FROM="datasets/queries_train.jsonl"
FLOORS=""                # 空 = 从 base.yaml 读
OVERSHOOT=1.4
MAX_ROUNDS=5
TOPUP_STEP=512
START_NUM=0              # 0 = 自动估
MAX_CONCURRENT=32
ACTOR_MODEL="deepseek-v4-pro"
OUT_ROOT="/mnt/afs_toolcall/sunhao4/agentic_cl_rollouts/real"
TAG=""
SMOKE=0
NO_BORROW=0
NO_WARMUP=0
NO_PARQUET=0

while [ $# -gt 0 ]; do
  case "$1" in
    --cold) COLD="$2"; shift 2;;
    --borrow-from) BORROW_FROM="$2"; shift 2;;
    --floors) FLOORS="$2"; shift 2;;
    --overshoot) OVERSHOOT="$2"; shift 2;;
    --max-rounds) MAX_ROUNDS="$2"; shift 2;;
    --topup-step) TOPUP_STEP="$2"; shift 2;;
    --start-num) START_NUM="$2"; shift 2;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2;;
    --actor-model) ACTOR_MODEL="$2"; shift 2;;
    --out) OUT_ROOT="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --smoke) SMOKE=1; OUT_ROOT="/mnt/afs_toolcall/sunhao4/agentic_cl_rollouts/smoke"; shift;;
    --no-borrow) NO_BORROW=1; shift;;
    --no-warmup) NO_WARMUP=1; shift;;
    --no-parquet) NO_PARQUET=1; shift;;
    *) echo "[pipeline] 未知参数: $1" >&2; exit 2;;
  esac
done

PY=/opt/conda/bin/python
[ -z "$TAG" ] && TAG="$(echo "$ACTOR_MODEL" | tr '/:.' '___')_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_ROOT/$TAG"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"

# 凭证 + PYTHONPATH（不带 verl，避免 scripts 命名冲突）
set -a; source scripts/load_tencent_env.sh 2>/dev/null; set +a
export PYTHONPATH="$ROOT:/mnt/afs_toolcall/sunhao4/workspace/LightLLM"

# floors 缺省从 base.yaml 读（bucket_floors）
if [ -z "$FLOORS" ]; then
  FLOORS="$("$PY" -c "
import re
for l in open('configs/base.yaml'):
    m=re.search(r'bucket_floors:\s*\[([0-9,\s]+)\]', l)
    if m: print(','.join(x.strip() for x in m.group(1).split(','))); break
")"
fi
echo "[pipeline] tag=$TAG out=$OUT model=$ACTOR_MODEL floors=$FLOORS overshoot=$OVERSHOOT smoke=$SMOKE" | tee "$LOG"

_smoke_flag=""; [ "$SMOKE" = "1" ] && _smoke_flag="--smoke"
_borrow_flag=""; [ "$NO_BORROW" = "1" ] && _borrow_flag="--no-borrow"

# ── 1. 构造补采就绪 queries（按桶分块 + borrow 去重）────────────────────────
echo "[pipeline] === 1/4 构造补采 queries ===" | tee -a "$LOG"
TOPUP_Q="$OUT/queries_topup.jsonl"
"$PY" scripts/build_topup_queries.py \
  --cold "$COLD" --borrow-from "$BORROW_FROM" --floors "$FLOORS" \
  --overshoot "$OVERSHOOT" $_borrow_flag --out "$TOPUP_Q" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 构造 queries 失败，停止" | tee -a "$LOG"; exit 3; }
TOTAL_ROWS=$(wc -l < "$TOPUP_Q")

# 首轮 num-queries：各 floor×overshoot 之和的估计（够覆盖一遍达标所需候选）
if [ "$START_NUM" -eq 0 ]; then
  START_NUM="$("$PY" -c "
fl=[int(x) for x in '$FLOORS'.split(',')]
print(min($TOTAL_ROWS, int(round(sum(fl)*$OVERSHOOT))))
")"
fi

# ── 2. 按桶补采循环 ─────────────────────────────────────────────────────
QC_OUT="$OUT/qc"
CLEAN=""
CUR_N="$START_NUM"
round=0
while [ "$round" -lt "$MAX_ROUNDS" ]; do
  round=$((round+1))
  echo "[pipeline] === 2/4 补采循环 round=$round  num-queries=$CUR_N/$TOTAL_ROWS ===" | tee -a "$LOG"

  # 采集（incremental 幂等：只补 MISSING/FAILED，同一 out_dir + 同一 queries 行序）
  "$PY" scripts/run_cold_start.py \
    --single-turn --actor-impl hermes_structured --actor-model "$ACTOR_MODEL" \
    --num-queries "$CUR_N" --max-concurrent "$MAX_CONCURRENT" --slot-timeout 900 \
    --collect-mode incremental $_smoke_flag \
    --out-dir "$OUT" --queries "$TOPUP_Q" \
    2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 采集失败，停止" | tee -a "$LOG"; exit 3; }

  RAW=$(find "$OUT" -name "grpo_hermes.jsonl" 2>/dev/null | head -1)
  [ -z "$RAW" ] && { echo "[pipeline] 未找到 grpo_hermes.jsonl，停止" | tee -a "$LOG"; exit 3; }
  echo "[pipeline] 采集产物: $RAW ($(wc -l < "$RAW") 行)" | tee -a "$LOG"

  # 质检（每轮重跑：LLMChecker resume 会复用已成功的轮次，只补新增/失败的）
  CONCURRENCY="$MAX_CONCURRENT" bash scripts/qc_cold_start.sh "$RAW" "$QC_OUT" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 质检失败，停止（不 warmup/parquet）" | tee -a "$LOG"; exit 4; }

  CLEAN=$(find "$QC_OUT" -name "*_llmchecked.jsonl" 2>/dev/null | head -1)
  [ -z "$CLEAN" ] && { echo "[pipeline] 未找到质检干净集，停止" | tee -a "$LOG"; exit 4; }

  # 按桶统计干净数，与 floor 比，算缺口
  GAP_JSON="$OUT/gaps_round${round}.json"
  "$PY" - "$CLEAN" "$FLOORS" "$GAP_JSON" <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, sys
from collections import Counter
clean, floors_csv, gap_out = sys.argv[1], sys.argv[2], sys.argv[3]
buckets = ['workflow','ops','qa','finance','office','communication','safety','coding','research']
floors = dict(zip(buckets, (int(x) for x in floors_csv.split(','))))
c = Counter()
for line in open(clean):
    line=line.strip()
    if not line: continue
    r=json.loads(line); c[(r.get('metadata') or {}).get('bucket')]+=1
gaps={}
print(f"  {'bucket':14} {'clean':>6} {'floor':>6} {'gap':>6}")
for b in buckets:
    have=c.get(b,0); g=max(0, floors[b]-have)
    if g>0: gaps[b]=g
    print(f"  {b:14} {have:>6} {floors[b]:>6} {('-'+str(g)) if g else 'OK':>6}")
json.dump(gaps, open(gap_out,'w'))
print(f"  未达标桶: {gaps if gaps else '无（全部达 floor）'}")
sys.exit(0 if not gaps else 7)   # 7 = 仍有缺口
PYEOF
  gap_rc="${PIPESTATUS[0]}"

  if [ "$gap_rc" -eq 0 ]; then
    echo "[pipeline] ✓ 所有桶达 floor（round=$round）" | tee -a "$LOG"
    break
  fi
  if [ "$CUR_N" -ge "$TOTAL_ROWS" ]; then
    echo "[pipeline] ⚠ 候选已耗尽（num=$CUR_N=total）仍有缺口，见 $GAP_JSON —— 停止补采" | tee -a "$LOG"
    break
  fi
  # 扩大覆盖：分块排列 → 提高 num-queries 自然覆盖缺量桶的更多候选
  CUR_N=$(( CUR_N + TOPUP_STEP )); [ "$CUR_N" -gt "$TOTAL_ROWS" ] && CUR_N="$TOTAL_ROWS"
done

if [ "$round" -ge "$MAX_ROUNDS" ]; then
  echo "[pipeline] ⚠ 达到 max-rounds=$MAX_ROUNDS，若仍有缺口见 $OUT/gaps_round${round}.json（不 fail，交人工决定）" | tee -a "$LOG"
fi
echo "[pipeline] 过质检干净集: $CLEAN ($(wc -l < "$CLEAN" 2>/dev/null) 行)" | tee -a "$LOG"

# ── 3. warmup 灌桶（冷启动最终产物）────────────────────────────────────────
if [ "$NO_WARMUP" = "1" ]; then
  echo "[pipeline] --no-warmup：到质检为止" | tee -a "$LOG"; exit 0
fi
echo "[pipeline] === 3/4 warmup 灌桶 ===" | tee -a "$LOG"
WARMUP_OUT="buffer_dumps/warmup_${TAG}.sqlite"
"$PY" scripts/warmup_buffer.py --input "$CLEAN" --out "$WARMUP_OUT" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] warmup 失败" | tee -a "$LOG"; exit 6; }
echo "[pipeline] warmup buffer → $WARMUP_OUT" | tee -a "$LOG"

# ── 4. 转 parquet（最后一步，只转过质检的，不淘汰数据）──────────────────────
if [ "$NO_PARQUET" = "1" ]; then
  echo "[pipeline] --no-parquet：到 warmup 为止" | tee -a "$LOG"; exit 0
fi
echo "[pipeline] === 4/4 转 parquet ===" | tee -a "$LOG"
"$PY" scripts/trajectory_to_parquet.py --input "$CLEAN" --out-dir datasets 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 转 parquet 失败" | tee -a "$LOG"; exit 5; }

echo "[pipeline] ✓ 全部完成 → warmup=$WARMUP_OUT | datasets/{train,val}.parquet | 日志 $LOG" | tee -a "$LOG"
