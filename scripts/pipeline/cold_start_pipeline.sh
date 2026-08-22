#!/usr/bin/env bash
# 冷启动数据流水线：按桶 floor 补采循环 → warmup 灌桶 → parquet，一条脚本串完，tmux 后台跑。
#
# 流程（补采直到每桶「过质检的干净数」≥ floor，parquet 后置因为它不淘汰数据）：
#   1. 构造：build_topup_queries.py → 仅 cold queries 按桶分块（不 borrow）
#   2. 补采 = 阶段A + 阶段B：
#      阶段A(cold-only)：run_cold_start(incremental) → qc_cold_start.sh(规则+LLM+purify)
#                        → 按桶统计干净数；不够就提高 num-queries，直到 cold 候选耗尽/全达标
#      阶段B(缺了才 borrow)：cold 采完仍缺的桶 → build_topup_queries --gaps 只借缺口桶、
#                        追加到 queries 尾部（record_id 跨轮去重）→ 继续采质检，直到达标/借尽
#   3. warmup：warmup_buffer.py 把过质检干净集按桶灌进 buffer_dumps/warmup.sqlite（冷启动最终产物）
#   4. parquet：trajectory_to_parquet.py 只转过质检的干净集（最后一步，不淘汰数据）
#
# 用法：
#   # 冒烟（少量 + 到质检为止，不 warmup/parquet）
#   bash scripts/pipeline/cold_start_pipeline.sh --smoke --topup-step 4 --max-rounds 1 --no-parquet
#   # 正式（按桶达 floor，dsv4，tmux 后台）
#   tmux new -d -s cold "bash scripts/pipeline/cold_start_pipeline.sh --max-concurrent 32"
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

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

COLD="datasets/queries_cold.jsonl"
BORROW_FROM="datasets/queries_train.jsonl"
FLOORS=""                # 空 = 从 base.yaml 读
OVERSHOOT=1.4
MAX_ROUNDS=5
TOPUP_STEP=512
START_NUM=0              # 0 = 自动估
MAX_CONCURRENT=32
ACTOR_MODEL="deepseek-v4-pro-202606"
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

PY="${PYTHON:-$(which python3 || which python)}"
[ -z "$TAG" ] && TAG="$(echo "$ACTOR_MODEL" | tr '/:.' '___')_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$OUT_ROOT/$TAG"
mkdir -p "$OUT"
LOG="$OUT/pipeline.log"

# 凭证 + PYTHONPATH（不带 verl，避免 scripts 命名冲突）
set -a; source scripts/env/load_tencent_env.sh 2>/dev/null; set +a
export PYTHONPATH="$ROOT/src:$ROOT:/mnt/afs_toolcall/sunhao4/workspace/LightLLM"

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
# 单轮采集（2026-07-23）：全链路单轮化——训练侧 verl 固定 batch 契约 vs 多轮变长产出
# 不可调和，已关闭 Questioner 追问（见 rollout/simulated_session.py docstring）。
# 冷采集同步单轮：只跑 seed query，不调 observer/questioner（单轮不需要它们）。
# 质检（规则+LLM+purify）仍照常跑。--single-turn 显式开启正式单轮；冒烟也走单轮。
SINGLE_TURN=1                       # 默认单轮（全链路单轮化）；--multi-turn 可回退（deprecated）
_usersim_flag="--no-usersim"        # 单轮 = 跳过 observer/questioner
[ "$SMOKE" = "1" ] && _smoke_flag="--smoke"
# NO_BORROW=1 时跳过阶段B（cold 缺就缺，记 warning），见段②守卫。

# ── 1. 构造 cold-only queries（按桶分块，不 borrow）─────────────────────────
echo "[pipeline] === 1/4 构造 cold-only queries ===" | tee -a "$LOG"
TOPUP_Q="$OUT/queries_topup.jsonl"
"$PY" scripts/data/build_topup_queries.py \
  --cold "$COLD" --floors "$FLOORS" --out "$TOPUP_Q" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 构造 queries 失败，停止" | tee -a "$LOG"; exit 3; }

QC_OUT="$OUT/qc"
CLEAN=""
GAP_JSON=""

# 采集(incremental 幂等) → 质检 → 按桶算缺口。用 $CUR_N 控本轮覆盖行数。
# 设 GAP_JSON / CLEAN 全局供后续用；返回码 0=全达标 7=仍有缺口。
collect_qc_gap() {
  local cur_n="$1" tag_round="$2"
  "$PY" scripts/collect/run_cold_start.py \
    $_usersim_flag --actor-impl hermes_structured --actor-model "$ACTOR_MODEL" \
    --num-queries "$cur_n" --max-concurrent "$MAX_CONCURRENT" --slot-timeout 900 \
    --collect-mode incremental $_smoke_flag \
    --out-dir "$OUT" --queries "$TOPUP_Q" \
    2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 采集失败，停止" | tee -a "$LOG"; exit 3; }

  RAW=$(find "$OUT" -name "grpo_hermes.jsonl" 2>/dev/null | head -1)
  [ -z "$RAW" ] && { echo "[pipeline] 未找到 grpo_hermes.jsonl，停止" | tee -a "$LOG"; exit 3; }
  echo "[pipeline] 采集产物: $RAW ($(wc -l < "$RAW") 行)" | tee -a "$LOG"

  CONCURRENCY="$MAX_CONCURRENT" bash scripts/pipeline/qc_cold_start.sh "$RAW" "$QC_OUT" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 质检失败，停止（不 warmup/parquet）" | tee -a "$LOG"; exit 4; }

  CLEAN=$(find "$QC_OUT" -name "*_llmchecked.jsonl" 2>/dev/null | head -1)
  [ -z "$CLEAN" ] && { echo "[pipeline] 未找到质检干净集，停止" | tee -a "$LOG"; exit 4; }

  GAP_JSON="$OUT/gaps_${tag_round}.json"
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
sys.exit(0 if not gaps else 7)
PYEOF
  return "${PIPESTATUS[0]}"
}

# ── 2. 补采：阶段 A（cold-only 循环）→ 阶段 B（缺了才 borrow）───────────────
# 阶段 A：只采 cold，提高 num-queries 直到 cold 候选耗尽或全达标。
echo "[pipeline] === 2/4 阶段A：cold-only 采集循环 ===" | tee -a "$LOG"
TOTAL_ROWS=$(wc -l < "$TOPUP_Q")
[ "$START_NUM" -eq 0 ] && START_NUM="$("$PY" -c "
fl=[int(x) for x in '$FLOORS'.split(',')]; print(min($TOTAL_ROWS, int(round(sum(fl)*$OVERSHOOT))))")"
CUR_N="$START_NUM"
gap_rc=7
roundA=0
while [ "$roundA" -lt "$MAX_ROUNDS" ]; do
  roundA=$((roundA+1))
  echo "[pipeline] 阶段A round=$roundA  num=$CUR_N/$TOTAL_ROWS(cold)" | tee -a "$LOG"
  collect_qc_gap "$CUR_N" "A${roundA}"; gap_rc=$?
  [ "$gap_rc" -eq 0 ] && { echo "[pipeline] ✓ cold 阶段全桶达 floor" | tee -a "$LOG"; break; }
  [ "$CUR_N" -ge "$TOTAL_ROWS" ] && { echo "[pipeline] cold 候选耗尽，仍缺 → 进入阶段B borrow" | tee -a "$LOG"; break; }
  CUR_N=$(( CUR_N + TOPUP_STEP )); [ "$CUR_N" -gt "$TOTAL_ROWS" ] && CUR_N="$TOTAL_ROWS"
done

# 阶段 B：cold 采完仍缺 → 按缺口只为缺口桶 borrow，追加到 queries 尾部，继续采。
if [ "$gap_rc" -ne 0 ] && [ "$NO_BORROW" != "1" ]; then
  echo "[pipeline] === 2/4 阶段B：按缺口 borrow 补缺 ===" | tee -a "$LOG"
  roundB=0
  while [ "$roundB" -lt "$MAX_ROUNDS" ]; do
    roundB=$((roundB+1))
    prev_rows="$TOTAL_ROWS"
    # 只为缺口桶从 train borrow，追加到 TOPUP_Q 尾部（record_id 跨轮去重）
    "$PY" scripts/data/build_topup_queries.py \
      --cold "$COLD" --borrow-from "$BORROW_FROM" --gaps "$GAP_JSON" \
      --append-to "$TOPUP_Q" --overshoot "$OVERSHOOT" 2>&1 | tee -a "$LOG"
    TOTAL_ROWS=$(wc -l < "$TOPUP_Q")
    if [ "$TOTAL_ROWS" -le "$prev_rows" ]; then
      echo "[pipeline] ⚠ train 无可借候选（缺口桶已借尽），停止 borrow" | tee -a "$LOG"; break
    fi
    echo "[pipeline] 阶段B round=$roundB  borrow 后 total=$TOTAL_ROWS" | tee -a "$LOG"
    collect_qc_gap "$TOTAL_ROWS" "B${roundB}"; gap_rc=$?
    [ "$gap_rc" -eq 0 ] && { echo "[pipeline] ✓ borrow 后全桶达 floor" | tee -a "$LOG"; break; }
  done
fi

if [ "$gap_rc" -ne 0 ]; then
  echo "[pipeline] ⚠ 仍有缺口，见 $GAP_JSON（不 fail，交人工决定）" | tee -a "$LOG"
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
"$PY" scripts/data/trajectory_to_parquet.py --input "$CLEAN" --out-dir datasets 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "[pipeline] 转 parquet 失败" | tee -a "$LOG"; exit 5; }

echo "[pipeline] ✓ 全部完成 → warmup=$WARMUP_OUT | datasets/{train,val}.parquet | 日志 $LOG" | tee -a "$LOG"
