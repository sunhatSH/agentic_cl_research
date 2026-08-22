#!/usr/bin/env bash
# 重采集废轨迹 + 替换原轨迹 + 重建 warmup sqlite。
#
# 流程：
#   1. waste_requery.jsonl（{line, query, bucket, ...}）→ requery_queries.jsonl（run_cold_start 输入格式）
#   2. run_cold_start.py hermes 单轮采集 242 条（32 并发）
#   3. 按 line 号把新轨迹替换回 cold_start_1429.jsonl
#   4. warmup_buffer.py 重建 warmup_1429.sqlite
#
# 后台跑（CLAUDE.md 强制长任务 tmux）：
#   tmux new -d -s requery 'bash scripts/collect/requery_waste.sh'
#
# 覆盖：OUT_DIR（采集输出目录）、MAX_CONCURRENT（并发，默认 32）、ACTOR_MODEL（默认 gpt-5.5）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
# run_cold_start.py 里 `from scripts.collect.xxx` 需 repo root 在 sys.path（其自带
# sys.path.insert 加的是 scripts/ 不是 repo root，故这里显式补 PYTHONPATH）
export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"

COLD_START="datasets/cold_start/cold_start_1429.jsonl"
WASTE="${WASTE:-datasets/cold_start/waste_requery.jsonl}"
REQUERY_QUERIES="datasets/cold_start/requery_queries.jsonl"
SQLITE="datasets/cold_start/warmup_1429.sqlite"
OUT_DIR="${OUT_DIR:-/mnt/afs_toolcall/sunhao4/agentic_cl_rollouts/requery}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
ACTOR_MODEL="${ACTOR_MODEL:-gpt-5}"
SLOT_TIMEOUT="${SLOT_TIMEOUT:-1800}"   # 单条 hermes 超时(s)；ops/finance 难任务需 >900
PY="${PY:-.venv/bin/python}"   # 采集需 e2b/tqdm，用项目 .venv（系统 python3 无 e2b）

echo "=== [requery] 凭证加载 ==="
source scripts/env/load_tencent_env.sh 2>/dev/null || true
[ -f .env ] && set -a && source .env && set +a || true

echo "=== [requery] 1/4 转换 waste → requery_queries ==="
WASTE="$WASTE" $PY - <<'PY'
import json
import os
from pathlib import Path
inp = Path(os.environ["WASTE"])
out = Path("datasets/cold_start/requery_queries.jsonl")
n = 0
with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        # record_id 编码原 line 号（替换映射用）；query 是首个 user 消息
        g.write(json.dumps({
            "record_id": f"row-{d['line']:08d}",
            "queries": [d["query"]],
            "bucket": d.get("bucket", ""),
        }, ensure_ascii=False) + "\n")
        n += 1
print(f"  转换 {n} 条 → {out}")
PY
N_QUERY=$(wc -l < "$REQUERY_QUERIES")

echo "=== [requery] 2/4 hermes 单轮采集（${N_QUERY} 条，${MAX_CONCURRENT} 并发，slot_timeout=${SLOT_TIMEOUT}s，model=${ACTOR_MODEL}）==="
$PY scripts/collect/run_cold_start.py \
  --queries "$REQUERY_QUERIES" \
  --num-queries "$N_QUERY" \
  --max-concurrent "$MAX_CONCURRENT" \
  --actor-model "$ACTOR_MODEL" \
  --actor-impl hermes_structured \
  --backend e2b \
  --template agentic-cl-sandbox \
  --no-usersim \
  --slot-timeout "$SLOT_TIMEOUT" \
  --collect-mode overwrite \
  --out-dir "$OUT_DIR" \
  --model-tag requery

# 采集输出：$OUT_DIR/trajectory/requery/grpo_hermes.jsonl
COLLECTED="$OUT_DIR/trajectory/requery/grpo_hermes.jsonl"

echo "=== [requery] 3/4 替换原轨迹（按 line 号）==="
$PY - "$COLLECTED" "$WASTE" "$COLD_START" <<'PY'
import json
import sys
from pathlib import Path

collected_path = Path(sys.argv[1])
waste_path = Path(sys.argv[2])
cold_path = Path(sys.argv[3])

# waste 顺序 == 采集输出顺序（run_cold_start 按 query_index 排序输出）
waste_lines = [json.loads(l)["line"] for l in open(waste_path, encoding="utf-8") if l.strip()]
new_trajs = [json.loads(l) for l in open(collected_path, encoding="utf-8") if l.strip()]

assert len(waste_lines) == len(new_trajs), (
    f"数量不匹配：waste={len(waste_lines)} vs collected={len(new_trajs)}"
)

old_trajs = [json.loads(l) for l in open(cold_path, encoding="utf-8") if l.strip()]
# line 号可能乱序，先建映射再整体替换（遍历一次，避免重复替换）
line_to_new = {ln: t for ln, t in zip(waste_lines, new_trajs)}
replaced = 0
for ln, t in line_to_new.items():
    if 0 <= ln < len(old_trajs):
        old_trajs[ln] = t
        replaced += 1
    else:
        print(f"  ⚠️ line {ln} 越界（cold_start 共 {len(old_trajs)} 条），跳过", file=sys.stderr)

# 写回（原子：tmp + replace）
tmp = cold_path.with_suffix(".jsonl.tmp")
with open(tmp, "w", encoding="utf-8") as f:
    for t in old_trajs:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
tmp.replace(cold_path)
print(f"  替换 {replaced} 条 → {cold_path}（共 {len(old_trajs)} 条）")
PY

echo "=== [requery] 4/4 重建 warmup sqlite ==="
$PY scripts/warmup_buffer.py \
  --input "$COLD_START" \
  --out "$SQLITE"

echo "=== [requery] DONE ==="
echo "  采集输出 : $COLLECTED"
echo "  冷启动   : $COLD_START"
echo "  sqlite   : $SQLITE"
