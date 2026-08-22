#!/usr/bin/env bash
# 新数据打标 pipeline
#   1. extract_new_queries: 从新轨迹 jsonl 提取 queries
#   2. label_new_queries:   tokenhub 并发打 bucket 标
#   3. report:              合并老数据 + 新数据的桶分布
#
# 用法:
#   bash scripts/pipeline/run_new_label.sh           # 全量
#   bash scripts/pipeline/run_new_label.sh --limit 100  # 冒烟
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY=/mnt/afs_toolcall/sunhao4/miniconda3/bin/python3
LIMIT="${1:-}"
if [ -n "$LIMIT" ]; then LIMIT="--limit $LIMIT"; fi

echo "===== Step 1: 提取 queries ====="
$PY "$ROOT/scripts/data/extract_new_queries.py"

echo ""
echo "===== Step 2: 打标(冒烟 100 条先测) ====="
$PY "$ROOT/scripts/data/label_new_queries.py" --limit 100 --workers 16 --resume

echo ""
echo "===== Step 3: 桶分布报告 ====="
$PY - <<'PYEOF'
import json
from collections import Counter

# 老数据
try:
    import pyarrow.parquet as pq
    t = pq.read_table("datasets/train_aligned.parquet")
    old = Counter(t.column("bucket").to_pylist())
except Exception as e:
    print(f"[warn] old data unavailable: {e}")
    old = Counter()

# 新打标数据
new = Counter()
with open("datasources/labeled/new_trajectories_labeled.jsonl") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        new[r.get("bucket", "?")] += 1

print(f"\n{'bucket':15s} {'old':>8} {'new':>8} {'total':>8} {'steps(×32)':>10}")
print("-" * 55)
for b in ["workflow","ops","qa","finance","office","communication","safety","coding","research"]:
    o = old.get(b, 0); n = new.get(b, 0); t = o + n
    print(f"  {b:15s} {o:8d} {n:8d} {t:8d} {t//32:10d}")
print(f"  {'TOTAL':15s} {sum(old.values()):8d} {sum(new.values()):8d} {sum(old.values())+sum(new.values()):8d}")
PYEOF
