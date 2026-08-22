#!/usr/bin/env bash
# After buffer-fill collection completes:
#   1. Filter out error trajectories from grpo_hermes.jsonl -> grpo_hermes_clean.jsonl
#   2. Check per-bucket clean count vs floors (cap/30)
#   3. For short buckets, borrow from queries_train.jsonl
#   4. Collect borrowed tasks (with --actor-impl hermes_structured)
#   5. Append borrowed trajectories to clean set
#   6. Remove borrowed records from queries_train.jsonl
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
TRAJ=/mnt/afs_toolcall/sunhao4/agentic_cl_rollouts/real/trajectory/gpt5/grpo_hermes.jsonl
CLEAN="${TRAJ%.jsonl}_clean.jsonl"
Q_BUF=datasets/queries_buffer.jsonl
Q_TRAIN=datasets/queries_train.jsonl

echo "=== Step 1: filter errors ==="
$PY -c "
import json
FLOORS={'workflow':163,'ops':145,'qa':131,'finance':97,'office':72,'communication':72,'safety':65,'coding':30,'research':53}
clean={}; dropped=0; kept=0
with open('$TRAJ','r',errors='replace') as f, open('$CLEAN','w',encoding='utf-8') as out:
    for l in f:
        l=l.strip()
        if not l: continue
        d=json.loads(l)
        e=d.get('error','')
        if e:
            dropped+=1; continue
        kept+=1
        b=d.get('bucket','?')
        clean[b]=clean.get(b,0)+1
        out.write(json.dumps(d,ensure_ascii=False)+'\n')
print(f'kept={kept}  dropped={dropped}')
print('per-bucket clean:')
for b,f in FLOORS.items():
    n=clean.get(b,0); ok='OK' if n>=f else f'SHORT by {f-n}'
    print(f'  {b:14s} clean={n:4d}  floor={f}  {ok}')
"

echo ""
echo "=== Step 2: borrow from training if needed ==="
$PY -c "
import json, random
random.seed(42)
FLOORS={'workflow':163,'ops':145,'qa':131,'finance':97,'office':72,'communication':72,'safety':65,'coding':30,'research':53}
# count clean trajectories
clean={}
with open('$CLEAN','r',errors='replace') as f:
    for l in f:
        d=json.loads(l); clean[d.get('bucket','?')]=clean.get(d.get('bucket','?'),0)+1
# compute shortfall
shortfall={}
for b,f in FLOORS.items():
    s=clean.get(b,0)
    if s<f: shortfall[b]=f-s
if not shortfall:
    print('all buckets meet floors — no borrowing needed')
else:
    print(f'shortfall: {shortfall}')
    # load training queries by bucket
    train_by_bucket={}
    with open('$Q_TRAIN','r',errors='replace') as f:
        for l in f:
            d=json.loads(l); b=d.get('bucket','?')
            train_by_bucket.setdefault(b,[]).append(d)
    # pick random queries
    borrowed=[]
    borrowed_ids=set()
    for b,need in shortfall.items():
        pool=train_by_bucket.get(b,[])
        if len(pool)<need: print(f'WARN: {b} needs {need} but training pool only has {len(pool)}')
        picks=random.sample(pool,min(need,len(pool)))
        for p in picks: borrowed_ids.add(p['record_id'])
        borrowed.extend(picks)
        print(f'  {b}: borrowed {len(picks)} from training')
    with open('/tmp/borrowed_queries.jsonl','w',encoding='utf-8') as out:
        for b in borrowed: out.write(json.dumps(b,ensure_ascii=False)+'\n')
    print(f'total borrowed: {len(borrowed)}')
    # update training queries — remove borrowed
    orig=0; kept=0
    with open('$Q_TRAIN','r',errors='replace') as f, \
         open('$Q_TRAIN.tmp','w',encoding='utf-8') as out:
        for l in f:
            d=json.loads(l); orig+=1
            if d['record_id'] not in borrowed_ids:
                out.write(l); kept+=1
    import os; os.rename('$Q_TRAIN.tmp','$Q_TRAIN')
    print(f'training set: {orig} -> {kept} (removed {orig-kept})')
"

echo ""
echo "=== Step 3: if borrowed, run collection ==="
if [ -f /tmp/borrowed_queries.jsonl ] && [ -s /tmp/borrowed_queries.jsonl ]; then
    N=$(wc -l < /tmp/borrowed_queries.jsonl)
    echo "borrowed $N tasks — running collection to fill gaps..."
    .venv/bin/python scripts/collect/run_cold_start.py \
        --num-queries "$N" --max-concurrent 16 \
        --actor-model openai/gpt-5 --model-tag gpt5_borrow \
        --actor-impl hermes_structured \
        --taskspecs-dir datasources/taskspecs_w3 \
        --queries /tmp/borrowed_queries.jsonl \
        --max-turns 20 --hermes-max-turns 30 --slot-timeout 900 \
        --collect-mode overwrite
    # append borrowed to clean
    cat "$TRAJ" >> "$CLEAN"
    echo "borrowed trajectories appended to $CLEAN"
else
    echo "no borrowing needed"
fi

echo "DONE. Clean trajectories: $CLEAN"
echo "Training queries updated: $Q_TRAIN"
