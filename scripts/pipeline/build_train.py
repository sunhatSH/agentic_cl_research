#!/usr/bin/env python3
"""Build final train.parquet: 5 buckets × 3200 rows, 4-6/4-7 intersection + gpt fallback."""
import json, re, math, itertools, random
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LABELED = ROOT / "data" / "labeled" / "new_trajectories_labeled.jsonl"
OUT = ROOT / "datasets" / "train_v2.parquet"

_REMINDER_RE = re.compile(r"<system-reminder>.*?(?:</system-reminder>|$)", re.DOTALL)
T = ["coding", "office", "ops", "research", "workflow"]
PER_BUCKET = 3200

SYSTEM_PROMPT = (
    "You are a capable autonomous agent. Complete the user's task using the available tools. "
    "Work independently — never ask the user for input, confirmation, or clarification. "
    "When faced with ambiguity or multiple options, pick the most reasonable or first option "
    "and proceed without hesitation."
)

# ── Load labeled data (non-empty, stripped) ──
labeled = {}
for line in LABELED.read_text().splitlines():
    if not line.strip(): continue
    d = json.loads(line)
    q = d.get("seed_query", "")
    if "<system-reminder>" in q:
        q = _REMINDER_RE.sub("", q).strip()
        if not q: continue  # skip empty after strip
    d["_clean_query"] = q
    labeled[d["record_id"]] = d

print(f"labeled: {len(labeled)} non-empty records")

# ── Load model scores (non-empty only) ──
grok = {}; gpt = {}
for line in open(ROOT / "data" / "labeled" / "difficulty_grok-4.5.jsonl"):
    d = json.loads(line.strip()); rid = d["record_id"]
    if rid in labeled and d.get("difficulty") is not None:
        grok[rid] = d["difficulty"]
for line in open(ROOT / "data" / "labeled" / "difficulty_gpt-5.6-luna.jsonl"):
    d = json.loads(line.strip()); rid = d["record_id"]
    if rid in labeled and d.get("difficulty") is not None:
        gpt[rid] = d["difficulty"]

common = set(grok) & set(gpt)
print(f"grok {len(grok):,}  gpt {len(gpt):,}  intersection {len(common):,}")

# ── Select records per bucket ──
selected = {}  # rid -> bucket
random.seed(42)

for b in T:
    # Candidates: records in this bucket that both models scored
    candidates_common = []
    candidates_gpt_only = []
    for rid in labeled:
        if labeled[rid]["bucket"] != b: continue
        if rid in grok and rid in gpt:
            candidates_common.append(rid)
        elif rid in gpt:
            candidates_gpt_only.append(rid)

    pool = []
    strategy = ""

    if b in ("coding", "office", "ops"):
        # 4-6 intersection
        for rid in candidates_common:
            gs, gp = grok[rid], gpt[rid]
            if 4 <= gs <= 6 and 4 <= gp <= 6:
                pool.append(rid)
        strategy = "4-6 intersection"
    elif b == "research":
        # 4-7 intersection
        for rid in candidates_common:
            gs, gp = grok[rid], gpt[rid]
            if 4 <= gs <= 7 and 4 <= gp <= 7:
                pool.append(rid)
        strategy = "4-7 intersection"
    elif b == "workflow":
        # intersection first, then gpt-only fallback: 5, then 4, 6, 7, 3
        for rid in candidates_common:
            gs, gp = grok[rid], gpt[rid]
            if 4 <= gs <= 7 and 4 <= gp <= 7:
                pool.append(rid)
        strategy = "4-7 intersection + gpt fallback"
        if len(pool) < PER_BUCKET:
            # gpt-only fallback, priority: 5, 4, 6, 7, 3
            gpt_fb = []
            for rid in candidates_gpt_only:
                gpt_fb.append((rid, gpt[rid]))
            for priority in [5, 4, 6, 7, 3]:
                gpt_fb.sort(key=lambda x: (0 if x[1] == priority else abs(x[1] - priority)))
                for rid, s in gpt_fb:
                    if rid not in pool:
                        pool.append(rid)
                        if len(pool) >= PER_BUCKET: break
                if len(pool) >= PER_BUCKET: break

    random.shuffle(pool)
    taken = pool[:PER_BUCKET]
    for rid in taken:
        selected[rid] = b
    print(f"  {b:>12}: pool={len(pool):,}  taken={len(taken)}  strategy={strategy}")

# ── Training order (max-distance) ──
with open(ROOT / "configs" / "bucket_coords_final.json") as f:
    coords = json.load(f)["coordinates"]

def dist(a, b):
    return math.sqrt(sum((coords[a][i] - coords[b][i]) ** 2 for i in range(7)))

best_order, best_min = None, -1
for perm in itertools.permutations(T):
    min_d = min(dist(perm[i], perm[i + 1]) for i in range(len(perm) - 1))
    if min_d > best_min:
        best_min, best_order = min_d, perm
print(f"\n训练序: {' → '.join(best_order)}")

# ── Build rows ──
rows = []
for b in best_order:
    for rid in selected:
        if selected[rid] != b: continue
        rec = labeled[rid]
        q = rec["_clean_query"]
        gs = grok.get(rid); gp = gpt.get(rid)
        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            "data_source": "agentic_cl",
            "reward_model": {
                "ground_truth": "",
                "style": "rule",
                "reward_fn": {"_function_name": "trainer.model_reward_omni.compute_score"},
            },
            "bucket": b,
            "extra_info": {
                "record_id": rid,
                "bucket": b,
                "queries": [q],
                "persona": "",
                "available_tools": [],
                "missing_info_slots": [],
                "safety_constraints": [],
                "difficulty_grok": str(gs) if gs else "",
                "difficulty_gpt": str(gp) if gp else "",
            },
        })

# Round to batch 32
n = (len(rows) // 32) * 32
rows = rows[:n]
rids = [r["extra_info"]["record_id"] for r in rows]
assert len(rids) == len(set(rids)), f"DUPLICATES: {len(rids)} vs {len(set(rids))}"

# Save
import pyarrow as pa, pyarrow.parquet as pq
table = pa.Table.from_pylist(rows)
pq.write_table(table, OUT)
pq.write_table(table, str(OUT).replace("train_v2", "train"))
pq.write_table(table, str(OUT).replace(".parquet", "_aligned.parquet").replace("train_v2", "train"))
with open(str(OUT).replace(".parquet", ".jsonl").replace("train_v2", "train"), "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

c = Counter(r["bucket"] for r in rows)
print(f"\n✅ {OUT} → {len(rows)} rows ({len(rows)//32} steps), 0 duplicates")
for b in best_order:
    print(f"  {b}: {c[b]} rows ({c[b]//32} steps)")
print(f"  train.parquet / train_aligned.parquet / train.jsonl 同步更新")
