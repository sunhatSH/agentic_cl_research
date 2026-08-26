#!/usr/bin/env python3
"""Build datasets/train_cl.parquet：第一步训练集 = coding + research 各 6400（400 step）。

coding 取难度 4-6；research 取 4-6，不够从 7 补。难度来自 difficulty_all.jsonl（双模型交集后
单一值）。训练序按 bucket 空间最大距离（coding→research）。shuffle=false，前 200 step coding、
后 200 step research。第二步的 7 桶数据集见 build_train_exp2.py（record_id 与本集不重合）。
"""
import json, re, math, itertools, random
from collections import defaultdict, Counter
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_normalize import normalize_paths, extract_gen_task_id, extract_ws_dir  # 路径归一化 + 抽任务id/ws标识

ROOT = Path(__file__).resolve().parent.parent.parent
LABELED = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
OUT = ROOT / "datasets" / "train_cl.parquet"

_REMINDER_RE = re.compile(r"<system-reminder>.*?(?:</system-reminder>|$)", re.DOTALL)
T = ["coding", "research"]
PER_BUCKET = 6400  # 200 step × 32 batch

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
    d["_gen_task_id"] = extract_gen_task_id(q)  # 归一化【前】抽 generated_tasks_hermes 任务id
    d["_ws_dir"] = extract_ws_dir(q)  # 归一化【前】抽 review 任务 workspace 快照标识(F5-review)
    q = normalize_paths(q)  # Windows E:\hermes\... → ./inputs/ 或 ./outputs/；ws → /home/user/workspace/
    d["_clean_query"] = q
    labeled[d["record_id"]] = d

print(f"labeled: {len(labeled)} non-empty records")

# ── Load difficulty (difficulty_all.jsonl 单一值 = 双模型交集后) ──
diff = {}
for line in open(ROOT / "datasources" / "labeled" / "difficulty_all.jsonl"):
    d = json.loads(line.strip()); rid = d["record_id"]
    if rid in labeled and d.get("difficulty") is not None:
        diff[rid] = d["difficulty"]
print(f"difficulty: {len(diff):,} records")

# ── Select records per bucket ──
selected = {}  # rid -> bucket
BATCH = 32
random.seed(42)


def _balanced_order(rids, diff_map, seed=42):
    """把一个桶内的 rids 按难度分层、层内 shuffle、再按比例交错，
    使每个 batch(32) 的难度配比≈全桶配比（每难度值内部随机）。

    做法：按难度分组→组内 shuffle→用"分数累加"式交错(类似 Bresenham/最大余数)，
    保证任意前缀里各难度占比都贴近全局占比，故每个连续 32 窗口配比稳定一致。
    """
    rng = random.Random(seed)
    groups = {}
    for rid in rids:
        groups.setdefault(diff_map[rid], []).append(rid)
    for d in groups:
        rng.shuffle(groups[d])
    total = len(rids)
    # 每难度的"发牌速率" = 该难度占比；用累加器决定下一个发哪个难度。
    diffs = sorted(groups)
    remaining = {d: len(groups[d]) for d in diffs}
    idx = {d: 0 for d in diffs}
    acc = {d: 0.0 for d in diffs}
    out = []
    for _ in range(total):
        # 给每个还有剩余的难度累加其速率，挑累加值最大的发一张（最大余数法）
        best, best_acc = None, -1.0
        for d in diffs:
            if remaining[d] <= 0:
                continue
            acc[d] += len(groups[d]) / total
            if acc[d] > best_acc:
                best, best_acc = d, acc[d]
        acc[best] -= 1.0
        out.append(groups[best][idx[best]])
        idx[best] += 1
        remaining[best] -= 1
    return out


for b in T:
    # Candidates: records in this bucket that have a difficulty label
    candidates = [rid for rid in labeled if labeled[rid]["bucket"] == b and rid in diff]

    pool = []
    strategy = ""

    if b == "coding":
        # 4-6 难度（coding 4-6 充足）
        pool = [rid for rid in candidates if 4 <= diff[rid] <= 6]
        strategy = "4-6"
    elif b == "research":
        # 4-6 优先，不够从 7 补
        pool_46 = [rid for rid in candidates if 4 <= diff[rid] <= 6]
        pool_7 = [rid for rid in candidates if diff[rid] == 7]
        pool = pool_46 + pool_7[: max(0, PER_BUCKET - len(pool_46))]
        strategy = "4-6 + 7 fallback"

    random.shuffle(pool)
    taken = pool[:PER_BUCKET]
    for rid in taken:
        selected[rid] = b
    print(f"  {b:>12}: pool={len(pool):,}  taken={len(taken)}  strategy={strategy}")

# ── Training order (max-distance) ──
with open(ROOT / "configs" / "bucket_coords.json") as f:
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
    bucket_rids = [rid for rid in selected if selected[rid] == b]
    # 难度分层交错：每个 batch(32) 难度配比≈全桶配比，层内随机（用户口径）。
    ordered_rids = _balanced_order(bucket_rids, diff, seed=42)
    for rid in ordered_rids:
        rec = labeled[rid]
        q = rec["_clean_query"]
        dd = diff[rid]
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
                "difficulty": str(dd),
                "gen_task_id": rec.get("_gen_task_id") or "",  # generated_tasks_hermes/<D>/<tid>/ 输入文件+answer_key 定位
                "ws_dir": rec.get("_ws_dir") or "",  # F5-review：review 任务 workspace 快照标识（datasources/review_ws/<ws_dir>/ws）
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
pq.write_table(table, OUT)  # train_cl.parquet
_json_out = str(OUT).replace(".parquet", ".jsonl")
with open(_json_out, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

c = Counter(r["bucket"] for r in rows)
print(f"\n✅ {OUT} → {len(rows)} rows ({len(rows)//32} steps), 0 duplicates")
for b in best_order:
    print(f"  {b}: {c[b]} rows ({c[b]//32} steps)")
print(f"  {_json_out} 同步更新")
