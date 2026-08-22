#!/usr/bin/env python3
"""Build train_exp2.parquet：第二步「9 桶少量边训边评」的训练集。

与第一步 train_cl.parquet（coding/research）按桶划分、完全不重合——这里取其余 7 桶
（office/ops/workflow/qa/finance/safety/communication）的中等难度(4-6)任务。

每桶取 min(中等难度全量, MAX_PER_BUCKET)，batch(32) 对齐。小桶（qa/communication 等
数据量少）有多少取多少；大桶（office/ops）封顶 MAX_PER_BUCKET（默认 1600 = 50 step × 32）。
"""
import json
import os
import random
from collections import Counter
from pathlib import Path
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from path_normalize import normalize_paths, extract_gen_task_id  # 路径归一化 + 抽任务id

ROOT = Path(__file__).resolve().parent.parent.parent
LABELED = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
DIFF = ROOT / "datasources" / "labeled" / "difficulty_all.jsonl"
OUT = ROOT / "datasets" / "train_exp2.parquet"

import re

_REMINDER_RE = re.compile(r"<system-reminder>.*?(?:</system-reminder>|$)", re.DOTALL)
T = ["office", "ops", "workflow", "qa", "finance", "safety", "communication"]
MAX_PER_BUCKET = int(os.environ.get("MAX_PER_BUCKET", "1600"))  # 1600 = 50 step × 32 batch

SYSTEM_PROMPT = (
    "You are a capable autonomous agent. Complete the user's task using the available tools. "
    "Work independently — never ask the user for input, confirmation, or clarification. "
    "When faced with ambiguity or multiple options, pick the most reasonable or first option "
    "and proceed without hesitation."
)

# ── Load labeled data（去 system-reminder 垃圾，同 build_train.py）──
labeled = {}
for line in LABELED.read_text().splitlines():
    if not line.strip():
        continue
    d = json.loads(line)
    q = d.get("seed_query", "")
    if "<system-reminder>" in q:
        q = _REMINDER_RE.sub("", q).strip()
        if not q:
            continue
    d["_gen_task_id"] = extract_gen_task_id(q)  # 归一化前抽任务id
    q = normalize_paths(q)  # Windows E:\hermes\... → ./inputs/ 或 ./outputs/
    d["_clean_query"] = q
    labeled[d["record_id"]] = d
print(f"labeled: {len(labeled)} non-empty records")

# ── Load difficulty（difficulty_all.jsonl 单一值）──
diff = {}
for line in DIFF.read_text().splitlines():
    d = json.loads(line)
    rid = d["record_id"]
    if rid in labeled and d.get("difficulty") is not None:
        diff[rid] = d["difficulty"]
print(f"difficulty: {len(diff):,} records")

# ── 与第一步 train_cl.parquet 的 record_id 去重（数据不重合）──
import pyarrow.parquet as pq

_cl = pq.read_table(ROOT / "datasets" / "train_cl.parquet")
_cl_rids = {str(x["record_id"]) for x in _cl.column("extra_info").to_pylist() if isinstance(x, dict)}
print(f"train_cl.parquet 已用 record_id: {len(_cl_rids):,}")

# ── Select per bucket（4-6 中等难度，batch 对齐）──
selected = {}
random.seed(42)


def _balanced_order(rids, diff_map, seed=42):
    """按难度分层、层内 shuffle、按比例交错，使每个 batch(32) 难度配比≈全桶配比。
    （同 build_train.py，最大余数法交错。）"""
    rng = random.Random(seed)
    groups = {}
    for rid in rids:
        groups.setdefault(diff_map[rid], []).append(rid)
    for d in groups:
        rng.shuffle(groups[d])
    total = len(rids)
    diffs = sorted(groups)
    remaining = {d: len(groups[d]) for d in diffs}
    idx = {d: 0 for d in diffs}
    acc = {d: 0.0 for d in diffs}
    out = []
    for _ in range(total):
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
    cands = [
        rid for rid in labeled
        if labeled[rid]["bucket"] == b and rid in diff and 4 <= diff[rid] <= 6 and rid not in _cl_rids
    ]
    random.shuffle(cands)
    taken = cands[:MAX_PER_BUCKET]
    taken = taken[: (len(taken) // 32) * 32]  # batch 32 对齐
    for rid in taken:
        selected[rid] = b
    print(f"  {b:>15}: pool={len(cands):,}  taken={len(taken)}  ({len(taken)//32} steps)")

# ── Build rows ──
rows = []
for b in T:
    bucket_rids = [rid for rid in selected if selected[rid] == b]
    ordered_rids = _balanced_order(bucket_rids, diff, seed=42)  # 难度分层交错
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
                "gen_task_id": rec.get("_gen_task_id") or "",
            },
        })

rids = [r["extra_info"]["record_id"] for r in rows]
assert len(rids) == len(set(rids)), "DUPLICATES"

import pyarrow as pa

table = pa.Table.from_pylist(rows)
pq.write_table(table, OUT)
_json_out = str(OUT).replace(".parquet", ".jsonl")
with open(_json_out, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

c = Counter(r["bucket"] for r in rows)
print(f"\n✅ {OUT} → {len(rows)} rows ({len(rows)//32} steps), 0 duplicates")
for b in T:
    print(f"  {b}: {c[b]} rows ({c[b]//32} steps)")
print(f"  {_json_out} 同步更新")
