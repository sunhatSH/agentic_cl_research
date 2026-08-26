#!/usr/bin/env python3
"""重构 train_cl.parquet:从 all_tasks_labeled 选 coding/office 各 6400,难度均分。

选题:
  - coding/office 各 6400
  - d4-6 优先, 不够 d7 补
  - 桶内每个 step(32 batch)难度配比 ≈ 全桶配比(最大余数法交错, 每个 step 难度均分)
  - 去重 by D_id, 只选有 files/ 的任务(产出型无文件不选)

重构:
  - record_id = D_id
  - prompt 单 user message(去 system, 用 Hermes 默认)
  - query 路径归一化(Windows→Linux, collapse double-bs, residual fix)
  - gen_task_id = D_id (cl_agent_dataset 从 taskspecs_w3/<D_id>/files 取文件)

输出: datasets/train_cl.parquet + datasets/train_cl.jsonl (原地覆盖, 先 .tmp 再 replace)

用法:
  python3 scripts/data/rebuild_train_cl_final.py            # 干跑
  python3 scripts/data/rebuild_train_cl_final.py --apply    # 落地
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from path_normalize import normalize_paths  # noqa: E402

LABELED = ROOT / "datasources" / "labeled" / "all_tasks_labeled.jsonl"
TASKSPECS = ROOT / "datasources" / "taskspecs_w3"
PARQUET = ROOT / "datasets" / "train_cl.parquet"
JSONL = ROOT / "datasets" / "train_cl.jsonl"

PER_BUCKET = 6400  # 200 step × 32 batch
BATCH = 32
SEED = 42


def normalize_query(q: str) -> str:
    """归一化 query 路径(Windows→Linux, collapse double-bs, residual fix)。"""
    if not q:
        return q
    s = re.sub(r"\\{2,}", r"\\", q)

    def to_bs(m):
        return m.group(0).replace("/", "\\")

    s = re.sub(r"[A-Za-z]:/hermes[\\/]+runtime[\\/]+(?:bigtasks|winruns)[\\/]+D\d+[\\/]+D[A-Za-z0-9_]+[\\/]+(?:inputs|ws)[\\/]+", to_bs, s, flags=re.IGNORECASE)
    s = re.sub(r"[A-Za-z]:/hermes[\\/]+runtime[\\/]+(?:bigtasks|winruns)[\\/]+D\d+[\\/]+D[A-Za-z0-9_]+[\\/]+", to_bs, s, flags=re.IGNORECASE)
    s = re.sub(r"[A-Za-z]:/hermes[\\/]+runtime[\\/]+", to_bs, s, flags=re.IGNORECASE)
    s = normalize_paths(s)
    s = re.sub(r"\./bigtasks\\D\d+\\[^\\]+\\inputs(?:\\|:)", "/home/user/workspace/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\D\d+\\[^\\]+\\ws(?:\\|:)", "/home/user/workspace/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./bigtasks\\D\d+\\[^\\]+\\", "/home/user/workspace/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\D\d+\\[^\\]+\\", "/home/user/workspace/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./bigtasks\\D\d+\\[^\\]+(?=\s|$|:|,)", "/home/user/workspace/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\D\d+\\[^\\]+(?=\s|$|:|,)", "/home/user/workspace/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\(?=$|\s|,|;)", "/home/user/workspace/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./bigtasks\\(?=$|\s|,|;)", "/home/user/workspace/", s, flags=re.IGNORECASE)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(/home/user/workspace/[^\s\"']*?)\\(?=[A-Za-z0-9_.])", r"\1/", s)
    return s


def _has_files(d_id: str) -> bool:
    """taskspecs_w3/<D_id>/files 存在且非空。"""
    d = os.path.join(str(TASKSPECS), d_id, "files")
    return os.path.isdir(d) and bool(os.listdir(d))


def _balanced_order(rids: list[str], diff_map: dict[str, int], seed: int = 42) -> list[str]:
    """桶内 rids 按难度分层、层内 shuffle、最大余数法交错,使每个 batch(32)难度配比≈全桶配比。

    与 build_train.py 的 _balanced_order 同口径。
    """
    rng = random.Random(seed)
    groups: dict[int, list[str]] = {}
    for rid in rids:
        groups.setdefault(diff_map[rid], []).append(rid)
    for d in groups:
        rng.shuffle(groups[d])
    total = len(rids)
    diffs = sorted(groups)
    remaining = {d: len(groups[d]) for d in diffs}
    idx = {d: 0 for d in diffs}
    acc = {d: 0.0 for d in diffs}
    out: list[str] = []
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # ── 1. 加载 all_tasks_labeled ──
    tasks: dict[str, dict] = {}
    with open(LABELED, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tasks[d["D_id"]] = d
    print(f"loaded {len(tasks)} tasks", flush=True)

    # ── 2. 选 coding/office, d4-6 优先 d7 补, 再不够 d3/d8 兜底（含 _s 产出型）──
    selected: dict[str, list[str]] = {}  # bucket -> [D_id]
    for bk in ("coding", "office"):
        # d4-6 优先 → d7 → d3 → d8 兜底
        pool_46 = [gid for gid, t in tasks.items()
                   if t.get("bucket") == bk and t.get("difficulty") is not None
                   and 4 <= t["difficulty"] <= 6]
        pool_7 = [gid for gid, t in tasks.items()
                  if t.get("bucket") == bk and t.get("difficulty") == 7]
        pool_3 = [gid for gid, t in tasks.items()
                  if t.get("bucket") == bk and t.get("difficulty") == 3]
        pool_8 = [gid for gid, t in tasks.items()
                  if t.get("bucket") == bk and t.get("difficulty") == 8]
        rng = random.Random(SEED)
        for p in (pool_46, pool_7, pool_3, pool_8):
            rng.shuffle(p)
        taken = pool_46[:PER_BUCKET]
        rem = PER_BUCKET - len(taken)
        taken += pool_7[:rem]
        rem = PER_BUCKET - len(taken)
        taken += pool_3[:rem]
        rem = PER_BUCKET - len(taken)
        taken += pool_8[:rem]
        selected[bk] = taken
        d46 = sum(1 for gid in taken if tasks[gid]["difficulty"] in (4, 5, 6))
        d7 = sum(1 for gid in taken if tasks[gid]["difficulty"] == 7)
        d38 = sum(1 for gid in taken if tasks[gid]["difficulty"] in (3, 8))
        print(f"  {bk}: pool d4-6={len(pool_46)} d7={len(pool_7)} d3={len(pool_3)} d8={len(pool_8)} "
              f"taken={len(taken)} (d4-6={d46} d7={d7} d3/d8={d38})", flush=True)

    # ── 3. 桶内难度均分(最大余数法交错) ──
    ordered: list[str] = []
    for bk in ("coding", "office"):
        diff_map = {gid: tasks[gid]["difficulty"] for gid in selected[bk]}
        ordered += _balanced_order(selected[bk], diff_map, seed=SEED)

    # ── 4. 构建行 ──
    SYSTEM_PROMPT = ""  # 单 user message, 无 system(用 Hermes 默认)
    rows = []
    for gid in ordered:
        t = tasks[gid]
        q = normalize_query(t["user_prompt"])
        rows.append({
            "prompt": [{"role": "user", "content": q}],
            "data_source": "agentic_cl",
            "reward_model": {
                "ground_truth": "",
                "style": "rule",
                "reward_fn": {"_function_name": "trainer.model_reward_omni.compute_score"},
            },
            "bucket": t["bucket"],
            "extra_info": {
                "record_id": gid,
                "bucket": t["bucket"],
                "queries": [q],
                "persona": "",
                "available_tools": [],
                "missing_info_slots": [],
                "safety_constraints": [],
                "difficulty": str(t["difficulty"]),
                "gen_task_id": gid,
                "ws_dir": "",
            },
        })

    # round to batch 32
    n = (len(rows) // BATCH) * BATCH
    rows = rows[:n]
    rids = [r["extra_info"]["record_id"] for r in rows]
    assert len(rids) == len(set(rids)), f"DUPLICATES: {len(rids)} vs {len(set(rids))}"

    # ── 5. 统计 ──
    c = Counter(r["bucket"] for r in rows)
    print(f"\n=== 重构结果 ===", flush=True)
    print(f"  总行数: {len(rows)} ({len(rows)//BATCH} steps)", flush=True)
    for b in ("coding", "office"):
        print(f"  {b}: {c[b]} rows ({c[b]//BATCH} steps)", flush=True)
        # 难度分布
        dv = Counter(r["extra_info"]["difficulty"] for r in rows if r["bucket"] == b)
        print(f"    难度: {dict(sorted(dv.items()))}", flush=True)
        # 每 step 难度配比(前 3 个 step)
        bk_rows = [r for r in rows if r["bucket"] == b]
        for si in range(min(3, len(bk_rows) // BATCH)):
            step = bk_rows[si * BATCH:(si + 1) * BATCH]
            sdv = Counter(r["extra_info"]["difficulty"] for r in step)
            print(f"    step {si}: {dict(sorted(sdv.items()))}", flush=True)

    # ── 6. 落地 ──
    if not args.apply:
        print("\n加 --apply 落地", flush=True)
        return 0

    # 读旧 schema
    old_t = pq.read_table(str(PARQUET))
    new_t = pa.Table.from_pylist(rows, schema=old_t.schema)
    tmp = PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(PARQUET)
    tmp_j = JSONL.with_suffix(".jsonl.tmp")
    with open(tmp_j, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_j.replace(JSONL)
    print(f"\n✅ 落地 {PARQUET.name} + {JSONL.name} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
