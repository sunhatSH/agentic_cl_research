#!/usr/bin/env python3
"""用 labeled 全量 coding/research d4-7 query 命中 generated_tasks_hermes 任务,配文件。

思路:
  - labeled 有 87671 unique record_id,其中 coding/research d4-7 = 22311 个(验证过桶+难度)
  - generated_tasks_hermes 有 47988 任务(有 inputs 文件)
  - 用 22311 个 query 去【搜索】47988 任务:70% token 重叠 = 命中
  - 命中 = 该 query 找到对应文件 → 任务拿到验证过的桶+难度 + 文件路径

输出: datasources/labeled/matched_tasks.jsonl
  每行 = 一个命中的 (query, task) 配对,带:验证桶+验证难度+文件路径
  只保留 coding/research d4-7。

用法:
  python3 scripts/data/match_queries_to_tasks.py            # 全量
  python3 scripts/data/match_queries_to_tasks.py --threshold 0.7
  python3 scripts/data/match_queries_to_tasks.py --limit 500  # 冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from path_normalize import extract_gen_task_id, normalize_paths  # noqa: E402

ALL_TASKS = ROOT / "datasources" / "labeled" / "all_tasks_with_bd.jsonl"
LABELED = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
DIFF = ROOT / "datasources" / "labeled" / "difficulty_all.jsonl"
OUT = ROOT / "datasources" / "labeled" / "matched_tasks.jsonl"

VALID_BUCKETS = {"workflow", "ops", "qa", "finance", "office",
                 "communication", "safety", "coding", "research"}


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
    s = re.sub(r"\./bigtasks\\D\d+\\[^\\]+\\inputs(?:\\|:)", "./inputs/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\D\d+\\[^\\]+\\ws(?:\\|:)", "./outputs/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./bigtasks\\D\d+\\[^\\]+\\", "./outputs/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\D\d+\\[^\\]+\\", "./outputs/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./bigtasks\\D\d+\\[^\\]+(?=\s|$|:|,)", "./inputs/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\D\d+\\[^\\]+(?=\s|$|:|,)", "./outputs/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./winruns\\(?=$|\s|,|;)", "./outputs/", s, flags=re.IGNORECASE)
    s = re.sub(r"\./bigtasks\\(?=$|\s|,|;)", "./inputs/", s, flags=re.IGNORECASE)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(\./(?:inputs|outputs)/[^\s\"']*?)\\(?=[A-Za-z0-9_.])", r"\1/", s)
    return s


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def like_match(qtoks: set[str], ttoks: set[str], threshold: float) -> bool:
    """70% token 重叠 = 命中。ratio = overlap / min(len_q, len_t)。"""
    if not qtoks or not ttoks:
        return False
    overlap = len(qtoks & ttoks)
    ratio = overlap / min(len(qtoks), len(ttoks))
    return ratio >= threshold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.7, help="token 重叠阈值")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条 query(冒烟)")
    args = ap.parse_args()

    # ── 1. 加载 47988 任务,建倒排索引 ──
    all_tasks = []
    with open(ALL_TASKS, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            t["_toks"] = tokenize(t["_nq"])
            all_tasks.append(t)
    print(f"loaded {len(all_tasks)} tasks", flush=True)

    inv_idx = defaultdict(set)
    for i, t in enumerate(all_tasks):
        for tok in t["_toks"]:
            inv_idx[tok].add(i)
    print(f"inverted index: {len(inv_idx)} tokens", flush=True)

    # ── 2. 加载 labeled 全量 coding/research d4-7 query(验证桶+难度)──
    # rid -> (bucket, difficulty), dedup
    rid_bd = {}
    with open(LABELED, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rid = d["record_id"]
            if rid not in rid_bd:
                rid_bd[rid] = (d.get("bucket"), d.get("seed_query", ""))
    diff = {}
    with open(DIFF, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            diff[d["record_id"]] = d.get("difficulty")

    queries = []
    for rid, (bk, sq) in rid_bd.items():
        if bk not in ("coding", "research"):
            continue
        dv = diff.get(rid)
        if dv is None or not (4 <= dv <= 7):
            continue
        nq = normalize_query(sq)
        queries.append({
            "record_id": rid,
            "bucket": bk,
            "difficulty": dv,
            "nq": nq,
            "toks": tokenize(nq),
        })
    if args.limit:
        queries = queries[: args.limit]
    print(f"loaded {len(queries)} coding/research d4-7 queries (threshold={args.threshold})", flush=True)

    # ── 3. 命中:每个 query 搜 47988 任务,70% token 重叠 = 命中 ──
    task_matched_by = defaultdict(list)  # task_idx -> list of query idx
    matched_queries = 0
    matched_with_files = 0
    for qi, q in enumerate(queries):
        if not q["toks"]:
            continue
        cand = set()
        for tok in q["toks"]:
            cand |= inv_idx.get(tok, set())
        hit = None
        for ti in cand:
            if like_match(q["toks"], all_tasks[ti]["_toks"], args.threshold):
                hit = ti
                break
        if hit is not None:
            task_matched_by[hit].append(qi)
            matched_queries += 1
            if all_tasks[hit]["has_inputs"]:
                matched_with_files += 1
        if (qi + 1) % 5000 == 0:
            print(f"  {qi+1}/{len(queries)} queries, matched {matched_queries}", flush=True)

    print(f"\n=== 命中结果 ===", flush=True)
    print(f"  queries matched: {matched_queries}/{len(queries)}", flush=True)
    print(f"  unique tasks hit: {len(task_matched_by)}", flush=True)
    print(f"  matched with files: {matched_with_files}", flush=True)

    # ── 4. 输出:被命中的任务,用 query 的验证标签 ──
    out_rows = []
    for ti, qis in task_matched_by.items():
        t = all_tasks[ti]
        if not t["has_inputs"]:
            continue
        # 一个任务可能被多 query 命中,每个 query 输出一行(query 验证标签)
        for qi in qis:
            q = queries[qi]
            out_rows.append({
                "D_id": t["D_id"],
                "domain": t["domain"],
                "bucket": q["bucket"],
                "difficulty": q["difficulty"],
                "label_source": "query_validated",
                "inputs_dir": t["inputs_dir"],
                "has_inputs": True,
                "user_prompt": t["user_prompt"],
                "matched_query_rid": q["record_id"],
                "query_text": q["nq"],
            })

    bk = Counter(r["bucket"] for r in out_rows)
    dv = Counter(r["difficulty"] for r in out_rows)
    print(f"\n=== 输出 (coding/research d4-7 + 有文件) ===", flush=True)
    print(f"  total (query,task) pairs: {len(out_rows)}", flush=True)
    print(f"  unique tasks: {len(set(r['D_id'] for r in out_rows))}", flush=True)
    print(f"  coding: {bk['coding']}, research: {bk['research']}", flush=True)
    print(f"  difficulty: {dict(sorted(dv.items(), key=lambda x: str(x[0])))}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅ wrote {OUT} ({len(out_rows)} pairs, {len(set(r['D_id'] for r in out_rows))} unique tasks)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
