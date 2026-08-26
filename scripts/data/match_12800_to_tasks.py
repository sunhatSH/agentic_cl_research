#!/usr/bin/env python3
"""用 12800 parquet query 命中 all_tasks_metadata.jsonl 的 D_id,找文件位置。

思路:
  - 12800 parquet query 有【已验证】bucket+difficulty
  - all_tasks_metadata.jsonl 有 47988 个 D_id + user_prompt + inputs_dir(文件位置)
  - 用 12800 query 去【搜索】47988 任务:70% token 重叠 = 命中
  - 命中 → query 拿到对应 D_id + inputs_dir(文件位置)
  - query 的验证桶+难度 + 文件位置 = 完整训练行

输出: datasources/labeled/matched_12800.jsonl
  每行 = 一个命中的 (query, D_id) 配对,带:验证桶+验证难度+文件路径+原 query
  只保留 coding/research。

用法:
  python3 scripts/data/match_12800_to_tasks.py            # 全量
  python3 scripts/data/match_12800_to_tasks.py --threshold 0.7
  python3 scripts/data/match_12800_to_tasks.py --limit 500  # 冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from path_normalize import normalize_paths  # noqa: E402

ALL_TASKS = ROOT / "datasources" / "labeled" / "all_tasks_metadata.jsonl"
PARQUET = ROOT / "datasets" / "train_cl.parquet"
OUT = ROOT / "datasources" / "labeled" / "matched_12800.jsonl"


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

    # ── 1. 加载 47988 任务,建倒排索引(token -> task idx)──
    all_tasks = []
    with open(ALL_TASKS, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            t["_nq"] = normalize_query(t["user_prompt"])
            t["_toks"] = tokenize(t["_nq"])
            all_tasks.append(t)
    print(f"loaded {len(all_tasks)} tasks", flush=True)

    inv_idx = defaultdict(set)
    for i, t in enumerate(all_tasks):
        for tok in t["_toks"]:
            inv_idx[tok].add(i)
    print(f"inverted index: {len(inv_idx)} tokens", flush=True)

    # ── 2. 加载 12800 parquet query(已验证 bucket+difficulty)──
    rows = pq.read_table(str(PARQUET)).to_pylist()
    queries = []
    for r in rows:
        ei = r["extra_info"]
        q = normalize_query(str((ei.get("queries") or [""])[0]))
        queries.append({
            "record_id": ei["record_id"],
            "bucket": r["bucket"],
            "difficulty": ei.get("difficulty"),
            "gen_task_id": ei.get("gen_task_id") or "",
            "nq": q,
            "toks": tokenize(q),
        })
    if args.limit:
        queries = queries[: args.limit]
    print(f"loaded {len(queries)} queries (threshold={args.threshold})", flush=True)

    # ── 3. 命中:每个 query 搜 47988 任务,70% token 重叠 = 命中 ──
    matched = 0
    matched_with_files = 0
    out_rows = []
    for qi, q in enumerate(queries):
        if not q["toks"]:
            out_rows.append({**q, "matched_D_id": "", "matched": False})
            continue
        # 候选:至少共享 1 token 的任务
        cand = set()
        for tok in q["toks"]:
            cand |= inv_idx.get(tok, set())
        # 70% 重叠
        hit = None
        for ti in cand:
            if like_match(q["toks"], all_tasks[ti]["_toks"], args.threshold):
                hit = ti
                break
        if hit is not None:
            t = all_tasks[hit]
            matched += 1
            if t["has_inputs"]:
                matched_with_files += 1
            out_rows.append({
                **q,
                "matched_D_id": t["D_id"],
                "matched_domain": t["domain"],
                "matched_inputs_dir": t["inputs_dir"],
                "matched_has_inputs": t["has_inputs"],
                "matched_task_prompt": t["user_prompt"],
                "matched": True,
            })
        else:
            out_rows.append({**q, "matched_D_id": "", "matched": False})
        if (qi + 1) % 2000 == 0:
            print(f"  {qi+1}/{len(queries)} matched {matched}", flush=True)

    print(f"\n=== 命中结果 ===", flush=True)
    print(f"  queries matched: {matched}/{len(queries)}", flush=True)
    print(f"  matched with files: {matched_with_files}", flush=True)

    # ── 4. 输出 ──
    bk = Counter(r["bucket"] for r in out_rows if r["matched"])
    print(f"  matched buckets: {dict(bk)}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out_rows:
            r2 = {k: (sorted(v) if isinstance(v, set) else v) for k, v in r.items()}
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    print(f"\n✅ wrote {OUT} ({len(out_rows)} rows, {matched} matched)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
