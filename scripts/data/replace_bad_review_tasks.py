#!/usr/bin/env python3
"""替换 train_cl 里"读文件但文件源缺失"的坏行(F9 收尾,2026-08-23)。

坏行 = query 引用 /home/user/workspace/ 或 ./inputs/ 读文件,但文件源不存在
(无 ws_dir 快照 / 无 gen_task_id inputs / 无 taskspecs_w3 files)。

替换:从 labeled 同桶、不在保留的好行里、unique 的题中,按难度 4-6 优先、其次 7
随机选(固定 seed),原位替换。新题走 path_normalize + 抽 gen_task_id/ws_dir。
保持结构(coding 6400 + research 6400 = 12800),不管每步难度均衡。

用法:
  python3 scripts/data/replace_bad_review_tasks.py            # 干跑
  python3 scripts/data/replace_bad_review_tasks.py --apply    # 落地
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
from path_normalize import extract_gen_task_id, extract_ws_dir, normalize_paths  # noqa: E402

PARQUET = ROOT / "datasets" / "train_cl.parquet"
LABELED = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
DIFF = ROOT / "datasources" / "labeled" / "difficulty_all.jsonl"
REVIEW_WS = ROOT / "datasources" / "review_ws"
SEED = 42
DIFF_PRIORITY = ["4", "5", "6", "7"]

SYSTEM_PROMPT = (
    "You are a capable autonomous agent. Complete the user's task using the available tools. "
    "Work independently — never ask the user for input, confirmation, or clarification. "
    "When faced with ambiguity or multiple options, pick the most reasonable or first option "
    "and proceed without hesitation."
)
_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def _has_ws(rid: str, idx: dict) -> bool:
    rel = idx.get(rid)
    return bool(rel and os.path.isdir(REVIEW_WS / rel / "ws"))


def _file_ok_q(q: str, rid: str, gid: str, idx: dict) -> bool:
    """query 若读文件,文件源是否存在。"""
    rw = "/home/user/workspace" in q
    ri = "./inputs" in q
    if not rw and not ri:
        return True
    if rw and _has_ws(rid, idx):
        return True
    if ri:
        if gid:
            dp = gid.split("_")[0]
            if os.path.isdir(ROOT / "datasources" / "generated_tasks_hermes" / dp / gid / "inputs"):
                return True
        if os.path.isdir(ROOT / "datasources" / "taskspecs_w3" / rid / "files"):
            return True
        if os.path.isdir(ROOT / "datasources" / "seed2traj_taskspecs" / rid / "files"):
            return True
        return False
    return not rw  # 读 workspace 但无 ws → 坏


def _build_row(rid: str, bucket: str, raw_q: str, dv: str) -> dict:
    q = raw_q
    if "<system-reminder>" in q:
        q = _REMINDER_RE.sub("", q).strip()
    gid = extract_gen_task_id(q) or ""
    ws = extract_ws_dir(q) or ""
    q = normalize_paths(q)
    return {
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
        "bucket": bucket,
        "extra_info": {
            "record_id": rid,
            "bucket": bucket,
            "queries": [q],
            "persona": "",
            "available_tools": [],
            "missing_info_slots": [],
            "safety_constraints": [],
            "difficulty": str(dv),
            "gen_task_id": gid,
            "ws_dir": ws,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    rng = random.Random(SEED)

    idx = json.load(open(REVIEW_WS / "index.json")) if (REVIEW_WS / "index.json").exists() else {}
    diff = {}
    for line in open(DIFF):
        d = json.loads(line.strip())
        diff[d["record_id"]] = str(d.get("difficulty"))

    t = pq.read_table(str(PARQUET))
    rows = t.to_pylist()

    # 坏行:读文件但缺源
    bad_idx = []
    for i, r in enumerate(rows):
        ei = r["extra_info"]
        rid = ei["record_id"]
        gid = ei.get("gen_task_id") or ""
        q = str((ei.get("queries") or [""])[0])
        if not _file_ok_q(q, rid, gid, idx):
            bad_idx.append(i)
    need = Counter(rows[i]["bucket"] for i in bad_idx)
    good_rids = {rows[i]["extra_info"]["record_id"] for i in range(len(rows)) if i not in set(bad_idx)}
    print(f"坏行 {len(bad_idx)}: {dict(need)}")

    # 替换池:labeled 同桶 + 不在好行 + unique + 【文件齐全】(不引入新缺源)。
    # 按 (bucket, is_non_unk, difficulty) 分组。优先级:非unk d4-6 → 非unk d7 → unk d4-6 → unk d7。
    pool: dict[tuple[str, bool, str], list[tuple[str, str]]] = defaultdict(list)
    seen = set()
    for line in open(LABELED):
        d = json.loads(line)
        rid = d["record_id"]
        bk = d.get("bucket")
        if bk not in ("coding", "research") or rid in good_rids or rid in seen:
            continue
        seen.add(rid)
        sq = d.get("seed_query", "")
        if "<system-reminder>" in sq:
            sq = _REMINDER_RE.sub("", sq).strip()
        gid = extract_gen_task_id(sq) or ""
        q = normalize_paths(sq)
        if not _file_ok_q(q, rid, gid, idx):
            continue  # 只选文件齐全的,不引入新缺源
        is_non_unk = not rid.startswith("unk_")
        dv = diff.get(rid, "?")
        pool[(bk, is_non_unk, dv)].append((rid, sq))
    for k in pool:
        rng.shuffle(pool[k])

    # 取题顺序:非unk d4-6 → 非unk d7 → unk d4-6 → unk d7
    ORDER = [
        (True, "4"), (True, "5"), (True, "6"), (True, "7"),
        (False, "4"), (False, "5"), (False, "6"), (False, "7"),
    ]
    picks: dict[str, list] = {}
    ok = True
    for bk, cnt in need.items():
        chosen = []
        for is_nu, dv in ORDER:
            while len(chosen) < cnt and pool[(bk, is_nu, dv)]:
                rid, sq = pool[(bk, is_nu, dv)].pop()
                chosen.append((rid, sq, dv))
        picks[bk] = chosen
        status = "✓" if len(chosen) == cnt else f"✗只凑到{len(chosen)}"
        n_non_unk = sum(1 for c in chosen if not c[0].startswith("unk_"))
        dstat = Counter(c[2] for c in chosen)
        print(f"  {bk}: 需{cnt} 选到{len(chosen)} {status} (非unk {n_non_unk}) 难度{dict(sorted(dstat.items()))}")
        if len(chosen) < cnt:
            ok = False
    if not ok:
        print("替换池不足,未落地。")
        return 1

    # 原位替换
    cursor = {bk: 0 for bk in picks}
    for i in bad_idx:
        bk = rows[i]["bucket"]
        rid, sq, dv = picks[bk][cursor[bk]]
        cursor[bk] += 1
        rows[i] = _build_row(rid, bk, sq, dv)

    # 校验
    all_rids = [r["extra_info"]["record_id"] for r in rows]
    dup = [x for x, c in Counter(all_rids).items() if c > 1]
    still_bad = sum(1 for r in rows
                    if not _file_ok_q(str((r["extra_info"].get("queries") or [""])[0]),
                                      r["extra_info"]["record_id"],
                                      r["extra_info"].get("gen_task_id") or "", idx))
    print(f"\n替换后 {len(rows)} 行, record_id 重复: {len(dup)}, 仍缺源: {still_bad}")

    if args.apply and not dup:
        new_t = pa.Table.from_pylist(rows, schema=t.schema)
        tmp = PARQUET.with_suffix(".parquet.tmp")
        pq.write_table(new_t, tmp)
        tmp.replace(PARQUET)
        jsonl = PARQUET.with_suffix(".jsonl")
        tmpj = jsonl.with_suffix(".jsonl.tmp")
        with open(tmpj, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmpj.replace(jsonl)
        print(f"✅ 落地 {PARQUET.name} + .jsonl")
    elif not args.apply:
        print("加 --apply 落地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
