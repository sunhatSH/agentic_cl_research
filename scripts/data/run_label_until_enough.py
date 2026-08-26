#!/usr/bin/env python3
"""分桶+难度双进程并行,够 1.5x 缺口即停。

每 10 分钟检查:
  - 已分桶的 coding/research/office 数量
  - 已打难度的 coding/research/office d4-7 数量
  - 缺口 = 未命中 4445 (coding 1564 + research 2881)
  - 1.5x 目标: coding 2346, research 4322 (若 research 不够,改 office+coding 各 6400)
  - 两个桶都够 1.5x → 停两个进程

启动:
  python3 scripts/data/run_label_until_enough.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from path_normalize import extract_gen_task_id  # noqa: E402

BUCKET_OUT = ROOT / "datasources" / "labeled" / "unlabeled_bucket.jsonl"
DIFF_OUT = ROOT / "datasources" / "labeled" / "unlabeled_difficulty.jsonl"
LABELED = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
DIFF_ALL = ROOT / "datasources" / "labeled" / "difficulty_all.jsonl"
LOG = ROOT / "logs" / "label_until_enough.log"

# 缺口(未命中 4445): coding 1564 + research 2881
# 1.5x 目标
TARGET_15X = {"coding": 2346, "research": 4322}
# 若 research 不够,改 office+coding 各 6400
FALLBACK_TARGET = {"coding": 6400, "office": 6400}

CHECK_INTERVAL = 600  # 10 min


def load_validated_bd() -> dict[str, tuple[str, int | None]]:
    """labeled 验证桶+难度 (D_id -> (bucket, difficulty))。"""
    rid_diff = {}
    if DIFF_ALL.is_file():
        with open(DIFF_ALL, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid_diff[d["record_id"]] = (d.get("bucket"), d.get("difficulty"))
    out = {}
    if not LABELED.is_file():
        return out
    with open(LABELED, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gid = extract_gen_task_id(d.get("seed_query", ""))
            if gid and gid not in out:
                bk = d.get("bucket")
                dv = rid_diff.get(d["record_id"], (None, None))[1]
                out[gid] = (bk, dv)
    return out


def load_ai_bucket() -> dict[str, str]:
    out = {}
    if BUCKET_OUT.is_file():
        with open(BUCKET_OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("bucket"):
                        out[d["D_id"]] = d["bucket"]
    return out


def load_ai_diff() -> dict[str, int]:
    out = {}
    if DIFF_OUT.is_file():
        with open(DIFF_OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("difficulty") is not None:
                        out[d["D_id"]] = d["difficulty"]
    return out


def check_enough():
    """返回 (bucket_counts, diff_d47_counts, enough)。"""
    validated = load_validated_bd()
    ai_bk = load_ai_bucket()
    ai_dv = load_ai_diff()

    # 合并桶(验证优先)
    all_bk = {}
    for gid, (bk, _) in validated.items():
        all_bk[gid] = bk
    for gid, bk in ai_bk.items():
        if gid not in all_bk:
            all_bk[gid] = bk

    # 合并难度(验证优先)
    all_dv = {}
    for gid, (_, dv) in validated.items():
        if dv is not None:
            all_dv[gid] = dv
    for gid, dv in ai_dv.items():
        if gid not in all_dv:
            all_dv[gid] = dv

    from collections import Counter
    bk_cnt = Counter(all_bk.values())
    # d4-7 且有桶
    d47_cnt = Counter()
    for gid, dv in all_dv.items():
        if dv is not None and 4 <= dv <= 7:
            b = all_bk.get(gid)
            if b:
                d47_cnt[b] += 1

    # 判断够不够 1.5x
    coding_ok = d47_cnt["coding"] >= TARGET_15X["coding"]
    research_ok = d47_cnt["research"] >= TARGET_15X["research"]
    # fallback: office+coding 各 6400
    fallback_ok = d47_cnt["coding"] >= FALLBACK_TARGET["coding"] and d47_cnt["office"] >= FALLBACK_TARGET["office"]
    enough = (coding_ok and research_ok) or fallback_ok
    return bk_cnt, d47_cnt, enough


def main() -> int:
    os.makedirs(ROOT / "logs", exist_ok=True)
    # 启动分桶+难度两个进程(后台)
    log_bk = open(ROOT / "logs" / "label_bucket.log", "a")
    log_dv = open(ROOT / "logs" / "label_diff.log", "a")
    p_bucket = subprocess.Popen(
        ["python3", str(ROOT / "scripts" / "data" / "label_unlabeled_tasks.py"), "--stage", "bucket"],
        stdout=log_bk, stderr=subprocess.STDOUT,
    )
    p_diff = subprocess.Popen(
        ["python3", str(ROOT / "scripts" / "data" / "label_unlabeled_tasks.py"), "--stage", "difficulty"],
        stdout=log_dv, stderr=subprocess.STDOUT,
    )
    print(f"started bucket pid={p_bucket.pid}, difficulty pid={p_diff.pid}", flush=True)

    with open(LOG, "a") as logf:
        logf.write(f"\n=== run started {time.strftime('%H:%M:%S')} ===\n")
        while True:
            bk_cnt, d47_cnt, enough = check_enough()
            msg = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"bucket: coding={bk_cnt['coding']} research={bk_cnt['research']} office={bk_cnt['office']} | "
                f"d4-7: coding={d47_cnt['coding']} research={d47_cnt['research']} office={d47_cnt['office']} | "
                f"target1.5x: coding{TARGET_15X['coding']}/research{TARGET_15X['research']} | "
                f"enough={enough}\n"
            )
            print(msg, flush=True)
            logf.write(msg)
            logf.flush()
            if enough:
                print("✅ 够 1.5x,停两个进程", flush=True)
                p_bucket.terminate()
                p_diff.terminate()
                p_bucket.wait(timeout=10)
                p_diff.wait(timeout=10)
                break
            # 检查进程是否还在
            if p_bucket.poll() is not None and p_diff.poll() is not None:
                print("两个进程都结束了", flush=True)
                break
            time.sleep(CHECK_INTERVAL)

    log_bk.close()
    log_dv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
