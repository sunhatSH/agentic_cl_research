#!/usr/bin/env python3
"""重构 train_cl.parquet:给 12800 行补全文件位置 + 修路径 + 改 record_id=D_id。

策略(优先级递减,命中即停):
  1. D_id 路径:query 里的 E:\\hermes\\...\\D<N>\\<D_id>\\inputs\\ → 直接抽 D_id (3632 行已有)
  2. 50% token 重叠:query 命中 all_tasks_metadata 的 task.json user_prompt
  命中 → 该行拿到 D_id + inputs_dir(文件位置),record_id 改 D_id,gen_task_id 填 D_id
  未命中 → 保留原样(gen_task_id 空,产出型不注入文件)

同时:
  - prompt 去掉 system,只留单 user message(用 Hermes 默认 system)
  - query 路径归一化(Windows→Linux, collapse double-bs, residual fix)
  - record_id = D_id (cl_agent_dataset 用 gen_task_id 定位文件)

输出: datasets/train_cl.parquet + datasets/train_cl.jsonl (原地覆盖,先写 .tmp 再 replace)

用法:
  python3 scripts/data/rebuild_train_cl.py            # 干跑(统计)
  python3 scripts/data/rebuild_train_cl.py --apply    # 落地
  python3 scripts/data/rebuild_train_cl.py --apply --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from path_normalize import extract_gen_task_id, normalize_paths  # noqa: E402

ALL_TASKS = ROOT / "datasources" / "labeled" / "all_tasks_metadata.jsonl"
PARQUET = ROOT / "datasets" / "train_cl.parquet"
JSONL = ROOT / "datasets" / "train_cl.jsonl"


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="落地(否则干跑)")
    ap.add_argument("--threshold", type=float, default=0.5, help="token 重叠阈值(策略2)")
    args = ap.parse_args()

    # ── 1. 加载 47988 任务,建倒排索引 ──
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

    # ── 2. 加载 12800 parquet ──
    t = pq.read_table(str(PARQUET))
    rows = t.to_pylist()
    print(f"loaded {len(rows)} parquet rows", flush=True)

    # ── 3. 逐行重构 ──
    by_strategy = Counter()
    out_rows = []
    for ri, r in enumerate(rows):
        ei = dict(r["extra_info"])
        raw_q = str((ei.get("queries") or [""])[0])
        nq = normalize_query(raw_q)

        # 策略1: D_id 路径(从原始 query 抽,归一化前)
        gid = extract_gen_task_id(raw_q) or ei.get("gen_task_id") or ""
        strategy = ""
        inputs_dir = ""
        if gid:
            # 验证文件存在
            dp = gid.split("_")[0]
            cand_dir = os.path.join(str(ROOT), "datasources", "generated_tasks_hermes", dp, gid, "inputs")
            if os.path.isdir(cand_dir) and os.listdir(cand_dir):
                inputs_dir = cand_dir
                strategy = "1_did_path"
            else:
                gid = ""  # 文件不存在,降级

        # 策略2: 50% token 重叠
        if not gid:
            qtoks = tokenize(nq)
            if qtoks:
                cand = set()
                for tok in qtoks:
                    cand |= inv_idx.get(tok, set())
                best_ratio = 0.0
                best_ti = None
                for ti in cand:
                    ttoks = all_tasks[ti]["_toks"]
                    overlap = len(qtoks & ttoks)
                    ratio = overlap / min(len(qtoks), len(ttoks)) if qtoks and ttoks else 0
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_ti = ti
                if best_ti is not None and best_ratio >= args.threshold:
                    gid = all_tasks[best_ti]["D_id"]
                    inputs_dir = all_tasks[best_ti]["inputs_dir"]
                    strategy = f"2_token_{best_ratio:.2f}"

        by_strategy[strategy or "0_unmatched"] += 1

        # 重构行:prompt 单 user,record_id=D_id,gen_task_id=D_id,query 已归一化
        new_ei = {
            "record_id": gid or ei.get("record_id", ""),
            "bucket": ei.get("bucket", ""),
            "queries": [nq],
            "persona": ei.get("persona", ""),
            "available_tools": ei.get("available_tools", []),
            "missing_info_slots": ei.get("missing_info_slots", []),
            "safety_constraints": ei.get("safety_constraints", []),
            "difficulty": ei.get("difficulty", ""),
            "gen_task_id": gid,
            "ws_dir": ei.get("ws_dir", ""),
        }
        new_row = {
            "prompt": [{"role": "user", "content": nq}],
            "data_source": r.get("data_source", "agentic_cl"),
            "reward_model": r.get("reward_model", {
                "ground_truth": "", "style": "rule",
                "reward_fn": {"_function_name": "trainer.model_reward_omni.compute_score"},
            }),
            "bucket": r.get("bucket", ""),
            "extra_info": new_ei,
        }
        out_rows.append(new_row)

    # ── 4. 统计 ──
    print(f"\n=== 重构统计 (threshold={args.threshold}) ===", flush=True)
    print(f"  总行数: {len(out_rows)}", flush=True)
    for s, c in sorted(by_strategy.items()):
        print(f"  {s}: {c}", flush=True)
    matched = sum(c for s, c in by_strategy.items() if s.startswith(("1_", "2_")))
    print(f"  命中(有文件): {matched}/{len(out_rows)}", flush=True)

    # 按桶
    bk_matched = Counter(r["bucket"] for r in out_rows if r["extra_info"]["gen_task_id"])
    print(f"  命中按桶: {dict(bk_matched)}", flush=True)

    # ── 5. 落地 ──
    if not args.apply:
        print("\n加 --apply 落地", flush=True)
        return 0

    new_t = pa.Table.from_pylist(out_rows, schema=t.schema)
    tmp = PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(PARQUET)
    tmp_j = JSONL.with_suffix(".jsonl.tmp")
    with open(tmp_j, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_j.replace(JSONL)
    print(f"\n✅ 落地 {PARQUET.name} + {JSONL.name} ({len(out_rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
