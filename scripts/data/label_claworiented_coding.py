#!/usr/bin/env python3
"""从 generalClaworiented 系列打桶标 → 提取 coding 类 → 带轨迹判难度。

背景:sweCoding 要剔除(稀疏抽取无法完成),coding 桶缺口 ~3000。从郑乃榕
generalClaworiented 系列里捞干净的 agentic coding 补。

流程(每条一记录,并发 64,断点续跑):
  1. 遍历指定的中小文件(跳过 claw_dpresearch_skill_* 研究大文件)。
  2. 【防污染】排除 ClawEval 评测任务:task_id 以 T<n>/C<n>/M<n> 开头的直接丢
     (synth_* 合成任务、无 task_id 的保留)。
  3. luna 按 BUCKET_SYSTEM(9桶)打桶标,只留 bucket==coding。
  4. 对 coding 记录,luna 带原始轨迹判难度(DIFF_PROMPT,1-10)。
  5. 落地 coding_candidates.jsonl:{src_file, row_idx, task_id, bucket, difficulty,
     query, n_msgs, has_tool}。GT 待产出后另做。

用法:
  python3 scripts/data/label_claworiented_coding.py            # 全量
  python3 scripts/data/label_claworiented_coding.py --limit-per-file 200  # 每文件抽样
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
from label_unlabeled_tasks import BUCKET_SYSTEM, CANONICAL  # noqa: E402
from score_lh_difficulty import DIFF_PROMPT, parse_json, traj_to_text  # noqa: E402

API_BASE = "https://tokenhub.sensetime.com/v1"
MODEL = "gpt-5.6-luna"
WORKERS = 64

GC = "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/generalClaworiented"
# 纳入扫描的中小 coding 相关文件(跳过 claw_dpresearch_skill_* 研究大文件、claw-eval-sft 纯评测轨迹)
SRC_FILES = [
    "generalClaworiented_claw-eval-sft-phase4_claude48opus_custom_mac_20260625",
    "generalClaworiented_claw-eval-gap-filling_claude48opus_custom_mac_20260706",
    "generalClaworiented_wild-claw-sft_claude48opus_openclaw260626_mac_20260626",
    "generalClaworiented_b1b2b3b4_claude48opus_custom_mac_20260630",
    "generalClaworiented_claw-aug-fullset2253_claude48opus_custom_mac_20260703",
    "B1_B2_claude_opus48_perfect_audit_Hermes_plus_b2_orig_repair_v3v4_strict17_20260618",
]

OUT = ROOT / "datasources" / "labeled" / "claworiented_coding_candidates.jsonl"
BUCKET_CACHE = ROOT / "datasources" / "labeled" / "claworiented_bucket_cache.jsonl"

# ClawEval 评测任务前缀:task_id 命中则丢(防训练/评测污染)
_EVAL_TASK_RE = re.compile(r"^[TCM]\d{2,3}([_a-z]|$)", re.IGNORECASE)


def load_key() -> str:
    key = os.environ.get("TOKENHUB_API_KEY", "") or os.environ.get("AGENT_MODEL_KEY", "")
    if not key:
        for envf in (ROOT / ".env", ROOT / "docker" / "sandbox" / "runtime.env"):
            if envf.is_file():
                for line in envf.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    for k in ("TOKENHUB_API_KEY=", "AGENT_MODEL_KEY="):
                        if s.startswith(k):
                            key = s.split("=", 1)[1].strip().strip("\"'")
                            break
                if key:
                    break
    if not key:
        sys.exit("ERROR: no TOKENHUB_API_KEY / AGENT_MODEL_KEY")
    return key


def resolve_path(name: str) -> Path:
    return Path(GC) / name / f"{name}.jsonl"


def first_user(msgs) -> str:
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content") or ""
            # 去掉 system-reminder / harness 引导段
            c = re.sub(r"<system-reminder>.*?</system-reminder>", "", c, flags=re.DOTALL)
            return c.strip()
    return ""


def is_eval_task(task_id: str) -> bool:
    return bool(task_id) and bool(_EVAL_TASK_RE.match(task_id))


def chat(client, messages, max_tokens=256, retries=3):
    for attempt in range(retries):
        try:
            resp = client.post(
                f"{API_BASE}/chat/completions",
                json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
                timeout=90.0,
            )
            resp.raise_for_status()
            c = resp.json()["choices"][0]["message"]["content"]
            if c and c.strip():
                return c
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return ""


def load_records(limit_per_file: int | None) -> list[dict]:
    """加载所有源文件的记录(带来源标记,排除评测任务)。"""
    recs = []
    excluded = 0
    for name in SRC_FILES:
        p = resolve_path(name)
        if not p.is_file():
            print(f"  [skip] 不存在: {name}", flush=True)
            continue
        n = 0
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if limit_per_file and n >= limit_per_file:
                    break
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                md = d.get("metadata", {}) if isinstance(d.get("metadata"), dict) else {}
                tid = md.get("task_id", "") or ""
                if is_eval_task(tid):
                    excluded += 1
                    continue
                msgs = d.get("messages", [])
                q = first_user(msgs)
                if not q:
                    continue
                recs.append({
                    "src": name,
                    "row": i,
                    "task_id": tid,
                    "query": q,
                    "traj": traj_to_text(msgs),
                    "n_msgs": len(msgs),
                    "has_tool": any(m.get("role") == "tool" for m in msgs) or any(m.get("tool_calls") for m in msgs),
                })
                n += 1
        print(f"  [{name[:45]}] 收 {n} 条", flush=True)
    print(f"排除评测任务(T/C/M 号): {excluded}", flush=True)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-file", type=int, default=None)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    key = load_key()
    recs = load_records(args.limit_per_file)
    print(f"\n候选记录: {len(recs)}", flush=True)

    # 断点续跑:已打桶的 (src,row) 跳过
    done_bucket = {}
    if BUCKET_CACHE.is_file():
        with open(BUCKET_CACHE, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_bucket[(d["src"], d["row"])] = d["bucket"]
                except Exception:  # noqa: BLE001
                    continue
    todo = [r for r in recs if (r["src"], r["row"]) not in done_bucket]
    print(f"待打桶: {len(todo)} ({len(done_bucket)} 已打)", flush=True)

    # ── 阶段1:打桶 ──
    BUCKET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if todo:
        client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(90.0))
        fout = open(BUCKET_CACHE, "a", encoding="utf-8")
        ok = 0
        t0 = time.time()

        def bucket_one(r):
            raw = chat(client, [
                {"role": "system", "content": BUCKET_SYSTEM},
                {"role": "user", "content": r["query"][:4000]},
            ])
            b = (parse_json(raw).get("bucket", "") or "").strip().lower()
            return r, b if b in CANONICAL else ""

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r, b in ex.map(bucket_one, todo):
                done_bucket[(r["src"], r["row"])] = b
                fout.write(json.dumps({"src": r["src"], "row": r["row"], "task_id": r["task_id"], "bucket": b}, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
                if ok % 500 == 0 or ok == len(todo):
                    print(f"  [bucket] {ok}/{len(todo)} rate={ok/(time.time()-t0):.1f}/s", flush=True)
        fout.close()

    bc = Counter(done_bucket.values())
    print(f"\n=== 桶分布 ===", flush=True)
    for b, n in bc.most_common():
        print(f"  {b or '(未知)'}: {n}", flush=True)

    # coding 记录
    coding = [r for r in recs if done_bucket.get((r["src"], r["row"])) == "coding"]
    print(f"\ncoding 类: {len(coding)}", flush=True)
    if not coding:
        print("无 coding,结束", flush=True)
        return 0

    # ── 阶段2:coding 带轨迹判难度 ──
    done_diff = {}
    if OUT.is_file():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_diff[(d["src"], d["row"])] = d
                except Exception:  # noqa: BLE001
                    continue
    todo_d = [r for r in coding if (r["src"], r["row"]) not in done_diff]
    print(f"待判难度: {len(todo_d)} ({len(done_diff)} 已判)", flush=True)

    if todo_d:
        client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(120.0))
        fout = open(OUT, "a", encoding="utf-8")
        ok = 0
        t0 = time.time()

        def diff_one(r):
            raw = chat(client, [{"role": "user", "content": f"任务执行轨迹:\n{r['traj'][:16000]}\n\n{DIFF_PROMPT}"}])
            diff = parse_json(raw).get("difficulty")
            try:
                diff = max(1, min(10, int(float(diff)))) if diff is not None else None
            except Exception:  # noqa: BLE001
                diff = None
            return r, diff

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r, diff in ex.map(diff_one, todo_d):
                rec = {
                    "src": r["src"], "row": r["row"], "task_id": r["task_id"],
                    "bucket": "coding", "difficulty": diff,
                    "n_msgs": r["n_msgs"], "has_tool": r["has_tool"],
                    "query": r["query"][:2000],
                }
                done_diff[(r["src"], r["row"])] = rec
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
                if ok % 200 == 0 or ok == len(todo_d):
                    print(f"  [diff] {ok}/{len(todo_d)} rate={ok/(time.time()-t0):.1f}/s", flush=True)
        fout.close()

    # 汇总
    all_coding = list(done_diff.values())
    dc = Counter(r["difficulty"] for r in all_coding if r.get("difficulty"))
    d47 = sum(v for k, v in dc.items() if 4 <= k <= 7)
    print(f"\n=== coding 难度分布(共 {len(all_coding)}) ===", flush=True)
    for d in sorted(k for k in dc if k):
        print(f"  d{d}: {dc[d]}", flush=True)
    print(f"  d4-7: {d47}", flush=True)
    # 按来源
    sc = Counter(r["src"][:40] for r in all_coding)
    print(f"\n按来源:", flush=True)
    for s, n in sc.most_common():
        print(f"  {s}: {n}", flush=True)
    print(f"\n清单: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
