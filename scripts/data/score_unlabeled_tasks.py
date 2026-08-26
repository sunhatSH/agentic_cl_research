#!/usr/bin/env python3
"""给 generated_tasks_hermes 里未在 labeled 出现过的任务打【桶 + 难度】标签。

只保留 coding / research 桶的任务（用户只要这两个）。难度 1-10 整数。
走 tokenhub gpt-5.6-luna，64 并发，断点续（已打标的 D_id 跳过）。

Input : datasources/labeled/unlabeled_tasks_to_score.jsonl  (26104 行, collect_unlabeled 产出)
Output: datasources/labeled/unlabeled_scored.jsonl          (每行 D_id+bucket+difficulty)
         只含 coding/research 的行。

用法:
  python3 scripts/data/score_unlabeled_tasks.py            # 全量,断点续
  python3 scripts/data/score_unlabeled_tasks.py --limit 50  # 冒烟
  python3 scripts/data/score_unlabeled_tasks.py --workers 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "datasources" / "labeled" / "unlabeled_tasks_to_score.jsonl"
OUT = ROOT / "datasources" / "labeled" / "unlabeled_scored.jsonl"
API_BASE = "https://tokenhub.sensetime.com/v1"
MODEL = "gpt-5.6-luna"
WORKERS = 64

# 9 桶定义（与 label_capability.py / buckets.json 一致）
BUCKET_DEFS = {
    "workflow": "多步骤工作流编排",
    "ops": "系统操作(文件读写/命令行/系统运维)",
    "qa": "问答检索(查事实/答问题/阅读理解)",
    "finance": "财务金融(贷款/税务/估值/采购)",
    "office": "办公文档(表格/报表/办公数据分析)",
    "communication": "沟通表达(邮件/内容创作/润色/翻译)",
    "safety": "安全合规(拒绝不安全请求/漏洞评估/合规审查)",
    "coding": "代码(编写/审查/调试/修复)",
    "research": "研究综合(多源信息综合成报告/简报)",
}

PROMPT = """你是能力分类器 + 任务难度评估器。给定一个 agent 任务,请:

1. 从下面的固定能力桶里选【唯一一个】最匹配的(按"完成任务所需的核心本领"选,不按话题):
{bucket_defs}

2. 评估任务【本身】的绝对完成难度(1-10 整数,不是评估回答质量):
   - 1-2: 极简单,单步完成
   - 3-4: 简单,少量步骤
   - 5-6: 中等,多步处理/一定推理
   - 7-8: 困难,多步骤依赖/复杂推理
   - 9-10: 极难,架构级规划/跨领域综合
   不要因"看起来专业"或"描述很长"就给高分;只看实际完成所需能力。

只输出 JSON,不要 markdown,不要解释:
{{"bucket": "<桶名>", "difficulty": <1-10整数>}}"""


def load_key() -> str:
    key = os.environ.get("TOKENHUB_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("TOKENHUB_API_KEY="):
                    key = s.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        sys.exit("ERROR: TOKENHUB_API_KEY not set (see .env)")
    return key


def chat(client, messages, retries=3):
    for attempt in range(retries):
        try:
            resp = client.post(
                f"{API_BASE}/chat/completions",
                json={"model": MODEL, "messages": messages, "max_tokens": 256, "temperature": 0.0},
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return ""


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        text = text[s : e + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="只打前 N 条(冒烟)")
    args = ap.parse_args()

    key = load_key()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # load all tasks to score
    all_tasks = []
    with open(IN, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            all_tasks.append(json.loads(line))
    if args.limit:
        all_tasks = all_tasks[: args.limit]

    # resume: skip already-scored D_id
    done = set()
    if OUT.is_file():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                done.add(d["D_id"])
    remaining = [t for t in all_tasks if t["D_id"] not in done]
    print(f"total {len(all_tasks)}, already scored {len(done)}, remaining {len(remaining)}", flush=True)
    if not remaining:
        print("nothing to do")
        return 0

    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(60.0))
    lock = threading.Lock()
    fout = open(OUT, "a", encoding="utf-8")
    ok = fail = 0
    kept = 0  # coding/research kept
    t0 = time.time()

    def score_one(task):
        nonlocal ok, fail, kept
        up = task["user_prompt"]
        # 截断超长 prompt(省 token)
        if len(up) > 4000:
            up = up[:4000] + "..."
        user_msg = f"任务:\n{up}\n\n{PROMPT.format(bucket_defs=chr(10).join(f'- {b}: {d}' for b, d in BUCKET_DEFS.items()))}"
        raw = chat(client, [{"role": "user", "content": user_msg}])
        parsed = parse_json(raw)
        bucket = parsed.get("bucket", "").strip().lower()
        diff = parsed.get("difficulty")
        try:
            diff = int(diff)
        except (TypeError, ValueError):
            diff = None
        res = {"D_id": task["D_id"], "domain": task["domain"], "bucket": bucket, "difficulty": diff}
        with lock:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            if diff is not None and bucket:
                ok += 1
                if bucket in ("coding", "research"):
                    kept += 1
            else:
                fail += 1
            done_n = ok + fail
            if done_n % 200 == 0 or done_n == len(remaining):
                el = time.time() - t0
                rate = done_n / el if el else 0
                eta = (len(remaining) - done_n) / rate if rate else 0
                print(
                    f"  {done_n}/{len(remaining)} ok={ok} fail={fail} kept(cr)={kept} "
                    f"rate={rate:.1f}/s eta={eta:.0f}s",
                    flush=True,
                )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(score_one, remaining))

    fout.close()
    print(f"\ndone: {ok} ok, {fail} fail, {kept} coding/research kept", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
