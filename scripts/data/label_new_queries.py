#!/usr/bin/env python3
"""对 queries_new.jsonl 打能力桶标签（tokenhub, 并发 16, 断点续）。

基于 label_capability.py 改造：输入从 taskspec YAML 换为 queries jsonl。
只打 bucket 标签；persona 留空（单轮不需要）；其余字段从老数据模板补。

用法:
  python scripts/data/label_new_queries.py --limit 10          # 冒烟
  python scripts/data/label_new_queries.py                     # 全量
  python scripts/data/label_new_queries.py --resume            # 断点续
"""
from __future__ import annotations

import argparse, json, os, sys, textwrap, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_IN = ROOT / "datasets" / "queries_new.jsonl"
OUT = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
API_BASE = "https://tokenhub.sensetime.com/v1"

CANONICAL_BUCKETS = ["workflow","ops","qa","finance","office","communication","safety","coding","research"]
BUCKET_DEFS = {
    "workflow":      "多步骤工作流编排（拆解目标、串联动作、条件分支与流程协调）",
    "ops":           "系统操作（文件读写、命令行/终端执行、系统运维、资源增删改查）",
    "qa":            "问答检索（查事实、答问题、阅读理解、记忆检索；即查即答，不产出长报告）",
    "finance":       "财务金融（贷款/税务/估值/ROI 计算、采购等按金融业务规则算账）",
    "office":        "办公文档（办公问答、表格/报表处理、办公数据分析）",
    "communication": "沟通表达（邮件撰写/分类、内容创作、润色改写、翻译）",
    "safety":        "安全合规（拒绝不安全请求、漏洞/威胁评估、合规审查、敏感操作把关）",
    "coding":        "代码（编写、审查、调试、修复代码，代码正确性推理）",
    "research":      "研究综合（查多源信息并综合成报告/简报/摘要；区别于 qa 的即查即答）",
}


def _load_key():
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
        sys.exit("ERROR: TOKENHUB_API_KEY not set")
    return key


def _chat(messages, model, key, max_tokens=256, retries=3):
    import httpx
    last = None
    for attempt in range(retries):
        try:
            resp = httpx.post(f"{API_BASE}/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                headers={"Authorization": f"Bearer {key}"}, timeout=120.0)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last


def _parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    start = text.find("{")
    if start > 0: text = text[start:]
    end = text.rfind("}")
    if end > 0: text = text[:end + 1]
    return json.loads(text)


_SYSTEM = textwrap.dedent("""\
    你是能力分类器。给你一个 agent 任务，请判断【完成它主要依赖哪一种能力】，
    从下面的固定能力桶里选【唯一一个】最匹配的。

    ## 能力桶（按能力划分，不是话题）
    {bucket_defs}

    ## 规则
    - 只选一个 bucket，必须是上面列表里的英文桶名之一。
    - 按"完成任务所需的核心本领"选，不要按业务话题。
    - 只输出 JSON: {{"bucket": "<桶名>", "reason": "一句话理由"}}，不要 markdown。
""")

_USER = textwrap.dedent("""\
    query: {query}
    available_tools: {tools}

    这个任务主要依赖哪个能力桶？只回 JSON。
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    key = _load_key()

    # use hardcoded bucket definitions (same as label_buckets.py)
    names = CANONICAL_BUCKETS
    defs = "\n".join(f"- {b}: {d}" for b, d in BUCKET_DEFS.items())
    print(f"[label] {len(names)} buckets: {names}")

    # load queries
    queries = []
    for line in QUERIES_IN.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        queries.append(json.loads(line))
    if args.limit > 0: queries = queries[:args.limit]
    print(f"[label] {len(queries)} queries loaded")

    # resume
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and OUT.is_file():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["record_id"])
        print(f"[label] resume: {len(done)} already done, skipping")

    todo = [q for q in queries if q["record_id"] not in done]
    print(f"[label] to label: {len(todo)} (workers={args.workers})")

    from tqdm import tqdm
    system = _SYSTEM.format(bucket_defs=defs)
    valid = set(names)
    lock = threading.Lock()
    fout = OUT.open("a", encoding="utf-8")
    counters = {"ok": 0, "fail": 0}

    def _label_one(q):
        user = _USER.format(query=q["seed_query"][:2000], tools=q.get("tools", [])[:20])
        rid = q["record_id"]
        try:
            out = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}], args.model, key)
            bucket = _parse_json(out).get("bucket", "")
        except Exception as exc:
            tqdm.write(f"FAIL {rid}: {exc}")
            with lock: counters["fail"] += 1
            return
        if bucket not in valid:
            try:
                retry_user = user + f"\n\n上次未给出有效桶。必须从 {names} 里选一个。只回 JSON。"
                out = _chat([{"role": "system", "content": system}, {"role": "user", "content": retry_user}], args.model, key)
                bucket = _parse_json(out).get("bucket", "")
            except Exception:
                bucket = ""
        if bucket not in valid:
            tqdm.write(f"WARN {rid}: invalid bucket={bucket!r}, marking unknown")
            bucket = "unknown"
        rec = {
            "record_id": rid,
            "bucket": bucket,
            "seed_query": q["seed_query"],
            "source_file": q.get("source_file", ""),
        }
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            counters["ok"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(_label_one, todo), total=len(todo), desc="labeling"))

    fout.close()
    print(f"\n[label] done: {counters['ok']} ok / {counters['fail']} fail → {OUT}")


if __name__ == "__main__":
    main()
