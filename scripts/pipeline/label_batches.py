#!/usr/bin/env python3
"""分批打标 + 每批后检查桶容量，达到目标自动停。

  目标: 每桶 ≥ TARGET 条（默认 1280 = 40 steps × 32）
  模型: claude-sonnet-4-6 (tokenhub), 并发 16
  断点续: 已打标的 record_id 自动跳过
"""
import json, os, sys, time, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_IN = ROOT / "datasets" / "queries_new.jsonl"
OUT = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
API_BASE = "https://tokenhub.sensetime.com/v1"
TARGET = 1280  # per bucket minimum (40 steps × 32)
BATCH_SIZE = 100  # label this many at a time, then check (incremental reporting)
# Only these buckets need more data; others already met target.  Can
# be overridden with --buckets on the command line.
FOCUS_BUCKETS = None  # None = all; set to ["ops","research"] to skip others

CANONICAL = ["workflow","ops","qa","finance","office","communication","safety","coding","research"]
BUCKET_DEFS = {
    "workflow": "多步骤工作流编排", "ops": "系统操作(文件读写/命令行/系统运维)",
    "qa": "问答检索(查事实/答问题/阅读理解)", "finance": "财务金融(贷款/税务/估值/采购)",
    "office": "办公文档(表格/报表/办公数据分析)", "communication": "沟通表达(邮件/内容创作/润色/翻译)",
    "safety": "安全合规(拒绝不安全请求/漏洞评估/合规审查)", "coding": "代码(编写/审查/调试/修复)",
    "research": "研究综合(多源信息综合成报告/简报)",
}
SYSTEM_PROMPT = (
    "你是能力分类器。给定一个agent任务，从下面的固定能力桶里选【唯一一个】最匹配的。\n"
    "只输出 JSON: {\"bucket\": \"<桶名>\", \"reason\": \"一句话理由\"}，不要markdown。\n\n"
    "## 能力桶（按能力划分，不是话题）\n"
    + "\n".join(f"- {b}: {d}" for b, d in BUCKET_DEFS.items())
    + "\n\n## 规则\n"
    "- 只选一个bucket，必须是上面列表里的英文桶名之一。\n"
    "- 按'完成任务所需的核心本领'选，不要按业务话题。\n"
)


def load_key():
    key = os.environ.get("TOKENHUB_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("TOKENHUB_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        sys.exit("ERROR: TOKENHUB_API_KEY not set")
    return key


def chat(client, model, messages, max_tokens=256, retries=3):
    import httpx
    last = None
    for attempt in range(retries):
        try:
            resp = client.post(f"{API_BASE}/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=120.0)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("empty response")
            return content
        except Exception as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1]
        if text.startswith("json"): text = text[4:]
    text = text.strip()
    s = text.find("{"); e = text.rfind("}")
    if s >= 0 and e > s: text = text[s:e+1]
    return json.loads(text)


def check_sufficient(out_path):
    """Return (sufficient: bool, counts: dict, shortfall: dict)."""
    counts = Counter()
    if out_path.is_file():
        for line in out_path.read_text().splitlines():
            if line.strip():
                b = json.loads(line).get("bucket", "")
                if b in CANONICAL:
                    counts[b] += 1
    shortfall = {b: max(0, TARGET - counts.get(b, 0)) for b in CANONICAL}
    sufficient = all(v == 0 for v in shortfall.values())
    return sufficient, counts, shortfall


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import httpx
    key = load_key()
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"})

    # load all queries
    all_queries = []
    for line in QUERIES_IN.read_text().splitlines():
        if not line.strip(): continue
        all_queries.append(json.loads(line))
    print(f"[pipeline] loaded {len(all_queries)} queries")

    # resume
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if OUT.is_file():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["record_id"])

    todo = [q for q in all_queries if q["record_id"] not in done_ids]
    print(f"[pipeline] {len(done_ids)} already done, {len(todo)} remaining")

    total_ok = len(done_ids)
    batch_num = 0

    while todo:
        batch = todo[:args.batch]
        todo = todo[args.batch:]
        batch_num += 1
        print(f"\n===== Batch {batch_num}: {len(batch)} queries =====")

        lock = threading.Lock()
        fout = OUT.open("a")
        ok, fail = 0, 0

        def label_one(q):
            user = f"query: {q['seed_query'][:2000]}\navailable_tools: {q.get('tools', [])[:20]}\n\n这个任务主要依赖哪个能力桶？只回 JSON。"
            rid = q["record_id"]
            try:
                raw = chat(client, args.model, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}])
                bucket = parse_json(raw).get("bucket", "")
            except Exception as exc:
                with lock:
                    nonlocal fail; fail += 1
                return
            if bucket not in CANONICAL:
                try:
                    raw = chat(client, args.model, [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user + f"\n\n上次未给出有效桶。必须从 {CANONICAL} 里选一个。只回 JSON。"}])
                    bucket = parse_json(raw).get("bucket", "")
                except Exception:
                    bucket = "unknown"
            if bucket not in CANONICAL:
                bucket = "unknown"
            rec = {"record_id": rid, "bucket": bucket, "seed_query": q["seed_query"], "source_file": q.get("source_file", "")}
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                nonlocal ok; ok += 1

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(label_one, batch))

        fout.close()
        total_ok += ok
        print(f"  ok={ok} fail={fail}")

        # check if sufficient
        sufficient, counts, shortfall = check_sufficient(OUT)
        print(f"\n  桶分布 (累计 {total_ok}):")
        for b in CANONICAL:
            c = counts.get(b, 0)
            sf = shortfall[b]
            print(f"    {b:15s}: {c:6d}  {'✓' if sf==0 else '需 '+str(sf)}")
        print(f"    {'TOTAL':15s}: {sum(counts.values()):6d}")

        if sufficient:
            print(f"\n✅ 全部 9 桶达到 {args.target} 条目标，停止打标。")
            break

    if not sufficient:
        print(f"\n⚠️ 已处理全部 {len(all_queries)} 条查询，以下桶未达标:")
        for b in CANONICAL:
            c = counts.get(b, 0)
            if c < args.target:
                print(f"  {b}: {c}/{args.target} (差 {args.target - c})")

    print(f"\n[pipeline] done. 总计: {total_ok} labeled → {OUT}")


if __name__ == "__main__":
    main()
