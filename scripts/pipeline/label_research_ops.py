#!/usr/bin/env python3
"""关键词预筛选 → tokenhub 打标 循环。
先关键词过滤出疑似 research/ops 的候选，再送 tokenhub 分类（prompt 不带关键词提示）。
每 5000 条一轮，报告分布，直到 research 和 ops 都达标。
"""
import json, os, sys, time, threading, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_IN = ROOT / "datasets" / "queries_new.jsonl"
OUT = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
API_BASE = "https://tokenhub.sensetime.com/v1"
BATCH = 5000
WORKERS = 100

# 目标：ops=4000 (125步), research=4000 (125步)
TARGETS = {"ops": 4000, "research": 4000}

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

# ── 关键词预筛选（不进入 prompt，只用于选候选 query）─────────────────
RESEARCH_KEYWORDS = [
    r'\breport\b', r'\bsummary\b', r'\bsummarize\b', r'\bsynthesize\b',
    r'\bresearch\b', r'\banalysis\b', r'\banalyze\b', r'\bfindings?\b',
    r'\bconclusions?\b', r'\boverview\b', r'\bbrief(ing)?\b',
    r'\bmulti(ple)?[ -]?sources?\b', r'\bcross[ -]?reference\b',
    r'\bcompile\b', r'\bconsolidate\b', r'\baggregate\b',
    r'\bgather\b.*\binformation\b', r'\bcollect\b.*\bdata\b',
    r'\bcomprehensive\b', r'\bin[ -]?depth\b',
    r'\breview\b.*\bliterature\b', r'\bwhite[ -]?paper\b',
    r'\bdashboard\b', r'\binsights?\b', r'\btrends?\b',
    r'\bwrite\b.*\breport\b', r'\bcreate\b.*\breport\b',
    r'\bgenerate\b.*\breport\b',
]
OPS_KEYWORDS = [
    r'\bterminal\b', r'\bcommand\b', r'\bbash\b', r'\bshell\b',
    r'\bexecute\b.*\bscript\b', r'\brun\b.*\bcommand\b',
    r'\binstall\b', r'\bdeploy\b', r'\bconfigure\b', r'\bsetup\b',
    r'\bdocker\b', r'\bkubectl\b', r'\bpackage\b', r'\bdependency\b',
    r'\bfile\b.*\bsystem\b', r'\bchmod\b', r'\bchown\b', r'\bsudo\b',
    r'\bprocess\b', r'\bdaemon\b', r'\bservice\b', r'\bsystemd\b',
    r'\bcron\b', r'\bport\b', r'\bfirewall\b', r'\bnetwork\b.*\bconfig\b',
    r'\blog\b.*\brotate\b', r'\bmonitor(ing)?\b', r'\balert\b',
    r'\bserver\b', r'\bmigration\b', r'\bbackup\b', r'\brestore\b',
    r'\bdisk\b', r'\bmount\b', r'\bswap\b',
]

def keyword_score(query_text):
    """Return (research_score, ops_score) based on keyword matches."""
    text = query_text.lower()
    rs = sum(1 for kw in RESEARCH_KEYWORDS if re.search(kw, text, re.IGNORECASE))
    os_ = sum(1 for kw in OPS_KEYWORDS if re.search(kw, text, re.IGNORECASE))
    return rs, os_

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

def chat(client, model, messages, max_tokens=256, retries=2):
    import httpx
    for attempt in range(retries):
        try:
            resp = client.post(f"{API_BASE}/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=60.0)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("empty response")
            return content
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None

def parse_json(text):
    if text is None: return {}
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"): text = text[4:]
    text = text.strip()
    s = text.find("{"); e = text.rfind("}")
    if s >= 0 and e > s: text = text[s:e+1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}

def current_counts():
    c = Counter()
    if OUT.is_file():
        for line in OUT.read_text().splitlines():
            if line.strip():
                b = json.loads(line).get("bucket", "")
                if b in CANONICAL:
                    c[b] += 1
    return c

def main():
    import argparse, httpx
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--target-ops", type=int, default=TARGETS["ops"])
    ap.add_argument("--target-research", type=int, default=TARGETS["research"])
    args = ap.parse_args()

    key = load_key()
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(60.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # load done ids
    done_ids = set()
    if OUT.is_file():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["record_id"])

    # load all queries, filter unlabeled
    all_queries = []
    for line in QUERIES_IN.read_text().splitlines():
        if not line.strip(): continue
        q = json.loads(line)
        if q["record_id"] not in done_ids:
            all_queries.append(q)
    print(f"[pipeline] {len(all_queries)} unlabeled queries remaining", flush=True)

    counts = current_counts()
    round_num = 0

    while True:
        op_n = counts.get("ops", 0)
        rs_n = counts.get("research", 0)
        if op_n >= args.target_ops and rs_n >= args.target_research:
            print(f"\n✅ 达标! ops={op_n} research={rs_n}", flush=True)
            break
        if not all_queries:
            print(f"\n⚠️ 无更多查询。ops={op_n}/{args.target_ops} research={rs_n}/{args.target_research}", flush=True)
            break

        round_num += 1
        # ── 关键词预筛选 ──
        scored = []
        for q in all_queries:
            rs, os_ = keyword_score(q["seed_query"])
            # Only keep queries with research or ops keyword hits
            if rs > 0 or os_ > 0:
                # Prioritize research (higher weight) since it's the bottleneck
                priority = rs * 3 + os_
                scored.append((priority, q))

        # Sort by priority (highest first), take top batch
        scored.sort(key=lambda x: -x[0])
        batch = [q for _, q in scored[:args.batch]]
        # Remove selected queries from pool
        selected_ids = {q["record_id"] for q in batch}
        all_queries = [q for q in all_queries if q["record_id"] not in selected_ids]

        print(f"\n===== Round {round_num}: {len(batch)} keyword-filtered queries =====", flush=True)
        if not batch:
            print("⚠️ No keyword-matched queries remaining. Falling back to unfiltered pool.", flush=True)
            batch = all_queries[:args.batch]
            all_queries = all_queries[args.batch:]

        # ── tokenhub 打标 ──
        lock = threading.Lock()
        fout = OUT.open("a")
        ok, fail = 0, 0
        new_ops, new_research = 0, 0

        def label_one(q):
            nonlocal ok, fail, new_ops, new_research
            user = f"query: {q['seed_query'][:2000]}\navailable_tools: {q.get('tools', [])[:20]}\n\n这个任务主要依赖哪个能力桶？只回 JSON。"
            rid = q["record_id"]
            bucket = "unknown"
            try:
                raw = chat(client, args.model, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user}])
                bucket = parse_json(raw).get("bucket", "unknown")
                if bucket not in CANONICAL:
                    raw = chat(client, args.model, [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user + f"\n\n上次未给出有效桶。必须从 {CANONICAL} 里选一个。只回 JSON。"}])
                    bucket = parse_json(raw).get("bucket", "unknown")
            except Exception:
                pass
            if bucket not in CANONICAL:
                bucket = "unknown"
            rec = {"record_id": rid, "bucket": bucket, "seed_query": q["seed_query"],
                   "source_file": q.get("source_file", "")}
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
                if bucket == "ops": new_ops += 1
                elif bucket == "research": new_research += 1

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(label_one, batch))

        fout.close()
        counts = current_counts()
        print(f"  ok={ok}  ops_+{new_ops}  research_+{new_research}", flush=True)
        print(f"  ops={counts.get('ops',0)} research={counts.get('research',0)}  "
              f"remaining unlabeled: {len(all_queries)}", flush=True)

    # Final report
    counts = current_counts()
    print(f"\n=== 最终分布 ===", flush=True)
    for b in CANONICAL:
        c = counts.get(b, 0)
        print(f"  {b:15s}: {c:6d} ({c//32} steps)", flush=True)
    print(f"  {'TOTAL':15s}: {sum(counts.values()):6d}", flush=True)

if __name__ == "__main__":
    main()
