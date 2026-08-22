#!/usr/bin/env python3
"""关键词预筛选 → tokenhub 打标 — finance + safety。
"""
import json, os, sys, time, threading, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
QUERIES_IN = ROOT / "datasets" / "queries_new.jsonl"
OUT = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
API_BASE = "https://tokenhub.sensetime.com/v1"
BATCH = 5000
WORKERS = 100
TARGET = 1500

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

FINANCE_KW = [
    r'\bloan\b', r'\btax\b', r'\baudit\b', r'\binvoice\b', r'\bvaluation\b',
    r'\bprocurement\b', r'\bpurchase[ -]?order\b', r'\baccounting\b',
    r'\bfinancial\b', r'\bbudget\b', r'\brevenue\b', r'\bexpense\b',
    r'\basset\b', r'\bliability\b', r'\bequity\b', r'\bdepreciation\b',
    r'\bamortization\b', r'\binterest\b.*\brate\b', r'\bcredit\b', r'\bdebit\b',
    r'\bledger\b', r'\bbalance\b.*\bsheet\b', r'\bincome\b.*\bstatement\b',
    r'\bcash\b.*\bflow\b', r'\binvestment\b', r'\bportfolio\b',
    r'\brisk\b.*\bassess', r'\binsurance\b', r'\bpayroll\b', r'\bpayable\b',
    r'\breceivable\b', r'\bfiscal\b', r'\bROI\b', r'\bNPV\b', r'\bEBITDA\b',
    r'\breconcili', r'\baccrual\b', r'\bcapital\b.*\bgain', r'\bdividend\b',
    r'\bmortgage\b', r'\blease\b', r'\bwarrant\b', r'\bderivative\b',
    r'\bexchange\b.*\brate', r'\bforex\b', r'\bcrypto\b',
    r'\bERP\b', r'\bSAP\b', r'\bQuickBooks?\b',
]
SAFETY_KW = [
    r'\bsecurity\b', r'\bvulnerability\b', r'\bexploit\b', r'\battack\b',
    r'\bmalicious\b', r'\bunsafe\b', r'\bdangerous\b', r'\bharmful\b',
    r'\billegal\b', r'\bcompliance\b', r'\baudit\b', r'\bpermission\b',
    r'\baccess\b.*\bcontrol\b', r'\bauthenticat', r'\bauthoriz',
    r'\bencrypt\b', r'\bdecrypt\b', r'\bfirewall\b', r'\bintrusion\b',
    r'\bbreach\b', r'\bthreat\b', r'\brisk\b', r'\bprivacy\b',
    r'\bGDPR\b', r'\bPII\b', r'\bsensitive\b.*\bdata\b', r'\brefuse\b',
    r'\breject\b', r'\bblock\b', r'\bdeny\b', r'\bpenetration\b',
    r'\bphishing\b', r'\bmalware\b', r'\bransomware\b', r'\bCVE\b',
    r'\bOWASP\b', r'\bXSS\b', r'\bSQL\b.*\binjection\b', r'\bCSRF\b',
    r'\bpatch\b', r'\bhardening\b', r'\bpentest\b', r'\bforensic\b',
    r'\bSOC2?\b', r'\bISO\b.*\b27001\b', r'\bHIPAA\b',
]

def keyword_score(text):
    text_l = text.lower()
    fs = sum(1 for kw in FINANCE_KW if re.search(kw, text_l))
    ss = sum(1 for kw in SAFETY_KW if re.search(kw, text_l))
    return fs, ss

def load_key():
    key = os.environ.get("TOKENHUB_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("TOKENHUB_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key: sys.exit("ERROR: TOKENHUB_API_KEY not set")
    return key

def chat(client, model, messages, retries=2):
    import httpx
    for attempt in range(retries):
        try:
            resp = client.post(f"{API_BASE}/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": 256}, timeout=60.0)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content and content.strip(): return content
        except Exception:
            if attempt < retries - 1: time.sleep(1)
    return None

def parse_json(text):
    if not text: return {}
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"): text = text[4:]
    s = text.find("{"); e = text.rfind("}")
    if s >= 0 and e > s: text = text[s:e+1]
    try: return json.loads(text)
    except json.JSONDecodeError: return {}

def counts():
    c = Counter()
    if OUT.is_file():
        for line in OUT.read_text().splitlines():
            if line.strip():
                b = json.loads(line).get("bucket", "")
                if b in CANONICAL: c[b] += 1
    return c

def main():
    import argparse, httpx
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--target", type=int, default=TARGET)
    args = ap.parse_args()
    key = load_key()
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(60.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)

    done_ids = set()
    if OUT.is_file():
        for line in OUT.read_text().splitlines():
            if line.strip(): done_ids.add(json.loads(line)["record_id"])

    all_queries = []
    for line in QUERIES_IN.read_text().splitlines():
        if not line.strip(): continue
        q = json.loads(line)
        if q["record_id"] not in done_ids: all_queries.append(q)
    print(f"[pipeline] {len(all_queries)} unlabeled queries", flush=True)

    c = counts()
    for round_num in range(1, 100):
        fn = c.get("finance", 0); sf = c.get("safety", 0)
        if fn >= args.target and sf >= args.target:
            print(f"\n✅ 达标! finance={fn} safety={sf}", flush=True); break
        if not all_queries:
            print(f"\n⚠️ 无更多查询。finance={fn}/{args.target} safety={sf}/{args.target}", flush=True); break

        # keyword pre-filter
        scored = []
        for q in all_queries:
            fs, ss = keyword_score(q["seed_query"])
            if fs > 0 or ss > 0: scored.append((fs + ss, q))
        scored.sort(key=lambda x: -x[0])
        batch = [q for _, q in scored[:args.batch]]
        sids = {q["record_id"] for q in batch}
        all_queries = [q for q in all_queries if q["record_id"] not in sids]
        if not batch:
            print("⚠️ No keyword-matched queries left.", flush=True); break

        print(f"\n===== Round {round_num}: {len(batch)} keyword-filtered queries =====", flush=True)

        lock = threading.Lock()
        fout = OUT.open("a")
        ok, fin, saf = 0, 0, 0
        def label_one(q):
            nonlocal ok, fin, saf
            user = f"query: {q['seed_query'][:2000]}\navailable_tools: {q.get('tools', [])[:20]}\n\n这个任务主要依赖哪个能力桶？只回 JSON。"
            bucket = "unknown"
            try:
                raw = chat(client, args.model, [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user}])
                bucket = parse_json(raw).get("bucket", "unknown")
                if bucket not in CANONICAL:
                    raw = chat(client, args.model, [{"role":"system","content":SYSTEM_PROMPT},
                        {"role":"user","content":user+f"\n\n上次未给出有效桶。必须从 {CANONICAL} 里选一个。只回 JSON。"}])
                    bucket = parse_json(raw).get("bucket", "unknown")
            except Exception: pass
            if bucket not in CANONICAL: bucket = "unknown"
            rec = {"record_id": q["record_id"], "bucket": bucket, "seed_query": q["seed_query"],
                   "source_file": q.get("source_file", "")}
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
                if bucket == "finance": fin += 1
                elif bucket == "safety": saf += 1
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(label_one, batch))
        fout.close()
        c = counts()
        print(f"  ok={ok}  finance_+{fin}  safety_+{saf}  finance={c.get('finance',0)} safety={c.get('safety',0)}", flush=True)

    c = counts()
    print(f"\n=== 最终 ===", flush=True)
    for b in CANONICAL: print(f"  {b:15s}: {c.get(b,0):6d}", flush=True)

if __name__ == "__main__": main()
