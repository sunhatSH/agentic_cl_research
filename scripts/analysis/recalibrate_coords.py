#!/usr/bin/env python3
"""从每桶各抽 20 条轨迹，让 LLM 打五维分，综合出新的能力空间坐标。

五维: tool_intensity / reasoning_depth / structure_rigidity / knowledge_domain / multi_step
每维 1-10 分，取均值作为该桶坐标。

用法: python scripts/analysis/recalibrate_coords.py [--samples 20] [--out configs/bucket_coords.json]
"""
import json, os, random, sys, time, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Load labeled trajectory data (from old taskspec + new labeled)
def load_all_labeled():
    """Return {bucket: [queries, ...]}"""
    by_bucket = defaultdict(list)
    # old labeled data
    old_file = ROOT / "data" / "labeled" / "taskspecs_labeled.jsonl"
    if old_file.is_file():
        for line in old_file.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            qs = r.get("queries", [])
            b = r.get("bucket", "")
            if b and qs:
                by_bucket[b].append(qs[0][:2000])
    # new labeled data
    new_file = ROOT / "data" / "labeled" / "new_trajectories_labeled.jsonl"
    if new_file.is_file():
        for line in new_file.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            b = r.get("bucket", "")
            q = r.get("seed_query", "")
            if b and q:
                by_bucket[b].append(q[:2000])
    return by_bucket

SYSTEM_PROMPT = """你是任务分析器。给定一个 agent 任务的描述(query)，请评估完成该任务所需的五种能力维度，每维给出 1-10 的整数分数。

1. tool_intensity: 需要多少工具调用？(1=不需要/几乎不需要工具, 10=高度依赖大量工具调用和工具链)
2. reasoning_depth: 需要多深的逻辑推理和思考？(1=简单/表面, 10=需要深度推理、数学推导、多跳逻辑)
3. structure_rigidity: 任务的结构化程度？(1=开放式/创意型, 10=高度结构化/遵循严格规则/格式)
4. knowledge_domain: 需要多少领域专业知识？(1=常识即可, 10=需要深厚专业领域知识)
5. multi_step: 任务是否需要多步骤分解完成？(1=一步即可完成, 10=必须多步分解、分阶段完成)

只输出 JSON: {"tool_intensity": <1-10>, "reasoning_depth": <1-10>, "structure_rigidity": <1-10>, "knowledge_domain": <1-10>, "multi_step": <1-10>}
不要 markdown，不要额外文字。"""

DIMS = ["tool_intensity", "reasoning_depth", "structure_rigidity", "knowledge_domain", "multi_step"]


def load_key():
    key = os.environ.get("TOKENHUB_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        for line in env.read_text().splitlines():
            if line.startswith("TOKENHUB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("\"'")
                break
    if not key: sys.exit("ERROR: TOKENHUB_API_KEY not set")
    return key


def chat(client, model, messages):
    import httpx
    for attempt in range(3):
        try:
            resp = client.post("https://tokenhub.sensetime.com/v1/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": 256}, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2: raise e
            time.sleep(2)


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```", 2); text = parts[1]
        if text.startswith("json"): text = text[4:]
    s = text.find("{"); e = text.rfind("}")
    if s >= 0 and e > s: text = text[s:e+1]
    return json.loads(text)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "configs" / "bucket_coords.json"))
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    data = load_all_labeled()
    print(f"[load] {len(data)} buckets: {dict((k, len(v)) for k, v in data.items())}")

    # sample N per bucket
    random.seed(42)
    samples = {}
    for b, queries in data.items():
        picked = random.sample(queries, min(args.samples, len(queries)))
        samples[b] = picked
        print(f"  {b}: {len(picked)} samples")

    import httpx
    key = load_key()
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"})
    lock = threading.Lock()

    # score each sample
    all_scores = defaultdict(list)  # bucket -> [{dim: score}, ...]
    ok, fail = 0, 0

    def score_one(item):
        bucket, query = item
        user = f"任务描述:\n{query[:3000]}\n\n请评估五维分数，只回 JSON。"
        try:
            raw = chat(client, args.model, [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}])
            scores = parse_json(raw)
            result = {d: float(scores.get(d, 5)) for d in DIMS}
            with lock:
                nonlocal ok; ok += 1
            return (bucket, result)
        except Exception as e:
            with lock:
                nonlocal fail; fail += 1
            return (bucket, None)

    items = [(b, q) for b, qs in samples.items() for q in qs]
    print(f"[scoring] {len(items)} items, model={args.model}, workers={args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for bucket, result in ex.map(score_one, items):
            if result is not None:
                all_scores[bucket].append(result)

    # aggregate: mean per dimension per bucket
    import statistics as st
    new_coords = {}
    print(f"\n[results] ok={ok} fail={fail}")
    print(f"\n{'bucket':15s} {'n':>4}", end="")
    for d in DIMS: print(f" {d[:6]:>7}", end="")
    print()
    for b in sorted(all_scores.keys()):
        scores = all_scores[b]; n = len(scores)
        means = {d: round(st.mean([s[d] for s in scores]), 1) for d in DIMS}
        new_coords[b] = [means[d] for d in DIMS]
        print(f"{b:15s} {n:4d}", end="")
        for d in DIMS: print(f" {means[d]:7.1f}", end="")
        print()

    # write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "dimensions": DIMS,
        "coordinates": new_coords,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
