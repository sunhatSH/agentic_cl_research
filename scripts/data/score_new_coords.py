#!/usr/bin/env python3
"""用 grok 给新数据(coding/research/office 各 100 条中等难度)打 7 维能力坐标。

复用 coords_pipeline.py 的 COORDS_PROMPT(7 维),看新数据下 office/coding/research 坐标 + 两两距离。

数据源: all_tasks_labeled.jsonl(已 merge 桶+难度) + all_tasks_metadata.jsonl(user_prompt)

用法:
  python3 scripts/data/score_new_coords.py            # grok-4.5, 每桶 100 条
  python3 scripts/data/score_new_coords.py --limit 10 # 冒烟
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "datasources" / "labeled" / "all_tasks_labeled.jsonl"
META = ROOT / "datasources" / "labeled" / "all_tasks_metadata.jsonl"
API_BASE = "https://tokenhub.sensetime.com/v1"
MODEL = "grok-4.5"
WORKERS = 32
PER_BUCKET = 100

DIMS = ["knowledge", "reasoning", "planning", "tool_use", "environment_action", "generation", "interaction"]

COORDS_PROMPT = """你是任务能力标注器。给定一个 agent 任务的完整描述，请评估完成该任务所需的七种能力维度，每维给出 1-10 的整数分数。
重要：给任务本身打分，不要从领域标签推断。独立评估每个维度。使用完整的 1-10 范围。
1 分=基本不需要。5-6 分=中等。8+分=强烈要求。9-10 分=明显是成功必要条件。

1. knowledge: 需要多少外部/专业/领域知识？1-2:纯机械 3-4:常识足够 5-6:大量领域知识 7-8:专业知识核心 9-10:专家级必需
2. reasoning: 需要多少逻辑推理、分析、计算？1-2:直接检索 3-4:简单推理 5-6:大量推理 7-8:复杂推理核心 9-10:高度复杂必需
3. planning: 需要多少多步规划？1-2:一步完成 3-4:几步 5-6:需依赖步骤规划 7-8:复杂分解 9-10:大量规划/依赖管理
4. tool_use: 需要多少工具调用(tool_call)？1-2:无/1-2次 3-4:3-5次 5-6:6-15次 7-8:16-30次/工具链 9-10:30+次/多种工具
5. environment_action: 需要多少文件系统/OS/沙箱操作？1-2:纯信息型 3-4:少量操作 5-6:有意义的系统更改 7-8:大量环境操作核心 9-10:根本上需要重大环境更改
6. generation: 需要多少内容生成？1-2:简单检索 3-4:简短输出 5-6:大量生成 7-8:复杂/长篇核心 9-10:大量高约束制品
7. interaction: 需要多少用户交互/上下文维护？1-2:自包含 3-4:少量上下文 5-6:有意义交互 7-8:多轮核心 9-10:持续适应性交互

只输出 JSON: {"knowledge":<1-10>,"reasoning":<1-10>,"planning":<1-10>,"tool_use":<1-10>,"environment_action":<1-10>,"generation":<1-10>,"interaction":<1-10>}
不要 markdown。"""


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
        sys.exit("ERROR: TOKENHUB_API_KEY not set")
    return key


def parse_json(text):
    text = text.strip()
    for c in ["```json", "```"]:
        text = text.replace(c, "")
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        text = text[s : e + 1]
    return json.loads(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每桶只抽 N 条(冒烟)")
    args = ap.parse_args()

    key = load_key()

    # 1. 加载 user_prompt + 桶
    meta = {}
    with open(META, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meta[d["D_id"]] = d.get("user_prompt", "")
    bk_dv = {}
    with open(LABELED, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            bk_dv[d["D_id"]] = (d.get("bucket"), d.get("difficulty"))

    # 2. 每桶抽 PER_BUCKET 条 d4-7
    random.seed(42)
    samples = {}
    for bk in ("coding", "research", "office"):
        pool = [gid for gid, (b, dv) in bk_dv.items() if b == bk and dv is not None and 4 <= dv <= 7 and gid in meta]
        random.shuffle(pool)
        n = args.limit if args.limit else PER_BUCKET
        samples[bk] = [(gid, meta[gid]) for gid in pool[:n]]
        print(f"  {bk}: 抽 {len(samples[bk])} 条 (d4-7 池 {len(pool)})", flush=True)

    # 3. grok 打 7 维坐标
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(120.0))
    per_bucket = defaultdict(lambda: defaultdict(list))

    def score_one(item):
        bk, (gid, up) = item
        up = up[:4000]
        user = f"任务描述:\n\n{up}\n\n请评估七维分数，只回 JSON。"
        for attempt in range(3):
            try:
                resp = client.post(
                    f"{API_BASE}/chat/completions",
                    json={"model": MODEL, "messages": [{"role": "system", "content": COORDS_PROMPT}, {"role": "user", "content": user}], "max_tokens": 256, "temperature": 0.0},
                    timeout=120.0,
                )
                resp.raise_for_status()
                scores = parse_json(resp.json()["choices"][0]["message"]["content"])
                return (bk, {d: max(1, min(10, int(float(scores.get(d, 5))))) for d in DIMS})
            except Exception:
                import time
                time.sleep(2)
        return (bk, None)

    flattened = [(bk, item) for bk in samples for item in samples[bk]]
    print(f"\n打坐标: {len(flattened)} 条, model={MODEL}, workers={WORKERS}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for bk, result in ex.map(score_one, flattened):
            if result:
                for d in DIMS:
                    per_bucket[bk][d].append(result[d])

    # 4. 平均坐标 + 两两距离
    coords = {}
    print(f"\n{'bucket':12s} {'n':>5} {'know':>5} {'reas':>5} {'plan':>5} {'tool':>5} {'env':>5} {'gen':>5} {'inter':>5}")
    for bk in sorted(per_bucket):
        means = {d: round(st.mean(per_bucket[bk][d]), 1) for d in DIMS}
        n = len(per_bucket[bk][DIMS[0]])
        print(f"{bk:12s} {n:5d} {means['knowledge']:5.1f} {means['reasoning']:5.1f} {means['planning']:5.1f} {means['tool_use']:5.1f} {means['environment_action']:5.1f} {means['generation']:5.1f} {means['interaction']:5.1f}")
        coords[bk] = [means[d] for d in DIMS]

    print("\n=== 两两距离 ===")
    import itertools
    for a, b in itertools.combinations(["coding", "research", "office"], 2):
        if a in coords and b in coords:
            dist = math.sqrt(sum((coords[a][i] - coords[b][i]) ** 2 for i in range(len(DIMS))))
            print(f"  {a} vs {b}: {dist:.4f}")

    # 输出
    out = {"dimensions": DIMS, "model": MODEL, "coordinates": coords}
    fp = ROOT / "configs" / "bucket_coords_new.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nsaved → {fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
