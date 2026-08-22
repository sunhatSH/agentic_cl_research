#!/usr/bin/env python3
"""Step1:打难度分(落盘) → Step2:筛中等(4-7) → Step3:三模型七维坐标。"""
import json, os, random, time, threading, statistics as st
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
WORKERS = 20
DIMS = ["knowledge","reasoning","planning","tool_use","environment_action","generation","interaction"]
DIFF_CACHE = ROOT / "datasources" / "labeled" / "difficulty_500.json"
OUT_DIR = ROOT / "configs"

DIFF_PROMPT = '评估该任务的绝对难度，给出 1-10 整数分数。1-2:非常简单 3-4:简单 5-6:中等 7-8:困难 9-10:非常困难。只输出 JSON: {"difficulty": <1-10>}'

COORDS_PROMPT = """你是任务能力标注器。给定一个 agent 任务的完整执行轨迹（messages，含工具调用和结果），请评估完成该任务所需的七种能力维度，每维给出 1-10 的整数分数。
重要：给任务本身打分，不要从领域标签推断。独立评估每个维度。使用完整的 1-10 范围。
1 分=基本不需要。5-6 分=中等。8+分=强烈要求。9-10 分=明显是成功必要条件。

1. knowledge: 需要多少外部/专业/领域知识？1-2:纯机械 3-4:常识足够 5-6:大量领域知识 7-8:专业知识核心 9-10:专家级必需
2. reasoning: 需要多少逻辑推理、分析、计算？1-2:直接检索 3-4:简单推理 5-6:大量推理 7-8:复杂推理核心 9-10:高度复杂必需
3. planning: 需要多少多步规划？1-2:一步完成 3-4:几步 5-6:需依赖步骤规划 7-8:复杂分解 9-10:大量规划/依赖管理
4. tool_use: 轨迹中实际工具调用(tool_call)频率和种类。1-2:无/1-2次 3-4:3-5次/1-2种 5-6:6-15次/3-4种 7-8:16-30次/工具链 9-10:30+次/多种工具
5. environment_action: 需要多少文件系统/OS/沙箱操作？1-2:纯信息型 3-4:少量操作 5-6:有意义的系统更改 7-8:大量环境操作核心 9-10:根本上需要重大环境更改
6. generation: 需要多少内容生成？1-2:简单检索 3-4:简短输出 5-6:大量生成 7-8:复杂/长篇核心 9-10:大量高约束制品
7. interaction: 需要多少用户交互/上下文维护？1-2:自包含 3-4:少量上下文 5-6:有意义交互 7-8:多轮核心 9-10:持续适应性交互

只输出 JSON: {\"knowledge\":<1-10>,\"reasoning\":<1-10>,\"planning\":<1-10>,\"tool_use\":<1-10>,\"environment_action\":<1-10>,\"generation\":<1-10>,\"interaction\":<1-10>}
不要 markdown。"""


def load_key():
    env = ROOT / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("TOKENHUB_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("KEY not found")


def load_trajectories():
    by_b = defaultdict(list)
    for name, path in [("cold", "datasets/cold_start/cold_start_1429.jsonl"),
                        ("old", "datasources/labeled/taskspecs_labeled.jsonl"),
                        ("new", "datasources/labeled/new_trajectories_labeled.jsonl")]:
        fp = ROOT / path
        if not fp.is_file(): continue
        for line in fp.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            b = (r.get("metadata") or {}).get("bucket", "") or r.get("bucket", "")
            if not b: continue
            if name == "cold":
                msgs = r.get("messages", [])
                if isinstance(msgs, str):
                    try: msgs = json.loads(msgs)
                    except: continue
                lines = []
                for m in msgs:
                    if m.get("role") == "user": lines.append(f"[user] {(m.get('content','') or '')[:1500]}")
                    elif m.get("role") == "assistant":
                        for tc in (m.get("tool_calls") or []):
                            fn = tc.get("function", {})
                            lines.append(f"[tool_call:{fn.get('name','?')}] {str(fn.get('arguments',''))[:500]}")
                    elif m.get("role") == "tool":
                        lines.append(f"[tool_result:{m.get('name','?')}] {(m.get('content','') or '')[:300]}")
                by_b[b].append("\n".join(lines))
            elif name == "old":
                qs = r.get("queries", [])
                by_b[b].append(f"Task: {qs[0][:2000] if qs else ''}")
            else:
                q = r.get("seed_query", "")
                if q: by_b[b].append(f"Query: {q[:2000]}")
    return by_b


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
    for c in ["```json","```"]: text = text.replace(c, "")
    s = text.find("{"); e = text.rfind("}")
    if s >= 0 and e > s: text = text[s:e+1]
    return json.loads(text)


def main():
    import httpx
    key = load_key()
    random.seed(42)
    data = load_trajectories()
    print(f"[load] {len(data)} buckets")

    # ===== Step 1: Difficulty scoring (cached) =====
    if DIFF_CACHE.is_file():
        print(f"\n[cache] loading difficulty from {DIFF_CACHE}")
        diff_cache = json.loads(DIFF_CACHE.read_text())
    else:
        print("\n===== Step 1: 打难度分(每桶500条,落盘到 difficulty_500.json) =====")
        client = httpx.Client(headers={"Authorization": f"Bearer {key}"})
        diff_cache = {}
        for b in sorted(data):
            pool = data[b][:500]
            items = [(b, t) for t in pool]

            def sd(item):
                b, t = item
                try:
                    msg = [{"role": "system", "content": DIFF_PROMPT}, {"role": "user", "content": f"轨迹:\n\n{t[:4000]}"}]
                    raw = chat(client, "claude-opus-5", msg)
                    return (b, int(float(parse_json(raw).get("difficulty", 5))))
                except:
                    return (b, -1)

            from tqdm import tqdm
            diffs = []
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for b, dif in tqdm(ex.map(sd, items), total=len(items), desc=f"  {b}"):
                    if dif >= 1: diffs.append(dif)
            diff_cache[b] = diffs
        DIFF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DIFF_CACHE.write_text(json.dumps(diff_cache))
        print(f"[cache] saved → {DIFF_CACHE}")

    # ===== Step 2: Filter medium (4-7) =====
    print("\n===== Step 2: 筛中等难度(4-7) =====")
    medium_trajs = {}
    for b in sorted(data):
        pool = data[b][:500]
        diffs = diff_cache.get(b, [])
        med_indices = [i for i, d in enumerate(diffs) if 4 <= d <= 7]
        med_trajs = [pool[i] for i in med_indices[:100]]
        medium_trajs[b] = med_trajs
        avg_d = st.mean(diffs) if diffs else 0
        print(f"  {b:15s}: {len(pool)} trajs, {len(diffs)} diff scores, {len(med_trajs)} medium (avg diff={avg_d:.1f})")

    # ===== Step 3: Coordinate scoring (3 models, medium only) =====
    print("\n===== Step 3: 三模型七维坐标(中等难度) =====")
    for model_name in ["claude-opus-5", "gpt-5.6-sol", "kimi-k3"]:
        print(f"\n--- {model_name} ---")
        client = httpx.Client(headers={"Authorization": f"Bearer {key}"})
        flattened = [(b, t) for b in sorted(medium_trajs) for t in medium_trajs[b]]

        lock = threading.Lock()
        per_bucket = defaultdict(lambda: defaultdict(list))
        ok, fail = 0, 0

        def so(item):
            b, traj = item
            try:
                user = f"任务轨迹:\n\n{traj[:16000]}\n\n请评估七维分数，只回 JSON。"
                msg = [{"role": "system", "content": COORDS_PROMPT}, {"role": "user", "content": user}]
                raw = chat(client, model_name, msg)
                scores = parse_json(raw)
                return (b, {d: max(1, min(10, int(float(scores.get(d, 5))))) for d in DIMS}, None)
            except:
                return (b, None, "err")

        from tqdm import tqdm
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for b, result, err in tqdm(ex.map(so, flattened), total=len(flattened), desc=f"  {model_name}"):
                with lock:
                    if result is not None:
                        ok += 1
                        for d in DIMS: per_bucket[b][d].append(result[d])
                    else:
                        fail += 1

        print(f"  ok={ok} fail={fail}")
        print(f"\n  {'bucket':15s} {'n':>5} {'know':>5} {'reas':>5} {'plan':>5} {'tool':>5} {'env':>5} {'gen':>5} {'inter':>5}")
        coords_out = {}
        for b in sorted(per_bucket):
            means = {d: round(st.mean(per_bucket[b][d]), 1) for d in DIMS}
            n = len(per_bucket[b][DIMS[0]])
            print(f"  {b:15s} {n:5d} {means['knowledge']:5.1f} {means['reasoning']:5.1f} {means['planning']:5.1f} {means['tool_use']:5.1f} {means['environment_action']:5.1f} {means['generation']:5.1f} {means['interaction']:5.1f}")
            coords_out[b] = [means[d] for d in DIMS]

        fp = OUT_DIR / f"bucket_coords_{model_name.replace('.','_')}.json"
        json.dump({"dimensions": DIMS, "model": model_name, "coordinates": coords_out}, fp.open("w"), ensure_ascii=False, indent=2)
        print(f"  saved → {fp}")

    print("\n[done]")


if __name__ == "__main__":
    main()
