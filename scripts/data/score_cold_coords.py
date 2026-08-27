#!/usr/bin/env python3
"""用冷启动数据(cold_start_1429.jsonl)逐条打 7 维能力分,按桶算术平均得新坐标。

复用 coords_pipeline.py 的 COORDS_PROMPT(7 维),grok-4.5 逐条打分。
数据全量(1421 条,含完整轨迹 messages),最终坐标 = 桶内算术平均。

输出: configs/bucket_coords.json

用法: python3 scripts/data/score_cold_coords.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
COLD = ROOT / "datasets" / "cold_start" / "cold_start_1429.jsonl"
OUT = ROOT / "configs" / "bucket_coords.json"
API_BASE = "https://tokenhub.sensetime.com/v1"
MODEL = "grok-4.5"
WORKERS = 64

DIMS = ["knowledge", "reasoning", "planning", "tool_use", "environment_action", "generation", "interaction"]

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


def traj_to_text(msgs) -> str:
    """轨迹 messages → 文本(复用 coords_pipeline.py 的格式)。

    2026-08-27: 补上 assistant 的 content 文本(思考/规划/最终答案);并【去掉所有截断】
    ——全轨迹发给模型(几十 K 也发),此前逐段截断(user1500/assistant1200/tool_call500/
    tool_result300)+整体16000 会丢大量执行细节,导致维度信号不全。system 仍不纳入
    (通用 harness 引导语,不承载任务能力信息、稀释信号)。
    """
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return ""
    if not isinstance(msgs, list):
        return ""
    lines = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            lines.append(f"[user] {m.get('content','') or ''}")
        elif m.get("role") == "assistant":
            content = m.get("content") or ""
            if content.strip():
                lines.append(f"[assistant] {content}")
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                lines.append(f"[tool_call:{fn.get('name','?')}] {str(fn.get('arguments',''))}")
        elif m.get("role") == "tool":
            lines.append(f"[tool_result:{m.get('name','?')}] {m.get('content','') or ''}")
    return "\n".join(lines)


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
    key = load_key()

    # 加载冷启动数据
    items = []  # (bucket, traj_text)
    with open(COLD, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            b = d.get("bucket") or (d.get("metadata") or {}).get("bucket") or ""
            traj = traj_to_text(d.get("messages", []))
            if b and traj:
                items.append((b, traj))
    print(f"加载 {len(items)} 条冷启动轨迹", flush=True)
    if not items:
        sys.exit("ERROR: 无数据")

    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(300.0))
    per_bucket = defaultdict(lambda: defaultdict(list))
    ok = fail = 0
    t0 = time.time()

    def score_one(item):
        nonlocal ok, fail
        b, traj = item
        user = f"任务轨迹:\n\n{traj}\n\n请评估七维分数，只回 JSON。"
        for attempt in range(3):
            try:
                resp = client.post(
                    f"{API_BASE}/chat/completions",
                    json={"model": MODEL, "messages": [{"role": "system", "content": COORDS_PROMPT}, {"role": "user", "content": user}], "max_tokens": 256, "temperature": 0.0},
                    timeout=300.0,
                )
                resp.raise_for_status()
                scores = parse_json(resp.json()["choices"][0]["message"]["content"])
                return (b, {d: max(1, min(10, int(float(scores.get(d, 5))))) for d in DIMS})
            except Exception:
                time.sleep(1.5)
        return (b, None)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for b, result in ex.map(score_one, items):
            if result:
                ok += 1
                for d in DIMS:
                    per_bucket[b][d].append(result[d])
            else:
                fail += 1
            n = ok + fail
            if n % 500 == 0 or n == len(items):
                el = time.time() - t0
                print(f"  {n}/{len(items)} ok={ok} fail={fail} rate={n/el:.1f}/s", flush=True)

    # 按桶算术平均
    import statistics as st
    coords = {}
    print(f"\n{'bucket':15s} {'n':>5} {'know':>5} {'reas':>5} {'plan':>5} {'tool':>5} {'env':>5} {'gen':>5} {'inter':>5}")
    for b in sorted(per_bucket):
        means = {d: round(st.mean(per_bucket[b][d]), 2) for d in DIMS}
        n = len(per_bucket[b][DIMS[0]])
        print(f"{b:15s} {n:5d} {means['knowledge']:5.2f} {means['reasoning']:5.2f} {means['planning']:5.2f} {means['tool_use']:5.2f} {means['environment_action']:5.2f} {means['generation']:5.2f} {means['interaction']:5.2f}")
        coords[b] = [means[d] for d in DIMS]

    OUT.write_text(json.dumps({"dimensions": DIMS, "model": MODEL, "coordinates": coords}, ensure_ascii=False, indent=2))
    print(f"\n✅ 保存 → {OUT} ({len(coords)} 桶)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
