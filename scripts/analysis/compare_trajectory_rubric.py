#!/usr/bin/env python3
"""对比旧 trajectory 打分 vs 新五维度 trajectory rubric：拿已落盘的 rollout 重新打分。

动机：trajectory 实测 mean=0.34 过低（judge 对 9B 过于苛刻）。2026-08-14 把 trajectory
从 REWARD_RUBRIC 拆成【独立 judge 调用】，用锚点制五维度 rubric（Tool/Efficiency/
Planning/Consistency/Recovery），加权聚合：
    trajectory = 0.20×Tool + 0.20×Efficiency + 0.25×Planning + 0.25×Consistency + 0.10×Recovery

本脚本抽 N 条 rollout，用【新五维度 rubric】单独 judge 打分，对比 rollout_status 里
已记录的【旧 trajectory 分数】，看新 rubric 抬了多少、是否更合理。

⚠️ judge = gpt-5.6-luna（走 https://tokenhub.sensetime.com/v1，TOKENHUB_API_KEY），本机配好
TOKENHUB_API_KEY 即可跑，不需要集群。旧 trajectory 分数直接读 rollout_status，不重打旧 rubric。

用法（集群）：
  python scripts/analysis/compare_trajectory_rubric.py --n 30 --exp qwen35_9b_b1_16gpu
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # agents/, datasources/
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # trainer/ 等 6 包


def _messages_to_trajectory(messages: list) -> str:
    """把 rollout 的 messages(list of dict) 序列化成 judge 的 trajectory 文本。"""
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str):
            content = content.strip()
        elif content is None:
            # tool_calls 消息：序列化 tool_calls
            tcs = m.get("tool_calls")
            if tcs:
                content = json.dumps(tcs, ensure_ascii=False)
            else:
                content = ""
        else:
            content = str(content)
        if content:
            lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--exp", type=str, default="qwen35_9b_b1_16gpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    # 1. record_id -> query 原文
    t = pq.read_table(ROOT / "datasets/train.parquet").to_pylist()
    query_map = {r["extra_info"].get("record_id", ""): r["extra_info"].get("queries", [""])[0] for r in t}

    # 2. 抽 rollout（含旧 trajectory 分数）
    rdir = ROOT / "rollouts/training" / args.exp
    records = []  # (task_id, query, messages, old_trajectory, old_reward)
    for f in sorted(rdir.glob("rollout_status-*.jsonl")):
        for line in open(f):
            q = json.loads(line.strip())
            tid = q["task_id"]
            query = query_map.get(tid, "")
            for r in q["rollouts"]:
                records.append((tid, query, r.get("messages", []), r.get("trajectory"), r.get("reward")))

    random.seed(args.seed)
    sample = random.sample(records, min(args.n, len(records)))
    print(f"抽 {len(sample)} 条 rollout（exp={args.exp}）\n")

    # 3. 用新五维度 rubric 重新打分
    from agents.prompts import TRAJECTORY_RUBRIC
    from trainer.model_reward import _TRAJECTORY_SYSTEM, aggregate_trajectory, get_judge

    judge = get_judge()
    print(f"judge: {type(judge).__name__}\n")

    results = []
    for i, (tid, query, messages, old_traj, old_reward) in enumerate(sample, 1):
        trajectory = _messages_to_trajectory(messages)
        verdict = judge.score(
            task=query or "(no query)",
            trajectory=trajectory,
            rubric=TRAJECTORY_RUBRIC,
            data_source="compare",
            system=_TRAJECTORY_SYSTEM,
        )
        new_traj = aggregate_trajectory(verdict)
        results.append((tid, old_traj, new_traj, old_reward, verdict))

    # 4. 输出对比
    print(f'{"#":>3} {"task_id":>12} {"旧traj":>8} {"新traj":>8} {"Δ":>7} {"旧reward":>9}')
    print("-" * 55)
    for i, (tid, ot, nt, rw, verdict) in enumerate(results, 1):
        otv = f"{ot:.3f}" if ot is not None else "?"
        ntv = f"{nt:.3f}" if nt is not None else "?"
        delta = f"{nt-ot:+.3f}" if (ot is not None and nt is not None) else "?"
        print(f"{i:>3} {tid[:12]:>12} {otv:>8} {ntv:>8} {delta:>7} {rw if rw is not None else '?':>9.3f}")

    # 汇总
    old_vals = [r[1] for r in results if r[1] is not None]
    new_vals = [r[2] for r in results if r[2] is not None]
    if old_vals and new_vals:
        import statistics
        print("-" * 55)
        print(f"旧 trajectory mean: {statistics.mean(old_vals):.3f}")
        print(f"新 trajectory mean: {statistics.mean(new_vals):.3f}")
        print(f"平均提升: {statistics.mean(new_vals)-statistics.mean(old_vals):+.3f}")

    # 可选落盘
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for tid, ot, nt, rw, verdict in results:
                f.write(json.dumps({"task_id": tid, "old_traj": ot, "new_traj": nt, "old_reward": rw, "verdict": verdict}, ensure_ascii=False) + "\n")
        print(f"\n落盘 → {out_path}")


if __name__ == "__main__":
    main()
