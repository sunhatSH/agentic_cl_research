#!/usr/bin/env python3
"""用【新五维度 trajectory rubric】给已落盘的 rollout 轨迹打分，看分布是否合理（校准验证）。

数据源：`rollouts/cold_start/grpo_hermes.jsonl`（采集阶段落盘的全量轨迹：messages 含
tool_calls/tool 结果，observer_reports 含 state_diff 环境取证）。训练 rollout_status 的
messages 因 v1 tag 未带 message_history 全空，故用这份采集轨迹验证新 rubric 的校准度。

trajectory rubric 输入三通道：TASK（首条 user 消息）、TRAJECTORY（messages 序列化）、
ENVIRONMENT DIFF（observer_reports 里的 state_diff，consistency 维度判据）。

judge = gpt-5.6-luna（tokenhub），本机配 TOKENHUB_API_KEY 即可跑，不需要集群。
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))          # agents/, datasources/
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # trainer/ 等 6 包


def _first_user_task(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
    return ""


def _serialize_messages(messages: list) -> str:
    """序列化 trajectory（含 tool 名/成功/结果、assistant tool_calls）。"""
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        content = m.get("content")
        tcs = m.get("tool_calls")
        if tcs:
            lines.append(f"[{role}] tool_calls: {json.dumps(tcs, ensure_ascii=False)}")
        if role == "tool":
            name = m.get("name", "")
            ok = m.get("success", "")
            prefix = f"[tool:{name}]" if name else "[tool]"
            if ok != "":
                prefix += f" (success={ok})"
            lines.append(f"{prefix} {str(content).strip()}")
            continue
        if isinstance(content, str):
            content = content.strip()
        else:
            content = str(content) if content is not None else ""
        if content:
            lines.append(f"[{role}] {content}")
    return "\n\n".join(lines)


def _state_diff(observer_reports: list) -> str:
    """聚合各 turn 的 state_diff 为一段环境取证（consistency 判据）。"""
    if not isinstance(observer_reports, list):
        return ""
    parts = []
    for r in observer_reports:
        if isinstance(r, dict) and r.get("state_diff"):
            parts.append(f"[turn {r.get('turn', '?')}]\n{str(r['state_diff']).strip()}")
    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--src", type=str, default="rollouts/cold_start/grpo_hermes.jsonl")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    with open(ROOT / args.src, encoding="utf-8", errors="replace") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    print(f"抽 {len(sample)} 条轨迹（{args.src} 共 {len(rows)} 条）\n")

    from agents.prompts import TRAJECTORY_RUBRIC
    from trainer.model_reward import _TRAJECTORY_SYSTEM, aggregate_trajectory, get_judge

    judge = get_judge()
    print(f"judge: {type(judge).__name__}\n")

    results = []
    for i, r in enumerate(sample, 1):
        msgs = r.get("messages") or []
        task = _first_user_task(msgs)
        trajectory = _serialize_messages(msgs)
        diff = _state_diff(r.get("observer_reports"))
        rubric = TRAJECTORY_RUBRIC
        if diff:
            rubric += "\n\n# Environment diff\n" + diff[:4000]
        verdict = judge.score(
            task=task or "(no task)",
            trajectory=trajectory,
            rubric=rubric,
            data_source="rollout_calib",
            system=_TRAJECTORY_SYSTEM,
        )
        traj = aggregate_trajectory(verdict)
        results.append((r.get("bucket", ""), traj, verdict))
        print(
            f"{i:>3} {str(r.get('bucket',''))[:10]:>10} traj={traj:.3f} "
            f"tool={verdict.get('tool',0):.2f} eff={verdict.get('efficiency',0):.2f} "
            f"plan={verdict.get('planning',0):.2f} cons={verdict.get('consistency',0):.2f} "
            f"rec={verdict.get('recovery',0):.2f}"
        )

    if not results:
        print("无结果")
        return
    trajs = [x[1] for x in results]
    print("\n" + "-" * 60)
    print(
        f"trajectory 聚合分: mean={statistics.mean(trajs):.3f} "
        f"median={statistics.median(trajs):.3f} min={min(trajs):.3f} max={max(trajs):.3f}"
    )
    for dim in ("tool", "efficiency", "planning", "consistency", "recovery"):
        vals = [x[2].get(dim, 0.0) for x in results]
        print(
            f"  {dim:>11}: mean={statistics.mean(vals):.3f} "
            f"nonzero={sum(1 for v in vals if v > 0)}/{len(vals)}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for bucket, traj, verdict in results:
                f.write(
                    json.dumps({"bucket": bucket, "trajectory": traj, "verdict": verdict},
                               ensure_ascii=False) + "\n"
                )
        print(f"\n落盘 → {out_path}")


if __name__ == "__main__":
    main()
