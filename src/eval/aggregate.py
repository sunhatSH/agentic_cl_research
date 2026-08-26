#!/usr/bin/env python3
"""评测结果聚合：读 cl_eval 输出的 per_task.json，算 pass@N / 均分 / 按桶分组。

cl_eval.py 跑完 195 题 × N 次后输出 per_task.json（每题一条，含 reward/task_done/
correctness/trajectory/safety 四维）。本脚本读它，算：
  - pass@N（N 次全过的题占比）
  - 均分（reward 均值）
  - 按桶分组（9 桶各自的 pass_rate / 均分 / 四维）
  - 与 baseline 对比（delta）

用法：
  python -m eval.aggregate --results eval/results/<model>/per_task.json
  python -m eval.aggregate --results eval/results/<model>/per_task.json --baseline eval/results/base/per_task.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any


def load_results(path: str) -> list[dict]:
    """读 per_task.json（cl_eval 输出）。每条含 task_id/reward/task_done/correctness/trajectory/safety/bucket。"""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return data


def aggregate(results: list[dict], baseline: list[dict] | None = None) -> dict[str, Any]:
    """算全局 pass@3(至少1次过) / pass^3(3次全过) / 均分 / 四维 + 与 baseline 对比。

    results 可能是两种格式：
    ① 每题一条（已聚合，task_done=1 表示该题通过）→ pass@3 = pass^3 = 通过率
    ② 每题 N 条（per-run，同一 task_id 多条）→ 按 task_id 分组算 pass@3 / pass^3
    """
    n = len(results)
    if n == 0:
        return {"error": "no results"}

    # 检测是否有 per-run 记录（同一 task_id 多条）
    from collections import Counter
    tid_counts = Counter(r.get("task_id", "") for r in results)
    has_per_run = any(c > 1 for c in tid_counts.values())

    if has_per_run:
        # per-run 模式：按 task_id 分组
        from collections import defaultdict
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            by_task[r["task_id"]].append(r)

        n_tasks = len(by_task)
        pass_at_3 = sum(1 for tid, runs in by_task.items()
                        if any(r.get("task_done", 0) >= 1.0 for r in runs))
        pass_caret_3 = sum(1 for tid, runs in by_task.items()
                           if all(r.get("task_done", 0) >= 1.0 for r in runs) and len(runs) > 0)
        # 均分：每题先算 run 内均分，再题间均分
        per_task_reward = [sum(r.get("reward", 0) for r in runs) / len(runs)
                           for runs in by_task.values()]
        mean_reward = sum(per_task_reward) / n_tasks if n_tasks else 0

        dims = {}
        for d in ["task_done", "correctness", "trajectory", "safety"]:
            per_task_vals = [sum(r.get(d, 0) for r in runs) / len(runs)
                             for runs in by_task.values()]
            dims[d] = {
                "mean": sum(per_task_vals) / n_tasks if n_tasks else 0,
                "zero_frac": sum(1 for v in per_task_vals if v == 0) / n_tasks if n_tasks else 0,
                "one_frac": sum(1 for v in per_task_vals if v >= 1.0) / n_tasks if n_tasks else 0,
            }
    else:
        # 每题一条模式
        n_tasks = n
        pass_at_3 = sum(1 for r in results if r.get("task_done", 0) >= 1.0)
        pass_caret_3 = pass_at_3  # 无 per-run，两者相同
        mean_reward = sum(r.get("reward", 0) for r in results) / n
        dims = {}
        for d in ["task_done", "correctness", "trajectory", "safety"]:
            vals = [r.get(d, 0) for r in results]
            dims[d] = {
                "mean": sum(vals) / n,
                "zero_frac": sum(1 for v in vals if v == 0) / n,
                "one_frac": sum(1 for v in vals if v >= 1.0) / n,
            }

    out = {
        "n_tasks": n_tasks,
        "pass_at_3": pass_at_3 / n_tasks if n_tasks else 0,   # 至少1次过（能力上限）
        "pass_caret_3": pass_caret_3 / n_tasks if n_tasks else 0,  # 3次全过（稳定性）
        "n_pass_at_3": pass_at_3,
        "n_pass_caret_3": pass_caret_3,
        "mean_reward": mean_reward,
        "dims": dims,
    }

    if baseline:
        base_map = {r["task_id"]: r for r in baseline}
        common = [r for r in results if r["task_id"] in base_map]
        if common:
            base_rewards = [base_map[r["task_id"]].get("reward", 0) for r in common]
            cur_rewards = [r.get("reward", 0) for r in common]
            base_pass = sum(1 for r in common if base_map[r["task_id"]].get("task_done", 0) >= 1.0)
            cur_pass = sum(1 for r in common if r.get("task_done", 0) >= 1.0)
            forgetting = sum(max(0, b - c) for b, c in zip(base_rewards, cur_rewards)) / len(common)
            out["baseline_comparison"] = {
                "n_common": len(common),
                "baseline_mean_reward": sum(base_rewards) / len(common),
                "current_mean_reward": sum(cur_rewards) / len(common),
                "reward_delta": sum(cur_rewards) / len(common) - sum(base_rewards) / len(common),
                "baseline_pass": base_pass / len(common),
                "current_pass": cur_pass / len(common),
                "pass_delta": cur_pass / len(common) - base_pass / len(common),
                "forgetting": forgetting,
                "cl_score": sum(cur_rewards) / len(common) - forgetting,
            }

    return out


def aggregate_by_bucket(results: list[dict], baseline: list[dict] | None = None) -> dict[str, dict]:
    """按 9 桶分组，每桶 pass@3 / pass^3 / 均分 / 四维。"""
    base_map = {r["task_id"]: r for r in (baseline or [])}
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_bucket[r.get("bucket", "Unlabeled")].append(r)

    table = {}
    for bucket, rows in sorted(by_bucket.items()):
        # 检测 per-run（同 task_id 多条）
        from collections import Counter
        tid_counts = Counter(r.get("task_id", "") for r in rows)
        has_per_run = any(c > 1 for c in tid_counts.values())

        if has_per_run:
            by_task: dict[str, list[dict]] = defaultdict(list)
            for r in rows:
                by_task[r["task_id"]].append(r)
            n = len(by_task)
            pass_at_3 = sum(1 for runs in by_task.values()
                           if any(r.get("task_done", 0) >= 1.0 for r in runs))
            pass_caret_3 = sum(1 for runs in by_task.values()
                               if all(r.get("task_done", 0) >= 1.0 for r in runs) and runs)
            per_task_rw = [sum(r.get("reward", 0) for r in runs) / len(runs) for runs in by_task.values()]
            mean_rw = sum(per_task_rw) / n if n else 0
            dim_vals = {}
            for d in ["task_done", "correctness", "trajectory", "safety"]:
                pt = [sum(r.get(d, 0) for r in runs) / len(runs) for runs in by_task.values()]
                dim_vals[d] = sum(pt) / n if n else 0
        else:
            n = len(rows)
            pass_at_3 = sum(1 for r in rows if r.get("task_done", 0) >= 1.0)
            pass_caret_3 = pass_at_3
            mean_rw = sum(r.get("reward", 0) for r in rows) / n if n else 0
            dim_vals = {d: sum(r.get(d, 0) for r in rows) / n if n else 0
                        for d in ["task_done", "correctness", "trajectory", "safety"]}

        row = {
            "n_tasks": n,
            "pass_at_3": pass_at_3 / n if n else 0,
            "pass_caret_3": pass_caret_3 / n if n else 0,
            "mean_reward": mean_rw,
            "task_done": dim_vals["task_done"],
            "correctness": dim_vals["correctness"],
            "trajectory": dim_vals["trajectory"],
            "safety": dim_vals["safety"],
        }
        if baseline:
            b_rows = [r for r in rows if r["task_id"] in base_map]
            if b_rows:
                bn = len(b_rows)
                b_rw = [base_map[r["task_id"]].get("reward", 0) for r in b_rows]
                b_ps = sum(1 for r in b_rows if base_map[r["task_id"]].get("task_done", 0) >= 1.0)
                row["baseline_reward"] = sum(b_rw) / bn
                row["baseline_pass"] = b_ps / bn
                row["reward_delta"] = row["mean_reward"] - sum(b_rw) / bn
                row["pass_delta"] = row["pass_at_3"] - b_ps / bn
        table[bucket] = row
    return table


def main():
    ap = argparse.ArgumentParser(description="聚合评测结果：pass@N / 均分 / 按桶")
    ap.add_argument("--results", required=True, help="per_task.json 路径（cl_eval 输出）")
    ap.add_argument("--baseline", default=None, help="baseline per_task.json（对比用）")
    ap.add_argument("--output", default=None, help="输出 scores.json 路径（默认同目录）")
    args = ap.parse_args()

    results = load_results(args.results)
    baseline = load_results(args.baseline) if args.baseline else None

    summary = aggregate(results, baseline)
    bucket_table = aggregate_by_bucket(results, baseline)

    out = {
        "results_file": args.results,
        "baseline_file": args.baseline,
        "summary": summary,
        "per_bucket": bucket_table,
    }

    out_path = args.output or str(Path(args.results).parent / "scores.json")
    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # 打印汇总表
    s = summary
    print(f"\n{'='*60}")
    print(f"  评测汇总 (n={s.get('n_tasks',0)} tasks)")
    print(f"{'='*60}")
    print(f"  pass@3  (至少1次过): {s.get('pass_at_3',0):.3f}  ({s.get('n_pass_at_3',0)}/{s.get('n_tasks',0)})")
    print(f"  pass^3  (3次全过):   {s.get('pass_caret_3',0):.3f}  ({s.get('n_pass_caret_3',0)}/{s.get('n_tasks',0)})")
    print(f"  均分 (mean_reward):  {s.get('mean_reward',0):.3f}")
    print(f"\n  四维均值:")
    for d in ["task_done", "correctness", "trajectory", "safety"]:
        dv = s.get("dims", {}).get(d, {})
        print(f"    {d:12s}: mean={dv.get('mean',0):.3f}  ==0:{dv.get('zero_frac',0)*100:.0f}%  ==1:{dv.get('one_frac',0)*100:.0f}%")
    if "baseline_comparison" in s:
        bc = s["baseline_comparison"]
        print(f"\n  vs baseline:")
        print(f"    reward_delta: {bc.get('reward_delta',0):+.3f}  (base={bc.get('baseline_mean_reward',0):.3f} → cur={bc.get('current_mean_reward',0):.3f})")
        print(f"    pass_delta:  {bc.get('pass_delta',0):+.3f}  (base={bc.get('baseline_pass',0):.3f} → cur={bc.get('current_pass',0):.3f})")
        print(f"    forgetting: {bc.get('forgetting',0):.3f}  cl_score: {bc.get('cl_score',0):.3f}")

    print(f"\n  按桶:")
    print(f"  {'bucket':<15s} {'n':>4s} {'pass@3':>8s} {'pass^3':>8s} {'mean_rw':>8s}")
    print(f"  {'-'*45}")
    for b, r in sorted(bucket_table.items()):
        print(f"  {b:<15s} {r.get('n_tasks',0):>4} {r.get('pass_at_3',0):>8.3f} {r.get('pass_caret_3',0):>8.3f} {r.get('mean_reward',0):>8.3f}")

    print(f"\n✅ scores 写入 {out_path}")


if __name__ == "__main__":
    main()
