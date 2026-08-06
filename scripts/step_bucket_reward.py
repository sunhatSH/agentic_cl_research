#!/usr/bin/env python3
"""按步展示训练 bucket 和 reward 信息。

用法:
  python scripts/step_bucket_reward.py [-n 10]

输出每 step 的 bucket、reward、loss、entropy。
"""
import json, sys
from collections import OrderedDict


def load_bucket_boundaries(parquet_path: str = "datasets/train.parquet") -> list:
    """返回 [(start_row, bucket_name, count), ...]"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        # fallback: from cold_start jsonl ordering
        with open("datasets/cold_start/cold_start_1429.jsonl") as f:
            buckets = []
            prev = None
            for line in f:
                r = json.loads(line)
                b = (r.get("metadata") or {}).get("bucket", "?")
                if b != prev:
                    buckets.append(b)
                    prev = b
            return buckets  # simplified, no row counts
    t = pq.read_table(parquet_path)
    buckets = t.column("bucket").to_pylist()
    prev = None
    boundaries = []
    for i, b in enumerate(buckets):
        if b != prev:
            boundaries.append((i, b, 0))
            prev = b
        boundaries[-1] = (boundaries[-1][0], boundaries[-1][1], boundaries[-1][2] + 1)
    return boundaries


def bucket_for_step(step: int, boundaries: list, batch_size: int = 32) -> str:
    start_row = (step - 1) * batch_size
    for row_start, name, count in boundaries:
        if start_row >= row_start and start_row < row_start + count:
            return name
    return "?"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    metrics_path = "logs/metrics/qwen35_9b_b1_16gpu/metrics.jsonl"

    boundaries = load_bucket_boundaries()

    steps = []
    with open(metrics_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            dd = d.get("data", d)
            s = d.get("step")
            b = bucket_for_step(s, boundaries)
            steps.append({
                "step": s,
                "bucket": b,
                "rew": dd.get("critic/rewards/mean"),
                "r_min": dd.get("critic/rewards/min"),
                "r_max": dd.get("critic/rewards/max"),
                "loss": dd.get("actor/loss"),
                "ent": dd.get("actor/entropy_loss"),
            })

    if n:
        steps = steps[-n:]

    # show bucket transitions
    prev_b = None
    print(f"{'step':>5} {'bucket':15s} {'rew':>7} {'r_range':16s} {'loss':>9} {'ent':>6}")
    print("-" * 65)
    for s in steps:
        marker = ""
        if s["bucket"] != prev_b and prev_b is not None:
            marker = " ← 换桶"
        rr = f"[{s['r_min']:.2f},{s['r_max']:.2f}]" if s['r_min'] is not None else "?"
        lo = f"{s['loss']:+.5f}" if s['loss'] is not None else "?"
        print(f"{s['step']:5d} {s['bucket']:15s} {s['rew']:7.4f} {rr:>16s} {lo:>9s} {s['ent']:6.4f}{marker}")
        prev_b = s["bucket"]


if __name__ == "__main__":
    main()
