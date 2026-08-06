#!/usr/bin/env python3
"""将 train.parquet 按桶截断为 batch_size=32 的整数倍，确保每步纯单桶。

用法:
  python scripts/align_bucket_batch.py [--batch 32] [--out datasets/train_aligned.parquet]
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BATCH = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="datasets/train.parquet")
    ap.add_argument("--out", default="datasets/train_aligned.parquet")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    t = pq.read_table(args.input)
    buckets = t.column("bucket").to_pylist()
    n = len(buckets)

    # find bucket boundaries
    prev = None
    segs = []  # [(start_row, bucket_name, count)]
    for i, b in enumerate(buckets):
        if b != prev:
            if segs:
                segs[-1] = (segs[-1][0], segs[-1][1], i - segs[-1][0])
            segs.append((i, b, 0))
            prev = b
    segs[-1] = (segs[-1][0], segs[-1][1], n - segs[-1][0])

    # truncate each segment to floor(count / batch) * batch
    keep_indices = []
    dropped_summary = []
    for start, name, count in segs:
        keep_count = (count // args.batch) * args.batch
        drop = count - keep_count
        keep_indices.extend(range(start, start + keep_count))
        if drop > 0:
            dropped_summary.append(f"  {name}: {count} → {keep_count} (drop {drop})")

    print(f"原始: {n} 行, {len(segs)} 个桶段")
    print(f"保留: {len(keep_indices)} 行 ({(len(keep_indices)/n*100):.1f}%)")
    print(f"丢弃:")
    for s in dropped_summary:
        print(s)

    # build new table with kept rows only
    t2 = t.take(keep_indices)
    pq.write_table(t2, args.out)
    print(f"写出: {args.out}")

    # Also dump summary
    buckets_after = [buckets[i] for i in keep_indices]
    print(f"\n对齐后桶步数 (每步 {args.batch} 行):")
    prev_b = None
    step_count = 0
    for b in buckets_after:
        if b != prev_b:
            if prev_b is not None:
                print(f"  {prev_b}: {step_count} steps ({step_count * args.batch} rows)")
            step_count = 0
            prev_b = b
        step_count += 1
    if prev_b:
        print(f"  {prev_b}: {step_count} steps ({step_count * args.batch} rows)")


if __name__ == "__main__":
    main()
