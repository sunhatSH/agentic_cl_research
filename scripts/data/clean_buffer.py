"""Batch-clean a cold-start buffer (SQLite snapshot).

Reads all trajectories from the buffer, applies ZW stripping + garble filtering
to each trajectory's messages, and writes a new clean buffer. Trajectories that
are too garbled are dropped.

Usage:
    python scripts/data/clean_buffer.py \
        --input logs/cold/buffer.sqlite \
        --output logs/cold/buffer_clean.sqlite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasources.cleaning import clean_messages
from replay_buffer.bucket import BucketReplayBuffer


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-clean a cold-start buffer.")
    ap.add_argument("--input", required=True, help="input buffer SQLite path")
    ap.add_argument("--output", required=True, help="output clean buffer SQLite path")
    ap.add_argument("--garble-threshold", type=float, default=0.05)
    ap.add_argument("--single-msg-threshold", type=float, default=0.20)
    ap.add_argument("--single-msg-policy", default="drop_all", choices=["drop_all", "drop_tail"])
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"[clean_buffer] ERROR: {in_path} not found", file=sys.stderr)
        sys.exit(1)

    # Load source buffer
    src = BucketReplayBuffer()
    src.load(in_path)
    src_stats = src.stats()
    print(f"[clean_buffer] loaded {src_stats['total_size']} trajectories from {in_path}", flush=True)

    # Create clean buffer with same config
    dst = BucketReplayBuffer(
        num_buckets=src.num_buckets,
        total_capacity=src.total_capacity,
        q_min=src.q_min,
        bucket_names=src.bucket_names,
        bucket_task_counts=src.bucket_task_counts,
        alpha=src.alpha,
    )

    total = added = dropped = 0
    for tid in src.store.all_ids():
        total += 1
        entry = src.store.get(tid)
        if entry is None:
            dropped += 1
            continue
        trajectory, meta = entry
        # Extract messages from the trajectory
        if isinstance(trajectory, list):
            messages = trajectory
        elif isinstance(trajectory, dict) and "messages" in trajectory:
            messages = trajectory["messages"]
        else:
            dropped += 1
            continue

        result = clean_messages(
            messages,
            garble_threshold=args.garble_threshold,
            single_msg_threshold=args.single_msg_threshold,
            single_msg_policy=args.single_msg_policy,
        )
        if result.dropped:
            dropped += 1
            continue

        # Update messages in the trajectory payload
        if isinstance(trajectory, list):
            cleaned_traj = result.messages
        else:
            cleaned_traj = dict(trajectory)
            cleaned_traj["messages"] = result.messages

        bucket = meta.get("bucket") or meta.get("bucket_name")
        dst.add_trajectory(cleaned_traj, bucket, meta)
        added += 1

    dst.dump(out_path)
    dst_stats = dst.stats()
    print(
        f"[clean_buffer] DONE: {added}/{total} kept, {dropped} dropped -> {out_path}\n"
        f"[clean_buffer] clean buffer stats: {dst_stats}",
        flush=True,
    )


if __name__ == "__main__":
    main()
