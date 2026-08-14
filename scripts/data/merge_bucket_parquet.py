#!/usr/bin/env python3
"""将各桶 train_<bucket>.parquet 按桶序拼接成一份混合 parquet。
用于 baseline 训练：按桶序排列数据 + verl shuffle=false，一次 cl_main 跑完，
无需多次调 cl_main / resume / per-bucket 目录。
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

BUCKET_ORDER = [
    "office", "research", "coding", "ops", "safety",
    "workflow", "finance", "communication", "qa",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="按桶序拼接 train parquet")
    ap.add_argument("--in-dir", required=True, help="含 train_<bucket>.parquet 的目录")
    ap.add_argument("--out-train", required=True, help="输出 train parquet 路径")
    ap.add_argument("--out-val", default="", help="可选的 val parquet 路径")
    args = ap.parse_args()

    indir = Path(args.in_dir)
    blocks = []
    for b in BUCKET_ORDER:
        f = indir / f"train_{b}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
            blocks.append(df)
            print(f"  {b}: {len(df)}")
        else:
            print(f"  {b}: (缺失)")

    if not blocks:
        print("ERROR: 无任何桶 parquet", file=sys.stderr)
        return 1

    merged = pd.concat(blocks, ignore_index=True)
    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out_train, index=False)
    print(f"[merge] {len(merged)} 行 → {args.out_train}")

    if args.out_val:
        import shutil
        val_src = indir / "val.parquet"
        if val_src.exists():
            shutil.copy(val_src, args.out_val)
            print(f"[merge] val → {args.out_val}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
