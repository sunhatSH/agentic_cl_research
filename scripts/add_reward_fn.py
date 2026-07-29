#!/usr/bin/env python3
"""给训练 parquet 的 reward_model 补 reward_fn 字段（迁移 §阶段3）。

omni RewardManager 从每条数据的 non_tensor_batch["reward_model"]["reward_fn"] 取打分函数
（omni.py:191，无 config 全局兜底，缺键会 KeyError）。现有 parquet 的 reward_model 是
{'ground_truth':'','style':'rule'}，缺 reward_fn。本脚本把 reward_fn 补成指向项目适配层的
dict（omni make_object_from_config 支持 _function_name 直接 import fqdn，不用改 verl 注册表）:

    reward_model["reward_fn"] = {"_function_name": "trainer.model_reward_omni.compute_score"}

用法:
  python scripts/add_reward_fn.py                       # 原地改 datasets/train.parquet(备份)
  python scripts/add_reward_fn.py --in X.parquet --out Y.parquet
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

REWARD_FN = {"_function_name": "trainer.model_reward_omni.compute_score"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="datasets/train.parquet")
    ap.add_argument("--out", dest="out", default=None, help="默认原地(先备份 .bak)")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out) if args.out else inp
    df = pd.read_parquet(inp)

    if "reward_model" not in df.columns:
        raise SystemExit(f"❌ {inp} 无 reward_model 列,列={list(df.columns)}")

    n_added, n_had = 0, 0

    def _patch(rm):
        nonlocal n_added, n_had
        rm = dict(rm) if isinstance(rm, dict) else {}
        if rm.get("reward_fn"):
            n_had += 1
        else:
            rm["reward_fn"] = REWARD_FN
            n_added += 1
        return rm

    df["reward_model"] = df["reward_model"].map(_patch)

    if out == inp and not args.no_backup:
        bak = inp.with_suffix(inp.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(inp, bak)
            print(f"备份 → {bak}")

    df.to_parquet(out, index=False)
    print(f"✅ {out}: 补 reward_fn {n_added} 行, 已有 {n_had} 行, 共 {len(df)} 行")
    print(f"   reward_fn = {REWARD_FN}")


if __name__ == "__main__":
    main()
