#!/usr/bin/env python3
"""Show current experiment progress with Unicode box-drawing tables."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/plot/ → repo root
LOG_DIR = ROOT / "logs" / "metrics"

EXPERIMENTS = {
    "B1": "qwen35_9b_b1_16gpu",
    "K2": "qwen35_9b_k2_16gpu",
    "R0": "qwen35_9b_r0-25k_16gpu",
}
K2_HAS_KL = True  # K2 has use_kl_loss=true

TRAIN_ORDER = ["coding", "office", "ops", "research", "workflow"]
BUCKET_STEPS = {b: i * 100 for i, b in enumerate(TRAIN_ORDER)}  # start step of each bucket

def bucket_at(step, total_steps=500):
    """Which bucket is being trained at this step (0-indexed)."""
    seg = total_steps // len(TRAIN_ORDER)
    idx = min((step - 1) // seg, len(TRAIN_ORDER) - 1)
    return TRAIN_ORDER[idx]


def fmt_table(name, metrics, has_kl):
    if not metrics:
        print(f"\n{name}: no metrics yet\n")
        return

    last = metrics[-1]
    bkt = bucket_at(last["step"])
    bkt_start = BUCKET_STEPS[bkt]
    bkt_end = bkt_start + 100
    progress = last["step"] - bkt_start

    hdr = f"{name} ({len(metrics)} steps, {bkt} 桶 {progress}/{bkt_end-bkt_start}"
    if has_kl:
        hdr += f", KL≈{metrics[-1].get('kl',0):.3f}"
    hdr += "):"

    if has_kl:
        print(f"\n{hdr}")
        # ┌──────┬────────┬─────────┬────────┬───────┬────────┐
        sep_top    = "  ┌──────┬────────┬─────────┬────────┬───────┬────────┐"
        sep_mid    = "  ├──────┼────────┼─────────┼────────┼───────┼────────┤"
        sep_bot    = "  └──────┴────────┴─────────┴────────┴───────┴────────┘"
        header_row = "  │ step │ reward │ pg_loss │   kl   │  ent  │  adv   │"
    else:
        print(f"\n{hdr}")
        sep_top    = "  ┌──────┬────────┬─────────┬───────┬────────┐"
        sep_mid    = "  ├──────┼────────┼─────────┼───────┼────────┤"
        sep_bot    = "  └──────┴────────┴─────────┴───────┴────────┘"
        header_row = "  │ step │ reward │ pg_loss │  ent  │  adv   │"

    print(sep_top)
    print(header_row)
    print(sep_mid)

    for i, m in enumerate(metrics):
        s  = m["step"]
        r  = m["reward"]
        pg = m["pg"]
        ent = m["ent"]
        adv = m["adv"]

        if has_kl:
            kl = m.get("kl", 0)
            row = f"  │ {s:4d} │ {r:6.4f} │ {pg:7.3f} │ {kl:6.4f} │ {ent:5.3f} │ {adv:6.3f} │"
        else:
            row = f"  │ {s:4d} │ {r:6.4f} │ {pg:7.3f} │ {ent:5.3f} │ {adv:6.3f} │"
        print(row)

        # separator after every row except last
        if i < len(metrics) - 1:
            print(sep_mid)

    print(sep_bot)


def main():
    for name, dirname in EXPERIMENTS.items():
        mlog = LOG_DIR / dirname / "metrics.jsonl"
        if not mlog.is_file():
            print(f"\n{name}: no metrics file")
            continue

        metrics = []
        with open(mlog) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)["data"]
                metrics.append({
                    "step": json.loads(line)["step"],
                    "reward": d.get("critic/rewards/mean", 0),
                    "pg": d.get("actor/pg_loss", 0),
                    "kl": d.get("actor/kl_loss", 0),
                    "ent": d.get("actor/entropy_loss", 0),
                    "adv": d.get("critic/advantages/mean", 0),
                })

        has_kl = (name == "K2")  # K2 has use_kl_loss=true
        fmt_table(name, metrics, has_kl)


if __name__ == "__main__":
    main()
