#!/usr/bin/env python3
"""把多个实验的 metrics.jsonl 叠在同一张图上对比(reward / loss 各一张)。

用途:同一组消融/baseline 的多个实验(如 4卡 b1/k1/k2/k3)横向对比 reward、loss
趋势。x 轴=step,每个实验一条线,图例带该实验最后 step(用户要求"用 step 作后缀区分")。

无 pandas 依赖(集群 .venv 只有 matplotlib)。兼容 verl FileLogger 嵌套 {"step":N,"data":{...}}
和扁平格式两种。

用法:
  python scripts/plot_metrics_compare.py \
      logs/metrics/qwen35_9b_b1_4gpu/metrics.jsonl \
      logs/metrics/qwen35_9b_k1_4gpu/metrics.jsonl ... \
      -o doc/debug/plots_4gpu --title-suffix "4gpu"
  # 实验名默认取 metrics.jsonl 父目录名;--labels 可覆盖(逗号分隔,与输入同序)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 画哪些指标:每个元素 (jsonl里的key, 输出文件名, 图标题, y轴标签)
METRICS = [
    ("critic/rewards/mean", "reward_mean", "Reward (critic/rewards/mean)", "reward"),
    ("actor/pg_loss", "pg_loss", "Policy Loss (actor/pg_loss)", "pg_loss"),
    ("actor/loss", "total_loss", "Total Loss (actor/loss)", "loss"),
    ("actor/replay_loss", "replay_loss", "Replay Loss (actor/replay_loss)", "replay_loss"),
    ("actor/kl_loss", "kl_loss", "KL Loss (actor/kl_loss)", "kl_loss"),
    ("actor/entropy_loss", "entropy_loss", "Entropy Loss (actor/entropy_loss)", "entropy"),
]


def load(path: Path):
    """→ (steps[list], data{key: list(值,与steps对齐,缺失None)})。"""
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    steps, data = [], {}
    for i, r in enumerate(rows):
        d = r.get("data", r)  # 嵌套 or 扁平
        step = r.get("step", i)
        steps.append(step)
        for k, v in d.items():
            if isinstance(v, (int, float)):
                data.setdefault(k, {})[step] = v
    return steps, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonls", nargs="+", help="多个 metrics.jsonl 路径")
    ap.add_argument("-o", "--outdir", default="doc/debug/plots_compare")
    ap.add_argument("--labels", default=None, help="逗号分隔的实验名(覆盖默认父目录名)")
    ap.add_argument("--title-suffix", default="", help="图标题后缀,如 4gpu")
    args = ap.parse_args()

    paths = [Path(p) for p in args.jsonls]
    if args.labels:
        labels = args.labels.split(",")
    else:
        # 默认取父目录名,去掉 qwen35_9b_ 前缀让图例简洁
        labels = [p.parent.name.replace("qwen35_9b_", "") for p in paths]

    series = []  # (label_with_step, steps, data)
    for p, lab in zip(paths, labels):
        if not p.exists():
            print(f"[warn] 跳过不存在: {p}", file=sys.stderr)
            continue
        steps, data = load(p)
        last = max(steps) if steps else 0
        series.append((f"{lab}@step{last}", steps, data))
        print(f"[load] {lab}: {len(steps)} 点, 到 step {last}")

    if not series:
        print("[error] 无有效数据", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = f" [{args.title_suffix}]" if args.title_suffix else ""

    made = []
    for key, fname, title, ylabel in METRICS:
        # 至少一个实验有这个 key 才画
        if not any(key in data for _, _, data in series):
            continue
        fig, ax = plt.subplots(figsize=(11, 5.5))
        for label, steps, data in series:
            if key not in data:
                continue
            pts = sorted(data[key].items())  # [(step, val)]
            xs = [s for s, _ in pts]
            ys = [v for _, v in pts]
            ax.plot(xs, ys, marker="", linewidth=1.4, alpha=0.85, label=label)
        ax.set_title(title + suffix)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        out = outdir / f"{fname}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        made.append(out)
        print(f"[plot] {out}")

    print(f"\n[done] {len(made)} 张图 → {outdir}/")


if __name__ == "__main__":
    main()
