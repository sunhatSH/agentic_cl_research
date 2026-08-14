#!/usr/bin/env python3
"""从 metrics.jsonl 画训练曲线(纯 stdlib + matplotlib,无 pandas 依赖)。

与 scripts/plot_metrics.py 等价,但去掉 pandas —— 集群 .venv 只有 matplotlib、没有
pandas/无外网装不上时用这个。

兼容两种 JSONL 格式:
  - 扁平:  {"step": N, "actor/loss": ..., "critic/rewards/mean": ...}
  - verl FileLogger 嵌套: {"step": N, "data": {"actor/loss": ..., ...}}

按 key 前缀(actor / critic / response_length / timing_s ...)分组,每组一张 PNG,
另出一张关键指标总览。

用法:
  python scripts/plot/plot_metrics_nopandas.py <jsonl> -o <outdir>
  python scripts/plot/plot_metrics_nopandas.py <jsonl> --groups actor,critic
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

KEY_OVERVIEW = [
    ("reward", "critic/rewards/mean"),
    ("pg_loss", "actor/pg_loss"),
    ("grad_norm", "actor/grad_norm"),
    ("entropy", "actor/entropy"),
    ("kl", "actor/ppo_kl"),
    ("resp_len", "response_length/mean"),
    ("adv", "critic/advantages/mean"),
    ("step_time", "timing_s/step"),
]


def load(path: Path):
    """→ (steps: list[float], cols: dict[name -> list[float|None]])."""
    rows = []
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "data" in d and isinstance(d["data"], dict):
            flat = {"step": d.get("step")}
            flat.update(d["data"])
            d = flat
        rows.append(d)
    if not rows:
        print("⚠️ 空 metrics 文件", file=sys.stderr)
        sys.exit(1)

    # 收集所有列名
    names = set()
    for r in rows:
        names.update(r.keys())
    names.discard("step")

    def num(v):
        try:
            if v is None or isinstance(v, bool):
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    steps = [num(r.get("step")) for r in rows]
    if any(s is None for s in steps):
        steps = list(range(1, len(rows) + 1))
    # 按 step 排序
    order = sorted(range(len(rows)), key=lambda i: steps[i])
    steps = [steps[i] for i in order]
    cols = {}
    for name in names:
        col = [num(rows[i].get(name)) for i in order]
        if any(v is not None for v in col):
            cols[name] = col
    return steps, cols


def _xy(steps, col):
    xs, ys = [], []
    for s, v in zip(steps, col):
        if v is not None:
            xs.append(s)
            ys.append(v)
    return xs, ys


def group_by_prefix(names):
    groups = {}
    for c in names:
        groups.setdefault(c.split("/")[0], []).append(c)
    return groups


def plot_panel(steps, cols, names, title, out_png):
    names = sorted(names)
    n = len(names)
    if n == 0:
        return
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for i, col in enumerate(names):
        ax = axes[i // ncol][i % ncol]
        xs, ys = _xy(steps, cols[col])
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  ✓ {out_png}  ({n} 子图)")


def plot_overview(steps, cols, out_png):
    picks = []
    for label, exact in KEY_OVERVIEW:
        col = exact if exact in cols else next(
            (c for c in cols if label.replace("_", "").lower()
             in c.replace("_", "").replace("/", "").lower()),
            None,
        )
        if col:
            picks.append((label, col))
    if not picks:
        return
    ncol = 4
    nrow = math.ceil(len(picks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3 * nrow), squeeze=False)
    for i, (label, col) in enumerate(picks):
        ax = axes[i // ncol][i % ncol]
        xs, ys = _xy(steps, cols[col])
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color="tab:blue")
        ax.set_title(f"{label}\n({col})", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    for j in range(len(picks), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("关键指标总览 (Key Metrics Overview)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  ✓ {out_png}  (总览 {len(picks)} 指标)")


def setup_font():
    zh_name = None
    for fp in (
        os.path.expanduser("~/.fonts/simhei.ttf"),
        os.path.expanduser("~/.local/share/fonts/simhei.ttf"),
    ):
        if os.path.exists(fp):
            try:
                fm.fontManager.addfont(fp)
                zh_name = fm.FontProperties(fname=fp).get_name()
                break
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = ([zh_name] if zh_name else []) + [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        print(f"❌ 找不到: {src}", file=sys.stderr)
        return 1
    outdir = Path(args.outdir) if args.outdir else src.parent / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    steps, cols = load(src)
    print(f"读入 {len(steps)} step, {len(cols)} 个数值指标 → {outdir}")

    setup_font()
    plot_overview(steps, cols, outdir / "00_overview.png")

    groups = group_by_prefix(cols.keys())
    want = set(args.groups.split(",")) if args.groups else None
    for pref, gcols in sorted(groups.items()):
        if want and pref not in want:
            continue
        plot_panel(steps, cols, gcols, f"{pref} 指标 ({len(gcols)})", outdir / f"{pref}.png")

    print(f"\n✅ 完成,图在 {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
