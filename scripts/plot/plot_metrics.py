#!/usr/bin/env python3
"""从 metrics.jsonl 画训练曲线(pandas + matplotlib),按指标类别分组成多张图。

兼容两种 JSONL 格式:
  - extract_metrics.py 产物(扁平):  {"step": N, "actor/loss": ..., "critic/rewards/mean": ...}
  - verl FileLogger 产物(嵌套):     {"step": N, "data": {"actor/loss": ..., ...}}

按 key 的前缀(actor / critic / response_length / timing_s / perf / global_seqlen ...)
分组,每组一张 PNG(组内每个标量指标一个子图),另出一张"关键指标总览"。

用法:
  python scripts/plot/plot_metrics.py logs/experiments/qwen35_9b_b1_16gpu/metrics.jsonl
  python scripts/plot/plot_metrics.py <jsonl> -o figs/            # 指定输出目录
  python scripts/plot/plot_metrics.py <jsonl> --groups actor,critic  # 只画部分组
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import pandas as pd

# "关键指标"总览图挑这些(按关键词模糊匹配第一个命中的列)
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


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 嵌套格式(verl FileLogger): {"step":N, "data":{...}} → 拍平
        if "data" in d and isinstance(d["data"], dict):
            flat = {"step": d.get("step")}
            flat.update(d["data"])
            d = flat
        rows.append(d)
    if not rows:
        print("⚠️ 空 metrics 文件", file=sys.stderr)
        sys.exit(1)
    df = pd.DataFrame(rows)
    # 只保留数值列 + step
    if "step" not in df.columns:
        df["step"] = range(1, len(df) + 1)
    df = df.sort_values("step").reset_index(drop=True)
    # 数值化(非数值列转 NaN 后丢弃)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "step" and df[c].notna().any()]


def group_by_prefix(cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for c in cols:
        pref = c.split("/")[0]
        groups.setdefault(pref, []).append(c)
    return groups


def plot_panel(df, cols, title, out_png):
    """一张图:cols 里每个指标一个子图(网格布局)。"""
    n = len(cols)
    if n == 0:
        return
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.2 * nrow), squeeze=False)
    for i, col in enumerate(cols):
        ax = axes[i // ncol][i % ncol]
        ax.plot(df["step"], df[col], marker="o", ms=3, lw=1.2)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    # 多出的空子图隐藏
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  ✓ {out_png}  ({n} 子图)")


def plot_overview(df, out_png):
    """关键指标总览:一张图 8 个核心指标。"""
    picks = []
    for label, exact in KEY_OVERVIEW:
        col = exact if exact in df.columns else next(
            (c for c in df.columns if label.replace("_", "").lower() in c.replace("_", "").replace("/", "").lower()),
            None,
        )
        if col and df[col].notna().any():
            picks.append((label, col))
    if not picks:
        return
    ncol = 4
    nrow = math.ceil(len(picks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3 * nrow), squeeze=False)
    for i, (label, col) in enumerate(picks):
        ax = axes[i // ncol][i % ncol]
        ax.plot(df["step"], df[col], marker="o", ms=3, lw=1.4, color="tab:blue")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="metrics.jsonl 路径")
    ap.add_argument("-o", "--outdir", default=None, help="输出目录(默认 <jsonl 同目录>/figs)")
    ap.add_argument("--groups", default=None, help="逗号分隔,只画这些前缀组(默认全画)")
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        print(f"❌ 找不到: {src}", file=sys.stderr)
        return 1
    outdir = Path(args.outdir) if args.outdir else src.parent / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    df = load(src)
    cols = numeric_cols(df)
    print(f"读入 {len(df)} step, {len(cols)} 个数值指标 → {outdir}")

    # 中文标题字体:主动注册 ~/.fonts 下的中文字体(集群无外网,SimHei 已在 ~/.fonts),
    # 再设为默认 sans-serif。缺失则回退 DejaVu(英文正常,中文显方框,不报错)。
    import os as _os
    import matplotlib.font_manager as _fm

    _zh_name = None
    for _fp in (
        _os.path.expanduser("~/.fonts/simhei.ttf"),
        _os.path.expanduser("~/.local/share/fonts/simhei.ttf"),
    ):
        if _os.path.exists(_fp):
            try:
                _fm.fontManager.addfont(_fp)
                _zh_name = _fm.FontProperties(fname=_fp).get_name()
                break
            except Exception:
                pass
    plt.rcParams["font.sans-serif"] = ([_zh_name] if _zh_name else []) + [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # 1) 关键指标总览
    plot_overview(df, outdir / "00_overview.png")

    # 2) 按前缀分组,每组一张
    groups = group_by_prefix(cols)
    want = set(args.groups.split(",")) if args.groups else None
    for pref, gcols in sorted(groups.items()):
        if want and pref not in want:
            continue
        plot_panel(df, gcols, f"{pref} 指标 ({len(gcols)})", outdir / f"{pref}.png")

    print(f"\n✅ 完成,图在 {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
