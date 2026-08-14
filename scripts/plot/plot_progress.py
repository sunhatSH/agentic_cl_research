#!/usr/bin/env python3
"""Plot training metrics — 每个实验一个文件夹，raw/EMA/SMA 各一张图，cross-experiment 对比。

每次运行产出:
  logs/experiments/figs/
    {exp}/              当前（旧图自动折叠到 {exp}_N/）
      {metric}_raw.png      原始数据
      {metric}_ema.png      EMA 平滑 (α=0.1)
      {metric}_sma.png      SMA 平滑 (window=10)
    compare/            跨实验 EMA 对比

旧图归档规则: 每次运行前，figs/ 下直接的 .png 和当前 {exp}/ 文件夹内容
自动移动到 {exp}_1/, {exp}_2/, ... 递增编号。compare/ 同理折叠到 compare_N/。
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/plot/ → repo root
LOG_DIR = ROOT / "logs" / "metrics"
FIG_DIR = ROOT / "logs" / "experiments" / "figs"

EXPERIMENTS = {
    "B1": ("qwen35_9b_b1_16gpu", False, "configs/run/b1_9b_16gpu.yaml"),
    "K2": ("qwen35_9b_k2_16gpu", True, "configs/run/k2_9b_16gpu.yaml"),
    "R0": ("qwen35_9b_r0-25k_16gpu", False, "configs/run/r0-25k_9b_16gpu.yaml"),
}

BUCKETS = {0: "coding", 100: "office", 200: "ops", 300: "research", 400: "workflow"}
COLORS = {"B1": "#2196F3", "K2": "#FF5722", "R0": "#4CAF50"}
EMA_ALPHA = 0.1
SMA_WINDOW = 10


def archive_subdir(base: Path, name: str) -> None:
    """将 figs/{name}/ 下的旧 PNG 移动到 figs/{name}_N/（N 自动递增）。
    也处理 figs/ 根目录下的孤立旧 PNG 和 name 文件夹。"""
    src = base / name
    if not src.exists() and not any(base.glob(f"{name}*")):
        return

    # 找到下一个可用编号
    n = 1
    while (base / f"{name}_{n}").exists():
        n += 1
    dst = base / f"{name}_{n}"
    dst.mkdir(parents=True, exist_ok=True)

    # 移动 src/ 下的 PNG + config.json
    if src.is_dir():
        for f in sorted(src.glob("*.png")) + sorted(src.glob("config.json")):
            shutil.move(str(f), str(dst / f.name))
        # 如果文件夹空了就删掉
        if not any(src.iterdir()):
            src.rmdir()

    # 也处理 figs/ 根目录下旧的 {name}_*.png 孤立文件
    for f in sorted(base.glob("*.png")):
        shutil.move(str(f), str(dst / f.name))


# ── 归档旧图 ──
for exp_name in EXPERIMENTS:
    archive_subdir(FIG_DIR, exp_name)
archive_subdir(FIG_DIR, "compare")


def ema(x, alpha=EMA_ALPHA):
    s = np.zeros_like(x)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
    return s


def sma(x, window=SMA_WINDOW):
    if len(x) < window:
        return x  # fallback
    return np.convolve(x, np.ones(window) / window, mode="valid")


def load_metrics(dirname):
    f = LOG_DIR / dirname / "metrics.jsonl"
    if not f.is_file():
        return None
    d = {
        "step": [],
        "reward": [],
        "pg_loss": [],
        "entropy": [],
        "kl": [],
        "adv": [],
        "reward_std": [],
        "group_std": [],
        "group_std_max": [],
        "lr": [],
    }
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            dd = j["data"]
            d["step"].append(j["step"])
            d["reward"].append(dd.get("critic/rewards/mean", 0))
            d["pg_loss"].append(dd.get("actor/pg_loss", 0))
            d["entropy"].append(dd.get("actor/entropy_loss", 0))
            d["kl"].append(dd.get("actor/kl_loss", 0))
            d["adv"].append(dd.get("critic/advantages/mean", 0))
            d["reward_std"].append(dd.get("cl/reward_std"))
            d["group_std"].append(dd.get("cl/group_reward_std"))
            d["group_std_max"].append(dd.get("cl/group_reward_std_max"))
            d["lr"].append(dd.get("actor/lr"))
    for k in list(d.keys()):
        d[k] = np.array(d[k], dtype=np.float64)
    return d


def has_std(data):
    if data is None:
        return False
    return not np.all(np.isnan(data["reward_std"]))


def load_config_summary(config_path):
    """读实验 yaml，提取关键超参（画图时写进 figs 目录供追溯）。"""
    p = ROOT / config_path
    if not p.is_file():
        return {"config": config_path, "error": "not found"}
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    a = cfg.get("actor_rollout_ref", {}) or {}
    actor = a.get("actor", {}) or {}
    rollout = a.get("rollout", {}) or {}
    data = cfg.get("data", {}) or {}
    trainer = cfg.get("trainer", {}) or {}
    cl = cfg.get("cl", {}) or {}
    optim = actor.get("optim", {}) or {}
    # 回放配置：replay_ratio(新:旧) 默认 5.0，replay_batch_size 是回放量上限
    replay_ratio = cl.get("replay_ratio", 5.0)
    replay_batch_size = cl.get("replay_batch_size", 64)
    return {
        "config": config_path,
        "lr": optim.get("lr"),
        "entropy_coeff": actor.get("entropy_coeff"),
        "use_kl_loss": actor.get("use_kl_loss"),
        "kl_loss_coef": actor.get("kl_loss_coef"),
        "rollout_n": rollout.get("n"),
        "train_batch_size": data.get("train_batch_size"),
        "train_files": data.get("train_files"),
        "total_training_steps": trainer.get("total_training_steps"),
        "lambda_replay": cl.get("lambda_replay"),
        "buffer_enabled": (cl.get("buffer", {}) or {}).get("enabled"),
        "replay_ratio": replay_ratio,  # 新:旧 比值
        "replay_fraction": round(1.0 / (float(replay_ratio) + 1.0), 4),  # 回放占比 = 1/(ratio+1)
        "replay_batch_size": replay_batch_size,  # 回放量上限
    }


def add_bucket_lines(ax, max_step=None):
    for bs in BUCKETS:
        if max_step is not None and bs > max_step:
            continue
        ax.axvline(bs, color="gray", ls="--", alpha=0.25, lw=0.6)


def save_fig(out_path, title, plot_fn):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plot_fn(ax)
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_single_metric(exp_dir, metric_key, title, ylim, xs, ys_raw, color, max_step, label_extra=""):
    """生成 {metric}_raw.png, {metric}_ema.png, {metric}_sma.png 三张图。"""
    label = f" {label_extra}" if label_extra else ""

    # raw
    save_fig(
        exp_dir / f"{metric_key}_raw.png",
        f"{title}{label} — raw",
        lambda ax: [
            ax.plot(xs, ys_raw, "o-", color=color, ms=3, lw=0.6, alpha=0.6),
            add_bucket_lines(ax, max_step),
            ylim and ax.set_ylim(*ylim),
        ],
    )

    # EMA
    if len(ys_raw) > 5:
        ys_e = ema(ys_raw)
        save_fig(
            exp_dir / f"{metric_key}_ema.png",
            f"{title}{label} — EMA (α={EMA_ALPHA})",
            lambda ax: [
                ax.plot(xs, ys_e, "-", color=color, lw=1.8),
                add_bucket_lines(ax, max_step),
                ylim and ax.set_ylim(*ylim),
            ],
        )

    # SMA
    if len(ys_raw) >= SMA_WINDOW:
        ys_s = sma(ys_raw)
        save_fig(
            exp_dir / f"{metric_key}_sma.png",
            f"{title}{label} — SMA (w={SMA_WINDOW})",
            lambda ax: [
                ax.plot(xs[SMA_WINDOW - 1 :], ys_s, "-", color=color, lw=1.8),
                add_bucket_lines(ax, max_step),
                ylim and ax.set_ylim(*ylim),
            ],
        )


# ── Load data ──
all_data = {}
for name, (dn, has_kl, cfg_path) in EXPERIMENTS.items():
    d = load_metrics(dn)
    if d is not None and len(d["step"]) > 0:
        all_data[name] = {"data": d, "has_kl": has_kl, "config": load_config_summary(cfg_path)}

if not all_data:
    print("No metrics found.")
    sys.exit(0)

max_step = max(d["data"]["step"][-1] for d in all_data.values())

METRICS = [
    ("reward", "Reward", (0.0, 1.05)),
    ("pg_loss", "Policy Gradient Loss", None),
    ("entropy", "Entropy Loss", None),
    ("adv", "Advantage", None),
]

# ── Per-experiment: raw/EMA/SMA for each metric ──
for name, info in all_data.items():
    d = info["data"]
    exp_dir = FIG_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    xs = d["step"]
    # 写配置摘要文件（追溯这张图在什么超参下画出）
    summary = dict(info["config"])
    # 用 metrics 里的真实 actor/lr 覆盖 yaml 静态值（命令行 --lr 覆盖后才是真实运行值）
    lr_vals = [x for x in d["lr"] if x is not None]
    if lr_vals:
        summary["lr_yaml"] = summary.get("lr")  # 保留 yaml 静态值对照
        summary["lr"] = float(lr_vals[-1])  # 真实运行 lr
    with open(exp_dir / "config.json", "w", encoding="utf-8") as cf:
        json.dump(summary, cf, ensure_ascii=False, indent=2, default=str)
    print(f"\n{name}/  (config → {exp_dir / 'config.json'})")

    for key, title, ylim in METRICS:
        plot_single_metric(exp_dir, key, f"{name} — {title}", ylim, xs, d[key], COLORS[name], max_step)
        print(f"  {key}_raw.png  {key}_ema.png  {key}_sma.png")

    # KL (if applicable)
    if info["has_kl"]:
        plot_single_metric(
            exp_dir, "kl", f"{name} — KL Divergence", None, xs, d["kl"], COLORS[name], max_step
        )
        print("  kl_raw.png  kl_ema.png  kl_sma.png")

    # ── std metrics ──
    if has_std(d):
        # reward_std: single line
        plot_single_metric(
            exp_dir, "reward_std", f"{name} — Reward Std", None, xs, d["reward_std"], COLORS[name], max_step
        )
        print("  reward_std_raw.png  reward_std_ema.png  reward_std_sma.png")

        # group_std + group_std_max: dual-line, three views
        for suffix, yf in [
            ("raw", lambda y: y),
            ("ema", ema),
            ("sma", lambda y: sma(y) if len(y) >= SMA_WINDOW else y),
        ]:
            stitle = f"{name} — Group Reward Std" + (
                f" (EMA α={EMA_ALPHA})"
                if suffix == "ema"
                else f" (SMA w={SMA_WINDOW})"
                if suffix == "sma"
                else " — raw"
            )
            if suffix == "sma" and len(d["group_std"]) < SMA_WINDOW:
                continue
            xp, y1, y2 = xs, yf(d["group_std"]), yf(d["group_std_max"])
            if suffix == "sma":
                xp = xs[SMA_WINDOW - 1 :]
            save_fig(
                exp_dir / f"group_std_{suffix}.png",
                stitle,
                lambda ax, xp=xp, y1=y1, y2=y2, c=COLORS[name]: [
                    ax.plot(xp, y1, "-", color=c, lw=1.8, label="group_std (mean)"),
                    ax.plot(xp, y2, "--", color="#795548", lw=1.5, label="group_std_max"),
                    add_bucket_lines(ax, max_step),
                    ax.legend(fontsize=9),
                ],
            )
        print("  group_std_raw.png  group_std_ema.png  group_std_sma.png")


# ── Cross-experiment comparison (EMA only) ──
cmp_dir = FIG_DIR / "compare"
exp_names = sorted(all_data.keys())
print("\ncompare/")

for key, title, ylim in METRICS + [("kl", "KL Divergence", None)]:
    if key == "kl" and not any(info["has_kl"] for info in all_data.values()):
        continue
    save_fig(
        cmp_dir / f"{key}_ema.png",
        f"{title} — EMA comparison",
        lambda ax, k=key, tl=title, yl=ylim: [
            (
                lambda: [
                    ax.plot(
                        info["data"]["step"], ema(info["data"][k]), "-", color=COLORS[name], lw=2, label=name
                    )
                    for name, info in all_data.items()
                    if k != "kl" or info["has_kl"]
                ]
            )(),
            add_bucket_lines(ax, max_step),
            yl and ax.set_ylim(*yl),
        ],
    )
    print(f"  {key}_ema.png")
