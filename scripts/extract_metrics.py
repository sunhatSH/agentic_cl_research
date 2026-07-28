#!/usr/bin/env python3
"""从 train.log 提取每 step 的训练 metrics 到 JSONL(离线,数据不丢)。

背景:实时落盘的 _log_training_metrics 取错了对象(hook 在 _update_actor 返回值上,
那里 metrics 未 reduce 且不含 reward/advantage —— 完整 metrics 在 verl fit 主循环,
外部 hook 够不到),故 logs/metrics/*.jsonl 从未生成。但 train.log 每 step 有一行
完整指标(verl logger 的 console 输出),本脚本从中解析还原。

行格式:  step:N - key:value - key:value - ...   (verl console logger)

用法:
  python scripts/extract_metrics.py logs/experiments/qwen35_9b_b1_16gpu/train.log
  python scripts/extract_metrics.py <log> -o out.jsonl        # 指定输出
  python scripts/extract_metrics.py <log> --print loss,reward # 顺带打印关键列趋势
"""
import argparse
import json
import re
import sys
from pathlib import Path

# 匹配 verl console 的 step 指标行:以 "step:<N> - " 开头(前面可能有 ray 进程前缀)
_STEP_LINE = re.compile(r"step:(\d+)\s+-\s+(.*)")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def parse_line(line: str):
    """从一行里抽出 (step, {metric: value})。非 step 行返回 None。"""
    line = _ANSI.sub("", line).rstrip("\n")
    # 去掉 ray 进程前缀,如 "(CLTaskRunner pid=5713) "
    line = re.sub(r"^\([^)]*\)\s*", "", line)
    m = _STEP_LINE.search(line)
    if not m:
        return None
    step = int(m.group(1))
    row = {"step": step}
    # 其余按 " - " 切成 key:value
    for tok in m.group(2).split(" - "):
        tok = tok.strip()
        if ":" not in tok:
            continue
        k, _, v = tok.partition(":")
        k, v = k.strip(), v.strip()
        try:
            row[k] = float(v) if ("." in v or "e" in v.lower() or "nan" in v.lower() or "inf" in v.lower()) else int(v)
        except ValueError:
            row[k] = v  # 非数值原样留(极少)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="train.log 路径")
    ap.add_argument("-o", "--out", default=None, help="输出 JSONL(默认 <log 同目录>/metrics.jsonl)")
    ap.add_argument("--print", dest="cols", default=None,
                    help="逗号分隔的关键词,顺带打印这些列趋势(如 loss,reward,grad_norm)")
    args = ap.parse_args()

    logp = Path(args.logfile)
    if not logp.exists():
        print(f"❌ 找不到日志: {logp}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else logp.parent / "metrics.jsonl"

    # 同一 step 可能多行(理论上一行),按 step 去重保留最后一次
    rows = {}
    for line in logp.open(encoding="utf-8", errors="replace"):
        r = parse_line(line)
        if r is not None:
            rows[r["step"]] = r

    if not rows:
        print("⚠️ 未解析到任何 step 指标行(日志可能还没完成第一个 step)", file=sys.stderr)
        return 1

    ordered = [rows[s] for s in sorted(rows)]
    with out.open("w", encoding="utf-8") as f:
        for r in ordered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 提取 {len(ordered)} 个 step → {out}")

    # 关键列趋势(可选)
    if args.cols:
        kws = [c.strip() for c in args.cols.split(",") if c.strip()]
        # 为每个关键词找匹配的完整 metric key(取第一个匹配)
        sample = ordered[-1]
        matched = {}
        for kw in kws:
            hit = next((k for k in sample if kw.lower() in k.lower() and isinstance(sample[k], (int, float))), None)
            if hit:
                matched[kw] = hit
        if matched:
            hdr = "step  " + "  ".join(f"{kw}({matched[kw]})" for kw in matched)
            print("\n" + hdr)
            for r in ordered:
                vals = "  ".join(f"{r.get(matched[kw], ''):.4g}" if isinstance(r.get(matched[kw]), (int, float)) else "-" for kw in matched)
                print(f"{r['step']:<5} {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
