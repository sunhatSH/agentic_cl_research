#!/usr/bin/env python3
"""冷启动轨迹可修复清洗（L2 前置）——只做不改变语义的安全修复。

针对 cold_start_1429.jsonl 的 L2 结构问题，做两类**确定安全**的清洗：
  1. 补 assistant.reasoning_content 空字段（validator 要求该字段存在，可为空串）。
  2. strip 特殊 token（``<think>`` / ``｜DSML｜`` 等 validator 拒绝的控制 token）——
     这些是模型输出里混入的框架/思考标记，去掉不改变任务语义。

【不做】的（留给 L3 语义质检判，或直接判废，不在此清洗）：
  - toolcall arguments 不符 schema（clarify choices 格式 / search_files 缺 pattern /
    patch 缺 mode）：这是 agent 真实调用行为，改参数会篡改轨迹语义。
  - toolcall 未在 tools 定义（web_search/exec/skill_search/think）：工具集不匹配，
    是采集时的工具配置问题，非单条可修。
  - 截断/没跑完/无效：这些是 L3 语义质检 + 重采集处理的对象。

保留行序（与 warmup sqlite 1:1 对应），逐行写出，不丢行。

用法：python scripts/collect/clean_cold_start.py [--in <jsonl>] [--out <jsonl>]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_IN = ROOT / "datasets" / "cold_start" / "cold_start_1429.jsonl"
_OUT = ROOT / "datasets" / "cold_start" / "cold_start_1429.cleaned.jsonl"

# validator 拒绝的特殊/控制 token（出现在 content / tool 返回里）。strip 掉不改语义。
_SPECIAL_TOKENS = ["｜DSML｜", "<think>", "</think>"]
# 非法控制字符（tool 返回里混入二进制/乱码）：U+007F DEL、U+FFFD 替换符、C0 控制符(除 \t\n\r)。
_CONTROL_CHARS = re.compile(r"[\x7f�\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_special(text: str) -> str:
    if not isinstance(text, str):
        return text
    for tok in _SPECIAL_TOKENS:
        if tok in text:
            text = text.replace(tok, "")
    return _CONTROL_CHARS.sub("", text)


def clean_record(d: dict) -> dict:
    msgs = d.get("messages") or []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        # 1. content strip 特殊 token（所有 role）
        if isinstance(m.get("content"), str):
            m["content"] = _strip_special(m["content"])
        # 2. assistant 补 reasoning_content 空字段
        if m.get("role") == "assistant":
            if "reasoning_content" not in m or m.get("reasoning_content") is None:
                m["reasoning_content"] = ""
            else:
                m["reasoning_content"] = _strip_special(str(m["reasoning_content"]))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(_IN))
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    n = 0
    n_special = 0
    n_reason = 0
    tmp = Path(args.out).with_suffix(".jsonl.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(args.inp, encoding="utf-8", errors="replace") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n += 1
            d = json.loads(line)
            # 统计（清洗前）
            raw = json.dumps(d.get("messages", []), ensure_ascii=False)
            if any(t in raw for t in _SPECIAL_TOKENS):
                n_special += 1
            if any(
                isinstance(m, dict) and m.get("role") == "assistant" and "reasoning_content" not in m
                for m in d.get("messages", [])
            ):
                n_reason += 1
            d = clean_record(d)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    import os

    os.replace(tmp, args.out)
    print(f"清洗 {n} 行 → {args.out}")
    print(f"  含特殊 token 的行: {n_special}（已 strip）")
    print(f"  缺 reasoning_content 的行: {n_reason}（已补空串）")


if __name__ == "__main__":
    main()
