#!/usr/bin/env python3
"""把冷采集数据里 assistant 消息的 `reasoning` 字段统一成 `reasoning_content`。

背景:2026-08-05 检查 cold_start_1429.jsonl 发现 assistant 消息同时带 `reasoning`
和 `reasoning_content` 两个字段(10776 条,其中 10674 值相同、102 条差异多为 None vs " ")。
用户要求推理字段保持一致、统一叫 `reasoning_content`。

规则:
- 同时有两者 -> 丢弃 `reasoning`(以 `reasoning_content` 为准)。
- 只有 `reasoning` -> 提升为 `reasoning_content`。
- 只有 `reasoning_content` -> 不动。
原文件先备份为 <path>.bak_before_reasoning_norm(幂等:已存在不覆盖)。
"""
import json
import os
import shutil
import sys
from pathlib import Path


def main(src: str) -> None:
    bak = src + ".bak_before_reasoning_norm"
    tmp = src + ".tmp"
    if not os.path.exists(bak):
        shutil.copy2(src, bak)
        print("backup ->", bak)

    n = dropped = kept = promoted = 0
    with open(src) as fin, open(tmp, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            msgs = rec["messages"]
            parsed = isinstance(msgs, str)
            if parsed:
                msgs = json.loads(msgs)
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                has_r = "reasoning" in m
                has_rc = "reasoning_content" in m
                if has_r and has_rc:
                    m.pop("reasoning")
                    dropped += 1
                    kept += 1
                elif has_r and not has_rc:
                    m["reasoning_content"] = m.pop("reasoning")
                    promoted += 1
                    kept += 1
                elif has_rc:
                    kept += 1
            if parsed:
                rec["messages"] = json.dumps(msgs, ensure_ascii=False)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    os.replace(tmp, src)
    print(f"records={n}  reasoning_content kept={kept}  dropped dup `reasoning`={dropped}  promoted lone={promoted}")

    bad = 0
    for line in Path(src).open():
        rec = json.loads(line)
        msgs = rec["messages"]
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        for m in msgs:
            if m.get("role") == "assistant" and "reasoning" in m:
                bad += 1
    print("VERIFY leftover bare `reasoning` keys:", bad, "(should be 0)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "datasets/cold_start/cold_start_1429.jsonl")
