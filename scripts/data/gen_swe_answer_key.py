#!/usr/bin/env python3
"""给 sweCoding 任务生成 answer_key.json(rubric 形式,SWE 专用 GT)。

从源 jsonl 的 user message 提取:
  - Interface hints    → rubric 条目(实现哪个接口/常量,签名是什么)
  - Additional requirements → rubric 条目(验收标准)

生成格式:
  {"type": "swe", "rubric": ["实现函数 decodeOsc52ClipboardData(...) ...", "...验收标准..."]}

judge 端(_load_ground_truth)识别 type=="swe" 走 rubric 逻辑,判"满足多少条验收标准"。

用法: python3 scripts/data/gen_swe_answer_key.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research")
OUT = ROOT / "datasources" / "taskspecs_w3"

SRC_LIST = [
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_opus5-reward1-cot-synth-pipelinev3_claude5opus_hermes-openhands-sdk1391_linux_20260824/sweCoding_opus5-reward1-cot-synth-pipelinev3_claude5opus_hermes-openhands-sdk1391_linux_20260824.jsonl",
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_opus5-reward1-cot-2batch_claude5opus_hermes-openhands-sdk1391_linux_20260814/sweCoding_opus5-reward1-cot-2batch_claude5opus_hermes-openhands-sdk1391_linux_20260814.jsonl",
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_rebench-v2-reward1_claude5opus_hermes_linux_20260807/sweCoding_rebench-v2-reward1_claude5opus_hermes_linux_20260807.jsonl",
]


def parse_interface_hints(text: str) -> list[str]:
    """把 Interface hints 文本转成 rubric 条目。"""
    rubric = []
    # 每条 hint 是 "Kind / name / path / Input: ... / Output: ... / desc" 一段(空行分隔)
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # 找 Kind 开头
        m = re.match(r"^(Function|Constant|Method|Class|Type|Interface|Variable|Enum)\s*/\s*(.+)$", block, re.DOTALL)
        if not m:
            continue
        kind = m.group(1)
        parts = [p.strip() for p in m.group(2).split("/")]
        if len(parts) < 2:
            continue
        name = parts[0]
        path = parts[1]
        # 剩余部分含 Input/Output/desc
        rest = "/".join(parts[2:]) if len(parts) > 2 else ""
        inp = ""
        outp = ""
        desc = ""
        mi = re.search(r"Input:\s*([^/]+?)(?:\s*/|$)", rest)
        mo = re.search(r"Output:\s*([^/]+?)(?:\s*/|$)", rest)
        if mi:
            inp = mi.group(1).strip()
        if mo:
            outp = mo.group(1).strip()
        # desc 是最后一段
        if "Output:" in rest:
            desc = rest.split("Output:", 1)[1]
            desc = re.sub(r"^[^/]*?/", "", desc).strip()
        sig = ""
        if inp or outp:
            sig = f" 签名 Input: {inp}, Output: {outp}" if inp else f" 输出: {outp}"
        rubric.append(f"实现{kind} {name}（{path}）：{desc}{sig}".strip())
    return rubric


def parse_requirements(text: str) -> list[str]:
    """把 Additional requirements 文本转成 rubric 条目。"""
    rubric = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            rubric.append(m.group(1).strip())
    return rubric


def main() -> int:
    n = 0
    idx = 0
    for src in SRC_LIST:
        with open(src) as f:
            for line in f:
                d = json.loads(line)
                msgs = d["messages"]
                # 找 user content
                content = ""
                for m in msgs:
                    if m["role"] == "user":
                        content = m["content"]
                        break
                if not content:
                    continue

                # 提取 issue title
                title = ""
                mti = re.search(r"## Title\s*\n(.+)", content)
                if mti:
                    title = mti.group(1).strip()

                # 提取 interface hints
                hints_text = ""
                mi = content.find("Interface hints:")
                mj = content.find("Additional requirements:")
                if mi >= 0 and mj > mi:
                    hints_text = content[mi + len("Interface hints:"):mj]

                # 提取 additional requirements
                req_text = ""
                mk = content.find("Make the minimal")
                if mj >= 0:
                    req_text = content[mj + len("Additional requirements:"):mk if mk > mj else None]

                rubric = parse_interface_hints(hints_text) + parse_requirements(req_text)
                if not rubric:
                    continue

                idx += 1
                rid = f"SWE_{idx:06d}"
                ak = {"type": "swe", "title": title, "rubric": rubric}
                d_out = OUT / rid
                if d_out.is_dir():
                    (d_out / "answer_key.json").write_text(
                        json.dumps(ak, ensure_ascii=False, indent=2), encoding="utf-8")
                    n += 1
        print(f"{Path(src).name}: 累计 {n} 个 answer_key", flush=True)

    print(f"\n✅ 生成 {n} 个 SWE answer_key.json (rubric 形式)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
