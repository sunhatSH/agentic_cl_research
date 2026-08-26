#!/usr/bin/env python3
"""改造 sweCoding 任务的 query + GT。

query 改造:
  - 去掉 <uploaded_files> 标签、SFT 引导语、行为约束(Make minimal / Solve task / git log)
  - 保留 issue + Interface hints + Additional requirements
  - /repo → ./inputs(训练时文件注入 ./inputs)
  - 加引导: "读 ./inputs/ 目录下的代码仓库,实现以下请求的改动"

GT 改造(生成 answer_key.json):
  - type=swe, rubric 形式
  - rubric = Interface hints 条目 + Additional requirements 条目
  - /repo → ./inputs

复用 rebuild_swecoding.py 的 rebuild_files 确定 SWE 序号(跳过无文件的样本,保证与已建 files 目录一致)。

用法: python3 scripts/data/rebuild_swe_query_gt.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

ROOT = Path("/mnt/afs_toolcall/sunhao4/workspace/agentic_cl_research")
OUT = ROOT / "datasources" / "taskspecs_w3"

SRC_LIST = [
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_opus5-reward1-cot-synth-pipelinev3_claude5opus_hermes-openhands-sdk1391_linux_20260824/sweCoding_opus5-reward1-cot-synth-pipelinev3_claude5opus_hermes-openhands-sdk1391_linux_20260824.jsonl",
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_opus5-reward1-cot-2batch_claude5opus_hermes-openhands-sdk1391_linux_20260814/sweCoding_opus5-reward1-cot-2batch_claude5opus_hermes-openhands-sdk1391_linux_20260814.jsonl",
    "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/sweCoding/sweCoding_rebench-v2-reward1_claude5opus_hermes_linux_20260807/sweCoding_rebench-v2-reward1_claude5opus_hermes_linux_20260807.jsonl",
]


def rebuild_files(msgs: list) -> tuple[OrderedDict, OrderedDict]:
    """同 rebuild_swecoding.py: 从轨迹重建文件,判断样本是否有文件。"""
    calls = {}
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]
                if fn["name"] == "file_editor":
                    args = fn.get("arguments", {})
                    if isinstance(args, dict):
                        calls[tc["id"]] = (args.get("command", ""), args.get("path", ""), args)
    results = {}
    for m in msgs:
        if m["role"] == "tool":
            results[m.get("tool_call_id", "")] = m.get("content", "")
    initial = OrderedDict()
    created = OrderedDict()
    for cid, (cmd, path, args) in calls.items():
        if not path or not path.startswith("/repo/"):
            continue
        rel = path[len("/repo/"):]
        if cmd == "create":
            created[rel] = args.get("file_text", "")
        elif cmd in ("view", "open"):
            res = results.get(cid, "")
            if res and "Error" not in res and not res.startswith("File "):
                initial.setdefault(rel, res)
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc["function"]
                if fn["name"] != "terminal":
                    continue
                args = fn.get("arguments", {})
                if not isinstance(args, dict):
                    continue
                cmd_str = args.get("command", "")
                res = results.get(tc["id"], "")
                if not res:
                    continue
                paths = re.findall(r"/repo/[\w./\-]+", cmd_str)
                if not paths:
                    paths = re.findall(r"(?:cat|sed\s+-n\s+[\d,]+p|head\s+-\d+)\s+([\w./\-]+\.\w+)", cmd_str)
                if len(paths) == 1:
                    p = paths[0]
                    rel = p[len("/repo/"):] if p.startswith("/repo/") else p
                    initial.setdefault(rel, res)
                elif len(paths) > 1:
                    parts = re.split(r"={3,}|-{5,}", res)
                    parts = [p.strip() for p in parts if p.strip()]
                    if len(parts) == len(paths):
                        for p, content in zip(paths, parts):
                            rel = p[len("/repo/"):] if p.startswith("/repo/") else p
                            initial.setdefault(rel, content)
    return initial, created


def replace_repo(text: str) -> str:
    """把真正的 /repo 目录引用 → ./inputs,不误伤 report/repository 等含 repo 的词。

    只匹配 /repo 后面跟 /、空白、引号、反引号、标点、结尾的(目录引用),
    不匹配 /report(后面跟 rt)。"""
    return re.sub(r"/repo(?=[/\s'\"`.,;:)\]}，。；：）】]|$)", "/home/user/workspace", text)


def parse_hint(block: str):
    """解析一条 interface hint,兼容两种格式,返回 (kind, name, path, inp, outp, desc) 或 None。

    格式1(标准): Kind / name / path / Input: ... / Output: ... / desc
    格式2(带字段名): Type: Kind / Name: `x` / Path: `x` / Input: ... / Output: ... / Description: ...

    用正则逐字段提取(字段值到下一个字段名或结尾),避免字段值里的 " / " 误分割。
    """
    block = block.strip().lstrip("- ").strip()
    if not block:
        return None

    has_field = bool(re.search(r"(Type|Kind|Name|Path|Description):", block))

    if has_field:
        def field(name: str) -> str:
            # 字段值到下一个 " / 字段名:" 或结尾,反引号剥离
            m = re.search(
                rf"(?:^|\s/\s){name}:\s*(.+?)(?=\s/\s(?:Type|Kind|Name|Path|Input|Output|Description):|$)",
                block,
            )
            return m.group(1).strip().strip("`") if m else ""

        kind = field("Type") or field("Kind")
        name = field("Name")
        path = field("Path")
        inp = field("Input")
        outp = field("Output")
        desc = field("Description")
    else:
        # 标准格式: Kind / name / path / Input: ... / Output: ... / desc
        parts = [p.strip() for p in block.split(" / ")]
        kind = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        path = parts[2] if len(parts) > 2 else ""
        inp = outp = desc = ""
        for p in parts[3:]:
            if p.startswith("Input:"):
                inp = p.split(":", 1)[1].strip()
            elif p.startswith("Output:"):
                outp = p.split(":", 1)[1].strip()
            else:
                desc = (desc + " " + p).strip()
    return kind, name, path, inp, outp, desc


def add_inputs_prefix(hints_text: str) -> str:
    """把 Interface hints 每条 hint 的 path(相对 repo 根)加 ./inputs/ 前缀,和文件注入目录对齐。"""
    out = []
    for block in hints_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        r = parse_hint(block)
        if r is None:
            out.append(block)
            continue
        kind, name, path, inp, outp, desc = r
        if not path:
            out.append(block)
            continue
        # path 加 /home/user/workspace/ 前缀,反引号原样保留(带字段名格式有反引号)
        if "`" in path:
            path = "`/home/user/workspace/" + path.strip("`") + "`"
        else:
            path = "/home/user/workspace/" + path.lstrip("/")
        # 按原格式重新拼回
        if "Type:" in block or "Name:" in block or "Path:" in block:
            seg = [f"Type: {kind}", f"Name: `{name}`" if name else f"Name: ",
                   f"Path: {path}"]
            if inp: seg.append(f"Input: {inp}")
            if outp: seg.append(f"Output: {outp}")
            if desc: seg.append(f"Description: {desc}")
            out.append("- " + " / ".join(seg))
        else:
            seg = [kind, name, path]
            if inp: seg.append(f"Input: {inp}")
            if outp: seg.append(f"Output: {outp}")
            if desc: seg.append(desc)
            out.append(" / ".join(seg))
    return "\n\n".join(out)


def build_query(content: str) -> str:
    """改造 query: 去标签/引导/行为约束, /repo→./inputs, 保留 issue+hints+requirements。"""
    # 提取 issue
    issue = ""
    mi = re.search(r"<issue_description>(.*?)</issue_description>", content, re.DOTALL)
    if mi:
        issue = mi.group(1).strip()
    else:
        issue = content.strip()

    # 提取 interface hints
    hints = ""
    hi = content.find("Interface hints:")
    hj = content.find("Additional requirements:")
    if hi >= 0 and hj > hi:
        hints = content[hi + len("Interface hints:"):hj].strip()

    # 提取 requirements
    req = ""
    hk = content.find("Make the minimal")
    if hj >= 0:
        req = content[hj + len("Additional requirements:"):hk if hk > hj else None].strip()

    # /repo → ./inputs(issue/req 里的绝对路径引用)
    issue = replace_repo(issue)
    req = replace_repo(req)
    # hints 的 path 加 ./inputs/ 前缀(相对路径,没有 /repo 前缀,单独处理)
    hints = add_inputs_prefix(hints)

    parts = ["读 /home/user/workspace/ 目录下的代码仓库，实现以下请求的改动：", "", issue]
    if hints:
        parts += ["", "Interface hints:", hints]
    if req:
        parts += ["", "Additional requirements:", req]
    return "\n".join(parts)


def parse_interface_hints(text: str) -> list[str]:
    """Interface hints → rubric 条目。兼容标准 + 带字段名两种格式。"""
    rubric = []
    for block in text.split("\n\n"):
        r = parse_hint(block)
        if r is None:
            continue
        kind, name, path, inp, outp, desc = r
        if not name and not desc:
            continue
        path = "/home/user/workspace/" + path.lstrip("/") if path else ""
        sig = ""
        if inp or outp:
            sig = f"；签名 Input: {inp}, Output: {outp}" if inp else f"；输出: {outp}"
        rubric.append(f"实现{kind} {name}（{path}）：{desc}{sig}".strip())
    return rubric


def parse_requirements(text: str) -> list[str]:
    rubric = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s*(.+)$", line)
        if m:
            rubric.append(m.group(1).strip())
    return rubric


def main() -> int:
    idx = 0
    n_query = 0
    n_gt = 0
    for src in SRC_LIST:
        with open(src) as f:
            for line in f:
                d = json.loads(line)
                msgs = d["messages"]
                content = ""
                for m in msgs:
                    if m["role"] == "user":
                        content = m["content"]
                        break
                if not content:
                    continue
                # 判断是否有文件(与 rebuild_swecoding.py 一致)
                initial, created = rebuild_files(msgs)
                if not initial and not created:
                    continue
                idx += 1
                rid = f"SWE_{idx:06d}"
                d_out = OUT / rid
                if not d_out.is_dir():
                    continue

                # 改造 query
                query = build_query(content)
                ts = yaml.safe_load((d_out / "taskspec.yaml").read_text(encoding="utf-8"))
                ts["seed_query"] = query
                ts["task_family"] = "sweCoding"
                ts["source"] = "sweCoding"
                (d_out / "taskspec.yaml").write_text(
                    yaml.safe_dump(ts, allow_unicode=True, sort_keys=False), encoding="utf-8")
                n_query += 1

                # 生成 GT(answer_key.json, rubric 形式)
                hints_text = ""
                hi = content.find("Interface hints:")
                hj = content.find("Additional requirements:")
                if hi >= 0 and hj > hi:
                    hints_text = content[hi + len("Interface hints:"):hj]
                req_text = ""
                hk = content.find("Make the minimal")
                if hj >= 0:
                    req_text = content[hj + len("Additional requirements:"):hk if hk > hj else None]
                # hints 的 path 由 parse_interface_hints 精确加前缀(不再 replace /repo,避免误伤 report)
                req_text = replace_repo(req_text)
                rubric = parse_interface_hints(hints_text) + parse_requirements(req_text)
                if rubric:
                    title = ""
                    mti = re.search(r"## Title\s*\n(.+)", content)
                    if mti:
                        title = mti.group(1).strip()
                    ak = {"type": "swe", "title": title, "rubric": rubric}
                    (d_out / "answer_key.json").write_text(
                        json.dumps(ak, ensure_ascii=False, indent=2), encoding="utf-8")
                    n_gt += 1
        print(f"{Path(src).name}: 累计 query={n_query}, gt={n_gt}", flush=True)

    print(f"\n✅ 改造 query={n_query}, 生成 GT={n_gt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
