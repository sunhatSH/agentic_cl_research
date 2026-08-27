#!/usr/bin/env python3
"""用 longhorizonCoding 替换有问题的 SWE 任务。

流程:
  1. 剔除 train_cl.parquet 里 241 个有问题的 SWE 任务(query 引用文件但 files/ 没有)
  2. 从 longhorizonCoding.jsonl 解析 523 条任务(C++→Rust / Python→JS 迁移),全 coding 桶
  3. 格式对齐:
     - query: Source 段(内嵌源码) + Requirements,路径 /workspace/dataset/ → /home/user/workspace/,/output → /home/user/workspace/
     - 文件: 源码内嵌在 query 里,不需要外部 files/(产出型,agent 从 query 读源码写迁移代码)
     - answer_key: type=longhorizon, rubric=Requirements 条目 + Test Cases(输入+期望输出)
     - record_id = LH_xxxxxx, gen_task_id = LH_xxxxxx
  4. 补到 coding 桶(优先 d4-6,其次 d7,用 longhorizonCoding 的 completion_score 当难度参考)
  5. 重建 parquet

用法: python3 scripts/data/replace_swe_with_longhorizon.py --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))
from path_normalize import normalize_paths  # noqa: E402

PARQUET = ROOT / "datasets" / "train_cl.parquet"
JSONL = ROOT / "datasets" / "train_cl.jsonl"
TASKSPECS = ROOT / "datasources" / "taskspecs_w3"
LONGHORIZON_SRC = "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/longhorizonCoding/longhorizonCoding_repozero-opus5cot-hybrid_claude5opus_custom-minisweagent_linux_20260824/longhorizonCoding_repozero-opus5cot-hybrid_claude5opus_custom-minisweagent_linux_20260824.jsonl"


def find_bad_swe(rows: list) -> set[str]:
    """找出有问题的 SWE D_id(query 引用文件但 files/ 没有)。"""
    bad = set()
    for r in rows:
        ei = r["extra_info"]
        gid = ei.get("gen_task_id") or ""
        if not gid.startswith("SWE_"):
            continue
        q = str((ei.get("queries") or [""])[0])
        p = TASKSPECS / gid / "files"
        has_files = p.is_dir() and bool(list(p.iterdir()))
        actual = set(os.listdir(p)) if has_files else set()
        refs = set(re.findall(
            r"[\w\-]+\.(?:csv|xlsx|xls|json|jsonl|sqlite|txt|pdf|docx|tsv|parquet|db|md|py|ts|js|go|rs|java|cpp|c|rb|sh|yml|yaml|toml|ini|cfg|xml|html|css|sql)\b",
            q, re.IGNORECASE))
        refs = {r for r in refs if not any(c in r for c in "*?") and len(r) > 3}
        missing = refs - actual
        real_missing = set()
        for m in missing:
            idx = q.find(m)
            if idx >= 0:
                ctx = q[max(0, idx - 40):idx + len(m) + 10].lower()
                if not any(k in ctx for k in ["write", "output", "save", "create", "generate", "produce", "写到", "输出", "保存", "生成", "创建"]):
                    real_missing.add(m)
        if real_missing or not has_files:
            bad.add(gid)
    return bad


def normalize_lh_query(content: str) -> str:
    """归一化 longhorizonCoding 的 user content。

    /workspace/dataset/ → /home/user/workspace/
    /output → /home/user/workspace/
    去掉 harness 引导语(Command Execution Rules / submit 等)
    """
    s = content
    # 路径归一化: /workspace/dataset/xxx → /home/user/workspace/xxx
    # 用负向后视 (?<!home/user) 避免二次匹配 /home/user/workspace/ 里的 /workspace/
    s = re.sub(r"/workspace/dataset/", "/home/user/workspace/", s)
    s = re.sub(r"(?<!home/user)/workspace/", "/home/user/workspace/", s)
    s = re.sub(r"(?<!home/user)/output(?=[/\s'\"`.,;:)\]}]|$)", "/home/user/workspace", s)
    # 去掉 harness 引导语(从 "## Command Execution Rules" 到结尾)
    s = re.sub(r"\n## Command Execution Rules.*$", "", s, flags=re.DOTALL)
    # 去掉 submit 引导
    s = re.sub(r"\nAfter implementing.*?(?:COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.*?)(?:\n|$)", "", s, flags=re.DOTALL)
    s = re.sub(r"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "", s)
    return s.strip()


def extract_test_cases(content: str) -> list[dict]:
    """从 user content 提取 Test Cases(输入参数 + 期望输出)→ GT checks。"""
    checks = []
    # C2Rust 格式: --- Example N ---\nInput Args: {}\nProgram Output: 4
    for m in re.finditer(
        r"---\s*Example\s+(\d+)\s*---\s*\n\s*Input Args:\s*(.+?)\n\s*Program Output:\s*(.+?)(?:\n|$)",
        content, re.DOTALL
    ):
        checks.append({
            "question": f"Example {m.group(1)}: Input Args={m.group(2).strip()}, expected output?",
            "answer": m.group(3).strip(),
        })
    # Py2JS 格式: # ===== Test Case N =====\n# Input:\n#   --a = 'xxx'\n# Output:\n#   b'yyy'\n# Return Code: 0
    for m in re.finditer(
        r"#\s*=+\s*\n#\s*Test Case\s+(\d+)\s*\n#\s*=+\s*\n#\s*Input:\s*\n(.*?)#\s*Output:\s*\n(.*?)(?:#\s*Return Code:|#$|\Z)",
        content, re.DOTALL
    ):
        inp = m.group(2).strip().replace("#", "").strip()
        out = m.group(3).strip().replace("#", "").strip()
        checks.append({
            "question": f"Test Case {m.group(1)}: Input={inp}, expected output?",
            "answer": out,
        })
    return checks


def extract_requirements(content: str) -> list[str]:
    """从 user content 提取 Requirements → rubric 条目。"""
    rubric = []
    # 找 Requirements 段
    mr = content.find("Requirements:")
    if mr < 0:
        mr = content.find("Task Requirements:")
    if mr < 0:
        return rubric
    req_seg = content[mr:]
    # 去掉 harness 引导
    req_seg = re.sub(r"\n## Command Execution Rules.*$", "", req_seg, flags=re.DOTALL)
    # 提取编号条目
    for m in re.finditer(r"^\d+\.\s*\*\*(.+?)\*\*:\s*(.+?)(?=\n\d+\.|\Z)", req_seg, re.DOTALL | re.MULTILINE):
        rubric.append(f"{m.group(1).strip()}: {m.group(2).strip()}")
    return rubric


def build_lh_tasks() -> list[dict]:
    """从 longhorizonCoding.jsonl 解析任务,格式对齐。"""
    tasks = []
    with open(LONGHORIZON_SRC, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            rid = f"LH_{i+1:06d}"
            # user content
            content = ""
            for m in d["messages"]:
                if m["role"] == "user":
                    content = m["content"]
                    break
            if not content:
                continue

            # query 归一化
            query = normalize_lh_query(content)

            # GT: Test Cases(checks) + Requirements(rubric)
            checks = extract_test_cases(content)
            requirements = extract_requirements(content)

            # 难度: 用 completion_score 做参考(1.0=成功,映射到 d5-6)
            cs = d.get("rounds_checker", {}).get("quality_difficulty", {}).get("completion", {}).get("completion_score", 1.0)
            # completion_score 1.0 → d5(中等), 0.8-1.0 → d6, <0.8 → d7
            if cs >= 0.95:
                difficulty = 5
            elif cs >= 0.8:
                difficulty = 6
            else:
                difficulty = 7

            # answer_key
            ak = {
                "type": "longhorizon",
                "checks": checks,
                "rubric": requirements,
            }

            tasks.append({
                "rid": rid,
                "query": query,
                "difficulty": difficulty,
                "answer_key": ak,
                "category": d.get("metadata", {}).get("category_original", ""),
            })
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    # 1. 加载当前 parquet
    t = pq.read_table(str(PARQUET))
    rows = t.to_pylist()
    print(f"当前 parquet: {len(rows)} 行", flush=True)

    # 2. 找有问题的 SWE
    bad_swe = find_bad_swe(rows)
    print(f"有问题的 SWE: {len(bad_swe)} 个(剔除)", flush=True)

    # 3. 剔除
    kept = [r for r in rows if r["extra_info"].get("gen_task_id", "") not in bad_swe]
    print(f"剔除后: {len(kept)} 行", flush=True)

    # 4. 解析 longhorizonCoding
    lh_tasks = build_lh_tasks()
    print(f"longhorizonCoding: {len(lh_tasks)} 条(全 coding)", flush=True)

    # 5. 补到 coding 桶(补到 6400)
    coding_now = sum(1 for r in kept if r["bucket"] == "coding")
    need = 6400 - coding_now
    print(f"coding 当前: {coding_now}, 需补: {need}", flush=True)

    lh_to_add = lh_tasks[:need]

    # 6. 构建 longhorizon 行
    lh_rows = []
    for task in lh_to_add:
        rid = task["rid"]
        q = task["query"]
        # 写 answer_key.json
        ak_path = TASKSPECS / rid / "answer_key.json"
        ak_path.parent.mkdir(parents=True, exist_ok=True)
        ak_path.write_text(json.dumps(task["answer_key"], ensure_ascii=False, indent=2), encoding="utf-8")

        lh_rows.append({
            "prompt": [{"role": "user", "content": q}],
            "data_source": "agentic_cl",
            "reward_model": {
                "ground_truth": "",
                "style": "rule",
                "reward_fn": {"_function_name": "trainer.model_reward_omni.compute_score"},
            },
            "bucket": "coding",
            "extra_info": {
                "record_id": rid,
                "bucket": "coding",
                "queries": [q],
                "persona": "",
                "available_tools": [],
                "missing_info_slots": [],
                "safety_constraints": [],
                "difficulty": str(task["difficulty"]),
                "gen_task_id": rid,
                "ws_dir": "",
            },
        })

    # 7. 合并
    final_rows = kept + lh_rows
    # round to batch 32
    n = (len(final_rows) // 32) * 32
    final_rows = final_rows[:n]

    # 统计
    c = Counter(r["bucket"] for r in final_rows)
    print(f"\n=== 最终 ===", flush=True)
    print(f"总行数: {len(final_rows)} ({len(final_rows)//32} steps)", flush=True)
    for b in ("coding", "office"):
        print(f"  {b}: {c[b]} rows ({c[b]//32} steps)", flush=True)

    if not args.apply:
        print("\n加 --apply 落地", flush=True)
        return 0

    new_t = pa.Table.from_pylist(final_rows, schema=t.schema)
    tmp = PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(PARQUET)
    tmp_j = JSONL.with_suffix(".jsonl.tmp")
    with open(tmp_j, "w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_j.replace(JSONL)
    print(f"\n✅ 落地 {PARQUET.name} + {JSONL.name} ({len(final_rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
