#!/usr/bin/env python3
"""给 longhorizonCoding 任务打难度(带原始轨迹)。

现状:train_cl.parquet 里 241 个 LH 任务的 difficulty 是按 completion_score 粗映射
(≥0.95→5),238/241 全挤在 d5——退化,没参考价值。

本脚本改用【原始轨迹】(longhorizonCoding.jsonl 的 messages,含 tool_call/tool_result)
作为上下文,复用 score_difficulty.py 的 DIFF_PROMPT(四维度 1-10),gpt-5.6-luna 逐条打分。
轨迹里工具调用次数/失败恢复/多步依赖是难度的直接证据,比 query 文本本身更准。

流程:
  1. 读 longhorizonCoding.jsonl,LH_{i+1:06d} ↔ 第 i 行
  2. messages → 轨迹文本(traj_to_text,与 score_cold_coords 同格式)
  3. query(任务描述,取第一个 user content 归一化) + 轨迹一起送模型
  4. 写 datasources/labeled/lh_difficulty.jsonl (LH_id + difficulty),断点续跑
  5. --apply 回填 train_cl.parquet + train_cl.jsonl 里 LH 行的 difficulty

用法:
  python3 scripts/data/score_lh_difficulty.py            # 只打分,写 lh_difficulty.jsonl
  python3 scripts/data/score_lh_difficulty.py --apply     # 打分(或复用已打)+ 回填 parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LONGHORIZON_SRC = "/mnt/afs_toolcall/zhengnairong/code/data_mllm_agent/agent/longhorizonCoding/longhorizonCoding_repozero-opus5cot-hybrid_claude5opus_custom-minisweagent_linux_20260824/longhorizonCoding_repozero-opus5cot-hybrid_claude5opus_custom-minisweagent_linux_20260824.jsonl"
PARQUET = ROOT / "datasets" / "train_cl.parquet"
JSONL = ROOT / "datasets" / "train_cl.jsonl"
OUT = ROOT / "datasources" / "labeled" / "lh_difficulty.jsonl"

API_BASE = "https://tokenhub.sensetime.com/v1"
MODEL = "gpt-5.6-luna"
WORKERS = 64

# 难度 prompt(从 label_unlabeled_tasks.py 完整复制,单一信源 score_difficulty.py)
DIFF_PROMPT = """你是一个严格、客观的任务难度评估器。

你的任务是评估【任务本身的绝对完成难度】，而不是评估回答质量、任务价值、任务长度或任务是否有趣。

请根据任务在真实完成过程中所要求的能力，对其进行 1-10 的整数难度评分。

## 一、评分原则

只评估"完成这个任务需要多大能力和工作量"，不要评估：

- 用户是否重要
- 任务是否有价值
- 最终答案写得好不好
- 文本本身有多长
- 任务描述写得是否复杂
- 你是否喜欢这个任务
- 是否存在潜在的复杂性但任务实际上并不需要

如果一个任务可以通过简单、直接的方法完成，即使涉及专业领域，也不要因为"专业"本身提高分数。

如果一个任务表面描述简单，但真正完成需要多轮操作、多步推理、复杂验证或较强专业能力，应提高分数。

---

## 二、从四个维度独立判断

### 1. 工具 / 操作复杂度

评估完成任务实际需要的工具、操作数量以及工具链复杂程度。

考虑：

- 是否需要工具/API
- 需要多少种不同工具
- 是否需要连续多步调用
- 工具之间是否存在依赖关系
- 是否需要根据前一步结果动态决定下一步操作
- 是否需要处理工具失败、异常或状态变化

参考：

- 1-2：无需工具，或单次简单操作
- 3-4：1-2 次简单工具调用
- 5-6：多个工具或多个连续操作
- 7-8：多工具协同、存在明显依赖或动态决策
- 9-10：复杂工具链、长流程、状态管理、失败恢复或复杂环境操作

注意：
"有工具可用"不等于"任务需要工具"。只按照完成任务实际需要的工具复杂度评分。


### 2. 推理深度

评估完成任务需要多少层次的思考、规划和信息关联。

考虑：

- 是否需要简单判断
- 是否需要多步逻辑推理
- 是否需要拆解问题
- 是否需要多跳信息关联
- 是否需要比较多个方案
- 是否需要处理约束、冲突或不确定性
- 是否需要制定长期或复杂执行计划

参考：

- 1-2：直接检索、识别、转换或单步判断
- 3-4：少量推理，2-3 个明确步骤
- 5-6：多步推理，需要综合多个条件
- 7-8：复杂多跳推理、规划、权衡或约束处理
- 9-10：深层问题分解、复杂规划、跨步骤依赖、开放式问题求解


### 3. 领域知识要求

评估任务完成时所需的专业知识门槛。

考虑：

- 是否需要通用常识即可完成
- 是否需要某一专业领域的基础知识
- 是否需要专业规则、概念或方法
- 是否需要较深入的专业知识
- 是否需要跨领域知识综合

参考：

- 1-2：通用常识即可
- 3-4：少量基础领域知识
- 5-6：需要明确的专业知识
- 7-8：需要较深入的专业知识或多个专业领域
- 9-10：高度专业、跨领域且需要深入知识

注意：
专业知识只是一个维度。
"需要金融知识"不代表任务本身就是高难度；如果只需要查一个定义，仍然可能只有 3-4 分。


### 4. 产出复杂度

评估完成任务所要求的最终产出复杂程度，而不是文本长度。

考虑：

- 是否只需要一个简单答案
- 是否需要结构化结果
- 是否需要完整分析
- 是否需要报告、方案或复杂文档
- 是否需要代码、配置、工程产物
- 是否需要可执行、可验证的结果
- 是否需要同时满足多个格式、功能和质量约束

参考：

- 1-2：一句话、简单答案、简单转换
- 3-4：短文本、简单结构化结果
- 5-6：较完整分析、脚本、结构化文档
- 7-8：复杂报告、完整代码、复杂方案或需要验证的产出
- 9-10：完整工程/系统、复杂架构、可部署产物或同时满足大量约束的复杂交付物

注意：
"文章很长"不等于"产出复杂"。只有当内容组织、正确性、结构、约束或可执行性显著增加难度时，才提高评分。


## 三、总体评分方法

先在脑中分别评估四个维度，再综合得到最终难度。

不要简单地取四个维度的平均值。

总体难度应重点考虑：

1. 是否存在真正的多步骤依赖；
2. 是否需要复杂推理或规划；
3. 是否存在较高的专业知识门槛；
4. 是否需要复杂、可验证或可执行的最终产出；
5. 四个维度之间是否存在叠加效应。

特别注意：

- 只有一个维度很高、其他维度都很低时，不要轻易给 8-10。
- 多个维度同时达到中高水平时，才应进入 7-8。
- 9-10 只用于真正具有高复杂度、强依赖、强专业性或开放式问题求解特征的任务。
- 简单任务即使描述很长，也可以是 1-3 分。
- 复杂任务即使描述很短，也可以是 8-10 分。


## 四、最终难度锚点

### 1-2：极简单
单步完成，几乎无需推理、工具或专业知识。

例如：
- 查询一个简单事实
- 简单翻译
- 格式转换
- 简单计算


### 3-4：简单
存在少量步骤或基础专业知识，但路径明确，基本没有复杂决策。

例如：
- 查找一个文件并摘要
- 安装一个软件包并验证
- 根据明确要求写一个简单脚本


### 5-6：中等
需要多步处理、一定推理、多个信息源或明确的专业知识。

例如：
- 分析日志定位常见问题
- 根据需求编写并测试脚本
- 综合多个资料完成分析


### 7-8：困难
存在明显的多步骤依赖、多工具协同、复杂推理、专业知识或高质量交付要求。

例如：
- 完成多源调研并形成结构化报告
- 分析复杂系统故障并提出修复方案
- 实现一个具有多个功能和约束的软件模块并测试


### 9-10：极难
只有在任务确实需要架构级规划、深层推理、跨领域综合、复杂环境操作、创造性问题求解或高复杂度工程交付时使用。

例如：
- 从零设计并实现复杂系统，同时完成测试和部署
- 解决开放式、跨领域且没有明确标准解的问题
- 设计复杂架构并处理大量相互依赖的约束


## 五、评分校准规则

请严格避免以下情况：

- 不要默认给 5 分
- 不要因为任务"看起来专业"就给高分
- 不要因为任务描述很长就给高分
- 不要因为需要多个步骤就直接给 7+
- 不要为了让 1-10 每个数字都出现而人为调整分数
- 不要偏好偶数，也不要偏好奇数
- 只根据任务的实际难度选择最合适的整数

对于相近难度的任务，应保持评分标准一致。

最终分数必须反映任务的绝对难度，而不是相对于其他任务的排名。

## 六、输出格式

只输出 JSON，不要输出任何解释：

{"difficulty": <1-10整数>}"""


def load_key() -> str:
    key = os.environ.get("TOKENHUB_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("TOKENHUB_API_KEY="):
                    key = s.split("=", 1)[1].strip().strip("\"'")
                    break
    if not key:
        sys.exit("ERROR: TOKENHUB_API_KEY not set (see .env)")
    return key


def traj_to_text(msgs) -> str:
    """轨迹 messages → 文本(与 score_cold_coords.py 同格式,含 tool_call/result)。"""
    if not isinstance(msgs, list):
        return ""
    lines = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            lines.append(f"[user] {(m.get('content', '') or '')[:2000]}")
        elif role == "assistant":
            c = m.get("content") or ""
            if c:
                lines.append(f"[assistant] {c[:600]}")
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                lines.append(f"[tool_call:{fn.get('name', '?')}] {str(fn.get('arguments', ''))[:500]}")
        elif role == "tool":
            lines.append(f"[tool_result:{m.get('name', '?')}] {(m.get('content', '') or '')[:300]}")
    return "\n".join(lines)


def parse_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        text = text[s : e + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def load_lh_tasks() -> list[dict]:
    """读 longhorizonCoding.jsonl → [{rid, traj_text, n_steps}]。"""
    tasks = []
    with open(LONGHORIZON_SRC, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            rid = f"LH_{i + 1:06d}"
            msgs = d.get("messages", [])
            traj = traj_to_text(msgs)
            if traj:
                tasks.append({"rid": rid, "traj": traj, "n_steps": d.get("total_steps", len(msgs))})
    return tasks


def score_all(tasks: list[dict], key: str, limit: int | None) -> dict[str, int]:
    """打分,断点续跑(已在 OUT 里的 rid 跳过)。返回 rid→difficulty。"""
    done: dict[str, int] = {}
    if OUT.is_file():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("difficulty"):
                        done[d["rid"]] = int(d["difficulty"])
                except Exception:  # noqa: BLE001
                    continue
    remaining = [t for t in tasks if t["rid"] not in done]
    if limit:
        remaining = remaining[:limit]
    print(f"[difficulty] {len(remaining)} 待打 ({len(done)} 已打)", flush=True)
    if not remaining:
        return done

    OUT.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(120.0))
    fout = open(OUT, "a", encoding="utf-8")
    ok = fail = 0
    t0 = time.time()

    def score_one(task):
        user = f"任务执行轨迹(含工具调用与结果):\n\n{task['traj'][:16000]}\n\n{DIFF_PROMPT}"
        for attempt in range(3):
            try:
                resp = client.post(
                    f"{API_BASE}/chat/completions",
                    json={"model": MODEL, "messages": [{"role": "user", "content": user}], "max_tokens": 256, "temperature": 0.0},
                    timeout=120.0,
                )
                resp.raise_for_status()
                parsed = parse_json(resp.json()["choices"][0]["message"]["content"])
                diff = parsed.get("difficulty")
                if diff is not None:
                    return task["rid"], max(1, min(10, int(float(diff))))
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (attempt + 1))
        return task["rid"], None

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for rid, diff in ex.map(score_one, remaining):
            if diff is not None:
                ok += 1
                done[rid] = diff
                fout.write(json.dumps({"rid": rid, "difficulty": diff}, ensure_ascii=False) + "\n")
                fout.flush()
            else:
                fail += 1
            n = ok + fail
            if n % 50 == 0 or n == len(remaining):
                el = time.time() - t0
                print(f"  {n}/{len(remaining)} ok={ok} fail={fail} rate={n / el:.1f}/s", flush=True)

    fout.close()
    print(f"[difficulty] done: ok={ok} fail={fail}", flush=True)
    return done


def apply_to_parquet(rid2diff: dict[str, int]) -> None:
    """回填 train_cl.parquet + train_cl.jsonl 里 LH 行的 difficulty。"""
    t = pq.read_table(str(PARQUET))
    rows = t.to_pylist()
    changed = 0
    for r in rows:
        ei = r["extra_info"]
        gid = ei.get("gen_task_id", "")
        if gid.startswith("LH_") and gid in rid2diff:
            new = str(rid2diff[gid])
            if ei.get("difficulty") != new:
                ei["difficulty"] = new
                changed += 1
    print(f"回填 {changed} 行 LH difficulty", flush=True)

    new_t = pa.Table.from_pylist(rows, schema=t.schema)
    tmp = PARQUET.with_suffix(".parquet.tmp")
    pq.write_table(new_t, tmp)
    tmp.replace(PARQUET)
    tmp_j = JSONL.with_suffix(".jsonl.tmp")
    with open(tmp_j, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_j.replace(JSONL)
    print(f"✅ 落地 {PARQUET.name} + {JSONL.name}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="打分后回填 parquet")
    ap.add_argument("--limit", type=int, default=None, help="只打前 N 条(调试)")
    args = ap.parse_args()

    key = load_key()
    tasks = load_lh_tasks()
    print(f"longhorizonCoding: {len(tasks)} 条(带轨迹)", flush=True)

    rid2diff = score_all(tasks, key, args.limit)

    # 分布
    c = Counter(rid2diff.values())
    print("\n=== 难度分布(全 LH)===", flush=True)
    for d in sorted(c):
        print(f"  d{d}: {c[d]}", flush=True)

    if args.apply:
        apply_to_parquet(rid2diff)
    else:
        print("\n加 --apply 回填 parquet", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
