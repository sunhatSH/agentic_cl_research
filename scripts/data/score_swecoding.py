#!/usr/bin/env python3
"""给 sweCoding 任务(issue 当 query)打桶标 + 难度标。

query = taskspecs_w3/SWE_xxx/taskspec.yaml 的 seed_query(GitHub issue 原文)。
复用原始 prompt:
  - 分桶: label_research_ops.py 的 SYSTEM_PROMPT(9 桶)
  - 难度: score_difficulty.py 的完整 DIFF_PROMPT(四维度 1-10)
单模型 gpt-5.6-luna,两阶段并行,断点续。

输出: datasources/labeled/swecoding_labeled.jsonl (SWE_id + bucket + difficulty)

用法:
  python3 scripts/data/score_swecoding.py            # 全量两阶段并行
  python3 scripts/data/score_swecoding.py --limit 20 # 冒烟
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASKSPECS = ROOT / "datasources" / "taskspecs_w3"
BUCKET_OUT = ROOT / "datasources" / "labeled" / "swecoding_bucket.jsonl"
DIFF_OUT = ROOT / "datasources" / "labeled" / "swecoding_difficulty.jsonl"
MERGED_OUT = ROOT / "datasources" / "labeled" / "swecoding_labeled.jsonl"

API_BASE = "https://tokenhub.sensetime.com/v1"
MODEL = "gpt-5.6-luna"
WORKERS = 64

BUCKET_DEFS = {
    "workflow": "多步骤工作流编排", "ops": "系统操作(文件读写/命令行/系统运维)",
    "qa": "问答检索(查事实/答问题/阅读理解)", "finance": "财务金融(贷款/税务/估值/采购)",
    "office": "办公文档(表格/报表/办公数据分析)", "communication": "沟通表达(邮件/内容创作/润色/翻译)",
    "safety": "安全合规(拒绝不安全请求/漏洞评估/合规审查)", "coding": "代码(编写/审查/调试/修复)",
    "research": "研究综合(多源信息综合成报告/简报)",
}
CANONICAL = list(BUCKET_DEFS.keys())

BUCKET_SYSTEM = (
    "你是能力分类器。给定一个agent任务，从下面的固定能力桶里选【唯一一个】最匹配的。\n"
    "只输出 JSON: {\"bucket\": \"<桶名>\", \"reason\": \"一句话理由\"}，不要markdown。\n\n"
    "## 能力桶（按能力划分，不是话题）\n"
    + "\n".join(f"- {b}: {d}" for b, d in BUCKET_DEFS.items())
    + "\n\n## 规则\n"
    "- 只选一个bucket，必须是上面列表里的英文桶名之一。\n"
    "- 按'完成任务所需的核心本领'选，不要按业务话题。\n"
)

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

先在脑中分别评估四个维度，再综合得到最终难度。不要简单地取四个维度的平均值。
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
例如：查询一个简单事实、简单翻译、格式转换、简单计算。

### 3-4：简单
存在少量步骤或基础专业知识，但路径明确，基本没有复杂决策。
例如：查找一个文件并摘要、安装一个软件包并验证、根据明确要求写一个简单脚本。

### 5-6：中等
需要多步处理、一定推理、多个信息源或明确的专业知识。
例如：分析日志定位常见问题、根据需求编写并测试脚本、综合多个资料完成分析。

### 7-8：困难
存在明显的多步骤依赖、多工具协同、复杂推理、专业知识或高质量交付要求。
例如：完成多源调研并形成结构化报告、分析复杂系统故障并提出修复方案、实现一个具有多个功能和约束的软件模块并测试。

### 9-10：极难
只有在任务确实需要架构级规划、深层推理、跨领域综合、复杂环境操作、创造性问题求解或高复杂度工程交付时使用。
例如：从零设计并实现复杂系统同时完成测试和部署、解决开放式跨领域且没有明确标准解的问题、设计复杂架构并处理大量相互依赖的约束。

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
        sys.exit("ERROR: TOKENHUB_API_KEY not set")
    return key


def load_tasks() -> list[dict]:
    tasks = []
    for d in sorted(TASKSPECS.iterdir()):
        if not d.name.startswith("SWE_"):
            continue
        f = d / "taskspec.yaml"
        if not f.is_file():
            continue
        try:
            ts = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        q = (ts.get("seed_query") or "").strip()
        if q:
            tasks.append({"id": d.name, "query": q})
    return tasks


def chat(client, messages, max_tokens=256, retries=3):
    for attempt in range(retries):
        try:
            resp = client.post(
                f"{API_BASE}/chat/completions",
                json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
                timeout=60.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content and content.strip():
                return content
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return ""


def parse_json(text):
    text = text.strip()
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


def run_bucket(tasks, key, workers, limit):
    done = set()
    if BUCKET_OUT.is_file():
        with open(BUCKET_OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    remaining = [t for t in tasks if t["id"] not in done]
    if limit:
        remaining = remaining[:limit]
    print(f"[bucket] {len(remaining)} to label ({len(done)} done)", flush=True)
    if not remaining:
        return
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(60.0))
    lock = threading.Lock()
    fout = open(BUCKET_OUT, "a", encoding="utf-8")
    ok = fail = 0

    def one(t):
        nonlocal ok, fail
        raw = chat(client, [{"role": "system", "content": BUCKET_SYSTEM}, {"role": "user", "content": t["query"][:4000]}])
        b = parse_json(raw).get("bucket", "").strip().lower()
        if b not in CANONICAL:
            b = ""
        with lock:
            fout.write(json.dumps({"id": t["id"], "bucket": b}, ensure_ascii=False) + "\n")
            fout.flush()
            if b:
                ok += 1
            else:
                fail += 1

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, remaining))
    fout.close()
    print(f"[bucket] done: ok={ok} fail={fail}", flush=True)


def run_difficulty(tasks, key, workers, limit):
    done = set()
    if DIFF_OUT.is_file():
        with open(DIFF_OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    remaining = [t for t in tasks if t["id"] not in done]
    if limit:
        remaining = remaining[:limit]
    print(f"[difficulty] {len(remaining)} to label ({len(done)} done)", flush=True)
    if not remaining:
        return
    client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=httpx.Timeout(60.0))
    lock = threading.Lock()
    fout = open(DIFF_OUT, "a", encoding="utf-8")
    ok = fail = 0

    def one(t):
        nonlocal ok, fail
        raw = chat(client, [{"role": "user", "content": f"任务:\n{t['query'][:4000]}\n\n{DIFF_PROMPT}"}])
        dv = parse_json(raw).get("difficulty")
        try:
            dv = int(dv)
        except (TypeError, ValueError):
            dv = None
        with lock:
            fout.write(json.dumps({"id": t["id"], "difficulty": dv}, ensure_ascii=False) + "\n")
            fout.flush()
            if dv is not None:
                ok += 1
            else:
                fail += 1

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, remaining))
    fout.close()
    print(f"[difficulty] done: ok={ok} fail={fail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    key = load_key()
    tasks = load_tasks()
    print(f"loaded {len(tasks)} SWE tasks", flush=True)

    # 两阶段并行(两个线程)
    t1 = threading.Thread(target=run_bucket, args=(tasks, key, args.workers, args.limit))
    t2 = threading.Thread(target=run_difficulty, args=(tasks, key, args.workers, args.limit))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # merge
    if not args.limit:
        bk = {}
        if BUCKET_OUT.is_file():
            for line in open(BUCKET_OUT, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    bk[d["id"]] = d.get("bucket", "")
        dv = {}
        if DIFF_OUT.is_file():
            for line in open(DIFF_OUT, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    dv[d["id"]] = d.get("difficulty")
        from collections import Counter
        bkc = Counter(bk.values())
        dvc = Counter(dv.values())
        print(f"\n[merge] bucket 分布: {dict(bkc)}")
        print(f"[merge] difficulty 分布: {dict(sorted(dvc.items(), key=lambda x: str(x[0])))}")
        with open(MERGED_OUT, "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps({"id": t["id"], "bucket": bk.get(t["id"], ""), "difficulty": dv.get(t["id"])}, ensure_ascii=False) + "\n")
        print(f"[merge] wrote {MERGED_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
