#!/usr/bin/env python3
"""Reward-model selection harness.

给 N 条真实轨迹,用**项目正式的 reward rubric**(agents.prompts.REWARD_RUBRIC,
含刚改的 safety 0/1 + trajectory 收敛"不加不该加的东西")让多个候选 judge 模型
分别打分,每条轨迹**重复打 K 次**,统计:

  · 每个维度(task_done / correctness / trajectory / safety)的均值 + 标准差
    -> 稳定度(std 越小越稳,一致性越好)
  · 聚合后最终 reward 的均值 + std
  · 每次调用的响应时间(latency)分布
  · JSON 解析失败率(judge_error / 截断)

数据源:datasets/cold_start/cold_start_1429.jsonl(OpenAI 完整轨迹)。
判分链路完全复用 trainer.model_reward + agents.prompts,所以这里量到的一致性
就是训练时真实会遇到的一致性。

跑法(需 SUFY_API_KEY;长任务请 tmux/nohup 后台):
  source scripts/load_training_env.sh   # 或手动 export SUFY_API_KEY=...
  PY=/mnt/afs_toolcall/sunhao4/miniconda3/bin/python3
  tmux new -d -s rmchoose "$PY rewardmodel_choose/compare_reward_models.py \
      --n-traj 20 --repeat 8 \
      --out rewardmodel_choose/results/run.json"

结果 JSON + 一张可读的 markdown 汇总写到 --out 同目录。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stats
import sys
import time
from pathlib import Path

import httpx

# 复用项目正式判分组件:rubric + prompt 组装 + 解析 + 聚合。这样"测的就是训练用的"。
ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (src/rewardmodel_choose → repo)
sys.path.insert(0, str(ROOT))          # agents/ 在仓库根
sys.path.insert(0, str(ROOT / "src"))  # trainer/ 等 6 包在 src/

from agents.prompts import REWARD_RUBRIC  # noqa: E402
from trainer.model_reward import (  # noqa: E402
    JUDGE_DIMENSIONS,
    aggregate,
    build_judge_prompt,
    parse_judge_output,
)

# 候选 judge 模型(sufy 需带厂商前缀)。用户指定这三个。
CANDIDATE_MODELS = [
    "deepseek/deepseek-v4-flash-20260731",
    "google/gemini-3.5-flash-lite",
    "stepfun/step-3.7-flash",
]

SUFY_BASE = "https://openai.sufy.com/v1"


def _serialize_trajectory(messages: list[dict]) -> tuple[str, str]:
    """(task, trajectory_text) from an OpenAI message list.

    task = 首条 user 内容;trajectory = 之后 assistant/tool 步骤的可读串
    (含 tool_calls 名/参数、tool 结果)。冷数据无沙箱 diff,rubric 允许缺 diff
    时从 trajectory + final answer 判分,故这里只喂 trajectory。
    """
    task = ""
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user" and not task:
            task = (m.get("content") or "").strip()
            continue
        if role == "assistant":
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"[assistant] {content}")
            for tc in m.get("tool_calls") or []:
                fn = (tc.get("function") or {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if isinstance(args, (dict, list)):
                    args = json.dumps(args, ensure_ascii=False)
                lines.append(f"[assistant->tool_call] {name}({str(args)[:600]})")
        elif role == "tool":
            name = m.get("name", "?")
            out = m.get("content") or ""
            lines.append(f"[tool:{name}] {str(out)[:800]}")
    return task, "\n".join(lines)


def _load_trajectories(path: Path, n: int) -> list[dict]:
    """取前若干条轨迹,尽量覆盖多个桶(每桶轮流取,保证维度多样)。"""
    by_bucket: dict[str, list[dict]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            msgs = r.get("messages")
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            if not msgs:
                continue
            b = (r.get("metadata") or {}).get("bucket", "?")
            by_bucket.setdefault(b, []).append({"bucket": b, "messages": msgs,
                                                "record_id": (r.get("metadata") or {}).get("query_index")})
    # round-robin across buckets for diversity
    picked: list[dict] = []
    buckets = sorted(by_bucket)
    idx = {b: 0 for b in buckets}
    while len(picked) < n and any(idx[b] < len(by_bucket[b]) for b in buckets):
        for b in buckets:
            if idx[b] < len(by_bucket[b]):
                picked.append(by_bucket[b][idx[b]])
                idx[b] += 1
                if len(picked) >= n:
                    break
    return picked


def _score_once(base: str, model: str, api_key: str, messages: list[dict],
                timeout: float, max_tokens: int,
                no_thinking: bool = False) -> tuple[dict | None, float, str]:
    """One judge call. Returns (verdict|None, latency_s, error_str).

    verdict None 表示解析失败/截断/IO 失败(算 judge_error)。
    no_thinking=True 时尝试关闭思考(sufy/OpenAI 兼容端点的常见参数)。
    """
    t0 = time.time()
    try:
        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if no_thinking:
            # 多家厂商关思考的参数名都塞进 extra_body,端点会忽略不认识的
            body["thinking"] = {"type": "disabled"}
            body["enable_thinking"] = False
            body["reasoning_effort"] = "none"
            body["chat_template_kwargs"] = {"enable_thinking": False}
        resp = httpx.post(
            f"{base.rstrip('/')}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        latency = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"].get("content")) or ""
        verdict, parsed = parse_judge_output(content)
        if not parsed:
            return None, latency, f"unparsed: {content[:80]!r}"
        return verdict, latency, ""
    except Exception as e:  # noqa: BLE001 -- record, never crash the sweep
        return None, time.time() - t0, f"{type(e).__name__}: {str(e)[:120]}"


def _agg(vals: list[float]) -> dict:
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": round(stats.mean(vals), 4),
        "std": round(stats.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/cold_start/cold_start_1429.jsonl")
    ap.add_argument("--n-traj", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--models", default=",".join(CANDIDATE_MODELS),
                    help="逗号分隔的候选模型(sufy 前缀)")
    ap.add_argument("--base", default=os.environ.get("REWARD_API_BASE", SUFY_BASE))
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="thinking 模型要留足推理预算(deepseek 尤其)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="关闭思考模式(thinking/enable_thinking/reasoning_effort 都塞进 extra_body)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="并发调用数(线程池);1=串行")
    ap.add_argument("--out", default="rewardmodel_choose/results/run.json")
    args = ap.parse_args()

    api_key = os.environ.get("SUFY_API_KEY") or os.environ.get("REWARD_API_KEY")
    if not api_key:
        sys.exit("需要 SUFY_API_KEY(或 REWARD_API_KEY);source scripts/load_training_env.sh 或手动 export")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    trajs = _load_trajectories(Path(args.data), args.n_traj)
    print(f"[load] {len(trajs)} 条轨迹, {len(models)} 个模型, 每条打 {args.repeat} 次")
    print(f"[rubric] 使用项目正式 REWARD_RUBRIC (len={len(REWARD_RUBRIC)}, safety 0/1)")

    # 预组装每条轨迹的 judge prompt(所有模型共用同一 prompt,保证公平)
    prompts = []
    for t in trajs:
        task, traj_text = _serialize_trajectory(t["messages"])
        msgs = build_judge_prompt(task=task, trajectory=traj_text, rubric=REWARD_RUBRIC)
        prompts.append({"bucket": t["bucket"], "messages": msgs})

    # results[model] = {dim: [all scores across traj*repeat], reward: [...],
    #                   latency: [...], errors: int, calls: int}
    from concurrent.futures import ThreadPoolExecutor

    def _call_one(p):
        return _score_once(args.base, model, api_key, p["messages"],
                           args.timeout, args.max_tokens, args.no_thinking)

    results: dict[str, dict] = {}
    for model in models:
        print(f"\n===== MODEL {model} =====")
        per_dim: dict[str, list[float]] = {d: [] for d in JUDGE_DIMENSIONS}
        rewards: list[float] = []
        latencies: list[float] = []
        # 逐轨迹稳定度:每条轨迹 repeat 次内的 std(体现"同一输入分数抖不抖")
        per_traj_reward_std: list[float] = []
        per_traj_dim_std: dict[str, list[float]] = {d: [] for d in JUDGE_DIMENSIONS}
        errors = 0
        calls = 0
        for i, p in enumerate(prompts):
            traj_rewards: list[float] = []
            traj_dims: dict[str, list[float]] = {d: [] for d in JUDGE_DIMENSIONS}
            # 同一轨迹的 repeat 次并发(并发度由 --concurrency 控制)
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                iter_out = list(ex.map(_call_one, [p] * args.repeat))
            for verdict, latency, _err in iter_out:
                calls += 1
                latencies.append(latency)
                if verdict is None:
                    errors += 1
                    continue
                r = aggregate(verdict)
                rewards.append(r)
                traj_rewards.append(r)
                for d in JUDGE_DIMENSIONS:
                    v = float(verdict.get(d, 0.0))
                    per_dim[d].append(v)
                    traj_dims[d].append(v)
            if len(traj_rewards) > 1:
                per_traj_reward_std.append(stats.pstdev(traj_rewards))
            for d in JUDGE_DIMENSIONS:
                if len(traj_dims[d]) > 1:
                    per_traj_dim_std[d].append(stats.pstdev(traj_dims[d]))
            print(f"  traj {i + 1}/{len(prompts)} [{p['bucket']}]: "
                  f"reward={_agg(traj_rewards)['mean']} "
                  f"(±{_agg(traj_rewards)['std']}), errs so far={errors}")

        results[model] = {
            "reward": _agg(rewards),
            "dims": {d: _agg(per_dim[d]) for d in JUDGE_DIMENSIONS},
            "latency_s": _agg(latencies),
            # 稳定度核心指标:同一轨迹重复打分的 std,跨轨迹取平均
            "within_traj_reward_std_mean": round(stats.mean(per_traj_reward_std), 4)
            if per_traj_reward_std else None,
            "within_traj_dim_std_mean": {
                d: (round(stats.mean(per_traj_dim_std[d]), 4) if per_traj_dim_std[d] else None)
                for d in JUDGE_DIMENSIONS
            },
            "judge_error_rate": round(errors / max(calls, 1), 4),
            "calls": calls,
            "errors": errors,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": {"n_traj": len(trajs), "repeat": args.repeat, "models": models,
                   "base": args.base, "max_tokens": args.max_tokens},
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\n[done] JSON -> {out}")

    # markdown 汇总
    md = _render_markdown(results, len(trajs), args.repeat)
    md_path = out.with_suffix(".md")
    md_path.write_text(md)
    print(f"[done] Markdown -> {md_path}")
    print("\n" + md)


def _render_markdown(results: dict, n_traj: int, repeat: int) -> str:
    lines = [
        f"# Reward 模型对比 (n_traj={n_traj}, repeat={repeat})",
        "",
        "稳定度 = 同一轨迹重复打分的标准差(within-traj std),越小越一致。",
        "",
        "## 总览",
        "",
        "| 模型 | reward 均值 | reward 全局 std | **稳定度(within-traj reward std)** | 延迟均值(s) | 延迟 max | judge_error 率 |",
        "|------|------|------|------|------|------|------|",
    ]
    for m, r in results.items():
        rw = r["reward"]
        lat = r["latency_s"]
        lines.append(
            f"| `{m}` | {rw['mean']} | {rw['std']} | **{r['within_traj_reward_std_mean']}** | "
            f"{lat['mean']} | {lat['max']} | {r['judge_error_rate']} |"
        )
    lines += ["", "## 各维度(均值 / 全局std / 稳定度within-traj-std)", ""]
    lines.append("| 模型 | " + " | ".join(JUDGE_DIMENSIONS) + " |")
    lines.append("|------|" + "------|" * len(JUDGE_DIMENSIONS))
    for m, r in results.items():
        cells = []
        for d in JUDGE_DIMENSIONS:
            dd = r["dims"][d]
            wstd = r["within_traj_dim_std_mean"][d]
            cells.append(f"{dd['mean']} / {dd['std']} / **{wstd}**")
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    lines += ["", "> 选型建议:优先 **稳定度(within-traj std)低** 且 judge_error 率低、",
              "> 延迟可接受的模型;reward 均值本身不决定优劣(不同模型尺度不同),",
              "> 关键看**同一输入是否给出一致分数**。"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
