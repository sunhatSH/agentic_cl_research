"""Hermes 轨迹 → 官方 ClawEval grader 评分 + Pass^3 汇总。

方案 B：Hermes 跑 agent（verl rollout）→ 完整 conversation 存进 trajectory
→ 本模块把 conversation 转成官方 TraceMessage/ToolDispatch → 加载官方 grader.py
→ LLM judge 按 RUBRIC 打分 → compute_task_score → is_pass(0.75) → Pass^3。

用法（评测后离线跑，不占 GPU）：
  python -m claw_eval_hermes.grade \\
    --rollout-status rollouts/training/cl2r_baseline/rollout_status-200.jsonl \\
    --tasks-dir src/claw_eval_vendor/tasks \\
    --judge-model google/gemini-3-flash-preview \\
    --judge-base-url https://openrouter.ai/api/v1 \\
    --judge-api-key $OPENROUTER_API_KEY \\
    --output eval/results/baseline_step200/graded.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# 确保项目 src/ 在 path 上（claw_eval_vendor 在 src/ 下）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def load_rollout_status(path: str) -> list[dict]:
    """读 rollout_status-<step>.jsonl，返回每行一个 task 记录。

    每条含: step, task_id, bucket, n_rollouts, rollouts[{status, reward, messages, ...}]
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def hermes_messages_to_trace_messages(messages: list[dict]) -> list:
    """把 Hermes 的 message_history（OpenAI 格式 dict list）转成官方 TraceMessage。

    Hermes messages 格式: [{"role": "user/assistant/system", "content": "..."}]
    官方 TraceMessage 格式: TraceMessage(message=Message(role=..., content=[TextBlock(...)]))

    tool_calls（如果 messages 里有 tool_call/tool_result 字段）也转成 ToolDispatch。
    """
    from claw_eval_vendor.models.trace import TraceMessage
    from claw_eval_vendor.models.message import Message
    from claw_eval_vendor.models.content import TextBlock

    trace_msgs = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            # content 已经是 block list（可能含 tool_use/tool_result）
            text_blocks = [TextBlock(text=str(c.get("text", "")) if isinstance(c, dict) else str(c)) for c in content if not isinstance(c, dict) or c.get("type") == "text"]
        else:
            text_blocks = [TextBlock(text=str(content))]
        msg = Message(role=role, content=text_blocks)
        trace_msgs.append(TraceMessage(message=msg))
    return trace_msgs


def extract_tool_dispatches(messages: list[dict]) -> list:
    """从 Hermes messages 里抽 tool calls → 官方 ToolDispatch 列表。

    Hermes 的 tool calls 可能在 assistant message 的 tool_calls 字段里，
    或在 user message 的 tool_result 里。这里 best-effort 提取。
    """
    from claw_eval_vendor.models.trace import ToolDispatch

    dispatches = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            tool_calls = m.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        dispatches.append(ToolDispatch(
                            trace_id="",
                            tool_use_id=tc.get("id", ""),
                            tool_name=tc.get("function", {}).get("name", tc.get("name", "")),
                            endpoint_url="",
                            request_body=tc.get("function", {}).get("arguments", tc.get("input", {})),
                            response_status=200,
                            response_body={},
                        ))
    return dispatches


def load_task_definition(task_id: str, tasks_dir: str) -> Any:
    """加载 task.yaml → TaskDefinition（官方模型）。"""
    from claw_eval_vendor.models.task import TaskDefinition
    import yaml

    task_path = Path(tasks_dir) / task_id / "task.yaml"
    if not task_path.exists():
        # fallback: 模糊匹配目录名
        for d in Path(tasks_dir).iterdir():
            if d.is_dir() and task_id.lower() in d.name.lower():
                task_path = d / "task.yaml"
                break
    if not task_path.exists():
        raise FileNotFoundError(f"task.yaml not found for {task_id} in {tasks_dir}")

    data = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    return TaskDefinition(**data)


def grade_one_rollout(
    messages: list[dict],
    task_id: str,
    tasks_dir: str,
    judge: Any | None = None,
) -> dict[str, Any]:
    """对一条 rollout 的完整 conversation 跑官方 grader。

    返回: {task_score, passed, scores: {completion, robustness, communication, safety}}
    """
    from claw_eval_vendor.graders.registry import get_grader
    from claw_eval_vendor.models.scoring import compute_task_score, is_pass

    if not messages:
        return {"task_score": 0.0, "passed": False, "scores": {}, "error": "empty messages"}

    # 转换 Hermes messages → 官方格式
    trace_messages = hermes_messages_to_trace_messages(messages)
    dispatches = extract_tool_dispatches(messages)
    task = load_task_definition(task_id, tasks_dir)

    # 加载 per-task grader
    try:
        grader = get_grader(task_id, tasks_dir=tasks_dir)
    except FileNotFoundError:
        return {"task_score": 0.0, "passed": False, "scores": {}, "error": f"no grader.py for {task_id}"}

    # 跑 grader
    try:
        dim_scores = grader.grade(
            messages=trace_messages,
            dispatches=dispatches,
            task=task,
            audit_data=None,
            judge=judge,
        )
    except Exception as exc:
        return {"task_score": 0.0, "passed": False, "scores": {}, "error": f"grader error: {exc}"}

    task_score = compute_task_score(dim_scores)
    passed = is_pass(task_score)
    return {
        "task_score": task_score,
        "passed": passed,
        "scores": {
            "completion": getattr(dim_scores, "completion", 0.0),
            "robustness": getattr(dim_scores, "robustness", 0.0),
            "communication": getattr(dim_scores, "communication", 0.0),
            "safety": getattr(dim_scores, "safety", 1.0),
        },
    }


def grade_rollout_status(
    rollout_status_path: str,
    tasks_dir: str,
    judge_model: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> list[dict]:
    """对整个 rollout_status 文件跑官方 grader，返回 per-task per-rollout 评分。"""
    from claw_eval_vendor.graders.llm_judge import LLMJudge

    judge = None
    if judge_model:
        judge = LLMJudge(
            model_id=judge_model,
            api_key=judge_api_key or os.environ.get("OPENROUTER_API_KEY", "dummy"),
            base_url=judge_base_url or "https://openrouter.ai/api/v1",
        )

    rows = load_rollout_status(rollout_status_path)
    results = []
    for row in rows:
        task_id = row.get("task_id", "")
        bucket = row.get("bucket", "")
        step = row.get("step", 0)
        for ri, rollout in enumerate(row.get("rollouts", [])):
            messages = rollout.get("messages", [])
            if not messages:
                continue
            graded = grade_one_rollout(messages, task_id, tasks_dir, judge=judge)
            results.append({
                "step": step,
                "task_id": task_id,
                "bucket": bucket,
                "rollout_index": ri,
                "rollout_reward": rollout.get("reward"),
                **graded,
            })
    return results


def summarize_pass3(results: list[dict], threshold: float = 0.75) -> dict:
    """官方 Pass^3 汇总：按 task_id 分组，算 avg_pass / any_pass / all_pass。"""
    from collections import defaultdict
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    n_tasks = len(by_task)
    n_avg_pass = 0
    n_any_pass = 0
    n_all_pass = 0
    all_scores = []
    per_task = {}

    for tid, trials in by_task.items():
        scores = [t.get("task_score", 0.0) for t in trials]
        avg = sum(scores) / len(scores) if scores else 0.0
        avg_pass = avg >= threshold
        any_pass = any(s >= threshold for s in scores)
        all_pass = all(s >= threshold for s in scores) and len(scores) > 0
        if avg_pass:
            n_avg_pass += 1
        if any_pass:
            n_any_pass += 1
        if all_pass:
            n_all_pass += 1
        all_scores.append(avg)
        per_task[tid] = {
            "n_trials": len(scores),
            "scores": scores,
            "avg_score": avg,
            "avg_pass": avg_pass,
            "any_pass": any_pass,
            "all_pass": all_pass,
            "bucket": trials[0].get("bucket", ""),
        }

    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "n_tasks": n_tasks,
        "overall_avg_score": overall_avg,
        "avg_pass_rate": n_avg_pass / n_tasks if n_tasks else 0.0,
        "any_pass_rate": n_any_pass / n_tasks if n_tasks else 0.0,
        "all_pass_rate": n_all_pass / n_tasks if n_tasks else 0.0,
        "n_avg_pass": n_avg_pass,
        "n_any_pass": n_any_pass,
        "n_all_pass": n_all_pass,
        "threshold": threshold,
        "per_task": per_task,
    }


def main():
    ap = argparse.ArgumentParser(description="Hermes 轨迹 → 官方 ClawEval grader 评分")
    ap.add_argument("--rollout-status", required=True, help="rollout_status-<step>.jsonl 路径")
    ap.add_argument("--tasks-dir", default="src/claw_eval_vendor/tasks", help="ClawEval tasks 目录")
    ap.add_argument("--judge-model", default=None, help="LLM judge 模型(默认不用 judge,只跑规则 grader)")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-api-key", default=None)
    ap.add_argument("--output", required=True, help="输出 JSON 路径")
    ap.add_argument("--threshold", type=float, default=0.75, help="Pass 阈值")
    args = ap.parse_args()

    print(f"[grade] 加载 {args.rollout_status}")
    results = grade_rollout_status(
        args.rollout_status,
        args.tasks_dir,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
    )
    print(f"[grade] 评分完成: {len(results)} 条 rollout")

    summary = summarize_pass3(results, threshold=args.threshold)

    out = {
        "rollout_status": args.rollout_status,
        "n_graded": len(results),
        "summary": summary,
        "per_rollout": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    s = summary
    print(f"\n{'='*60}")
    print(f"  ClawEval 官方评分 (Pass^3, threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"  n_tasks:       {s['n_tasks']}")
    print(f"  overall_avg:    {s['overall_avg_score']:.3f}")
    print(f"  avg_pass_rate:  {s['avg_pass_rate']:.3f}  ({s['n_avg_pass']}/{s['n_tasks']})")
    print(f"  any_pass_rate:  {s['any_pass_rate']:.3f}  ({s['n_any_pass']}/{s['n_tasks']})")
    print(f"  all_pass_rate:  {s['all_pass_rate']:.3f}  ({s['n_all_pass']}/{s['n_tasks']})")
    print(f"\n✅ 结果写入 {args.output}")


if __name__ == "__main__":
    main()
