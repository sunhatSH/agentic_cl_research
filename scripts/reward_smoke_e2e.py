#!/usr/bin/env python3
"""End-to-end reward smoke: API-as-actor → local sandbox rollout → observer diff → reward judge → score.

目的（2026-07-31）：证明 reward 全 0 的两个根因已修好——
  1. judge key 透传（本脚本在 driver 进程直接跑，key 从 .env/env 读，天然可用；
     训练侧的修复是 verl_runner._passthrough 加 SUFY_API_KEY，见 memory/reward-zero-two-causes.md）
  2. thinking judge 的 max_tokens 4096→16384（model_reward.OpenAIJudgeClient）

本脚本【不碰 verl / 不上 GPU】，用真实链路的项目侧组件跑一遍：
  - 【API 代替 actor】：OpenAIChatClient(agents.yaml questioner 或 REWARD/ACTOR env) 包成 GenerateFn，
    驱动 rollout.collect.make_react_agent_fn 的 ReAct 循环（真在 local 沙箱里 run_code）
  - SessionSandboxPool(backend="local") 起 N 个 slot（每个独立持久 workdir）
  - run_simulated_session：跑 rollout → observer.snapshot/diff（确定性取证）→ score_followup(真 judge) → 打分
  - 打印每个 slot 的 reward + 最终 winner 的分，证明 reward 非 0

用法：
  source scripts/env/load_training_env.sh   # 载入 SUFY_API_KEY
  python scripts/reward_smoke_e2e.py [--slots 4] [--actor-model qwen/qwen3.6-plus] [--task "..."]
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import OpenAIChatClient
from agents.config import resolve_role
from agents.observer import Observer
from agents.personas import sample_persona
from rollout.collect import GenStep, make_react_agent_fn
from rollout.session_pool import SessionSandboxPool
from rollout.simulated_session import run_simulated_session


def _make_actor_generate_fn(client: OpenAIChatClient, max_tokens: int):
    """Wrap an API chat client as a GenerateFn: messages -> GenStep(text)."""

    def generate_fn(messages):
        text = client.chat(messages, max_tokens=max_tokens)
        # No token ids/logprobs from a plain API endpoint; the reward path only
        # needs the assistant text + tool observations, which the ReAct loop
        # reconstructs. response_ids left empty (mask/logprob not used by judge).
        return GenStep(text=text or "", response_ids=[], logprobs=[])

    return generate_fn


def _resolve_actor(actor_model: str | None) -> OpenAIChatClient:
    """Build the API actor client. Reuse the questioner endpoint (sufy) by default
    so we don't need a separate ACTOR_* env; override model via --actor-model."""
    role = resolve_role("questioner")
    ep = role.flat_endpoints()[0]
    model = actor_model or ep.model
    print(f"[smoke] actor(API) = {model} @ {ep.base_url}  key_len={len(ep.api_key)}")
    return OpenAIChatClient(base_url=ep.base_url, model=model, api_key=ep.api_key, temperature=0.7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4, help="并发 slot 数（rollout.n 的等价物）")
    ap.add_argument("--actor-model", default=None, help="actor 用的 API 模型名（默认用 questioner 端点模型）")
    ap.add_argument("--actor-max-tokens", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=4, help="ReAct 每 slot 最多轮数")
    ap.add_argument(
        "--task",
        default=(
            "在当前工作目录用 python 创建文件 report.csv，内容两行：\n"
            "metric,value\nchurn,0.42\n"
            "然后再创建 summary.md，写一句话总结这个 csv。用工具执行，别只描述。"
        ),
        help="seed query（actor 要在沙箱里完成的任务）",
    )
    args = ap.parse_args()

    # judge key 检查（本脚本在 driver 进程，直接读 env；缺了就明确报出来）
    if not os.environ.get("SUFY_API_KEY", "").strip():
        print("[smoke] ⚠ SUFY_API_KEY 未设置 → judge 会用 sk-local 打 sufy 401 → reward 归 0。")
        print("[smoke]   请先 `source scripts/env/load_training_env.sh` 再跑。")
    from trainer.model_reward import get_judge

    judge = get_judge()
    print(f"[smoke] reward judge = {judge.model} @ {judge.base_url}  "
          f"max_tokens={judge.max_tokens}  key_len={len(judge.api_key)}")

    actor_client = _resolve_actor(args.actor_model)
    generate_fn = _make_actor_generate_fn(actor_client, args.actor_max_tokens)
    agent_fn = make_react_agent_fn(generate_fn, max_turns=args.max_turns)

    pool = SessionSandboxPool(slots=args.slots, backend="local")
    observer = Observer(use_llm=False)  # 确定性取证（不额外烧 observer 模型），reward judge 才是重点
    from agents.questioner import Questioner

    persona = sample_persona(random.Random(0))
    print(f"[smoke] slots={args.slots} backend=local persona={persona.name}")
    print(f"[smoke] task = {args.task!r}\n")

    result = run_simulated_session(
        pool,
        seed_query=args.task,
        agent_fn=agent_fn,
        persona=persona,
        observer=observer,
        questioner=Questioner(client=actor_client),
        reward_judge=judge,
        seed=0,
    )

    print("\n" + "=" * 70)
    print("每个 slot 的 reward 与 judge verdict：")
    rewards = []
    for i, t in enumerate(result.trajectories):
        v = t.meta.get("reward_verdict", {}) or {}
        rewards.append(t.reward)
        print(
            f"  slot{i}: reward={t.reward!r:>8}  "
            f"score={v.get('score')} completion={v.get('completion')} "
            f"safety={v.get('safety')} robustness={v.get('robustness')} "
            f"judge_error={v.get('judge_error')} gated={v.get('gated')}"
        )
    numeric = [r for r in rewards if isinstance(r, (int, float))]
    nonzero = [r for r in numeric if r and r > 0]
    print("=" * 70)
    print(f"总结：{len(result.trajectories)} 条轨迹, "
          f"{len(numeric)} 条有数值 reward, {len(nonzero)} 条 reward>0")
    if numeric:
        print(f"       max reward = {max(numeric):.4f}  mean = {sum(numeric)/len(numeric):.4f}")
    if nonzero:
        print("✅ reward 非 0 —— rollout→reward 链路打通，两个根因修复有效。")
        return 0
    print("❌ reward 仍全 0/None —— 打印上面每条的 judge_error/gated 定位（401? 截断? gated?）。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
