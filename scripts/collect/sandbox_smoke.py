"""Sandbox execute + sample smoke (mirrors ags-cookbook examples/mini-rl).

Proves the rollout collection loop end-to-end WITHOUT the real sandbox:
  query -> M rollouts (group) -> sandbox executes the agent's tool code
        -> reward -> GRPO advantage -> winner固化 -> per-query domain (bucket).

Backends:
  --backend local  (default) runs Python in a local subprocess (no network/SDK).
  --backend e2b     uses the real Tencent Agent Runtime (needs E2B_API_KEY /
                    E2B_DOMAIN + network); identical loop, one-line swap.
  --backend aliyun  Alibaba 无影 AgentBay slot -- interface/registry ready, the
                    vendor impl is a STUB (留空) to fill in on the cluster.

This is a stand-in policy (mock model). On the cluster the policy is the real
Qwen3.6-27B; everything else (parsing, sandbox exec, GRPO, domain tag) is reused.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.sandbox_client import grpo_advantages, make_sandbox, select_winner
from trainer.domain_tagging import build_domain_instruction, parse_domain

_TOOLCALL_RE = re.compile(r"<toolcall>\s*(\{.*\})\s*</toolcall>", re.S)


def parse_tool_call(model_output: str):
    m = _TOOLCALL_RE.search(model_output)
    if not m:
        raise ValueError("toolcall not found")
    obj = json.loads(m.group(1))
    return obj["tool"], obj["code"]


def mock_policy(query: str, expected: int, rng: random.Random) -> str:
    """Stand-in for the trained model: emits a tool call + a <task_domain> tag.

    ~30% of samples emit buggy code (wrong op) so the group has reward variance
    -> GRPO advantage / winner selection is meaningful.
    """
    a, b, c = 23, 17, 19
    buggy = rng.random() < 0.3
    expr = f"({a} * {b}) + {c}" if buggy else f"({a} * {b}) - {c}"
    return (
        f"我来计算：{query}\n"
        f'<toolcall>{{"tool": "sandbox.exec_python", "code": "print({expr})"}}</toolcall>\n'
        "这是一道结构化业务规则计算题。\n"
        "<task_domain>Finance</task_domain>"
    )


def run_one(sandbox, query: str, expected: int, rng: random.Random) -> dict:
    """One rollout: policy -> parse -> sandbox exec -> reward + domain."""
    out = mock_policy(query, expected, rng)
    _tool, code = parse_tool_call(out)
    res = sandbox.run_code(code)
    answer = res.stdout.strip()
    reward = 1.0 if answer == str(expected) else 0.0
    return {
        "model_output": out,
        "observation": answer or res.stderr.strip(),
        "reward": reward,
        "domain": parse_domain(out),
    }


def rollout_group(query: str, expected: int, m: int, backend: str, seed: int = 0) -> list[dict]:
    """GRPO group: M sandboxes for one query, each an independent rollout."""
    rng = random.Random(seed)
    trajs = []
    for i in range(m):
        sb = make_sandbox(backend)
        try:
            t = run_one(sb, query, expected, random.Random(rng.randint(0, 1 << 30)))
        finally:
            sb.kill()
        t["trajectory_id"] = f"{query[:8]}-{i}"
        trajs.append(t)
    return trajs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="local", choices=["local", "e2b", "aliyun"])
    ap.add_argument("-m", "--group-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    query, expected = "计算：23 × 17 − 19", 23 * 17 - 19

    print(f"[domain instruction injected into rollout system prompt]\n{build_domain_instruction()}\n")
    print(f"=== Rollout group (backend={args.backend}, M={args.group_size}) for: {query} ===")
    trajs = rollout_group(query, expected, args.group_size, args.backend, args.seed)

    rewards = [t["reward"] for t in trajs]
    advs = grpo_advantages(rewards)
    winner = select_winner(rewards, [t["trajectory_id"] for t in trajs])
    for i, t in enumerate(trajs):
        mark = " <-- winner固化为新母版" if i == winner else ""
        print(
            f"  [{t['trajectory_id']}] obs={t['observation']!r} "
            f"reward={t['reward']} adv={advs[i]:+.2f} domain={t['domain']}{mark}"
        )
    print(
        f"\nexpected={expected}  winner={trajs[winner]['trajectory_id']}  "
        f"-> bucket={trajs[winner]['domain']}  (B×M trajectories route into the 9-bucket buffer)"
    )


if __name__ == "__main__":
    main()
