"""Trajectory collection: native fields from the framework, not a proxy (Gap D).

Decision (doc/Sandbox_Agent架构.md §3): we orchestrate the 16×8 + winner-sync
session loop, but every single generation step is produced by the RL framework's
native generate (verl AgentLoopOutput: prompt_ids / response_ids / response_mask /
rollout_log_probs). This module models that boundary with a ``GenerateFn`` and
assembles a ReAct loop into a buffer-ready ``Trajectory``:

    messages ──GenerateFn──> GenStep(response tokens + logprobs)   [mask=1]
            ──parse tool_call──> sandbox.run_code ──> observation  [mask=0]
            ──append observation, loop until final / max_turns──

In real training ``GenerateFn`` wraps verl's rollout generate (token+logprob come
for free); here it is injectable so the whole chain unit-tests off-GPU. The
assembled trajectory carries the verl-native fields that
``trainer/trajectory_adapter`` / the buffer consume (``original_logprobs`` etc.).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rollout.session_pool import Trajectory
from trainer.domain_tagging import build_domain_instruction, parse_domain

_TOOLCALL_RE = re.compile(r"<toolcall>\s*(\{.*?\})\s*</toolcall>", re.S)


@dataclass
class GenStep:
    """One framework generation step (mirrors verl AgentLoopOutput fields)."""

    text: str
    response_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    # Actor-side usage from the API response (0 when not available).
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # response_mask for these tokens is implicitly 1 (model-generated).


# GenerateFn(messages) -> GenStep  (real: verl generate; test: mock)
GenerateFn = Callable[[list[dict[str, Any]]], GenStep]
# ToolExec(client, tool, code) -> (observation_text, observation_token_ids)
ToolExec = Callable[[Any, str, str], tuple[str, list[int]]]


def default_tool_exec(client: Any, tool: str, code: str) -> tuple[str, list[int]]:
    """Run code in the sandbox; tokenize observation as length placeholder.

    Real training swaps in the tokenizer; the mask (0 for observation) is what
    matters for training, not the placeholder ids here.
    """
    res = client.run_code(code)
    obs = (res.stdout or res.stderr or "").strip()
    obs_ids = [0] * len(obs.split())
    return obs, obs_ids


def parse_tool_call(text: str) -> tuple[str, str] | None:
    m = _TOOLCALL_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except (TypeError, ValueError):
        return None
    if "tool" not in obj or "code" not in obj:
        return None
    return str(obj["tool"]), str(obj["code"])


_REACT_SYSTEM_PROMPT = (
    "You are a coding assistant with access to a Python sandbox. "
    "When you need to execute code, output it in this exact format:\n"
    '<toolcall>{"tool": "python", "code": "your code here"}</toolcall>\n'
    "The sandbox will run the code and return the output as a user message "
    "prefixed with '[Sandbox Output]'. You can make multiple tool calls "
    "across turns. When done, provide your final answer without <toolcall> tags."
    # The buffer routes every trajectory into one of 7 capability buckets via a
    # <task_domain> tag parsed from the trajectory text (trainer.domain_tagging).
    # Without this instruction the model never emits the tag, so parse_domain
    # returns None and ingest_trajectories SKIPS every trajectory (B12). Append
    # the canonical domain-tagging instruction so cold rollouts are bucketable.
    "\n\n" + build_domain_instruction()
)


def make_react_agent_fn(
    generate_fn: GenerateFn,
    *,
    tool_exec: ToolExec = default_tool_exec,
    max_turns: int = 6,
    default_bucket: str | None = None,
    system_prompt: str | None = _REACT_SYSTEM_PROMPT,
):
    """Build an AgentFn for SessionSandboxPool that runs a native-collection ReAct loop.

    The returned trajectory concatenates, across turns:
      - generated tokens (mask=1) + their logprobs   -> trainable
      - observation tokens        (mask=0)            -> context only
    and records messages (assistant + tool) for transcript / bucket tagging.

    Args:
        system_prompt: Injected as the first system message so base models know
            the <toolcall> format. Default: a minimal ReAct instruction. Set to
            None to disable (for verl training where the model already knows the
            format, or when the model's own system prompt covers tool use).
    """

    def agent_fn(
        client: Any,
        query: str,
        state: Any,
        slot_idx: int,
        history: list[dict[str, Any]] | None = None,
    ) -> Trajectory:
        # 正史 = prior winners' messages (§3 ①) used ONLY as the generation
        # prefix; the returned trajectory records just THIS query's turns so the
        # pool can accumulate session_history without double-counting.
        prefix: list[dict[str, Any]] = list(history or [])
        if system_prompt and not any(m.get("role") == "system" for m in prefix):
            prefix.insert(0, {"role": "system", "content": system_prompt})
        turns: list[dict[str, Any]] = [{"role": "user", "content": query}]
        all_resp_ids: list[int] = []
        all_logprobs: list[float] = []
        response_mask: list[int] = []
        full_text_parts: list[str] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0

        for _turn in range(max_turns):
            step = generate_fn(prefix + turns)
            turns.append({"role": "assistant", "content": step.text})
            all_resp_ids.extend(step.response_ids)
            all_logprobs.extend(step.logprobs)
            response_mask.extend([1] * len(step.response_ids))
            full_text_parts.append(step.text)
            prompt_tokens_total += step.prompt_tokens
            completion_tokens_total += step.completion_tokens

            call = parse_tool_call(step.text)
            if call is None:
                break  # no tool call -> final answer
            tool, code = call
            obs, obs_ids = tool_exec(client, tool, code)
            # Use role='user' for sandbox observations (not role='tool').
            # OpenAI-compatible APIs require that 'tool' role messages follow
            # assistant messages with structured 'tool_calls'; our <toolcall>
            # XML doesn't satisfy this. role='user' works universally.
            turns.append({"role": "user", "content": f"[Sandbox Output]\n{obs}"})
            all_resp_ids.extend(obs_ids)
            all_logprobs.extend([0.0] * len(obs_ids))  # not policy tokens
            response_mask.extend([0] * len(obs_ids))    # masked out of loss
            full_text_parts.append(obs)

        full_text = "\n".join(full_text_parts)
        bucket = parse_domain(full_text) or default_bucket

        # advance logical disk state (mock: agent may have written files)
        new_state = dict(state) if isinstance(state, dict) else {}

        return Trajectory(
            slot_idx=slot_idx,
            trajectory_id="",
            messages=turns,
            reward=None,  # scorer fills this in (Gap A)
            response_token_ids=all_resp_ids,
            logprobs=all_logprobs,
            bucket=bucket,
            next_state=new_state,
            meta={"response_mask": response_mask, "num_turns": len(full_text_parts),
                  "prompt_tokens": prompt_tokens_total, "completion_tokens": completion_tokens_total},
        )

    return agent_fn


def trajectory_to_buffer_item(traj: Trajectory) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    """Map a collected Trajectory -> (payload, bucket, metadata) for buffer.add_trajectory.

    Metadata carries the verl-native priority signals the buffer expects:
    ``original_logprobs`` (snapshot policy logprob) + ``success_rate``.
    """
    payload = {
        "messages": traj.messages,
        "response_token_ids": traj.response_token_ids,
        "response_mask": traj.meta.get("response_mask", []),
    }
    meta = {
        "trajectory_id": traj.trajectory_id,
        "reward": traj.reward,
        "original_logprobs": traj.logprobs,
        "success_rate": traj.meta.get("success_rate"),
        "num_turns": traj.meta.get("num_turns"),
        "prompt_tokens": traj.meta.get("prompt_tokens", 0),
        "completion_tokens": traj.meta.get("completion_tokens", 0),
    }
    return payload, traj.bucket, meta


def ingest_trajectories(
    buffer: Any,
    trajectories: Sequence[Trajectory],
    *,
    valid_buckets: Sequence[str] | None = None,
) -> dict[str, int]:
    """Route collected trajectories into the 7-bucket buffer.

    Trajectories whose bucket is unresolved / not valid are SKIPPED (B12), never
    dumped into a default bucket. Returns counts {added, skipped}.
    """
    added = 0
    skipped = 0
    valid = set(valid_buckets) if valid_buckets is not None else None
    for traj in trajectories:
        payload, bucket, meta = trajectory_to_buffer_item(traj)
        if not bucket or (valid is not None and bucket not in valid):
            skipped += 1
            continue
        buffer.add_trajectory(payload, bucket, meta)
        added += 1
    return {"added": added, "skipped": skipped}
