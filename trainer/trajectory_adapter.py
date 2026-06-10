"""Convert verl rollout batches to replay-buffer trajectories.

Keeps verl TensorDict / DataProto layout out of ``replay_buffer/``.
Trajectories are stored as OpenAI-style message lists when available.
"""

from __future__ import annotations

from typing import Any


def _get_non_tensor(batch: Any, key: str, default=None):
    """Read from DataProto.non_tensor_batch or plain dict."""
    if batch is None:
        return default
    if hasattr(batch, "non_tensor_batch"):
        return batch.non_tensor_batch.get(key, default)
    if isinstance(batch, dict):
        return batch.get(key, default)
    return default


def extract_trajectories_from_batch(batch: Any, default_bucket: str = "Workflow") -> list[tuple[Any, str, dict]]:
    """Extract (trajectory, bucket, metadata) tuples from a verl training batch.

    Expected optional ``non_tensor_batch`` fields per sample:
        - ``messages``: OpenAI chat message list
        - ``bucket`` / ``category``: capability bucket name
        - ``reward``: scalar reward for R5 / logging
        - ``task_id``: source task identifier

    When ``messages`` is absent, falls back to storing raw prompt/response text
    if present in ``non_tensor_batch``.
    """
    messages_list = _get_non_tensor(batch, "messages")
    buckets = _get_non_tensor(batch, "bucket") or _get_non_tensor(batch, "category")
    rewards = _get_non_tensor(batch, "reward")
    task_ids = _get_non_tensor(batch, "task_id")
    prompts = _get_non_tensor(batch, "prompt")
    responses = _get_non_tensor(batch, "response")

    if messages_list is not None:
        n = len(messages_list)
    elif prompts is not None:
        n = len(prompts)
    elif responses is not None:
        n = len(responses)
    else:
        return []

    out: list[tuple[Any, str, dict]] = []
    for i in range(n):
        meta: dict[str, Any] = {}
        bucket = default_bucket
        if buckets is not None and i < len(buckets):
            bucket = buckets[i] or default_bucket
        if rewards is not None and i < len(rewards):
            meta["reward"] = float(rewards[i])
        if task_ids is not None and i < len(task_ids):
            meta["task_id"] = task_ids[i]
            meta["pattern_id"] = task_ids[i]

        if messages_list is not None:
            trajectory = messages_list[i]
            if isinstance(trajectory, dict) and "messages" in trajectory:
                trajectory = trajectory["messages"]
        elif prompts is not None or responses is not None:
            prompt = prompts[i] if prompts is not None and i < len(prompts) else ""
            response = responses[i] if responses is not None and i < len(responses) else ""
            trajectory = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        else:
            continue

        meta.setdefault("messages", trajectory if isinstance(trajectory, list) else [])
        out.append((trajectory, bucket, meta))
    return out


def replay_sample_to_metadata(replay_item: tuple[str, Any, dict]) -> dict:
    """Normalize a buffer sample ``(tid, trajectory, meta)`` for TokenWeighting."""
    _tid, trajectory, meta = replay_item
    out = dict(meta)
    if "messages" not in out:
        if isinstance(trajectory, list):
            out["messages"] = trajectory
        elif isinstance(trajectory, dict) and "messages" in trajectory:
            out["messages"] = trajectory["messages"]
    return out
