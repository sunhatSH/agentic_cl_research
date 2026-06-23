"""Convert verl rollout batches to replay-buffer trajectories.

Keeps verl TensorDict / DataProto layout out of ``replay_buffer/``.
Trajectories are stored as OpenAI-style message lists when available.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _get_non_tensor(batch: Any, key: str, default=None):
    """Read from DataProto.non_tensor_batch or plain dict."""
    if batch is None:
        return default
    if hasattr(batch, "non_tensor_batch"):
        return batch.non_tensor_batch.get(key, default)
    if isinstance(batch, dict):
        return batch.get(key, default)
    return default


def _get_tensor(batch: Any, key: str):
    """Read a per-row tensor field from DataProto.batch (TensorDict) or dict.

    Returns the tensor/array (indexable by row) or None.
    """
    if batch is None:
        return None
    inner = getattr(batch, "batch", None)
    if inner is not None:
        try:
            if key in inner:
                return inner[key]
        except (TypeError, KeyError):
            pass
    if isinstance(batch, dict):
        return batch.get(key)
    return None


def _row_to_list(tensor_row) -> list[float] | None:
    """Best-effort convert a tensor / array row to a python float list."""
    if tensor_row is None:
        return None
    if hasattr(tensor_row, "tolist"):
        try:
            return [float(x) for x in tensor_row.tolist()]
        except (TypeError, ValueError):
            return None
    try:
        return [float(x) for x in tensor_row]
    except (TypeError, ValueError):
        return None


def _first_present(batch: Any, keys):
    """Return the first non-None per-row tensor among candidate keys."""
    for k in keys:
        t = _get_tensor(batch, k)
        if t is not None:
            return t
    return None


def _domain_from_messages(trajectory: Any, valid_buckets: list[str] | None) -> str | None:
    """Recover the LLM-emitted <task_domain> tag from a trajectory's assistant text."""
    from trainer.domain_tagging import parse_domain

    if not isinstance(trajectory, list):
        return None
    for msg in reversed(trajectory):  # tag is emitted last -> scan from the end
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        domain = parse_domain(content, valid_buckets) if isinstance(content, str) else None
        if domain is not None:
            return domain
    return None


def extract_trajectories_from_batch(
    batch: Any, default_bucket: str | None = None, valid_buckets: list[str] | None = None
) -> list[tuple[Any, str, dict]]:
    """Extract (trajectory, bucket, metadata) tuples from a verl training batch.

    Expected optional ``non_tensor_batch`` fields per sample:
        - ``messages``: OpenAI chat message list
        - ``bucket`` / ``category``: capability bucket name
        - ``reward``: scalar reward for R5 / logging
        - ``task_id``: source task identifier

    When ``messages`` is absent, falls back to storing raw prompt/response text
    if present in ``non_tensor_batch``.

    Bug B12: rows without a resolvable bucket are SKIPPED (and counted in a
    warning) rather than silently dumped into a default "Workflow" bucket,
    which would distort the quota distribution. Pass ``default_bucket`` only
    when a deliberate catch-all bucket is wanted.
    """
    messages_list = _get_non_tensor(batch, "messages")
    # NB: non_tensor fields are numpy arrays -> never use ``a or b`` (ambiguous
    # truth value). Fall back to "category" only when "bucket" is truly absent.
    buckets = _get_non_tensor(batch, "bucket")
    if buckets is None:
        buckets = _get_non_tensor(batch, "category")
    rewards = _get_non_tensor(batch, "reward")
    task_ids = _get_non_tensor(batch, "task_id")
    prompts = _get_non_tensor(batch, "prompt")
    responses = _get_non_tensor(batch, "response")
    # Priority signal sources (bug D2 -- previously never backfilled, so
    # forgetting_risk / within_bucket_difficulty were always 0 in training):
    #   - rollout log-probs become `original_logprobs` (the snapshot of the
    #     policy at insert time; forgetting_risk later compares it to the
    #     current policy's log-probs via buffer.update_priority).
    #   - group-internal pass rate becomes `success_rate` (within_bucket_difficulty).
    rollout_logprobs = _first_present(batch, ("rollout_log_probs", "old_log_probs", "old_log_prob"))
    success_rates = _get_non_tensor(batch, "success_rate")
    # Response token ids per row -> needed by TokenWeighting (U-shape block
    # weights). Available from the rollout responses tensor; trimmed below by
    # response_mask so padding is not counted as response tokens.
    response_ids = _get_tensor(batch, "responses")
    response_mask = _get_tensor(batch, "response_mask")

    if messages_list is not None:
        n = len(messages_list)
    elif prompts is not None:
        n = len(prompts)
    elif responses is not None:
        n = len(responses)
    else:
        return []

    out: list[tuple[Any, str, dict]] = []
    skipped = 0
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
        if rollout_logprobs is not None and i < len(rollout_logprobs):
            lp = _row_to_list(rollout_logprobs[i])
            if lp is not None:
                meta["original_logprobs"] = lp
        if success_rates is not None and i < len(success_rates):
            try:
                meta["success_rate"] = float(success_rates[i])
            except (TypeError, ValueError):
                pass
        if response_ids is not None and i < len(response_ids):
            ids = _row_to_list(response_ids[i])
            if ids is not None:
                if response_mask is not None and i < len(response_mask):
                    mask = _row_to_list(response_mask[i]) or []
                    # ids/mask may differ in length under padding; keep the
                    # overlap (truncate to shorter) rather than crash ingestion.
                    ids = [int(t) for t, m in zip(ids, mask, strict=False) if m]
                else:
                    ids = [int(t) for t in ids]
                meta["response_token_ids"] = ids

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

        # Resolve bucket: explicit non_tensor field wins; otherwise recover the
        # LLM-emitted <task_domain> tag from the trajectory (domain_tagging).
        # Rows with no resolvable bucket are skipped (bug B12), never defaulted.
        if bucket is None:
            bucket = _domain_from_messages(trajectory, valid_buckets)
        if bucket is None:
            skipped += 1
            continue
        meta["bucket"] = bucket

        meta.setdefault("messages", trajectory if isinstance(trajectory, list) else [])
        out.append((trajectory, bucket, meta))

    if skipped:
        logger.warning(
            "extract_trajectories_from_batch: skipped %d/%d trajectories with no "
            "resolvable bucket (no 'bucket'/'category' field). They were NOT added "
            "to the replay buffer to avoid distorting quota.",
            skipped,
            n,
        )
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
