"""Gradient-carrying replay batch construction (concatenation design).

The supervised replay term ``-log pi(a|s) * w`` must be differentiable w.r.t.
the current policy. That is only true if the replay tokens go through the SAME
forward pass that produces ``model_output`` inside the actor update. Therefore
we APPEND replay rows to the actor's training batch (rather than precomputing
detached log-probs); ``cl_loss`` then selects the replay rows out of
``model_output["log_probs"]`` -- which carry gradient -- and applies the weights.

Layout contract (fields added to the batch TensorDict):
    is_replay            : [N] bool   -- True for appended replay rows
    replay_token_weights : [N, T]     -- per-token weight, 0 for non-replay rows
    advantages           : replay rows set to 0 so verl ppo_loss contributes 0

Pure helpers (``align_token_weights``, ``select_replay_rows``) are unit-tested
without verl. ``build_replay_rows`` is verl/tokenizer specific and is validated
on the GPU cluster (see doc/Progress.md verl checklist).
"""

from __future__ import annotations

from typing import Any

IS_REPLAY_KEY = "is_replay"
REPLAY_WEIGHTS_KEY = "replay_token_weights"


def align_token_weights(weights_per_traj, num_rows, seq_len, device=None, dtype=None):
    """Turn ragged per-trajectory weight lists into a padded ``[num_rows, seq_len]`` tensor.

    Args:
        weights_per_traj: list (len R) of per-token weight sequences (variable length),
            or None. Each inner item is a list/sequence of floats.
        num_rows: total rows in the batch (RL + replay). Non-replay rows get 0.
        seq_len: response length T (columns).
        device / dtype: target tensor spec.

    Returns:
        ``[num_rows, seq_len]`` tensor. The LAST ``len(weights_per_traj)`` rows are
        filled with the (right-truncated / zero-padded) replay weights, matching the
        append-at-end convention used by ``build_replay_rows``.
    """
    import torch

    dtype = dtype or torch.float32
    out = torch.zeros((num_rows, seq_len), device=device, dtype=dtype)
    if not weights_per_traj:
        return out

    r = len(weights_per_traj)
    start = num_rows - r
    for i, w in enumerate(weights_per_traj):
        if w is None:
            continue
        row = start + i
        n = min(len(w), seq_len)
        if n > 0:
            out[row, :n] = torch.as_tensor(w[:n], device=device, dtype=dtype)
    return out


def select_replay_rows(log_probs, response_mask, token_weights, is_replay):
    """Compute ``masked_mean(-log_probs * w)`` over replay rows only.

    All tensors are ``[N, T]`` (or broadcastable); ``is_replay`` is ``[N]`` bool.
    Returns a scalar tensor with gradient flowing through ``log_probs``.
    """
    import torch

    if is_replay is None or is_replay.sum() == 0:
        return log_probs.sum() * 0.0

    rows = is_replay.to(torch.bool)
    lp = log_probs[rows]
    mask = response_mask[rows].to(torch.bool)
    if token_weights is not None:
        w = token_weights[rows].to(dtype=lp.dtype, device=lp.device)
        loss_mat = -lp * w
    else:
        loss_mat = -lp

    denom = mask.sum().clamp(min=1)
    return (loss_mat * mask).sum() / denom


def build_replay_rows(
    samples: list[tuple[str, Any, dict]],
    token_weights: Any,
    tokenizer: Any,
    max_length: int = 4096,
) -> dict[str, Any]:
    """Tokenize replay message lists into verl-compatible row tensors.

    Returns a dict of tensors to be concatenated onto the actor batch:
        input_ids, attention_mask, position_ids, responses, response_mask,
        old_log_probs (placeholder), advantages (zeros), replay_token_weights, is_replay.

    NOTE: verl's exact batch schema (no_padding vs padded, multi-turn) is validated
    on the GPU cluster. This builds the right-padded dense form; adapt in
    ``CLTaskRunner`` if the configured engine expects the nested/no-padding form.
    """
    import torch

    input_ids_rows: list[list[int]] = []
    resp_mask_rows: list[list[int]] = []
    weight_lists: list[list[float]] = []

    tw = token_weights if token_weights is not None else [None] * len(samples)
    for (_tid, _traj, meta), w in zip(samples, tw):
        messages = meta.get("messages") or []
        if not messages:
            continue
        enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if isinstance(enc, dict):
            ids = enc["input_ids"]
        else:
            ids = enc
        ids = list(ids)[:max_length]
        # Response = assistant/tool spans; without offset map we approximate by
        # treating the full sequence after the first user turn as response. The
        # precise span is recomputed on the cluster from chat template offsets.
        resp_mask = [1] * len(ids)
        input_ids_rows.append(ids)
        resp_mask_rows.append(resp_mask)
        weight_lists.append(list(w) if w is not None else [1.0] * len(ids))

    if not input_ids_rows:
        return {}

    seq_len = max(len(r) for r in input_ids_rows)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def _pad(rows, fill):
        return torch.tensor([r + [fill] * (seq_len - len(r)) for r in rows])

    input_ids = _pad(input_ids_rows, pad_id)
    response_mask = _pad(resp_mask_rows, 0)
    attention_mask = (input_ids != pad_id).long()
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.clamp(min=0)
    weights = align_token_weights(weight_lists, len(input_ids_rows), seq_len)
    n = len(input_ids_rows)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": input_ids,
        "response_mask": response_mask,
        "old_log_probs": torch.zeros((n, seq_len)),
        "advantages": torch.zeros((n, seq_len)),
        REPLAY_WEIGHTS_KEY: weights,
        IS_REPLAY_KEY: torch.ones(n, dtype=torch.bool),
    }
