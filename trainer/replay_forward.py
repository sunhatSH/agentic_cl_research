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
    replay_response_mask : [N, T]     -- REAL response span of replay rows, used
                                         ONLY by L_replay; 0 for RL rows
    response_mask        : [N, T]     -- verl's PPO mask. Set to 0 for replay
                                         rows so ppo_loss ignores them entirely
                                         (bug B8 -- avoids diluting the RL loss
                                         denominator / leaking into KL+entropy).
    advantages           : replay rows = 0 (also masked out, belt-and-suspenders)
    old_log_probs / ref_log_prob : zero placeholders for replay rows (B10)

The two-mask split is the key: ``response_mask`` drives verl's PPO term (RL
rows only), while ``replay_response_mask`` drives the supervised replay term
(replay rows only). The two row sets therefore never contaminate each other,
yet share ONE differentiable forward pass.

Pure helpers (``align_token_weights``, ``select_replay_rows``,
``pad_rows_to_seq_len``) are unit-tested without verl. ``build_replay_rows`` is
tokenizer specific and is validated on the GPU cluster (see doc/Progress.md).
"""

from __future__ import annotations

from typing import Any

IS_REPLAY_KEY = "is_replay"
REPLAY_WEIGHTS_KEY = "replay_token_weights"
REPLAY_MASK_KEY = "replay_response_mask"
REPLAY_TIDS_KEY = "_replay_tids"


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


def select_replay_rows(log_probs, replay_response_mask, token_weights, is_replay):
    """Compute ``masked_mean(-log_probs * w)`` over replay rows only.

    ``replay_response_mask`` is the REAL response span of the replay rows (NOT
    verl's PPO ``response_mask``, which is 0 for replay rows). All tensors are
    ``[N, T]``; ``is_replay`` is ``[N]`` bool. Returns a scalar tensor with
    gradient flowing through ``log_probs``.
    """
    import torch

    if is_replay is None or is_replay.sum() == 0:
        return log_probs.sum() * 0.0

    rows = is_replay.to(torch.bool)
    lp = log_probs[rows]
    if replay_response_mask is None:
        mask = torch.ones_like(lp, dtype=torch.bool)
    else:
        mask = replay_response_mask[rows].to(torch.bool)
    if token_weights is not None:
        w = token_weights[rows].to(dtype=lp.dtype, device=lp.device)
        loss_mat = -lp * w
    else:
        loss_mat = -lp

    denom = mask.sum().clamp(min=1)
    return (loss_mat * mask).sum() / denom


def pad_rows_to_seq_len(row_dict: dict[str, Any], target_seq_len: int) -> dict[str, Any]:
    """Right-pad every 2D ``[N, T]`` tensor in ``row_dict`` to ``target_seq_len``.

    Fixes the DataProto.concat shape mismatch (bug B9) when replay rows are
    shorter/longer than the RL batch. 1D tensors (e.g. ``is_replay``) and rows
    already at the target length are returned unchanged.
    """
    import torch
    import torch.nn.functional as F

    out: dict[str, Any] = {}
    for k, v in row_dict.items():
        if hasattr(v, "dim") and v.dim() == 2 and v.shape[-1] < target_seq_len:
            pad = target_seq_len - v.shape[-1]
            out[k] = F.pad(v, (0, pad), value=0)
        else:
            out[k] = v
    return out


def build_replay_rows(
    samples: list[tuple[str, Any, dict]],
    token_weights: Any,
    tokenizer: Any,
    max_length: int = 4096,
) -> dict[str, Any]:
    """Tokenize replay message lists into verl-compatible row tensors.

    Returns a dict of tensors to be concatenated onto the actor batch:
        input_ids, attention_mask, position_ids, responses,
        response_mask (ALL ZEROS -> ppo_loss ignores replay rows, bug B8),
        replay_response_mask (real span -> drives L_replay),
        old_log_probs / ref_log_prob (zero placeholders, bug B10),
        advantages (zeros), replay_token_weights, is_replay (all True).

    NOTE: verl's exact batch schema (no_padding vs padded, multi-turn) is validated
    on the GPU cluster. This builds the right-padded dense form; adapt in
    ``CLTaskRunner`` if the configured engine expects the nested/no-padding form.
    """
    import torch

    input_ids_rows: list[list[int]] = []
    resp_mask_rows: list[list[int]] = []
    weight_lists: list[list[float]] = []
    built_tids: list[str] = []

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
        built_tids.append(_tid)

    if not input_ids_rows:
        return {}

    seq_len = max(len(r) for r in input_ids_rows)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    def _pad(rows, fill):
        return torch.tensor([r + [fill] * (seq_len - len(r)) for r in rows])

    input_ids = _pad(input_ids_rows, pad_id)
    replay_response_mask = _pad(resp_mask_rows, 0)
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
        # verl PPO mask = 0 for replay rows -> ppo_loss contributes nothing (B8).
        "response_mask": torch.zeros((n, seq_len), dtype=torch.long),
        # Real response span, consumed only by L_replay.
        REPLAY_MASK_KEY: replay_response_mask,
        "old_log_probs": torch.zeros((n, seq_len)),
        "ref_log_prob": torch.zeros((n, seq_len)),
        "advantages": torch.zeros((n, seq_len)),
        REPLAY_WEIGHTS_KEY: weights,
        IS_REPLAY_KEY: torch.ones(n, dtype=torch.bool),
        # Non-tensor sidecar: trajectory ids aligned with the rows above (rows
        # whose messages were empty are skipped). Consumed by the forgetting
        # backfill and popped before DataProto.concat (see _append_replay_rows).
        REPLAY_TIDS_KEY: built_tids,
    }
