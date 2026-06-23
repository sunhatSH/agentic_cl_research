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

    Bug-2b fix: a replay row must be SHAPE-compatible with a verl rollout row so
    that ``_update_actor``'s ``left_right_2_no_padding`` (run on the whole
    concatenated batch) and ``ppo_loss``'s ``no_padding_2_padding`` both work. A
    rollout row is a left-padded ``prompt`` segment + a right-padded ``response``
    segment; ``no_padding_2_padding`` reads ``prompts.shape[1]`` /
    ``responses.shape[1]`` / ``attention_mask`` to slice the response log-probs.
    So we split each replay trajectory into (prompt tokens, response tokens) and
    emit:
        prompts          : [n, P]  left-padded prompt segment
        responses        : [n, R]  right-padded response segment (assistant/tool)
        input_ids        : [n, P+R] = prompts ++ responses
        attention_mask   : [n, P+R] real-token mask (left+right padding -> 0)
        position_ids     : [n, P+R] cumsum of attention_mask
        response_mask    : [n, R] ALL ZEROS  -> ppo_loss ignores replay rows (B8)
        replay_response_mask : [n, R] real response span -> drives L_replay
        replay_token_weights : [n, R] per-token weight (response-aligned)
        old_log_probs / ref_log_prob / advantages : [n, R] zeros (B10)
        is_replay        : [n] all True

    Response-length alignment (R) matters: ``no_padding_2_padding`` returns dense
    ``[bsz, max_response_len]`` log-probs, so the replay mask / weights are R-wide,
    NOT (P+R)-wide.
    """
    import torch

    prompt_rows: list[list[int]] = []
    resp_rows: list[list[int]] = []
    weight_lists: list[list[float]] = []
    built_tids: list[str] = []

    tw = token_weights if token_weights is not None else [None] * len(samples)
    for (_tid, _traj, meta), w in zip(samples, tw, strict=True):
        messages = meta.get("messages") or []
        if not messages:
            continue
        prompt_ids, resp_ids = _split_prompt_response(messages, tokenizer, max_length)
        if not prompt_ids or not resp_ids:
            # no_padding_2_padding asserts prompt_len > 0; skip degenerate rows
            continue
        prompt_rows.append(prompt_ids)
        resp_rows.append(resp_ids)
        # weight aligned to the RESPONSE segment only
        if w is not None:
            weight_lists.append(list(w)[: len(resp_ids)])
        else:
            weight_lists.append([1.0] * len(resp_ids))
        built_tids.append(_tid)

    if not prompt_rows:
        return {}

    n = len(prompt_rows)
    P = max(len(r) for r in prompt_rows)
    R = max(len(r) for r in resp_rows)
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # prompt: LEFT-padded; response: RIGHT-padded (verl rollout convention).
    prompts = torch.full((n, P), pad_id, dtype=torch.long)
    responses = torch.full((n, R), pad_id, dtype=torch.long)
    prompt_mask = torch.zeros((n, P), dtype=torch.long)
    resp_attn = torch.zeros((n, R), dtype=torch.long)
    replay_response_mask = torch.zeros((n, R), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompt_rows, resp_rows, strict=True)):
        prompts[i, P - len(p) :] = torch.tensor(p, dtype=torch.long)
        prompt_mask[i, P - len(p) :] = 1
        responses[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        resp_attn[i, : len(r)] = 1
        replay_response_mask[i, : len(r)] = 1

    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat([prompt_mask, resp_attn], dim=1)
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    weights = align_token_weights(weight_lists, n, R)

    return {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        # verl PPO mask = 0 for replay rows -> ppo_loss contributes nothing (B8).
        "response_mask": torch.zeros((n, R), dtype=torch.long),
        # Real response span (R-wide, matches no_padding_2_padding output).
        REPLAY_MASK_KEY: replay_response_mask,
        "old_log_probs": torch.zeros((n, R)),
        "ref_log_prob": torch.zeros((n, R)),
        "advantages": torch.zeros((n, R)),
        REPLAY_WEIGHTS_KEY: weights,
        IS_REPLAY_KEY: torch.ones(n, dtype=torch.bool),
        # Non-tensor sidecar: trajectory ids aligned with the rows above. Consumed
        # by the forgetting backfill and popped before DataProto.concat.
        REPLAY_TIDS_KEY: built_tids,
    }


def _split_prompt_response(
    messages: list[dict], tokenizer: Any, max_length: int
) -> tuple[list[int], list[int]]:
    """Split a chat trajectory into (prompt_ids, response_ids).

    Prompt = everything up to and including the first user turn (the task);
    response = the assistant/tool turns the policy is trained on. Uses the
    tokenizer chat template; the exact assistant-span recovery (offset map) is
    refined on the GPU cluster, but the prompt/response BOUNDARY here is what
    no_padding_2_padding needs to slice correctly.
    """
    # First user turn (inclusive) is the prompt; the rest is the response.
    split = 1
    for i, m in enumerate(messages):
        if m.get("role") in ("assistant", "tool"):
            split = i
            break
    else:
        split = len(messages)
    prompt_msgs = messages[:split] or messages[:1]
    resp_msgs = messages[split:]

    def _enc(msgs, add_gen):
        if not msgs:
            return []
        enc = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_gen)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc
        return list(ids)

    prompt_ids = _enc(prompt_msgs, add_gen=True)[:max_length]
    resp_ids = _enc(resp_msgs, add_gen=False)[: max(0, max_length - len(prompt_ids))]
    return prompt_ids, resp_ids
