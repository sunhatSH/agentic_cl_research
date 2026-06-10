"""Continual-Learning loss assembled on top of verl's ppo_loss.

Standard verl loss signature:
    loss_fn(config, model_output, data, dp_group=None) -> (loss_tensor, metrics_dict)

Composition:
    L_cl = lambda_1 * L_rl + lambda_2 * L_kl + lambda_3 * L_replay + lambda_4 * L_ent

L_rl / L_kl / L_ent are produced by verl's ``ppo_loss``. L_replay is computed
here from the replay rows that ``install_buffer_hooks`` appended to the batch:
``model_output["log_probs"]`` already covers those rows WITH gradient, so the
replay term is differentiable (see ``trainer/replay_forward.py``).

The buffer object never enters this closure -- it lives only on the driver. The
loss reads everything it needs from ``model_output`` / ``data``.

See ``doc/VerlIntegration.md`` section 3.2.
"""

from __future__ import annotations

from typing import Any

from trainer.replay_forward import IS_REPLAY_KEY, REPLAY_WEIGHTS_KEY, select_replay_rows


def _get_non_tensor(data, key: str, default=None):
    try:
        from verl.utils import tensordict_utils as tu

        return tu.get_non_tensor_data(data=data, key=key, default=default)
    except ImportError:
        return default


def _data_get(data, key, default=None):
    if hasattr(data, "get"):
        try:
            return data.get(key, default)
        except TypeError:
            val = data.get(key)
            return val if val is not None else default
    return default


def compute_replay_loss(model_output, data):
    """Differentiable replay loss over appended replay rows.

    Pulls ``log_probs`` from ``model_output`` (gradient-carrying), and
    ``response_mask`` / ``replay_token_weights`` / ``is_replay`` from ``data``.
    Returns a scalar tensor (0 when there are no replay rows).
    """
    log_probs = model_output["log_probs"] if isinstance(model_output, dict) else model_output.get("log_probs")
    is_replay = _data_get(data, IS_REPLAY_KEY)
    response_mask = _data_get(data, "response_mask")
    token_weights = _data_get(data, REPLAY_WEIGHTS_KEY)
    return select_replay_rows(log_probs, response_mask, token_weights, is_replay)


def _replay_is_empty(data) -> bool:
    is_replay = _data_get(data, IS_REPLAY_KEY)
    if is_replay is None:
        return True
    try:
        return bool(is_replay.sum() == 0)
    except AttributeError:
        return not any(bool(x) for x in is_replay)


def make_cl_loss(
    replay_enabled: bool = False,
    lambda_replay: float = 0.5,
    replay_batch_size: int = 32,
    use_token_weighting: bool = True,
    weighting_scheme: str = "W2",
):
    """Build a verl-compatible loss function.

    Args:
        replay_enabled: whether to compile the replay branch. When False the
            returned closure is RL+KL+entropy only (no replay fields touched).
        lambda_replay: lambda_3 weight on L_replay.
        replay_batch_size / use_token_weighting / weighting_scheme: kept for
            metric/debug parity; the actual sampling + weighting happens on the
            driver (``install_buffer_hooks``), not in this closure.

    The buffer is intentionally NOT a parameter -- it must not be pickled to the
    actor worker via ``set_loss_fn``.
    """
    enabled = bool(replay_enabled) and lambda_replay > 0.0

    def cl_loss_no_replay(config, model_output, data, dp_group=None):
        """RL + KL + entropy only. Used when replay is disabled (B1 / K*)."""
        from verl.workers.utils.losses import ppo_loss

        rl_loss, metrics = ppo_loss(config, model_output, data, dp_group)
        metrics["actor/replay_loss"] = 0.0
        metrics["actor/replay_enabled"] = 0.0
        return rl_loss, metrics

    def cl_loss_with_replay(config, model_output, data, dp_group=None):
        """RL + KL + entropy + differentiable replay over appended rows."""
        from verl.workers.utils.losses import ppo_loss

        rl_loss, metrics = ppo_loss(config, model_output, data, dp_group)

        if _replay_is_empty(data):
            metrics["actor/replay_loss"] = 0.0
            metrics["actor/replay_enabled"] = 1.0
            metrics["actor/replay_empty"] = 1.0
            return rl_loss, metrics

        replay_loss = compute_replay_loss(model_output, data)
        total = rl_loss + lambda_replay * replay_loss
        metrics["actor/replay_loss"] = float(replay_loss.detach().item())
        metrics["actor/replay_enabled"] = 1.0
        metrics["actor/replay_empty"] = 0.0
        return total, metrics

    return cl_loss_with_replay if enabled else cl_loss_no_replay


def build_weighting_from_cfg(cl_cfg) -> Any | None:
    """Construct TokenWeighting from ``cfg.cl.weighting``."""
    wcfg = cl_cfg.get("weighting", {}) or {}
    scheme = wcfg.get("scheme", "W2")
    from replay_buffer.weighting import TokenWeighting

    return TokenWeighting(
        scheme=scheme,
        gamma=float(wcfg.get("gamma", 0.88)),
        delta=float(wcfg.get("delta", 0.88)),
        segmenter=wcfg.get("segmenter", "message_block"),
        role_boundaries=list(wcfg.get("role_boundaries", ["assistant", "tool"])),
        equal_length_K=int(wcfg.get("equal_length_K", 20)),
        long_block_threshold=int(wcfg.get("long_block_threshold", 100)),
        clip_quantiles=tuple(wcfg.get("clip_quantiles", [0.05, 0.95])),
    )
