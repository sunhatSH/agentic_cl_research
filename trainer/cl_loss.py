"""Continual-Learning loss assembled on top of verl's ppo_loss.

Standard verl loss signature:
    loss_fn(config, model_output, data, dp_group=None) -> (loss_tensor, metrics_dict)

Composition:
    L_cl = lambda_1 * L_rl + lambda_2 * L_kl + lambda_3 * L_replay + lambda_4 * L_ent

- L_rl      : delegated to verl's ``ppo_loss`` (handles GRPO advantage etc.).
- L_kl      : verl already supports it -- toggled via config.use_kl_loss /
              kl_loss_coef / kl_loss_type. We only need to choose reverse KL
              and the right pi_ref ckpt.
- L_replay  : SFT-style supervised replay over batch sampled from BucketReplayBuffer.
              Implemented in this file -- one extra forward on the replay batch,
              then -log pi(a|s) * w_t per token.
- L_ent     : verl already supports it via config.entropy_coeff (lambda_4=0.001 fixed).

Performance — zero-coefficient short-circuit (MANDATORY):
    Project rule (see project memory `short_circuit_zero_coefficient`):
    any loss term whose coefficient is 0 MUST be skipped end-to-end:
        - no buffer sampling
        - no model forward
        - no KL computation against pi_ref
    Across 20 ablation experiments most lambdas are zero in most runs
    (B1 has lambda_2 = lambda_3 = 0; K* have lambda_3 = 0; R* have
    lambda_2 = 0). Failing to short-circuit wastes GPU time without
    affecting results.

Token weight w_t comes from ``replay_buffer.weighting.TokenWeighting`` (W2 scheme,
U-shaped block weight: (gamma^block + delta^(K_i - block)) / 2, with K_i derived
from per-trajectory message-block segmentation -- assistant / tool chat turns,
not XML tags. Equal-length K=20 fallback when message format is unavailable or
K_i == 1.

See ``doc/VerlIntegration.md`` section 3.2 for the skeleton this implements.
"""

from __future__ import annotations

from typing import Any


def make_cl_loss(
    buffer: Any | None = None,
    lambda_replay: float = 0.5,
    replay_batch_size: int = 32,
    use_token_weighting: bool = True,
    weighting_scheme: str = "W2",
    weighting: Any | None = None,
):
    """Build a verl-compatible loss function with replay added.

    Args:
        buffer: ``BucketReplayBuffer`` instance, or None when lambda_replay=0.
        lambda_replay: lambda_3, weight on L_replay. **If 0, replay path is
                       compiled away** -- no buffer sampling, no replay forward.
        replay_batch_size: number of replay trajectories per step (ignored
                           when lambda_replay=0).
        use_token_weighting: if True, apply token-level weights; else uniform.
        weighting_scheme: 'W0' or 'W2'. Selects the TokenWeighting scheme.
        weighting: optional pre-built TokenWeighting instance; if None one is
                   constructed lazily from weighting_scheme.

    Returns:
        callable with signature
            (config, model_output, data, dp_group) -> (loss, metrics)

    The returned closure dispatches to a branch chosen at construction time
    based on whether lambda_replay > 0. This avoids per-step branching cost
    AND removes the need for replay-related fields in the `data` TensorDict
    for the zero-replay experiments.
    """
    replay_enabled = lambda_replay > 0.0 and buffer is not None
    if replay_enabled and weighting is None:
        from replay_buffer.weighting import TokenWeighting
        weighting = TokenWeighting(scheme=weighting_scheme)

    def cl_loss_no_replay(config, model_output, data, dp_group=None):
        """RL + KL + entropy only. No buffer touch. Used when lambda_replay=0."""
        from verl.workers.utils.losses import ppo_loss
        rl_loss, metrics = ppo_loss(config, model_output, data, dp_group)
        metrics["actor/replay_loss"] = 0.0
        metrics["actor/replay_enabled"] = 0.0
        return rl_loss, metrics

    def cl_loss_with_replay(config, model_output, data, dp_group=None):
        """RL + KL + entropy + replay. Replay batch sampled per call."""
        from verl.workers.utils.losses import ppo_loss
        rl_loss, metrics = ppo_loss(config, model_output, data, dp_group)

        replay_batch = buffer.sample(replay_batch_size)
        if not replay_batch:
            # Buffer empty (early training). Skip replay this step.
            metrics["actor/replay_loss"] = 0.0
            metrics["actor/replay_enabled"] = 1.0
            metrics["actor/replay_empty"] = 1.0
            return rl_loss, metrics

        replay_inputs = [meta for _, _, meta in replay_batch]
        token_weights = weighting.compute(replay_inputs) if use_token_weighting else None
        replay_loss = compute_replay_loss(
            model_output=model_output,
            replay_batch=replay_inputs,
            replay_token_weights=token_weights,
        )

        total = rl_loss + lambda_replay * replay_loss
        metrics["actor/replay_loss"] = float(getattr(replay_loss, "item", lambda: replay_loss)())
        metrics["actor/replay_enabled"] = 1.0
        metrics["actor/replay_empty"] = 0.0
        return total, metrics

    return cl_loss_with_replay if replay_enabled else cl_loss_no_replay


def compute_replay_loss(model_output, replay_batch, replay_token_weights):
    """One forward over replay_batch -> -log pi(a|s) * w_t aggregated.

    Two staging options (see ``doc/VerlIntegration.md`` section 4):
      (A) call model(...) inside cl_loss directly.
      (B) precompute log_probs in trainer layer via engine.infer_batch and
          pass them in via the ``data`` TensorDict.

    Pending integration with verl's engine API. Caller must NOT invoke
    this when lambda_replay == 0 -- the make_cl_loss closure handles that
    short-circuit.
    """
    raise NotImplementedError(
        "Replay forward pending verl engine integration; see VerlIntegration.md §4."
    )
