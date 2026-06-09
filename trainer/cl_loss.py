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

Token weight w_t comes from ``replay_buffer.weighting.TokenWeighting`` (W2 scheme,
U-shaped block weight: (gamma^block + delta^(K_i - block)) / 2, with K_i derived
from per-trajectory action-block segmentation -- structural tags like
<think> / <toolcall> / <observation> / <final_answer>. Action-block segmenter
is pending real rollout data; equal-length K=20 fallback is in place.

See ``doc/VerlIntegration.md`` section 3.2 for the skeleton this implements.
"""


def make_cl_loss(
    buffer,
    lambda_replay: float = 0.5,
    replay_batch_size: int = 32,
    use_token_weighting: bool = True,
    weighting_scheme: str = "W2",
):
    """Build a verl-compatible loss function with replay added.

    Args:
        buffer: ``BucketReplayBuffer`` instance.
        lambda_replay: lambda_3, weight on L_replay.
        replay_batch_size: number of replay trajectories per step.
        use_token_weighting: if True, apply W2 per-token weights; else uniform (W0).
        weighting_scheme: 'W0' or 'W2'.

    Returns:
        callable with signature
            (config, model_output, data, dp_group) -> (loss, metrics)
    """

    def cl_loss(config, model_output, data, dp_group=None):
        raise NotImplementedError

    return cl_loss


def compute_replay_loss(model, replay_batch, replay_token_weights):
    """One forward over replay_batch -> -log pi(a|s) * w_t aggregated.

    Note: this is the place where the FSDP / gradient-accumulation risk
    flagged in ``doc/VerlIntegration.md`` section 4 needs to be validated.
    Two options for staging the replay forward:
      (A) call model(...) inside cl_loss directly.
      (B) precompute log_probs in trainer layer via engine.infer_batch and
          pass them in via the ``data`` TensorDict.
    """
    raise NotImplementedError
