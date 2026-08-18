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

import logging
from typing import Any

from trainer.replay_forward import (
    IS_REPLAY_KEY,
    REPLAY_MASK_KEY,
    REPLAY_WEIGHTS_KEY,
    _replay_debug,
    _replay_debug_enabled,
    _summarize,
    select_replay_rows,
)

logger = logging.getLogger(__name__)


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


def _safe_scalar_repr(tensor) -> str:
    """把布尔/标量张量安全地打成一行诊断（优先 .sum()，失败退回 repr）。"""
    if tensor is None:
        return "None"
    try:
        s = tensor.sum()
        return f"sum={int(s.item())}"
    except Exception:  # noqa: BLE001 -- 诊断兜底
        return repr(tensor)


def compute_replay_loss(model_output, data):
    """Differentiable replay loss over appended replay rows.

    Pulls ``log_probs`` from ``model_output`` (gradient-carrying), and
    ``replay_response_mask`` / ``replay_token_weights`` / ``is_replay`` from
    ``data``. The replay term uses ``replay_response_mask`` (the real span),
    NOT the PPO ``response_mask`` -- the latter is 0 for replay rows so verl's
    ppo_loss skips them (bug B8). Returns a scalar (0 when no replay rows).

    Bug-2 fix: verl's engine produces ``model_output["log_probs"]`` as a
    NestedTensor (jagged, ``use_remove_padding=True`` default) covering the WHOLE
    sequence, NOT a dense ``[N, T]`` response tensor. We MUST run it through
    ``no_padding_2_padding`` first -- exactly as verl's own ``ppo_loss`` does on
    its first line -- to get dense ``[bsz, max_response_len]`` log-probs before
    selecting replay rows. ``replay_response_mask`` / ``replay_token_weights``
    are response-length aligned to match.
    """
    log_probs = model_output["log_probs"] if isinstance(model_output, dict) else model_output.get("log_probs")
    is_replay = _data_get(data, IS_REPLAY_KEY)
    replay_mask = _data_get(data, REPLAY_MASK_KEY)
    token_weights = _data_get(data, REPLAY_WEIGHTS_KEY)

    if _replay_debug_enabled():
        # 密集前 dump：定位「log_probs≈0」还是「mask/weights≈0」，以及 weights 的归一化尺度。
        _replay_debug(
            "compute_replay_loss(密集前): is_replay.sum()=%s "
            "log_probs=%s | replay_mask=%s | token_weights=%s"
            % (
                _safe_scalar_repr(is_replay),
                _summarize(log_probs),
                _summarize(replay_mask),
                _summarize(token_weights),
            )
        )

    # Restore dense [bsz, max_response_len] (same as ppo_loss line 1). On a real
    # verl batch log_probs is a NestedTensor; off-cluster mocks pass a dense
    # tensor, so only convert when the verl helper is importable and the tensor
    # is nested / needs slicing.
    log_probs = _to_dense_response_logprobs(log_probs, data)
    # 回放行变长后 mask/weights 也是 nested（response 对齐）→ 转 dense 对齐 log_probs。
    replay_mask = _to_dense_response(replay_mask, log_probs)
    token_weights = _to_dense_response(token_weights, log_probs)

    if _replay_debug_enabled():
        _replay_debug(
            "compute_replay_loss(密集后): log_probs=%s | replay_mask=%s | token_weights=%s"
            % (
                _summarize(log_probs),
                _summarize(replay_mask),
                _summarize(token_weights),
            )
        )

    return select_replay_rows(log_probs, replay_mask, token_weights, is_replay)


def _to_dense_response(tensor, dense_log_probs):
    """把 response 对齐的 nested tensor 转 dense [N, max_response_len]（right-pad 0），
    对齐已转 dense 的 log_probs。非 nested（本机 mock / dense 输入）原样返回。"""
    import torch

    if tensor is None or not getattr(tensor, "is_nested", False):
        return tensor
    n = tensor.size(0)
    t = dense_log_probs.shape[1] if dense_log_probs is not None and dense_log_probs.dim() == 2 else None
    if t is None:
        return tensor
    return torch.nested.to_padded_tensor(tensor, padding=0, output_size=(n, t))


def _to_dense_response_logprobs(log_probs, data):
    """Convert verl NestedTensor log_probs to dense [bsz, max_response_len].

    No-op when verl is not importable (unit tests pass dense tensors directly)
    or when ``data`` lacks the prompts/responses metadata the converter needs.
    """
    try:
        from verl.workers.utils.padding import no_padding_2_padding
    except ImportError:
        return log_probs
    is_nested = getattr(log_probs, "is_nested", False)
    has_meta = hasattr(data, "__contains__") and ("responses" in data)
    if not (is_nested or has_meta):
        return log_probs
    try:
        return no_padding_2_padding(log_probs, data)
    except Exception:  # noqa: BLE001 -- mocks / dense inputs fall through unchanged
        logger.warning(
            "no_padding_2_padding failed — returning unconverted log_probs "
            "to replay loss. This is expected during mock tests but may "
            "produce silently wrong replay loss on the cluster.",
            exc_info=True,
        )
        return log_probs


def _replay_is_empty(data) -> bool:
    is_replay = _data_get(data, IS_REPLAY_KEY)
    if is_replay is None:
        return True
    try:
        return bool(is_replay.sum() == 0)
    except AttributeError:
        return not any(bool(x) for x in is_replay)


def _resolve_ppo_loss(actor_cfg):
    """Return a ``partial(ppo_loss, config=ActorConfig)`` callable for RL term.

    Bug-1 fix: verl's engine calls the registered loss as
    ``loss_function(model_output=..., data=..., dp_group=...)`` -- it does NOT
    pass ``config`` (verl curries it via ``partial(ppo_loss, config=actor_config)``,
    engine_workers.py:584). When we override the loss via ``set_loss_fn`` we lose
    that binding, so we rebuild it here on the worker: ``actor_cfg`` is the
    (pickle-safe) OmegaConf ``actor_rollout_ref.actor`` subtree, converted to an
    ``ActorConfig`` exactly as verl does (engine_workers.py:543). Cached per call
    site via a closure cell. Returns None when verl is absent (mock tests).
    """
    try:
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config.actor import ActorConfig  # noqa: F401
        from verl.workers.utils.losses import ppo_loss
    except ImportError:
        return None
    from functools import partial

    config = omega_conf_to_dataclass(actor_cfg) if actor_cfg is not None else None
    return partial(ppo_loss, config=config)


def make_cl_loss(
    replay_enabled: bool = False,
    lambda_replay: float = 0.5,
    replay_batch_size: int = 512,
    use_token_weighting: bool = True,
    weighting_scheme: str = "W2",
    actor_cfg: Any = None,
    base_loss_fn: Any = None,
):
    """Build a verl-compatible loss function.

    Args:
        replay_enabled: whether to compile the replay branch. When False the
            returned closure is RL+KL+entropy only (no replay fields touched).
        lambda_replay: lambda_3 weight on L_replay.
        replay_batch_size / use_token_weighting / weighting_scheme: kept for
            metric/debug parity; the actual sampling + weighting happens on the
            driver (``install_buffer_hooks``), not in this closure.
        actor_cfg: OmegaConf ``actor_rollout_ref.actor`` subtree, used to rebuild
            the ``ActorConfig`` that the RL term (``ppo_loss``) needs -- because
            verl calls the loss WITHOUT a ``config`` arg (bug-1). Pickle-safe.
        base_loss_fn: optional pre-bound RL loss ``fn(model_output, data,
            dp_group)``. When given it is used directly (tests / custom); else a
            ``partial(ppo_loss, config=ActorConfig(actor_cfg))`` is rebuilt lazily
            on the worker.

    The buffer is intentionally NOT a parameter -- it must not be pickled to the
    actor worker via ``set_loss_fn``. The closure signature matches verl's call
    convention: ``(model_output, data, dp_group=None)`` -- NO ``config`` arg.
    """
    enabled = bool(replay_enabled) and lambda_replay > 0.0
    # Lazily resolved on first call (on the worker) so ActorConfig is built in the
    # worker process, not pickled across Ray.
    _rl_cell: dict[str, Any] = {"fn": base_loss_fn}

    def _rl_loss(model_output, data, dp_group):
        if _rl_cell["fn"] is None:
            _rl_cell["fn"] = _resolve_ppo_loss(actor_cfg)
        fn = _rl_cell["fn"]
        if fn is None:  # verl absent (mock test path)
            lp = (
                model_output["log_probs"] if isinstance(model_output, dict) else model_output.get("log_probs")
            )
            return lp.sum() * 0.0, {}
        return fn(model_output=model_output, data=data, dp_group=dp_group)

    def cl_loss_no_replay(model_output, data, dp_group=None):
        """RL + KL + entropy only. Used when replay is disabled (B1 / K*)."""
        rl_loss, metrics = _rl_loss(model_output, data, dp_group)
        metrics["actor/replay_loss"] = 0.0
        metrics["actor/replay_enabled"] = 0.0
        return rl_loss, metrics

    def cl_loss_with_replay(model_output, data, dp_group=None):
        """RL + KL + entropy + differentiable replay over appended rows."""
        rl_loss, metrics = _rl_loss(model_output, data, dp_group)

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
