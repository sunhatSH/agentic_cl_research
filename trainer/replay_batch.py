"""Driver-side replay batch preparation.

Samples replay trajectories from the buffer, computes per-token weights, and
builds verl-compatible replay rows to APPEND to the actor training batch (so the
replay log-probs are produced -- with gradient -- by the actor forward pass).

See ``trainer/replay_forward.py`` for the gradient rationale and row layout.
"""

from __future__ import annotations

from typing import Any

from trainer.replay_forward import build_replay_rows
from trainer.trajectory_adapter import replay_sample_to_metadata


def effective_replay_batch_size(buffer: Any, batch_size: int, warmup_size: int) -> int:
    """Linearly ramp the replay batch size from 0 to ``batch_size`` while the
    buffer fills (cold-start warmup, doc/CL_Update_Sunhao.md "冷启动处理").

    ratio = min(1, total_size / warmup_size); effective = round(batch_size * ratio).
    ``warmup_size <= 0`` disables the ramp (always full batch_size). An empty
    buffer yields 0 -> the caller adds no replay rows (replay_ratio = 0).
    """
    total = len(buffer.store)
    if total == 0:
        return 0
    if warmup_size and warmup_size > 0:
        ratio = min(1.0, total / warmup_size)
        return int(round(batch_size * ratio))
    return batch_size


def prepare_replay_rows(
    buffer: Any,
    weighting: Any | None,
    tokenizer: Any,
    batch_size: int,
    max_length: int = 4096,
    warmup_size: int = 0,
) -> dict[str, Any]:
    """Sample + weight + tokenize replay rows. Empty dict when nothing to add.

    ``warmup_size`` ramps the effective batch size while the buffer is still
    filling (see ``effective_replay_batch_size``).
    """
    if buffer is None or batch_size <= 0 or tokenizer is None:
        return {}

    eff = effective_replay_batch_size(buffer, batch_size, warmup_size)
    if eff <= 0:
        return {}

    samples = buffer.sample(eff)
    if not samples:
        return {}

    replay_inputs = [replay_sample_to_metadata(s) for s in samples]
    token_weights = weighting.compute(replay_inputs) if weighting is not None else None
    return build_replay_rows(samples, token_weights, tokenizer, max_length=max_length)
