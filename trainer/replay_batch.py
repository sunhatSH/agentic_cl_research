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


def prepare_replay_rows(
    buffer: Any,
    weighting: Any | None,
    tokenizer: Any,
    batch_size: int,
    max_length: int = 4096,
) -> dict[str, Any]:
    """Sample + weight + tokenize replay rows. Empty dict when nothing to add."""
    if buffer is None or batch_size <= 0 or tokenizer is None:
        return {}

    samples = buffer.sample(batch_size)
    if not samples:
        return {}

    replay_inputs = [replay_sample_to_metadata(s) for s in samples]
    token_weights = weighting.compute(replay_inputs) if weighting is not None else None
    return build_replay_rows(samples, token_weights, tokenizer, max_length=max_length)
