"""Paper-evidence hooks: buffer-dynamics logging + forgetting-risk backfill.

These utilities turn the buffer's internal state into experiment evidence that
the CL paper needs but that the core loss / buffer logic does not itself emit:

1. ``flatten_buffer_stats`` + ``BufferStatsLogger`` -- per-step buffer dynamics
   (per-bucket size / fill ratio / eviction / reservoir-reject curves, active
   priority-signal weights). Surfaced to verl metrics (wandb) AND a sidecar
   JSONL for offline plotting.
2. ``per_row_masked_mean`` + ``backfill_forgetting`` -- recompute
   ``forgetting_risk`` (the a1=0.5 priority signal) by comparing each replayed
   trajectory's stored ``original_logprobs`` against the CURRENT policy's
   log-probs. Without this the signal stays ~0 and R4-vs-R5 cannot be argued.

Everything except the actual current-logprob *forward* is pure and unit-tested
off-GPU; the forward (``compute_replay_current_logprobs``) uses verl's existing
``compute_log_prob`` worker call and is validated on the cluster.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def flatten_buffer_stats(stats: dict, prefix: str = "buffer/") -> dict[str, float]:
    """Flatten ``BucketReplayBuffer.stats()`` into scalar wandb metrics.

    No key contains "max"/"min" so ``verl.utils.metric.reduce_metrics`` reduces
    each via ``np.mean`` (scalar-safe). Per-bucket dicts are expanded to
    ``{prefix}size/<bucket>`` style keys; active-signal weights are emitted as
    ``{prefix}signal_weight/<signal>`` so e.g. diversity=0 is verifiable.
    """
    out: dict[str, float] = {}
    out[f"{prefix}total_size"] = float(stats.get("total_size", 0))
    out[f"{prefix}total_capacity"] = float(stats.get("total_capacity", 0))

    per_bucket = stats.get("per_bucket", {}) or {}
    fill_ratios: list[float] = []
    for name, b in per_bucket.items():
        out[f"{prefix}size/{name}"] = float(b.get("size", 0))
        fr = float(b.get("fill_ratio", 0.0))
        out[f"{prefix}fill_ratio/{name}"] = fr
        fill_ratios.append(fr)
        out[f"{prefix}evictions/{name}"] = float(b.get("evictions", 0))
    if fill_ratios:
        out[f"{prefix}fill_ratio_mean"] = sum(fill_ratios) / len(fill_ratios)

    for name, n in (stats.get("reservoir_rejected", {}) or {}).items():
        out[f"{prefix}reservoir_rejected/{name}"] = float(n)

    for sig, w in (stats.get("priority_active_signals", {}) or {}).items():
        out[f"{prefix}signal_weight/{sig}"] = float(w)

    return out


class BufferStatsLogger:
    """Append flattened buffer stats as one JSON line per logged step.

    Sidecar to wandb so the buffer-dynamics figures survive even if the run's
    wandb history is trimmed. Safe to construct eagerly; the file/dir is created
    lazily on first ``log``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._initialized = False

    def log(self, step: int, stats: dict) -> None:
        if not self._initialized:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._initialized = True
        # ``step`` last so the training step wins over buffer's internal step
        # (they coincide after buffer.set_step, but be explicit).
        record = {**stats, "step": int(step)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def per_row_masked_mean(values: Any, mask: Any) -> list[float]:
    """Per-row mean of ``values`` over positions where ``mask`` is truthy.

    ``values`` / ``mask`` are ``[N, T]`` tensors. Rows with an all-zero mask
    return 0.0 (no observed tokens). Returned as a plain ``list[float]`` so the
    result can cross the Ray driver boundary and feed ``backfill_forgetting``.
    """

    m = mask.to(dtype=values.dtype)
    denom = m.sum(dim=-1).clamp(min=1.0)
    row_mean = (values * m).sum(dim=-1) / denom
    return [float(x) for x in row_mean.tolist()]


def backfill_forgetting(buffer: Any, tids: list[str], current_means: list[float]) -> int:
    """Write current-policy log-prob means back so ``forgetting_risk`` activates.

    For each replayed trajectory we store ``current_logprobs`` as the per-row
    mean repeated to the length of the stored ``original_logprobs``. Because
    ``Priority.forgetting_risk`` averages ``orig[i] - curr[i]``, a constant
    ``curr`` yields exactly ``mean(orig) - mean(curr)`` -- the per-trajectory
    log-prob drift, i.e. how much the current policy has "forgotten" the
    trajectory since it entered the buffer. ``update_priority`` recomputes the
    fused priority immediately. Returns the number of trajectories updated.
    """
    updated = 0
    for tid, mean in zip(tids, current_means, strict=True):
        meta = buffer.store.get_metadata(tid)
        if meta is None:
            continue
        orig = meta.get("original_logprobs")
        length = len(orig) if orig else 1
        buffer.update_priority(tid, current_logprobs=[float(mean)] * length)
        updated += 1
    return updated


def compute_replay_current_logprobs(trainer: Any, replay_rows: dict[str, Any]):
    """Current-policy per-row mean log-prob for replay rows (GPU path).

    Uses verl's existing ``actor_rollout_wg.compute_log_prob`` (a no-backward
    forward) over a standalone DataProto built from ``replay_rows``. Returns a
    ``list[float]`` aligned with the replay rows, or ``None`` if verl / the
    worker group is unavailable (off-GPU tests, fake trainers). The exact
    DataProto schema is validated on the cluster; failures degrade to ``None``
    so a missing forward never crashes the training step.
    """
    try:
        from verl import DataProto
    except ImportError:
        return None
    wg = getattr(trainer, "actor_rollout_wg", None)
    if wg is None or not hasattr(wg, "compute_log_prob"):
        return None

    rows = {k: v for k, v in replay_rows.items() if not k.startswith("_")}
    mask = rows.get(REPLAY_MASK_KEY)
    if mask is None:
        return None
    try:
        dp = DataProto.from_single_dict(rows)
        out = wg.compute_log_prob(dp)
        log_probs = out.batch["old_log_probs"]
    except Exception:
        return None
    return per_row_masked_mean(log_probs, mask)


# Imported lazily to keep this module importable without trainer.replay_forward
# pulling torch at module load (mirrors the rest of trainer/).
from trainer.replay_forward import REPLAY_MASK_KEY  # noqa: E402
