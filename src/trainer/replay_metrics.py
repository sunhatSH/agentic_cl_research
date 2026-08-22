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


def compute_replay_current_logprobs_v1(trainer: Any, replay_meta: Any, replay_mask: Any, temperature: float = 1.0):
    """Current-policy per-row mean log-prob for replay rows (v1 KVBatchMeta path).

    v1 (custom_sync / transfer_queue / KVBatchMeta) 下 replay 行已由
    ``cl_replay_hook_v1._append_replay_rows_v1`` 写进 tq（train 分区），所以这里走 verl v1
    原生的 ``actor_rollout_wg.compute_log_prob(KVBatchMeta)`` → ``tq.kv_batch_get`` 读回
    ``log_probs``（nested 全序列）→ ``response_from_nested`` 按 ``response_mask`` 的长度切出
    response 段 → dense 后按 ``replay_mask`` 取每行 masked mean。与 verl 的
    ``_compute_old_log_prob`` / ``_compute_ref_log_prob`` 同一套契约（v0 的 DataProto 路径见
    ``compute_replay_current_logprobs``，此处是它的 v1 版本）。

    ``replay_mask`` = ``build_replay_rows`` 产出的 dense ``[n, R]`` ``replay_response_mask``
    （真实 response span）。任一环节失败（verl/tq 不可用、worker 无 compute_log_prob）降级
    ``None``，不崩训练。
    """
    try:
        import torch
        import transfer_queue as tq
        from verl.workers.utils.padding import response_from_nested
    except ImportError:
        return None
    if replay_meta is None or replay_mask is None:
        return None
    wg = getattr(trainer, "actor_rollout_wg", None)
    if wg is None or not hasattr(wg, "compute_log_prob"):
        return None

    # forward-only 元数据（对齐 verl _compute_ref_log_prob：不算 loss / 不算 entropy）。
    extra_info = dict(getattr(replay_meta, "extra_info", None) or {})
    extra_info.update({"calculate_entropy": False, "compute_loss": False, "temperature": temperature})
    replay_meta.extra_info = extra_info

    try:
        wg.compute_log_prob(replay_meta)
        data = tq.kv_batch_get(
            keys=replay_meta.keys,
            partition_id=replay_meta.partition_id,
            select_fields=["log_probs", "response_mask"],
        )
        # log_probs 是 nested 全序列（prompt+response）；response_mask 只取 nested 长度切 response 段。
        log_probs = response_from_nested(data["log_probs"], data["response_mask"])
        # nested → dense [n, max_resp_len]；与 build_replay_rows 的 replay_mask [n,R] 对齐（同批同 R）。
        dense_lp = torch.nested.to_padded_tensor(log_probs.float(), padding=0.0)
        # 防御性宽度对齐（正常同 R 应一致；不一致时不崩，截断/补零后仍算 mean）。
        w = replay_mask.shape[1]
        if dense_lp.shape[1] > w:
            dense_lp = dense_lp[:, :w]
        elif dense_lp.shape[1] < w:
            dense_lp = torch.nn.functional.pad(dense_lp, (0, w - dense_lp.shape[1]))
        return per_row_masked_mean(dense_lp, replay_mask)
    except Exception:
        return None


# Imported lazily to keep this module importable without trainer.replay_forward
# pulling torch at module load (mirrors the rest of trainer/).
from trainer.replay_forward import REPLAY_MASK_KEY  # noqa: E402
