"""Top-level replay buffer with 9 capability buckets.

Implements the design from ``doc/BucketDesign.md``:
- 9 buckets (workflow / ops / qa / finance / office /
  communication / safety / coding / research), not by difficulty.
- Quota allocation: q_min hard floor + sqrt-weighted soft target.
- In-bucket eviction only -- no cross-bucket displacement.
- Trajectory metadata: trajectory_id, bucket, priority, insert_step,
  last_replay_step, replay_count, token_length, pattern_id, etc.

Quota formula (sub-linear weighting):
    soft_target_i = q_min + (C - K * q_min) * n_i^alpha / sum_j(n_j^alpha)
    where n_i = bucket_task_counts[i], C = total_capacity, K = num_buckets,
    alpha < 1 dampens the dominance of large buckets.
"""

from __future__ import annotations

import pickle
import random
import sqlite3
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from replay_buffer.eviction import Eviction
from replay_buffer.priority import Priority
from replay_buffer.store import TrajectoryStore


def allocate_quota(
    total_capacity: int,
    q_min: int,
    bucket_task_counts: Sequence[int],
    alpha: float = 0.5,
) -> list[int]:
    """Compute soft_target per bucket.

    soft_target_i = q_min + remaining * n_i^alpha / sum_j(n_j^alpha)
    where remaining = total_capacity - K * q_min.

    Args:
        total_capacity: C, total trajectory slots across all buckets.
        q_min: hard floor reserved per bucket.
        bucket_task_counts: n_i for each bucket; relative size weights.
        alpha: sub-linear exponent. alpha=1.0 = proportional; alpha=0.5
               (default) = square-root weighting; alpha=0.0 = uniform.

    Returns:
        List of soft_target values, length = len(bucket_task_counts).
        Sums to total_capacity (modulo integer rounding -- the largest
        bucket absorbs the rounding residue).
    """
    k = len(bucket_task_counts)
    if total_capacity < k * q_min:
        raise ValueError(f"total_capacity ({total_capacity}) must be >= K * q_min ({k * q_min})")
    weights = [n**alpha for n in bucket_task_counts]
    s = sum(weights)
    remaining = total_capacity - k * q_min
    targets = [q_min + int(remaining * w / s) for w in weights]
    residue = total_capacity - sum(targets)
    if residue != 0:
        idx = max(range(k), key=lambda i: weights[i])
        targets[idx] += residue
    return targets


class BucketReplayBuffer:
    """9-bucket replay buffer.

    Args:
        num_buckets: number of buckets, default 9 (overridable for R0 single-buffer ablation).
        total_capacity: C, total trajectory slots (10k-50k).
        q_min: hard floor per bucket.
        bucket_names: list of K capability names.
        bucket_task_counts: list of n_i, used for quota allocation.
        alpha: sub-linear weighting exponent in quota formula (default 0.5).
        priority: Priority instance (anti-forgetting); pass RewardPriority for R5.
        eviction_type: 'priority' (default) or 'reservoir' (R0 CLEAR baseline).
        within_bucket_sampling: 'priority' (default) or 'uniform' (R0 / R3).
        seed: optional RNG seed for reproducible reservoir / sampling.

    Single-bucket mode (R0 CLEAR baseline): pass ``num_buckets=1``. When the
    default 9 ``bucket_names`` / ``bucket_task_counts`` are inherited from a
    multi-bucket config, they are automatically collapsed to one ``"All"``
    bucket whose task count is the sum -- so ``configs/phase3/r0.yaml`` can set
    only ``num_buckets: 1`` without redefining the name/count lists (bug A2).
    """

    def __init__(
        self,
        num_buckets: int = 9,
        total_capacity: int = 25000,
        q_min: int = 2000,
        bucket_names: Sequence[str] | None = None,
        bucket_task_counts: Sequence[int] | None = None,
        alpha: float = 0.5,
        priority: Priority | None = None,
        eviction_type: str = "priority",
        within_bucket_sampling: str = "priority",
        seed: int | None = None,
    ):
        if bucket_names is None:
            bucket_names = [
                "workflow",
                "ops",
                "qa",
                "finance",
                "office",
                "communication",
                "safety",
                "coding",
                "research",
            ]
        if bucket_task_counts is None:
            bucket_task_counts = [56, 44, 36, 20, 11, 11, 9, 2, 6]

        # Single-bucket collapse: tolerate inheriting multi-bucket name/count
        # lists when num_buckets == 1 (R0 CLEAR baseline).
        if num_buckets == 1 and (len(bucket_names) != 1 or len(bucket_task_counts) != 1):
            bucket_names = ["All"]
            bucket_task_counts = [sum(bucket_task_counts)]

        if len(bucket_names) != num_buckets or len(bucket_task_counts) != num_buckets:
            raise ValueError("bucket_names and bucket_task_counts must each have length num_buckets")

        self.num_buckets = num_buckets
        self.total_capacity = total_capacity
        self.q_min = q_min
        self.bucket_names = list(bucket_names)
        self.bucket_task_counts = list(bucket_task_counts)
        self.alpha = alpha
        self.eviction_type = eviction_type
        self.within_bucket_sampling = within_bucket_sampling

        targets = allocate_quota(total_capacity, q_min, bucket_task_counts, alpha)
        self.soft_target = dict(zip(self.bucket_names, targets, strict=True))

        self._rng = random.Random(seed)
        self.store = TrajectoryStore(backend="memory")
        self.priority_fn = priority or Priority()
        self.eviction = Eviction(
            q_min=q_min,
            soft_target=self.soft_target,
            eviction_type=eviction_type,
            rng=self._rng,
        )
        # Persistent sampler so starvation / last-sample state survives across
        # sample() calls (bug A4 -- previously a fresh sampler was built each call).
        from replay_buffer.sampler import TwoLevelSampler

        self._sampler = TwoLevelSampler(
            self,
            within_bucket_sampling=within_bucket_sampling,
            rng=self._rng,
        )

        self._step = 0
        self._seen_counts: dict[str, int] = {b: 0 for b in self.bucket_names}
        self._eviction_counts: dict[str, int] = {b: 0 for b in self.bucket_names}
        self._rejected_counts: dict[str, int] = {b: 0 for b in self.bucket_names}

    def set_step(self, step: int) -> None:
        """Trainer calls this each step so insert/replay step counters stay in sync."""
        self._step = step

    def _bucket_view(self, bucket: str) -> dict:
        """Snapshot of bucket state for Priority computation.

        Lazily collected -- only the fields Priority actually reads.
        """
        return {
            "pattern_counts": self.store.pattern_counts(bucket=bucket),
            # peer_embeddings populated only on demand; expensive to gather.
            "peer_embeddings": [],
        }

    def add_trajectory(self, trajectory: Any, bucket: str, metadata: dict | None = None) -> str:
        """Insert a trajectory into the named bucket; evict in-bucket if full.

        Args:
            trajectory: opaque payload (token ids, message list, etc.).
            bucket: bucket name; must be in self.bucket_names.
            metadata: optional dict with pattern_id, original_logprobs,
                      success_rate, embedding, etc. Priority is computed
                      from these.

        Returns:
            trajectory_id assigned to this entry.
        """
        if bucket not in self.soft_target:
            raise ValueError(f"unknown bucket {bucket!r}")

        meta = dict(metadata or {})
        meta.setdefault("bucket", bucket)
        meta.setdefault("insert_step", self._step)
        meta.setdefault("last_replay_step", -1)
        meta.setdefault("replay_count", 0)

        base_prio = self.priority_fn.compute(meta, self._bucket_view(bucket))
        meta["priority"] = self.eviction.boost_initial_priority(self.store, bucket, base_prio)

        tid = meta.get("trajectory_id") or f"traj_{uuid.uuid4().hex[:12]}"
        meta["trajectory_id"] = tid

        self._seen_counts[bucket] += 1

        if self.eviction_type == "reservoir":
            return self._reservoir_add(tid, trajectory, meta, bucket)

        # Priority eviction: evict the lowest-priority in-bucket trajectory
        # FIRST when the bucket is at/over its soft target, then insert, so the
        # steady-state bucket size never exceeds soft_target (bug A5).
        while self.store.bucket_size(bucket) >= self.soft_target[bucket]:
            if self.store.bucket_size(bucket) <= self.q_min:
                break
            victim = self.eviction.select_victim(self.store, bucket)
            if victim is None:
                break
            self.store.delete(victim)
            self._eviction_counts[bucket] += 1

        self.store.put(tid, trajectory, meta)
        return tid

    def _reservoir_add(self, tid: str, trajectory: Any, meta: dict, bucket: str) -> str:
        """Classic reservoir sampling acceptance (R0 CLEAR baseline).

        While the bucket has room, accept directly. Once full, accept the new
        trajectory with probability ``cap / n_seen`` and evict a uniformly
        random existing trajectory; otherwise reject the new one. This gives
        each streamed trajectory an equal long-run retention probability --
        the "random discard" behaviour CLEAR relies on.
        """
        cap = self.soft_target[bucket]
        size = self.store.bucket_size(bucket)
        if size < cap:
            self.store.put(tid, trajectory, meta)
            return tid
        n_seen = self._seen_counts[bucket]
        if self._rng.random() < cap / max(n_seen, 1):
            # Reservoir manages capacity itself; the q_min hard floor (a
            # bucketed-mode concept) must NOT block the random replacement,
            # so pick the victim directly rather than via Eviction.select_victim.
            ids = self.store.list_by_bucket(bucket)
            if ids:
                self.store.delete(self._rng.choice(ids))
                self._eviction_counts[bucket] += 1
            self.store.put(tid, trajectory, meta)
            return tid
        self._rejected_counts[bucket] += 1
        return tid

    def add_trajectories(self, batch) -> list[str]:
        """Bulk add. Each item is (trajectory, bucket, metadata)."""
        return [self.add_trajectory(t, b, m) for t, b, m in batch]

    def sample(self, batch_size: int):
        """Two-level sampling -- delegates to the persistent TwoLevelSampler.

        The sampler is created once in ``__init__`` so starvation_boost and
        last-sample bookkeeping persist across calls (bug A4).
        """
        return self._sampler.sample(batch_size)

    def update_priority(self, trajectory_id: str, **signal_updates) -> None:
        """Recompute priority after new signals (e.g. updated current_logprobs)."""
        meta = self.store.get_metadata(trajectory_id)
        if meta is None:
            return
        meta.update(signal_updates)
        new_prio = self.priority_fn.compute(meta, self._bucket_view(meta["bucket"]))
        self.store.update_metadata(trajectory_id, priority=new_prio, **signal_updates)

    def mark_replayed(self, trajectory_ids: Sequence[str]) -> None:
        """Bump replay_count and last_replay_step for sampled trajectories."""
        for tid in trajectory_ids:
            meta = self.store.get_metadata(tid)
            if meta is None:
                continue
            self.store.update_metadata(
                tid,
                last_replay_step=self._step,
                replay_count=meta.get("replay_count", 0) + 1,
            )

    def stats(self) -> dict:
        """Per-bucket fill ratio, total size, eviction counters."""
        per_bucket = {}
        for name in self.bucket_names:
            size = self.store.bucket_size(name)
            target = self.soft_target[name]
            per_bucket[name] = {
                "size": size,
                "soft_target": target,
                "fill_ratio": size / target if target else 0.0,
                "evictions": self._eviction_counts[name],
            }
        active_signals = (
            self.priority_fn.active_signals() if hasattr(self.priority_fn, "active_signals") else {}
        )
        return {
            "total_size": len(self.store),
            "total_capacity": self.total_capacity,
            "per_bucket": per_bucket,
            "step": self._step,
            "eviction_type": self.eviction_type,
            "within_bucket_sampling": self.within_bucket_sampling,
            "priority_active_signals": active_signals,
            "reservoir_rejected": dict(self._rejected_counts),
        }

    # ------------------------------------------------------------------ #
    # Persistence (SQLite snapshot)                                       #
    # ------------------------------------------------------------------ #
    _STATE_KEYS = ("_step", "_seen_counts", "_eviction_counts", "_rejected_counts")

    def dump(self, path: str | Path) -> None:
        """Snapshot the whole buffer (trajectories + counters) to ``path``.

        Trajectories go through ``TrajectoryStore.save_sqlite``; the buffer's
        own counters (step / seen / eviction / rejected) are stored in a
        sidecar ``buffer_state`` table inside the SAME SQLite file, so a single
        file fully restores the buffer (doc/Progress.md "Buffer 持久化").
        """
        path = Path(path)
        self.store.save_sqlite(path)
        state = {k: getattr(self, k) for k in self._STATE_KEYS}
        conn = sqlite3.connect(str(path))
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS buffer_state (state BLOB)")
            conn.execute("DELETE FROM buffer_state")
            conn.execute(
                "INSERT INTO buffer_state VALUES (?)",
                (pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL),),
            )
            conn.commit()
        finally:
            conn.close()

    def load(self, path: str | Path) -> None:
        """Restore buffer trajectories + counters from a ``dump`` snapshot."""
        path = Path(path)
        self.store.load_sqlite(path)
        conn = sqlite3.connect(str(path))
        try:
            row = conn.execute("SELECT state FROM buffer_state").fetchone()
        finally:
            conn.close()
        if row is not None:
            state = pickle.loads(row[0])
            for k, v in state.items():
                setattr(self, k, v)
