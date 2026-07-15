"""In-bucket eviction strategy.

Rules (from ``doc/BucketDesign.md``):
- Bucket not full -> accept new trajectory directly.
- Bucket full -> evict the lowest-priority trajectory IN THIS BUCKET ONLY.
- Cross-bucket displacement is forbidden.
- Empty bucket first reception -> accept with elevated initial priority
  (boost for pioneering samples).

Two-tier sizing:
- hard floor (bucket_floors) -- never evicted below this regardless of priority.
- soft target (formula) -- exceeding accelerates eviction, below accelerates intake.
"""

from __future__ import annotations

import random

from replay_buffer.store import TrajectoryStore


class Eviction:
    """In-bucket eviction policy.

    Args:
        soft_target: dict mapping bucket name -> int soft quota. Computed by
                     BucketReplayBuffer's quota allocator. Eviction triggers
                     when bucket_size > soft_target[bucket].
        floors: per-bucket hard floor. Eviction never reduces a bucket below
                its floor. Required (no scalar fallback).
        eviction_type: 'priority' (default) evicts the lowest-priority
                       trajectory in the bucket. 'reservoir' evicts a
                       uniformly random trajectory -- used by the R0 CLEAR
                       baseline (random discard).
        pioneer_boost: priority added to the first few trajectories entering
                       a near-empty bucket, so they survive until comparable
                       peers arrive. Default 0.5.
        pioneer_threshold: bucket sizes below this get the boost. Default 10.
        rng: optional random.Random for reproducible reservoir eviction.
    """

    def __init__(
        self,
        soft_target: dict[str, int],
        floors: dict[str, int] | None = None,
        eviction_type: str = "priority",
        pioneer_boost: float = 0.5,
        pioneer_threshold: int = 10,
        rng: random.Random | None = None,
    ):
        if eviction_type not in ("priority", "reservoir"):
            raise ValueError(f"unknown eviction_type {eviction_type!r}")
        self.floors = dict(floors or {})
        self.soft_target = dict(soft_target)
        self.eviction_type = eviction_type
        self.pioneer_boost = pioneer_boost
        self.pioneer_threshold = pioneer_threshold
        self.rng = rng or random.Random()

    def _floor(self, bucket: str) -> int:
        """Hard floor for this bucket."""
        return self.floors.get(bucket, 0)

    def should_evict(self, store: TrajectoryStore, bucket: str) -> bool:
        """Return True iff bucket size exceeds its soft_target AND has room
        above the bucket's hard floor to evict from."""
        size = store.bucket_size(bucket)
        floor = self._floor(bucket)
        target = self.soft_target.get(bucket, floor)
        return size > target and size > floor

    def select_victim(self, store: TrajectoryStore, bucket: str) -> str | None:
        """Return the trajectory_id to evict.

        - 'priority' mode: lowest-priority trajectory in the bucket.
        - 'reservoir' mode: a uniformly random trajectory in the bucket.

        Returns None if bucket is at or below its hard floor (cannot evict further).
        """
        if store.bucket_size(bucket) <= self._floor(bucket):
            return None
        if self.eviction_type == "reservoir":
            ids = store.list_by_bucket(bucket)
            return self.rng.choice(ids) if ids else None
        victims = store.bottom_k_priority(bucket, k=1)
        return victims[0] if victims else None

    def boost_initial_priority(self, store: TrajectoryStore, bucket: str, base_priority: float) -> float:
        """Apply pioneering-sample boost when bucket is near-empty.

        Rationale: when a new bucket starts receiving its first trajectories,
        their priority computed against an empty bucket may be artificially
        low (no neighbours for rarity/diversity). Boost ensures they survive
        long enough to bootstrap the bucket.
        """
        size = store.bucket_size(bucket)
        if size < self.pioneer_threshold:
            scale = 1.0 - (size / self.pioneer_threshold)
            return base_priority + self.pioneer_boost * scale
        return base_priority
