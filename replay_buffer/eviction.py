"""In-bucket eviction strategy.

Rules (from ``doc/BucketDesign.md``):
- Bucket not full -> accept new trajectory directly.
- Bucket full -> evict the lowest-priority trajectory IN THIS BUCKET ONLY.
- Cross-bucket displacement is forbidden.
- Empty bucket first reception -> accept with elevated initial priority
  (boost for pioneering samples).

Two-tier sizing:
- hard floor (q_min) -- never evicted below this regardless of priority.
- soft target (formula) -- exceeding accelerates eviction, below accelerates intake.
"""

from __future__ import annotations

from replay_buffer.store import TrajectoryStore


class Eviction:
    """In-bucket eviction policy.

    Args:
        q_min: hard floor per bucket. Eviction never reduces a bucket below
               q_min, even if its priority is the lowest globally.
        soft_target: dict mapping bucket name -> int soft quota. Computed by
                     BucketReplayBuffer's quota allocator. Eviction triggers
                     when bucket_size > soft_target[bucket].
        pioneer_boost: priority added to the first few trajectories entering
                       a near-empty bucket, so they survive until comparable
                       peers arrive. Default 0.5.
        pioneer_threshold: bucket sizes below this get the boost. Default 10.
    """

    def __init__(
        self,
        q_min: int,
        soft_target: dict[str, int],
        pioneer_boost: float = 0.5,
        pioneer_threshold: int = 10,
    ):
        self.q_min = q_min
        self.soft_target = dict(soft_target)
        self.pioneer_boost = pioneer_boost
        self.pioneer_threshold = pioneer_threshold

    def should_evict(self, store: TrajectoryStore, bucket: str) -> bool:
        """Return True iff bucket size exceeds its soft_target AND has room
        above q_min to evict from."""
        size = store.bucket_size(bucket)
        target = self.soft_target.get(bucket, self.q_min)
        return size > target and size > self.q_min

    def select_victim(self, store: TrajectoryStore, bucket: str) -> str | None:
        """Return the trajectory_id to evict (lowest priority in bucket).

        Returns None if bucket is at or below q_min (cannot evict further).
        """
        if store.bucket_size(bucket) <= self.q_min:
            return None
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
