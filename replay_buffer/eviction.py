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


class Eviction:
    """In-bucket eviction policy.

    Args:
        q_min: hard floor per bucket.
        soft_target: per-bucket soft quota (computed by quota allocator).
    """

    def __init__(self, q_min: int, soft_target: dict):
        raise NotImplementedError

    def should_evict(self, bucket) -> bool:
        """Return True if bucket size is above soft_target and over q_min."""
        raise NotImplementedError

    def select_victim(self, bucket):
        """Return the trajectory to evict (lowest priority, never below q_min)."""
        raise NotImplementedError

    def boost_initial_priority(self, trajectory, bucket) -> float:
        """Apply pioneering-sample boost when bucket is near-empty."""
        raise NotImplementedError
