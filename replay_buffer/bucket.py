"""Top-level replay buffer with 7 capability buckets.

Implements the design from ``doc/BucketDesign.md``:
- 7 buckets by capability (Workflow / SysOps / Dialogue / Finance /
  Communication / Knowledge / OfficeQA), not by difficulty.
- Quota allocation: q_min hard floor + sqrt-weighted soft target.
- In-bucket eviction only -- no cross-bucket displacement.
- Trajectory metadata: trajectory_id, bucket, priority, insert_step,
  last_replay_step, replay_count, token_length, pattern_id, etc.
"""


class BucketReplayBuffer:
    """7-bucket replay buffer.

    Args:
        num_buckets: number of buckets, fixed at 7.
        total_capacity: C, total trajectory slots (10k-50k).
        q_min: hard floor per bucket.
        bucket_names: list of 7 capability names.
        bucket_task_counts: list of n_i, used for quota allocation.
        alpha: sub-linear weighting exponent in quota formula (default 0.5).
    """

    def __init__(
        self,
        num_buckets: int = 7,
        total_capacity: int = 25000,
        q_min: int = 2000,
        bucket_names=None,
        bucket_task_counts=None,
        alpha: float = 0.5,
    ):
        raise NotImplementedError

    def add_trajectory(self, trajectory, bucket_name: str) -> None:
        """Insert a trajectory into the named bucket; evict in-bucket if full."""
        raise NotImplementedError

    def add_trajectories(self, batch) -> None:
        """Bulk add; routes each trajectory to its bucket."""
        raise NotImplementedError

    def sample(self, batch_size: int):
        """Two-level sampling -- delegates to TwoLevelSampler."""
        raise NotImplementedError

    def compute_token_weights(self, replay_batch):
        """Compute per-token w_t for the replay batch -- delegates to TokenWeighting."""
        raise NotImplementedError

    def stats(self) -> dict:
        """Return per-bucket fill ratios, total size, eviction counters."""
        raise NotImplementedError
