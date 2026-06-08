"""Two-level sampling: pick bucket, then pick trajectory within bucket.

Bucket-level: mixed strategy -- partly proportional to soft_target quota (covers
large buckets) and partly uniform (covers long-tail capabilities). Buckets that
have not been replayed for a long time receive a starvation_boost.

Within-bucket: priority-weighted random sampling, NOT top-k greedy. Greedy
top-k repeats "star trajectories" and reduces diversity.
"""


class TwoLevelSampler:
    """Two-level sampler for replay batches.

    Args:
        buffer: BucketReplayBuffer instance.
        bucket_mix_ratio: weight of quota-proportional vs uniform bucket sampling.
        starvation_boost: extra weight for long-unsampled buckets.
    """

    def __init__(self, buffer, bucket_mix_ratio: float = 0.7, starvation_boost: float = 0.1):
        raise NotImplementedError

    def sample(self, batch_size: int):
        """Return a list of trajectories sampled across buckets."""
        raise NotImplementedError

    def _sample_buckets(self, n: int):
        """Pick n bucket choices using mixed quota + uniform strategy."""
        raise NotImplementedError

    def _sample_within_bucket(self, bucket, k: int):
        """Pick k trajectories from bucket via priority-weighted random sampling."""
        raise NotImplementedError
