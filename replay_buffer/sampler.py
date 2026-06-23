"""Two-level sampling: pick bucket, then pick trajectory within bucket.

Bucket-level: mixed strategy -- partly proportional to soft_target quota (covers
large buckets) and partly uniform (covers long-tail capabilities). Buckets that
have not been replayed for a long time receive a starvation_boost.

Within-bucket: priority-weighted random sampling, NOT top-k greedy. Greedy
top-k repeats "star trajectories" and reduces diversity.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def _weighted_choice_without_replacement(
    rng: random.Random,
    items: Sequence,
    weights: Sequence[float],
    k: int,
) -> list:
    """Sample k items without replacement, with probability proportional to weights.

    Uses the standard ``random.choices`` approach with manual deduplication
    -- adequate for k << len(items), which is our case (replay_batch ~ 32,
    bucket size ~ thousands).
    """
    if k >= len(items):
        return list(items)
    out = []
    chosen = set()
    items_list = list(items)
    weights_list = list(weights)
    tries = 0
    max_tries = k * 10
    while len(out) < k and tries < max_tries:
        idx = rng.choices(range(len(items_list)), weights=weights_list, k=1)[0]
        if idx not in chosen:
            chosen.add(idx)
            out.append(items_list[idx])
        tries += 1
    if len(out) < k:
        for i, item in enumerate(items_list):
            if i not in chosen:
                out.append(item)
                if len(out) >= k:
                    break
    return out


class TwoLevelSampler:
    """Two-level sampler for replay batches.

    Args:
        buffer: BucketReplayBuffer instance.
        bucket_mix_ratio: weight of quota-proportional vs uniform bucket
                          sampling. 0.7 (default) = 70% quota, 30% uniform.
                          Set to 1.0 to disable uniform (large buckets dominate).
        starvation_boost: extra weight added to buckets whose last sampled
                          step is far in the past. 0.1 (default).
        starvation_window: steps before a bucket is considered starved.
        within_bucket_sampling: 'priority' (default) priority-weighted random;
                          'uniform' ignores priority (R0 / R3 baselines).
        rng: optional random.Random for reproducibility.
    """

    def __init__(
        self,
        buffer,
        bucket_mix_ratio: float = 0.7,
        starvation_boost: float = 0.1,
        starvation_window: int = 50,
        within_bucket_sampling: str = "priority",
        rng: random.Random | None = None,
    ):
        if within_bucket_sampling not in ("priority", "uniform"):
            raise ValueError(f"unknown within_bucket_sampling {within_bucket_sampling!r}")
        self.buffer = buffer
        self.bucket_mix_ratio = bucket_mix_ratio
        self.starvation_boost = starvation_boost
        self.starvation_window = starvation_window
        self.within_bucket_sampling = within_bucket_sampling
        self.rng = rng or random.Random()
        self._last_bucket_sample_step: dict[str, int] = {}

    def sample(self, batch_size: int):
        """Return a list of (trajectory_id, trajectory, metadata) sampled across buckets.

        Marks sampled trajectories with mark_replayed so last_replay_step
        and replay_count stay accurate.
        """
        if batch_size <= 0:
            return []

        bucket_picks = self._sample_buckets(batch_size)
        out = []
        for bucket in bucket_picks:
            tid = self._sample_one_within_bucket(bucket)
            if tid is None:
                continue
            traj, meta = self.buffer.store.get(tid)
            out.append((tid, traj, meta))
            self._last_bucket_sample_step[bucket] = self.buffer._step

        self.buffer.mark_replayed([tid for tid, _, _ in out])
        return out

    def _sample_buckets(self, n: int) -> list[str]:
        """Pick n bucket choices using mixed quota + uniform strategy.

        bucket_weight(b) = mix * (soft_target[b] / C)
                         + (1-mix) * (1/K)
                         + starvation_boost  (if starved)
        """
        names = self.buffer.bucket_names
        k = len(names)
        c = self.buffer.total_capacity
        mix = self.bucket_mix_ratio

        weights = []
        cur_step = self.buffer._step
        for b in names:
            size = self.buffer.store.bucket_size(b)
            if size == 0:
                weights.append(0.0)
                continue
            quota_w = self.buffer.soft_target[b] / c if c else 0.0
            uniform_w = 1.0 / k
            w = mix * quota_w + (1.0 - mix) * uniform_w
            last_step = self._last_bucket_sample_step.get(b, -1)
            if cur_step - last_step > self.starvation_window:
                w += self.starvation_boost
            weights.append(w)

        if sum(weights) == 0:
            return []
        return self.rng.choices(names, weights=weights, k=n)

    def _sample_one_within_bucket(self, bucket: str) -> str | None:
        """Pick one trajectory_id from bucket.

        priority-weighted random sampling by default; uniform random when
        ``within_bucket_sampling == 'uniform'`` (R0 / R3 baselines).
        """
        ids = self.buffer.store.list_by_bucket(bucket)
        if not ids:
            return None
        if self.within_bucket_sampling == "uniform":
            return self.rng.choice(ids)
        priorities = [max(self.buffer.store.get_metadata(tid)["priority"], 1e-9) for tid in ids]
        return self.rng.choices(ids, weights=priorities, k=1)[0]

    def _sample_within_bucket(self, bucket: str, k: int) -> list[str]:
        """Pick k trajectories from bucket via priority-weighted random sampling
        without replacement."""
        ids = self.buffer.store.list_by_bucket(bucket)
        if not ids:
            return []
        priorities = [max(self.buffer.store.get_metadata(tid)["priority"], 1e-9) for tid in ids]
        return _weighted_choice_without_replacement(self.rng, ids, priorities, k)
