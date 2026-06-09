"""Anti-forgetting priority for buffer trajectories.

Priority is NOT reward absolute value. See ``doc/CL_Update_Sunhao.md`` for the
full rationale: a reward-based priority degrades the buffer into a sliding
window since reward rises monotonically with training.

priority_i = alpha_1 * forgetting_risk
           + alpha_2 * rarity
           + alpha_3 * diversity
           + alpha_4 * within_bucket_difficulty

Default fusion weights: (0.4, 0.2, 0.2, 0.2) -- forgetting_risk dominates.

Each component is normalized to roughly [0, 1] so the weighted sum is
interpretable and the weights are comparable. None of the components
depends on reward absolute value (R5 reward-based priority is implemented
as a separate ``RewardPriority`` class for ablation, not via the weight
tuple).
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class Priority:
    """Compute and update per-trajectory priority by 4-signal fusion.

    Args:
        alpha: 4-tuple of fusion weights (forgetting, rarity, diversity, difficulty).
               Must sum to 1.0 (enforced; raises ValueError otherwise).
    """

    def __init__(self, alpha: Sequence[float] = (0.4, 0.2, 0.2, 0.2)):
        if len(alpha) != 4:
            raise ValueError(f"alpha must be a 4-tuple, got {alpha!r}")
        s = sum(alpha)
        if not math.isclose(s, 1.0, abs_tol=1e-6):
            raise ValueError(f"alpha must sum to 1.0, got {s}")
        self.alpha = tuple(alpha)

    def compute(self, trajectory, bucket_view) -> float:
        """Composite priority for a trajectory in its bucket.

        Args:
            trajectory: dict-like with fields used by the four signals
                        (current_logprobs, original_logprobs, pattern_id,
                        embedding, success_rate, ...). Missing fields
                        contribute 0 to their component.
            bucket_view: dict-like view of the bucket the trajectory lives
                        in -- exposes pattern_counts, peer_embeddings,
                        peer_success_rates. Implemented in bucket.py.

        Returns:
            float in roughly [0, 1].
        """
        a1, a2, a3, a4 = self.alpha
        # Short-circuit: if a weight is 0, skip the corresponding signal entirely.
        # Same project-wide rule as L_cl composition.
        f = self.forgetting_risk(trajectory) if a1 > 0 else 0.0
        r = self.rarity(trajectory, bucket_view) if a2 > 0 else 0.0
        d = self.diversity(trajectory, bucket_view) if a3 > 0 else 0.0
        diff = self.within_bucket_difficulty(trajectory, bucket_view) if a4 > 0 else 0.0
        return a1 * f + a2 * r + a3 * d + a4 * diff

    def forgetting_risk(self, trajectory) -> float:
        """Logprob drift from when the trajectory was first seen.

        forgetting_risk = clip(mean(orig_logprob - current_logprob), 0, 1)

        Positive drift = model now assigns LOWER prob to the action than it
        used to = sign of forgetting on this trajectory. Negative drift =
        model improved on this pattern; no replay needed.
        """
        orig = trajectory.get("original_logprobs")
        curr = trajectory.get("current_logprobs")
        if orig is None or curr is None or len(orig) == 0:
            return 0.0
        n = min(len(orig), len(curr))
        if n == 0:
            return 0.0
        drift = sum((orig[i] - curr[i]) for i in range(n)) / n
        return max(0.0, min(1.0, drift))

    def rarity(self, trajectory, bucket_view) -> float:
        """Inverse frequency of pattern_id within bucket.

        rarity = 1 / log(1 + count_of_this_pattern_in_bucket)
        """
        pid = trajectory.get("pattern_id")
        if pid is None:
            return 0.0
        counts = bucket_view.get("pattern_counts", {})
        count = counts.get(pid, 0)
        return 1.0 / math.log(2.0 + count)

    def diversity(self, trajectory, bucket_view) -> float:
        """Mean cosine distance to peer embeddings in bucket.

        diversity = 1 - max_cosine_sim(traj_emb, peer_embs)
        High = trajectory is unlike anything else in bucket.
        """
        emb = trajectory.get("embedding")
        peers = bucket_view.get("peer_embeddings", [])
        if emb is None or not peers:
            return 0.0
        max_sim = 0.0
        norm_e = math.sqrt(sum(x * x for x in emb)) or 1.0
        for p in peers:
            norm_p = math.sqrt(sum(x * x for x in p)) or 1.0
            dot = sum(emb[i] * p[i] for i in range(min(len(emb), len(p))))
            sim = dot / (norm_e * norm_p)
            if sim > max_sim:
                max_sim = sim
        return max(0.0, 1.0 - max_sim)

    def within_bucket_difficulty(self, trajectory, bucket_view) -> float:
        """Boundary / hard cases within bucket -- favours non-trivial samples.

        difficulty = 1 - success_rate (peer-relative).
        Trajectories that succeed only sometimes are more valuable for
        replay than trivially-easy or impossibly-hard ones.
        """
        sr = trajectory.get("success_rate")
        if sr is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - sr))


class RewardPriority:
    """Reward-based priority for the R5 ablation (vs anti-forgetting Priority).

    Used ONLY in Phase 3 / R5 to verify that reward-based priority degrades
    the buffer into a sliding window as training reward rises. NOT the
    project's main scheme.
    """

    def compute(self, trajectory, bucket_view) -> float:
        return float(trajectory.get("reward", 0.0))
