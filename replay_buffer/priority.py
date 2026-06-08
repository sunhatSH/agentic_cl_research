"""Anti-forgetting priority for buffer trajectories.

Priority is NOT reward absolute value. See ``doc/CL_Update_Sunhao.md`` for the
full rationale: a reward-based priority degrades the buffer into a sliding
window since reward rises monotonically with training.

priority_i = alpha_1 * forgetting_risk
           + alpha_2 * rarity
           + alpha_3 * diversity
           + alpha_4 * within_bucket_difficulty

Default fusion weights: (0.4, 0.2, 0.2, 0.2) -- forgetting_risk dominates.
"""


class Priority:
    """Compute and update per-trajectory priority.

    Args:
        alpha: 4-tuple of fusion weights (forgetting, rarity, diversity, difficulty).
    """

    def __init__(self, alpha=(0.4, 0.2, 0.2, 0.2)):
        raise NotImplementedError

    def compute(self, trajectory, bucket) -> float:
        """Compute composite priority for a trajectory in its bucket."""
        raise NotImplementedError

    def forgetting_risk(self, trajectory, current_policy_logprobs) -> float:
        """Estimate how much the model has regressed on this trajectory."""
        raise NotImplementedError

    def rarity(self, trajectory, bucket) -> float:
        """Score by inverse frequency of pattern_id / template_id within bucket."""
        raise NotImplementedError

    def diversity(self, trajectory, bucket) -> float:
        """Score by dissimilarity to existing bucket trajectories (e.g., embedding distance)."""
        raise NotImplementedError

    def within_bucket_difficulty(self, trajectory, bucket) -> float:
        """Relative difficulty within the bucket -- favours boundary / complex cases."""
        raise NotImplementedError
