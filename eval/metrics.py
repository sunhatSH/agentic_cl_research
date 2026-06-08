"""CL evaluation metrics (see doc/CL_Update_Sunhao.md "评测指标体系").

Metrics:
    new_task_performance  -- reward / pass rate on new tasks
    old_task_forgetting   -- drop on old-task scores vs previous stage
    cl_score              -- new_task_perf - alpha * old_task_forgetting (alpha=1.0)
    kl_trend              -- D_KL(pi_new || pi_ref) over steps
    replay_to_rl_ratio    -- L_replay / L_rl trend
    advantage_distribution-- mean / variance per phase
    grad_norm_per_loss    -- L2 norm of each loss component
    output_entropy        -- H(pi_new(.|s)) over rollout states (Echo Trap early-warning)
    trajectory_diversity  -- distinct-n / self-BLEU on the 2 trajectories per query
"""


def new_task_performance(rollouts):
    raise NotImplementedError


def old_task_forgetting(current_scores, previous_scores):
    raise NotImplementedError


def cl_score(new_perf: float, forgetting: float, alpha: float = 1.0) -> float:
    raise NotImplementedError


def output_entropy(logits, mask) -> float:
    raise NotImplementedError


def trajectory_diversity(trajectories_per_query) -> dict:
    """Return distinct-n and self-BLEU between trajectories of the same query."""
    raise NotImplementedError
