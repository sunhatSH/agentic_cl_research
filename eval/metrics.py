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

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def new_task_performance(rollouts: Sequence[dict]) -> float:
    """Mean reward / pass rate on new-task rollouts.

    Args:
        rollouts: each item carries at least {'reward': float} or
                  {'passed': bool}. Mixed payloads use 'reward' if present
                  else 1.0/0.0 from 'passed'.
    """
    if not rollouts:
        return 0.0
    total = 0.0
    for r in rollouts:
        if "reward" in r:
            total += float(r["reward"])
        else:
            total += 1.0 if r.get("passed") else 0.0
    return total / len(rollouts)


def old_task_forgetting(current_scores: dict[str, float], previous_scores: dict[str, float]) -> float:
    """Mean drop on tasks present in both score dicts.

    forgetting = mean_{task in both} max(0, previous - current)

    Negative drops (improvement) are clamped to 0 -- forgetting is a
    one-sided metric. Returns 0 when no overlap.
    """
    common = set(current_scores) & set(previous_scores)
    if not common:
        return 0.0
    drops = [max(0.0, previous_scores[t] - current_scores[t]) for t in common]
    return sum(drops) / len(drops)


def cl_score(new_perf: float, forgetting: float, alpha: float = 1.0) -> float:
    """new_task_perf - alpha * forgetting. Project default alpha = 1.0."""
    return new_perf - alpha * forgetting


def output_entropy(logits: Sequence[Sequence[float]], mask: Sequence[int] | None = None) -> float:
    """Token-mean entropy H(p) = -sum p log p, in nats.

    Args:
        logits: [seq_len, vocab] list of token logits.
        mask: optional [seq_len] 0/1 mask; 1 = include token, 0 = skip.

    Used as Echo Trap early-warning -- entropy collapse precedes training
    breakdown in multi-turn agent RL.
    """
    if not logits:
        return 0.0
    total_h = 0.0
    n = 0
    for t, row in enumerate(logits):
        if mask is not None and not mask[t]:
            continue
        # Softmax row -> p
        m = max(row)
        exps = [math.exp(x - m) for x in row]
        z = sum(exps)
        h = 0.0
        for e in exps:
            p = e / z
            if p > 0:
                h -= p * math.log(p)
        total_h += h
        n += 1
    return total_h / n if n else 0.0


def _ngrams(tokens: Sequence, n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _distinct_n(trajectories: Sequence[Sequence], n: int) -> float:
    """distinct-n = (# unique n-grams) / (# total n-grams) across trajectories.

    Higher = more diverse. Range [0, 1].
    """
    all_grams = []
    for tokens in trajectories:
        all_grams.extend(_ngrams(tokens, n))
    if not all_grams:
        return 0.0
    return len(set(all_grams)) / len(all_grams)


def _self_bleu_pair(a: Sequence, b: Sequence, n: int = 4) -> float:
    """Modified n-gram precision of a against b. Range [0, 1].

    Lightweight (no smoothing, no brevity penalty) -- enough for relative
    self-BLEU comparison within a query.
    """
    if len(a) < n:
        return 0.0
    grams_a = _ngrams(a, n)
    grams_b = Counter(_ngrams(b, n))
    if not grams_a:
        return 0.0
    overlap = 0
    for g in grams_a:
        if grams_b[g] > 0:
            overlap += 1
            grams_b[g] -= 1
    return overlap / len(grams_a)


def trajectory_diversity(trajectories_per_query: Sequence[Sequence[Sequence]]) -> dict[str, float]:
    """distinct-n and self-BLEU between trajectories of the same query.

    Args:
        trajectories_per_query: list of length Q; each element is a list of
                                M trajectories (M = traj/query, default 8),
                                each trajectory is a token sequence.

    Returns:
        {'distinct_1': ..., 'distinct_4': ..., 'self_bleu_4': ...}
        self_bleu averaged across all pairs within each query, then across queries.
    """
    d1_vals, d4_vals, bleu_vals = [], [], []
    for trajs in trajectories_per_query:
        if len(trajs) < 2:
            continue
        d1_vals.append(_distinct_n(trajs, 1))
        d4_vals.append(_distinct_n(trajs, 4))
        # Mean self-BLEU across all ordered pairs.
        bleu = 0.0
        pairs = 0
        for i, a in enumerate(trajs):
            for j, b in enumerate(trajs):
                if i == j:
                    continue
                bleu += _self_bleu_pair(a, b, n=4)
                pairs += 1
        if pairs:
            bleu_vals.append(bleu / pairs)

    return {
        "distinct_1": sum(d1_vals) / len(d1_vals) if d1_vals else 0.0,
        "distinct_4": sum(d4_vals) / len(d4_vals) if d4_vals else 0.0,
        "self_bleu_4": sum(bleu_vals) / len(bleu_vals) if bleu_vals else 0.0,
    }
