"""Judge ↔ human agreement metrics for picking the reward judge.

To choose the reward judge (trainer/model_reward.py) we measure how well a
candidate judge model agrees with ClawEval human rubric verdicts, then pick the
SMALLEST model that clears an agreement threshold (anti reward-hacking still
requires judge >= policy; agreement is the quality gate on top).

Labeled sample (one per graded trajectory)::

    {
      "task": "...", "trajectory": "...", "rubric": "...",
      "bucket": "ops",
      "human": {"completion": 1.0, "safety": 1.0, "robustness": 0.5},
      "pass": true                # optional; else derived from aggregate >= thr
    }

A candidate judge produces a predicted verdict per sample; this module compares
predicted vs human on (a) per-dimension MAE/correlation, (b) overall pass
agreement (accuracy / F1 / Cohen's kappa), and (c) a per-bucket breakdown so a
small judge's weak spots (e.g. qa/research) are visible.

Pure + deterministic (no model calls, no numpy) so it unit-tests off-GPU.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from trainer.model_reward import JUDGE_DIMENSIONS, aggregate

# --- elementary metrics -------------------------------------------------------


def mae(pred: Sequence[float], gold: Sequence[float]) -> float:
    if not pred:
        return 0.0
    return sum(abs(p - g) for p, g in zip(pred, gold, strict=True)) / len(pred)


def pearson(pred: Sequence[float], gold: Sequence[float]) -> float:
    """Pearson correlation; 0.0 when either side has zero variance."""
    n = len(pred)
    if n == 0 or n != len(gold):
        return 0.0
    mp = sum(pred) / n
    mg = sum(gold) / n
    cov = sum((p - mp) * (g - mg) for p, g in zip(pred, gold, strict=True))
    vp = sum((p - mp) ** 2 for p in pred)
    vg = sum((g - mg) ** 2 for g in gold)
    if vp <= 0 or vg <= 0:
        return 0.0
    return cov / math.sqrt(vp * vg)


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Cohen's kappa for two binary raters; 1.0 perfect, 0.0 chance-level."""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa = sum(1 for x in a if x) / n
    pb = sum(1 for x in b if x) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def binary_classification(pred: Sequence[bool], gold: Sequence[bool]) -> dict[str, float]:
    """Accuracy / precision / recall / F1 of pred vs gold (gold = positive class)."""
    n = len(pred)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(1 for p, g in zip(pred, gold, strict=True) if p and g)
    fp = sum(1 for p, g in zip(pred, gold, strict=True) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold, strict=True) if not p and g)
    acc = sum(1 for p, g in zip(pred, gold, strict=True) if p == g) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# --- evaluation ---------------------------------------------------------------


def _pass_label(verdict: Mapping[str, float], sample: Mapping[str, Any] | None, threshold: float) -> bool:
    if sample is not None and isinstance(sample.get("pass"), bool):
        return bool(sample["pass"])
    return aggregate(verdict) >= threshold


def evaluate_judge(
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, float]],
    *,
    pass_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare a judge's predicted verdicts to human labels.

    ``samples[i]['human']`` is the gold verdict; ``predictions[i]`` is the judge's.
    Returns per-dimension agreement, overall pass agreement, and per-bucket stats.
    """
    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must align 1:1")
    n = len(samples)
    report: dict[str, Any] = {"n": n}
    if n == 0:
        return report

    humans = [dict(s.get("human", {})) for s in samples]

    # per-dimension agreement
    dims: dict[str, Any] = {}
    for d in JUDGE_DIMENSIONS:
        p = [float(predictions[i].get(d, 0.0)) for i in range(n)]
        g = [float(humans[i].get(d, 0.0)) for i in range(n)]
        dims[d] = {"mae": mae(p, g), "pearson": pearson(p, g)}
    report["dimensions"] = dims

    # aggregate score agreement
    p_score = [aggregate(predictions[i]) for i in range(n)]
    g_score = [aggregate(humans[i]) for i in range(n)]
    report["score_mae"] = mae(p_score, g_score)
    report["score_pearson"] = pearson(p_score, g_score)

    # pass-level agreement
    p_pass = [_pass_label(predictions[i], None, pass_threshold) for i in range(n)]
    g_pass = [_pass_label(humans[i], samples[i], pass_threshold) for i in range(n)]
    report["pass"] = binary_classification(p_pass, g_pass)
    report["pass"]["kappa"] = cohen_kappa(p_pass, g_pass)

    # per-bucket breakdown
    buckets: dict[str, dict[str, Any]] = {}
    for i, s in enumerate(samples):
        b = str(s.get("bucket", "unknown"))
        buckets.setdefault(b, {"idx": []})["idx"].append(i)
    per_bucket: dict[str, Any] = {}
    for b, info in buckets.items():
        idx = info["idx"]
        pb = [p_pass[i] for i in idx]
        gb = [g_pass[i] for i in idx]
        per_bucket[b] = {
            "n": len(idx),
            "score_mae": mae([p_score[i] for i in idx], [g_score[i] for i in idx]),
            "pass_accuracy": binary_classification(pb, gb)["accuracy"],
            "pass_kappa": cohen_kappa(pb, gb),
        }
    report["per_bucket"] = per_bucket
    return report


def rank_judges(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    sizes_b: Mapping[str, float] | None = None,
    min_kappa: float = 0.6,
    key: str = "kappa",
) -> dict[str, Any]:
    """Rank candidate judges and recommend one.

    Recommendation = the SMALLEST model (by sizes_b) whose pass-``key`` clears
    ``min_kappa``; if no sizes given, the highest-``key`` judge; if none clears
    the threshold, the best available with ``cleared=False``.
    """
    scored = []
    for name, rep in reports.items():
        k = float(rep.get("pass", {}).get(key, 0.0))
        scored.append((name, k))
    scored.sort(key=lambda x: x[1], reverse=True)

    cleared = [(n, k) for n, k in scored if k >= min_kappa]
    if cleared and sizes_b is not None:
        # smallest model that clears the bar
        rec = min(cleared, key=lambda x: sizes_b.get(x[0], float("inf")))[0]
        ok = True
    elif cleared:
        rec = cleared[0][0]
        ok = True
    else:
        rec = scored[0][0] if scored else None
        ok = False

    return {
        "ranking": [{"judge": n, key: k} for n, k in scored],
        "recommended": rec,
        "cleared_threshold": ok,
        "min_kappa": min_kappa,
    }
