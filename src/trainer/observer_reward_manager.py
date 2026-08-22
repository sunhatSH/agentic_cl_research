# Copyright 2026. Continual-Learning over Agentic LLM.
#
# Licensed under the Apache License, Version 2.0.
"""Observer-aware reward manager -- CURRENTLY NOT WIRED (kept for reference/backup).

⚠️ 2026-07-27: training reward is now computed INLINE during rollout
(cl_rollout_manager.generate_sequences -> _score_all_slots(observer diff + judge)
-> t.reward -> trajectories_to_dataproto writes rm_scores). verl reads rm_scores
directly (use_rm=False), so it NEVER calls a reward manager -- this class is on
verl's separate "reward-loop worker" path, which we do not use. base.yaml's
reward_manager is plain ``naive`` (never invoked since rm_scores already exists).
This file is retained only as a reference for the alternative "route reward
through verl's RewardLoopWorker + ObserverRewardManager" design; DO NOT assume
it runs. If you re-enable it, set reward.reward_manager to source=importlib +
this module, and make generate_sequences STOP writing rm_scores (else double reward).

Observer-aware reward manager: fold the per-row observer diff into extra_info.

verl scores each row via ``compute_score(data_source, solution_str,
ground_truth, extra_info)``, reading ``extra_info`` from the dataset. Our rollout
manager (``trainer/cl_rollout_manager.py``) carries the Observer's before/after
sandbox diff back on a SEPARATE non_tensor key ``observer_report`` -- it CANNOT
reuse ``extra_info`` (that key belongs to the dataset and would collide on
``DataProto.union`` at ray_trainer.py:1448). This manager bridges the two: for
each row, it copies ``observer_report`` into that row's ``extra_info`` so the
training judge (``trainer/model_reward.py::compute_score``) grounds *completion*
on real state, not the actor's self-report.

The Observer never scores; it only supplies evidence. This manager is the wiring
that delivers that evidence to the one judge that actually drives training.

verl 0.8.0's reward runs in a SEPARATE Ray process (``RewardLoopWorker``), so a
``@register`` decorator (which only populates the registry in the process that
imports it) is NOT enough -- the worker never imported this module and raised
"Unknown reward manager: cl_observer". We therefore load via ``source: importlib``
(``load_extern_object("pkg://trainer.observer_reward_manager", "ObserverRewardManager")``),
which resolves directly from the module in EVERY process. The ``@register`` below
is kept as a harmless fallback for ``source: register`` callers. We subclass the
experimental ``NaiveRewardManager`` and override only ``run_single`` to inject
observer_report, inheriting its exact ``__init__`` / async contract.

Config (configs/base.yaml)::

    reward:
      reward_manager:
        source: importlib
        name: ObserverRewardManager
        module:
          path: pkg://trainer.observer_reward_manager

When a row has no ``observer_report`` (empty diff / cold path), behaviour is
identical to naive.
"""

from __future__ import annotations

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


@register("cl_observer")
class ObserverRewardManager(NaiveRewardManager):
    """Naive (experimental async) reward manager + per-row Observer-diff injection.

    Inherits ``NaiveRewardManager.__init__`` verbatim (config, tokenizer,
    compute_score, ...). Only ``run_single`` is wrapped: it folds this row's
    ``observer_report`` into ``extra_info`` before delegating, so the unchanged
    scoring path passes the diff evidence to ``compute_score``.
    """

    async def run_single(self, data: DataProto) -> dict:
        # NaiveRewardManager.run_single reads extra_info from data[-1:][0]'s
        # non_tensor_batch. Fold observer_report into it there. Mutating the
        # sliced item's extra_info dict (a per-row object) is local to this
        # reward call and never touches the dataset's shared dicts.
        item = data[-1:][0]
        nt = item.non_tensor_batch
        report = nt.get("observer_report")
        if report:
            extra = nt.get("extra_info")
            extra = dict(extra) if isinstance(extra, dict) else {}
            extra["observer_report"] = str(report)
            nt["extra_info"] = extra
        return await super().run_single(data)
