# Copyright 2026. Continual-Learning over Agentic LLM.
#
# Licensed under the Apache License, Version 2.0.
"""Observer-aware reward manager: fold the per-row observer diff into extra_info.

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

verl 0.8.0 resolves reward managers from the experimental async registry
(``source: register`` -> ``verl.experimental.reward_loop.reward_manager``). We
subclass its ``NaiveRewardManager`` and override only ``run_single`` to inject
observer_report, so we inherit its exact ``__init__`` / batching / async
contract and stay robust to the rest of that path. Registered as
``cl_observer``; select via ``reward.reward_manager.name: cl_observer`` (source
stays ``register``). Importing this module triggers registration -- ensure it is
imported before the trainer resolves the reward manager (done in
``trainer/verl_runner.py``).

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
