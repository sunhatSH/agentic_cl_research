"""Fully-Async-Policy integration plan + scaffold for CL training.

The project's MAIN deployment is verl's *Fully Async Policy* with a
disaggregated 40 (rollout) + 24 (train) GPU split (see
``doc/CL_Update_Sunhao.md`` § GPU 资源分配). This module wires the CL buffer +
loss into ``verl.experimental.fully_async_policy.FullyAsyncTrainer`` WITHOUT
forking verl, mirroring the synchronous ``trainer/verl_runner.py`` path.

How fully-async differs from the synchronous RayPPOTrainer (and what it means
for CL):

1. Trainer and rollouter are SEPARATE Ray actors. The trainer pulls assembled
   batches from a ``MessageQueue`` (``_get_samples_from_queue``) instead of a
   torch DataLoader. Trajectories can be STALE: produced by an older parameter
   version, bounded by ``async_training.staleness_threshold`` (target 0.3) and
   parameter sync cadence ``async_training.trigger_parameter_sync_step``.
2. ``required_samples = ppo_mini_batch_size * async_training.require_batches``.
3. The actor update still flows through the actor worker group, which still
   exposes ``set_loss_fn`` -- so CL loss injection is IDENTICAL to the sync
   path (``inject_cl_loss``).
4. The buffer lives on the trainer actor. Because ``FullyAsyncTrainer`` is
   itself a ``@ray.remote`` actor, the CL hooks must be installed INSIDE the
   actor, not patched from the driver. We therefore SUBCLASS the trainer with
   a thin CL mixin (``make_cl_fully_async_trainer_cls``) rather than
   monkey-patching a remote handle.

CL hook placement (per training step, in ``_fit_update_actor``):
  - PRE : append gradient-carrying replay rows to the actor batch
          (same ``prepare_replay_rows`` + ``_append_replay_rows`` helpers).
  - POST: ingest the step's freshly-generated trajectories into the buffer and
          sync ``buffer.set_step(current_param_version)``.

Staleness interaction: replay rows are by construction off-policy supervised
targets (no importance ratio), so buffer staleness is orthogonal to
``staleness_threshold`` -- the threshold governs only the on-policy RL rows.
``partial_rollout`` (resumed truncated trajectories) only affects how the
rollouter produces a trajectory; by the time a trajectory reaches the buffer
it is complete, so no special handling is needed on the CL side.

This module is import-safe without verl (the verl import is deferred into the
factory). The pure ``cl_actor_update`` wrapper is unit-tested without verl.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def cl_actor_update(
    batch: Any,
    original_update: Callable[[Any], Any],
    *,
    buffer: Any,
    lambda_replay: float,
    prepare_replay_rows: Callable[..., dict],
    append_replay_rows: Callable[[Any, dict], Any],
    extract_trajectories: Callable[[Any], list],
    weighting: Any,
    tokenizer: Any,
    replay_batch_size: int,
    replay_warmup_size: int,
    current_step: int,
) -> Any:
    """Run one CL-augmented actor update (shared by sync + async runners).

    PRE: append replay rows (when lambda_replay > 0 and rows are available).
    Calls ``original_update``. POST: ingest the batch's trajectories into the
    buffer and advance the buffer step. Pure w.r.t. verl -- all verl-specific
    behaviour is passed in as callables, so this is unit-testable with fakes.
    """
    if buffer is not None and lambda_replay > 0:
        replay_rows = prepare_replay_rows(
            buffer,
            weighting,
            tokenizer,
            replay_batch_size,
            warmup_size=replay_warmup_size,
        )
        if replay_rows:
            batch = append_replay_rows(batch, replay_rows)

    result = original_update(batch)

    if buffer is not None:
        buffer.set_step(current_step)
        for trajectory, bucket, meta in extract_trajectories(batch):
            buffer.add_trajectory(trajectory, bucket, meta)
    return result


def make_cl_fully_async_trainer_cls(buffer: Any, cfg: Any):
    """Return a ``FullyAsyncTrainer`` subclass with CL buffer + loss hooks.

    Usage on the cluster (sketch)::

        from trainer.cl_main import build_buffer
        buffer = build_buffer(cfg)
        TrainerCls = make_cl_fully_async_trainer_cls(buffer, cfg)
        trainer = TrainerCls.remote(config=cfg, ...)   # @ray.remote actor
        ray.get(trainer.init_workers.remote())
        # set_loss_fn + hooks are applied inside init_workers below.
        ray.get(trainer.fit.remote())

    The verl import is deferred to call time so this module imports without verl.
    """
    import ray
    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer

    from trainer.cl_loss import build_weighting_from_cfg
    from trainer.replay_batch import prepare_replay_rows
    from trainer.trajectory_adapter import extract_trajectories_from_batch
    from trainer.verl_runner import _append_replay_rows, make_cl_loss_from_cfg

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    replay_batch_size = int(cl.get("replay_batch_size", 512))
    replay_warmup_size = int(cl.get("replay_warmup_size", 0))
    weighting = build_weighting_from_cfg(cl) if lambda_replay > 0 else None

    # FullyAsyncTrainer is a Ray ActorClass (@ray.remote); Ray forbids
    # subclassing an actor class directly. Inherit from the UNDERLYING plain
    # class, then re-apply @ray.remote (matching verl's num_cpus=10).
    base_cls = getattr(getattr(FullyAsyncTrainer, "__ray_metadata__", None), "modified_class", None)
    if base_cls is None:
        raise TypeError(
            f"Cannot unwrap Ray actor class {FullyAsyncTrainer}. "
            "`__ray_metadata__.modified_class` not found. "
            "Check verl's fully_async_trainer for the underlying plain class, "
            "or update the extraction path for the installed Ray/verl version."
        )

    class CLFullyAsyncTrainer(base_cls):
        async def init_workers(self):
            await super().init_workers()
            # Same injection point as the sync path -- the actor worker group
            # exposes set_loss_fn regardless of sync/async.
            self.actor_wg.set_loss_fn(make_cl_loss_from_cfg(cfg))

        def _fit_update_actor(self, batch):
            return cl_actor_update(
                batch,
                super()._fit_update_actor,
                buffer=buffer,
                lambda_replay=lambda_replay,
                prepare_replay_rows=prepare_replay_rows,
                append_replay_rows=_append_replay_rows,
                extract_trajectories=extract_trajectories_from_batch,
                weighting=weighting,
                tokenizer=getattr(self, "tokenizer", None),
                replay_batch_size=replay_batch_size,
                replay_warmup_size=replay_warmup_size,
                # In fully-async, the param version is the meaningful "step".
                current_step=self.current_param_version,
            )

    return ray.remote(num_cpus=10)(CLFullyAsyncTrainer)
