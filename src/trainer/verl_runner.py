"""verl RayPPOTrainer construction and CL hook installation.

Wraps ``verl.trainer.main_ppo.TaskRunner`` so buffer + custom loss stay on the
same Ray actor as the trainer driver (buffer is not serialized across Ray).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf





def merge_verl_config(cl_cfg: Any) -> Any:
    """Ensure OmegaConf object is compatible with verl RayPPOTrainer.

    verl expects a full Hydra-style config. Experiment yamls inherit
    ``configs/base.yaml``; additional verl fields should be supplied via
    experiment overrides or a verl defaults yaml on the cluster.
    """
    if not OmegaConf.is_config(cl_cfg):
        cl_cfg = OmegaConf.create(cl_cfg)
    OmegaConf.set_struct(cl_cfg, False)
    return cl_cfg


def make_cl_loss_from_cfg(cfg: Any):
    """Build ``make_cl_loss`` closure from OmegaConf ``cfg.cl`` section.

    The buffer is never captured here -- the closure is pickled to the actor
    worker via ``set_loss_fn`` and must stay free of the (driver-only) buffer.

    Passes the ``actor_rollout_ref.actor`` subtree so the closure can rebuild the
    ``ActorConfig`` that verl's ``ppo_loss`` needs (bug-1: verl calls the loss
    without a ``config`` arg). This subtree is plain OmegaConf -> pickle-safe.
    """
    from trainer.cl_loss import make_cl_loss

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    scheme = (cl.get("weighting", {}) or {}).get("scheme", "W2")
    actor_cfg = cfg.get("actor_rollout_ref", {}).get("actor", None)
    return make_cl_loss(
        replay_enabled=lambda_replay > 0.0,
        lambda_replay=lambda_replay,
        replay_batch_size=int(cl.get("replay_batch_size", 512)),
        use_token_weighting=scheme != "W0",
        weighting_scheme=scheme,
        actor_cfg=actor_cfg,
    )


def inject_cl_loss(trainer: Any, cfg: Any, buffer: Any | None = None) -> None:
    """Call ``actor_rollout_wg.set_loss_fn`` after ``init_workers``.

    ``buffer`` is accepted for signature parity but intentionally unused -- it
    must not travel into the loss closure.
    """
    trainer.actor_rollout_wg.set_loss_fn(make_cl_loss_from_cfg(cfg))


def compute_std_metrics(batch: Any) -> dict[str, float]:
    """Dispersion metrics verl's ``compute_data_metrics`` does NOT emit.

    verl logs reward/advantage mean/max/min but no std, and no GRPO
    group-level spread. Those are exactly what we watch to catch a collapsing
    policy (advantage std → 0) or degenerate groups (all trajectories in a
    prompt's group scoring identically → 0 learning signal). Emitted under the
    ``cl/`` namespace so they never collide with verl's native keys.

    Computed from the RL batch AFTER ``compute_advantage`` (verl fit runs it
    before ``_update_actor``), so ``token_level_rewards`` / ``advantages`` /
    ``response_mask`` are present, and ``non_tensor_batch["uid"]`` identifies
    the GRPO group. Returns ``{}`` when torch or the required fields are absent
    (off-cluster / unexpected layout) rather than raising.

    Keys:
      - cl/reward_std          : std of per-sequence reward (sum over tokens)
      - cl/reward_mean         : mean of per-sequence reward (cross-check vs verl)
      - cl/advantage_std       : std of valid (masked) advantages
      - cl/group_reward_std    : mean over groups of the within-group reward std
                                 (GRPO "group std" -- 0 => degenerate groups)
      - cl/group_reward_std_max: worst (largest) within-group reward std
      - cl/num_groups          : number of distinct uids in the batch
    """
    try:
        import torch
    except ImportError:
        return {}
    bb = getattr(batch, "batch", None)
    if bb is None:
        return {}
    try:
        tlr = bb.get("token_level_rewards")
        resp_mask = bb.get("response_mask")
        adv = bb.get("advantages")
    except Exception:  # noqa: BLE001 -- TensorDict access variability
        tlr = resp_mask = adv = None
    if tlr is None:
        return {}

    out: dict[str, float] = {}
    seq_reward = tlr.sum(dim=-1).float()  # [B]
    if seq_reward.numel() > 0:
        out["cl/reward_std"] = float(seq_reward.std(unbiased=False).item())
        out["cl/reward_mean"] = float(seq_reward.mean().item())

    if adv is not None and resp_mask is not None:
        valid = torch.masked_select(adv, resp_mask.bool())
        if valid.numel() > 0:
            out["cl/advantage_std"] = float(valid.std(unbiased=False).item())

    # GRPO group spread: std of per-sequence reward within each uid group.
    uids = None
    nt = getattr(batch, "non_tensor_batch", None)
    if isinstance(nt, dict):
        uids = nt.get("uid")
    if uids is not None and seq_reward.numel() == len(uids):
        groups: dict[Any, list[float]] = {}
        for u, r in zip(list(uids), seq_reward.tolist(), strict=False):
            groups.setdefault(u, []).append(r)
        stds = []
        for vals in groups.values():
            if len(vals) > 1:
                t = torch.tensor(vals)
                stds.append(float(t.std(unbiased=False).item()))
            else:
                stds.append(0.0)
        if stds:
            out["cl/group_reward_std"] = float(sum(stds) / len(stds))
            out["cl/group_reward_std_max"] = float(max(stds))
        out["cl/num_groups"] = float(len(groups))
    return out


def _persist_winners(winners: list, exp_name: str, step: int) -> None:
    """Write winner trajectories as JSONL for offline analysis."""
    if not winners:
        return
    import json
    from pathlib import Path

    out_dir = Path(f"rollouts/training/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"step-{step}.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for traj, bucket, meta in winners:
            tid = meta.get("task_id", "")
            row = {
                "task_id": tid,
                "bucket": bucket,
                "reward": meta.get("reward"),
                "messages": traj.get("messages", []),
                "step": step,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[persist] {len(winners)} winners → {out_file}", flush=True)


def _log_training_metrics(result, exp_name: str, step: int) -> None:
    """DELETED / DEAD CODE (2026-07-28).

    此函数从 `_update_actor` 返回值的 meta_info["metrics"] 取指标 —— 但那里的 metrics
    是【未 reduce 的原始态】且【只含 actor 子集】(pg_loss/grad_norm/entropy);而
    reward/advantage/response_length 等是 verl 在 fit 主循环用 compute_data_metrics 单独
    算的,不在 update_actor 返回里。外部 hook 够不到 fit 的局部 metrics 字典 →
    metrics 恒空 → `if not metrics: return` 直接跳过 → logs/metrics/*.jsonl 从未生成。

    已改用 verl 原生 FileLogger(config logger:[...,file] + env VERL_FILE_LOGGER_PATH):
    它在 fit 主循环拿到聚合后的完整 metrics 并每 step 实时写 JSONL(buffering=0)。
    保留此桩仅为兼容可能的旧引用;不再调用。
    """
    return


def install_buffer_hooks(trainer: Any, buffer: Any | None, cfg: Any) -> None:
    """Patch RayPPOTrainer to feed trajectories into buffer and pre-stage replay.

    Hooks ``_update_actor`` (pre: attach replay sidecar; post: ingest batch)
    and ``fit`` step counter (sync buffer step).
    """
    if buffer is None:
        return

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    replay_batch_size = int(cl.get("replay_batch_size", 512))
    replay_warmup_size = int(cl.get("replay_warmup_size", 0))
    # Paper-evidence cadences (0 disables). See doc/Progress.md / replay_metrics.
    stats_log_freq = int(cl.get("buffer_stats_log_freq", 1))
    forgetting_update_freq = int(cl.get("forgetting_update_freq", 1))
    trainer_cfg = cfg.get("trainer", {}) or {}
    save_freq = int(trainer_cfg.get("save_freq", 0))
    exp_name = trainer_cfg.get("experiment_name", "cl")
    weighting = None
    if lambda_replay > 0:
        from trainer.cl_loss import build_weighting_from_cfg

        weighting = build_weighting_from_cfg(cl)

    from trainer.replay_batch import prepare_replay_rows
    from trainer.replay_forward import REPLAY_TIDS_KEY
    from trainer.replay_metrics import (
        BufferStatsLogger,
        backfill_forgetting,
        compute_replay_current_logprobs,
        flatten_buffer_stats,
    )
    from trainer.trajectory_adapter import extract_trajectories_from_batch

    stats_logger = BufferStatsLogger(f"logs/buffer_stats/{exp_name}.jsonl")
    tokenizer = getattr(trainer, "tokenizer", None)
    original_update = trainer._update_actor

    def _merge_buffer_metrics(result, stats) -> None:
        """Surface flattened buffer stats into verl's logged metrics (wandb)."""
        meta = getattr(result, "meta_info", None)
        if isinstance(meta, dict) and isinstance(meta.get("metrics"), dict):
            meta["metrics"].update(flatten_buffer_stats(stats))

    def patched_update(batch):
        # Keep a handle to the ORIGINAL rollout batch: the post-append batch
        # also contains replay rows (is_replay=True) which must NOT be ingested
        # back into the buffer as if they were fresh trajectories.
        rl_batch = batch
        # 1. Pre: append gradient-carrying replay rows to the actor batch.
        replay_rows: dict = {}
        if lambda_replay > 0:
            replay_rows = prepare_replay_rows(
                buffer,
                weighting,
                tokenizer,
                replay_batch_size,
                warmup_size=replay_warmup_size,
            )
            if replay_rows:
                shuffle_seed = int(getattr(trainer, "global_steps", 0) or 0)
                batch = _append_replay_rows(batch, replay_rows, shuffle_seed=shuffle_seed)

        result = original_update(batch)

        step = getattr(trainer, "global_steps", buffer._step)
        buffer.set_step(step)

        # 每 step metrics 落盘改用 verl 原生 FileLogger(logger:[...,file] + VERL_FILE_LOGGER_PATH),
        # 它拿到 verl fit 聚合后的完整 metrics(reward/advantage/loss 全套)并实时写 JSONL。
        # 旧的 _log_training_metrics 已删:它 hook 在 _update_actor 返回值上,那里 metrics 未 reduce
        # 且不含 reward/advantage(在 fit 主循环 compute_data_metrics 算),取不到 → 从未生成文件。

        # 1b. Dispersion metrics verl doesn't emit (reward/adv std + GRPO group
        #     spread). Computed from the RL batch (advantages already present)
        #     and merged into the SAME meta_info["metrics"] channel the buffer
        #     stats use, so verl's FileLogger picks them up. cl/ namespaced.
        try:
            std_metrics = compute_std_metrics(rl_batch)
            if std_metrics:
                meta = getattr(result, "meta_info", None)
                if isinstance(meta, dict) and isinstance(meta.get("metrics"), dict):
                    # Match the buffer-stats path: insert as scalars (reduce_metrics
                    # is scalar-safe via np.mean; no key contains max/min except the
                    # explicit *_max which reduce_metrics will np.max harmlessly).
                    meta["metrics"].update(std_metrics)
        except Exception:  # noqa: BLE001 -- metrics must never crash training
            pass

        # 2. Post: activate forgetting_risk -- recompute current-policy log-probs
        #    for the just-replayed trajectories and backfill their priority.
        tids = replay_rows.get(REPLAY_TIDS_KEY) if replay_rows else None
        if tids and forgetting_update_freq > 0 and step % forgetting_update_freq == 0:
            means = compute_replay_current_logprobs(trainer, replay_rows)
            if means is not None:
                backfill_forgetting(buffer, tids, means)

        # 3. Post: GRPO produces 8 trajectories per query for advantage
        #    computation, but only WINNERS (max reward per task_id) go into the
        #    buffer for replay — losers would dilute anti-forgetting quality.
        from datasources.cleaning import strip_zw

        # Group by task_id to pick winners
        groups: dict[str, list[tuple[Any, str, dict]]] = {}
        for trajectory, bucket, meta in extract_trajectories_from_batch(
            rl_batch, valid_buckets=getattr(buffer, "bucket_names", None)
        ):
            for msg in trajectory.get("messages", []):
                if isinstance(msg.get("content"), str):
                    msg["content"] = strip_zw(msg["content"])
            tid = meta.get("task_id") or ""
            groups.setdefault(tid, []).append((trajectory, bucket, meta))

        for tid, candidates in groups.items():
            best = max(candidates, key=lambda x: float(x[2].get("reward", 0) or 0))
            buffer.add_trajectory(*best)

        # 3b. Persist winners to JSONL (offline analysis / reproducibility).
        all_rows = [best for best in groups.values()]
        _persist_winners(all_rows, exp_name, step)

        # 4. Post: buffer-dynamics evidence (wandb metrics + sidecar JSONL).
        if stats_log_freq > 0 and step % stats_log_freq == 0:
            stats = buffer.stats()
            _merge_buffer_metrics(result, stats)
            stats_logger.log(step, stats)

        # 5. Post: periodic full buffer snapshot for reproducibility / analysis.
        if save_freq > 0 and step > 0 and step % save_freq == 0:
            snap = Path(f"buffer_dumps/{exp_name}-step-{step}.sqlite")
            snap.parent.mkdir(parents=True, exist_ok=True)
            buffer.dump(snap)

        return result

    trainer._update_actor = patched_update


def _append_replay_rows(batch: Any, replay_rows: dict[str, Any], shuffle_seed: int = 0) -> Any:
    """Concatenate replay row tensors onto a verl DataProto batch.

    Replay rows carry ``is_replay=True``, a real ``replay_response_mask`` (with
    ``response_mask`` = 0 so ppo_loss ignores them, bug B8), zeroed advantages,
    and per-token ``replay_token_weights``. RL rows are back-filled with the
    mirror-image zero fields so the two row sets are rectangular and never
    contaminate each other.

    Alignment (bug B9 / bug-2b): verl rollout rows are LEFT-padded prompt + RIGHT
    -padded response. Replay rows from ``build_replay_rows`` share that layout. We
    align the two sets on BOTH dims independently -- prompts left-padded to max P,
    responses right-padded to max R -- then rebuild ``input_ids`` / ``attention_mask``
    / ``position_ids`` so ``no_padding_2_padding`` slices the response correctly.
    The exact engine concat path is validated on the GPU cluster (doc/Progress.md).
    """
    try:
        import torch
        import torch.nn.functional as F
        from verl import DataProto
    except ImportError:
        return batch

    from trainer.replay_forward import (
        IS_REPLAY_KEY,
        REPLAY_MASK_KEY,
        REPLAY_TIDS_KEY,
        REPLAY_WEIGHTS_KEY,
    )

    # Non-tensor sidecar (trajectory ids) never enters the TensorDict.
    replay_rows = {k: v for k, v in replay_rows.items() if k != REPLAY_TIDS_KEY}

    bb = batch.batch
    if "prompts" not in bb or "responses" not in bb:
        # Unexpected layout (no prompt/response split) -> skip replay append
        # rather than corrupt the batch; logged upstream by metrics.
        return batch

    n_rl = len(bb["responses"])
    P_rl, R_rl = bb["prompts"].shape[-1], bb["responses"].shape[-1]
    P_re, R_re = replay_rows["prompts"].shape[-1], replay_rows["responses"].shape[-1]
    P, R = max(P_rl, P_re), max(R_rl, R_re)

    # Response-width fields (right-pad), prompt-width fields (left-pad).
    resp_width_keys = {
        "responses",
        "response_mask",
        REPLAY_MASK_KEY,
        REPLAY_WEIGHTS_KEY,
        "old_log_probs",
        "ref_log_prob",
        "advantages",
    }

    def _align(rows: dict[str, Any]) -> dict[str, Any]:
        out = dict(rows)
        # left-pad prompts
        if out["prompts"].shape[-1] < P:
            out["prompts"] = F.pad(out["prompts"], (P - out["prompts"].shape[-1], 0), value=0)
        # right-pad response-width fields
        for k in resp_width_keys:
            v = out.get(k)
            if v is not None and hasattr(v, "dim") and v.dim() == 2 and v.shape[-1] < R:
                out[k] = F.pad(v, (0, R - v.shape[-1]), value=0)
        # rebuild full-sequence fields from aligned prompt + response segments
        prompt_attn = (out["prompts"] != 0).long()
        # attention over the response segment = response tokens that are real
        # (pad id 0 for placeholder positions in both RL and replay rows).
        resp_real = (out["responses"] != 0).long()
        out["input_ids"] = torch.cat([out["prompts"], out["responses"]], dim=1)
        out["attention_mask"] = torch.cat([prompt_attn, resp_real], dim=1)
        out["position_ids"] = (out["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
        return out

    # Back-fill RL rows with mirror-image replay fields (response-width).
    bb[IS_REPLAY_KEY] = torch.zeros(n_rl, dtype=torch.bool)
    bb[REPLAY_WEIGHTS_KEY] = torch.zeros((n_rl, R_rl), dtype=torch.float32)
    bb[REPLAY_MASK_KEY] = torch.zeros((n_rl, R_rl), dtype=torch.long)

    # Align RL batch tensors in place (prompts left, responses right).
    rl_aligned = _align({k: bb[k] for k in bb.keys()})
    for k, v in rl_aligned.items():
        bb[k] = v

    replay_rows = _align(replay_rows)
    replay_dp = DataProto.from_single_dict(replay_rows)

    # DataProto.concat requires both sides to share non_tensor_batch keys with a
    # length == batch size. Replay rows carry no rollout non-tensor columns
    # (messages/bucket/...), so back-fill placeholder columns of length n_replay.
    # These placeholders are never read: replay rows are excluded from buffer
    # ingest (patched_update uses the pre-append rl_batch).
    rl_non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
    if rl_non_tensor:
        import numpy as np

        n_replay = len(replay_dp)
        for k, arr in rl_non_tensor.items():
            trailing = getattr(arr, "shape", (0,))[1:]
            replay_dp.non_tensor_batch[k] = np.full((n_replay, *trailing), None, dtype=object)

    merged = DataProto.concat([batch, replay_dp])

    # Shuffle rows so the replay block (concatenated at the tail) is spread
    # across ALL mini batches. verl slices mini batches sequentially WITHOUT
    # shuffle (tensordict_utils.make_iterator -> DataLoader shuffle defaults
    # False), so without this the replay rows would land only in the final
    # ceil(n_replay / ppo_mini_batch_size) mini batches -- every earlier mini
    # batch's optimizer.step() would carry ZERO replay gradient, splitting PPO
    # learning and CL anti-forgetting apart in time. A per-step permutation
    # (seeded off the buffer step for reproducibility) distributes replay rows
    # uniformly so each mini batch's combined loss (rl + lambda_3 * replay)
    # sees replay. Row-level shuffle does not touch the loss composition, so
    # lambda_3 semantics are unchanged.
    n_total = len(merged)
    gen = torch.Generator()
    gen.manual_seed(int(shuffle_seed))
    merged.reorder(torch.randperm(n_total, generator=gen))
    return merged


class CLTaskRunner:
    """Drop-in replacement for ``verl.trainer.main_ppo.TaskRunner`` with CL hooks.

    Usage from ``cl_main``::

        run_cl_ppo(cfg)  # ray.init + remote CLTaskRunner.run
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    # Delegate worker registration to verl TaskRunner helpers
    def add_actor_rollout_worker(self, config):
        from verl.trainer.main_ppo import TaskRunner

        return TaskRunner.add_actor_rollout_worker(self, config)

    def add_critic_worker(self, config):
        from verl.trainer.main_ppo import TaskRunner

        TaskRunner.add_critic_worker(self, config)

    def init_resource_pool_mgr(self, config):
        from verl.trainer.main_ppo import TaskRunner

        return TaskRunner.init_resource_pool_mgr(self, config)

    def add_reward_model_resource_pool(self, config):
        from verl.trainer.main_ppo import TaskRunner

        TaskRunner.add_reward_model_resource_pool(self, config)

    def add_teacher_model_resource_pool(self, config):
        from verl.trainer.main_ppo import TaskRunner

        TaskRunner.add_teacher_model_resource_pool(self, config)

    def add_ref_policy_worker(self, config, ref_policy_cls):
        from verl.trainer.main_ppo import TaskRunner

        return TaskRunner.add_ref_policy_worker(self, config, ref_policy_cls)

    def run(self, config, resume_from: str | None = None) -> None:
        """Mirror ``TaskRunner.run`` with CL buffer + loss injection."""
        import socket
        from pprint import pprint

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.config import validate_config
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        config = merge_verl_config(config)
        print(f"CLTaskRunner hostname: {socket.gethostname()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        # Rebuild config from a plain container so that list-valued nodes (e.g.
        # trainer.logger = ['console','swanlab']) become native lists instead of
        # ListConfig nodes.  OmegaConf 2.3 resolve() rejects ListConfig as a
        # "non-primitive" value; recreating from container sidesteps that.
        config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        OmegaConf.resolve(config)

        from trainer.cl_main import build_buffer

        # Import the custom reward manager so its @register("cl_observer") fires
        # BEFORE RayPPOTrainer resolves reward.reward_manager.name. It folds the
        # per-row observer diff (non_tensor "observer_report") into extra_info so
        # the training judge grounds completion on real state. Import is a no-op
        # when verl's reward-loop registry is unavailable (off-cluster).
        try:
            import trainer.observer_reward_manager  # noqa: F401
        except Exception as exc:  # noqa: BLE001 -- registry absent off-cluster
            print(f"[cl] observer reward manager not registered ({exc}); using configured manager", flush=True)

        buffer = build_buffer(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        # verl HARD-REQUIRES a non-empty val dataloader: RayPPOTrainer._create_dataloader
        # always builds val from config.data.val_files and asserts len>=1
        # (ray_trainer.py:341,382) -- regardless of test_freq/val_before_train. When
        # we run WITHOUT a usable validation set, verl's create_rl_dataset(None/missing)
        # dies (TypeError on None; FileNotFoundError on a non-existent path). We don't
        # do verl's in-loop validation anyway (test_freq=-1, val_before_train=false;
        # evaluation is offline-after-training per CLAUDE.md), so alias val_files ->
        # train_files whenever val is EMPTY *or points at a missing file* (the latter
        # is what crashed k2: val_files set but datasets/val.parquet doesn't exist).
        # Set it ON config.data so verl's internal re-build (:341) sees it too.
        import os

        val_files = config.data.get("val_files", None)
        _val_missing = bool(val_files) and isinstance(val_files, str) and not os.path.exists(val_files)
        if not val_files or _val_missing:
            reason = "empty" if not val_files else f"missing file ({val_files})"
            OmegaConf.update(config, "data.val_files", config.data.train_files, force_add=True)
            print(
                f"[cl] data.val_files {reason} -> aliasing to train_files (verl requires a "
                "loadable val dataloader; in-loop validation is disabled, so this "
                "placeholder is never used to validate).",
                flush=True,
            )

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 用 recipe_custom 的 RayPPOTrainerV1(继承标准 RayPPOTrainer,构造签名一致)。
        # 它在 init_workers 里把 rollout 换成 recipe_custom 原生 agent_loop
        # (RolloutManager) —— 从而 rollout 层交给 verl 处理(自动带 logprob、工具输出
        # 截断、rollout_correction、Qwen3.5 GDN 变长打包),我们的 CL loss/buffer 注入
        # 全部照旧(它用 DataProto、_update_actor(batch) 旧签名)。见 plan swift-juggling-toast。
        # 回退:recipe_custom 不可用(off-cluster)时退回标准 RayPPOTrainer,保持本机可 import。
        _TrainerCls = RayPPOTrainer
        try:
            from recipe_custom.ray_trainer_v1 import RayPPOTrainerV1

            _TrainerCls = RayPPOTrainerV1
            print("[cl] 使用 recipe_custom.RayPPOTrainerV1(原生 agent_loop rollout)", flush=True)
        except Exception as exc:  # noqa: BLE001 -- recipe_custom absent off-cluster
            print(
                f"[cl] recipe_custom 不可用({exc}),回退标准 RayPPOTrainer(自写 rollout)",
                flush=True,
            )

        trainer = _TrainerCls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        inject_cl_loss(trainer, config, buffer)
        install_buffer_hooks(trainer, buffer, config)
        # Resume: verl's fit() -> _load_checkpoint() does this NATIVELY. There is no
        # public trainer.load_checkpoint() (only private _load_checkpoint(self), no
        # args) -- calling it (the old code) raised AttributeError on every 2nd/resume
        # run. Instead drive verl's own resume: an explicit path -> resume_mode=
        # resume_path + resume_from_path; otherwise the default resume_mode=auto
        # already auto-finds the latest ckpt in default_local_dir. Either way fit()
        # loads it; we must NOT call any load here.
        if resume_from:
            OmegaConf.update(config, "trainer.resume_mode", "resume_path", force_add=True)
            OmegaConf.update(config, "trainer.resume_from_path", str(resume_from), force_add=True)
        trainer.fit()


def _inject_cl_into_v1_trainer(trainer: Any, config: Any, buffer: Any | None = None) -> None:
    """Inject CL loss + buffer hooks into a v1 trainer after ``trainer.init()``.

    v1 (custom_sync) 路线：CL loss 经 ``actor_rollout_wg.set_loss_fn`` 注入（与 v0
    同机制，``engine_workers.py:496``；CL loss 已是 v1 签名
    ``fn(config, model_output, data, dp_group)``，见 ``trainer/cl_loss.py``）。

    9桶 buffer（阶段 D）：用【我们自己的】方案 —— hook custom_sync 的
    ``_update_actor(batch: KVBatchMeta, metrics)``（cl_replay_hook_v1.install_buffer_hooks_v1），
    PRE 掺回放行(kv_batch_put+concat)、POST 抽 winner 入 9桶 buffer。【不】子类化 v1
    ReplayBuffer（那是 online off-policy 采样器，维度不同）。b1 baseline buffer=None
    → hook 空转，只有 CL loss。R 系列 buffer.enabled=true 才触发。
    """
    trainer.actor_rollout_wg.set_loss_fn(make_cl_loss_from_cfg(config))
    print("[cl] v1: CL loss 已注入 (actor_rollout_wg.set_loss_fn)", flush=True)

    from trainer.cl_replay_hook_v1 import install_buffer_hooks_v1

    install_buffer_hooks_v1(trainer, buffer, config)


def run_cl_ppo(cfg: Any, resume_from: str | None = None) -> None:
    """Initialize Ray and run CL training via verl **原生 main_ppo (v1)** 入口.

    对齐参考脚本 debug_rl_qwen35_9b.sh：``python -m verl.trainer.main_ppo`` +
    ``trainer.use_v1=True`` + ``trainer.v1.trainer_mode=custom_sync`` +
    ``agent_loop_manager_class=RemoteAgentLoopManager``。我们不重写训练框架，只用
    verl 原生 ``run_ppo(config, task_runner_class=...)`` 的 recipe 钩子（main_ppo.py:40
    "For recipe to change TaskRunner"）注入 CL 资产。见 plan swift-juggling-toast 阶段 A。
    """
    import os

    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo

    cfg = merge_verl_config(cfg)
    if resume_from:
        OmegaConf.update(cfg, "trainer.resume_mode", "resume_path", force_add=True)
        OmegaConf.update(cfg, "trainer.resume_from_path", str(resume_from), force_add=True)

    # 自定义 env 开关(CL_FAKE_ROLLOUT 等)透传到 Ray worker —— worker 不继承 driver
    # shell env，只有 runtime_env.env_vars 列出的才传得进。放进 cfg.ray_kwargs 让
    # run_ppo 的 ray.init 带上（run_ppo 内部 merge default_runtime_env + cfg 的）。
    _passthrough = {
        _k: os.environ[_k]
        for _k in ("CL_FAKE_ROLLOUT", "CL_FAKE_ROLLOUT_LEN")
        if os.environ.get(_k) is not None
    }
    # ★ PYTHONPATH 透传到所有 ray actor(含 AgentSessionWorker):verl 的
    # get_ppo_ray_runtime_env 不传 PYTHONPATH,而 AgentSessionWorker 里 create_hook
    # 要 import trainer.observer_hook.ObserverDiffHook(FQN hook)+ shim/verl 也需在
    # 子进程 path 上。不传则 worker ImportError(2026-07-29 observer hook 定位)。
    _pp = os.environ.get("PYTHONPATH")
    if _pp:
        _passthrough["PYTHONPATH"] = _pp
    # ★ VERL_USE_EXTERNAL_MODULES 透传到所有 ray actor(含 AgentSessionWorker):worker
    # 不继承 driver shell env,而 verl/__init__ 靠此 env 决定 import 哪些外部模块。不透传
    # 则 worker 里 os.getenv 为空 → 外部 patch 模块(rollout.e2b_http1_patch 关 e2b http2 /
    # trainer.observer_hook_register 让 create_hook 认 FQN hook)在 worker 进程根本不 import,
    # 而这两个 patch 的目标(建沙箱 / create_hooks)恰恰都在 worker 执行 → 静默失效。
    # 透传后外部模块在【每个】verl 进程 import verl 时都跑,patch 落到 worker。
    _ext = os.environ.get("VERL_USE_EXTERNAL_MODULES")
    if _ext:
        _passthrough["VERL_USE_EXTERNAL_MODULES"] = _ext
    # ★ 内存分配器 env 透传到所有 ray actor(含 GatewayActor):worker 不继承 driver
    # shell env,_train_impl.sh 里 export 的 LD_PRELOAD/MALLOC_* 到不了 Ray worker,
    # 必须经 runtime_env.env_vars 才生效。不透传则 gateway 仍走 glibc 默认 arena →
    # 长跑碎片累积 OOM(2026-07-31 qwen35_9b_b1_4gpu step53 节点 512GB 打满,2 个
    # GatewayActor 各 ~180GB)。见 memory/gateway-oom-jemalloc.md。
    for _mk in ("LD_PRELOAD", "MALLOC_CONF", "MALLOC_ARENA_MAX", "MALLOC_TRIM_THRESHOLD_"):
        _mv = os.environ.get(_mk)
        if _mv is not None:
            _passthrough[_mk] = _mv
    # ★ reward judge 凭证透传到所有 ray actor(含 AgentSessionWorker 起的 RewardLoopWorker):
    # reward 走 omni → trainer.model_reward_omni → model_reward.get_judge() →
    # agents/config.resolve_judge() 读 TOKENHUB_API_KEY(configs/agents.yaml reward 段 key_env)。
    # worker 不继承 driver shell env,缺 key 时 _resolve_key 静默返回 "sk-local"(不报错)→
    # judge 用假 key 打 tokenhub → 401 → compute_score except 兜住 → judge_error=1 → 每条
    # reward 恒 0(reward=0-from-step-1 bug,2026-07-31 定案)。故必须经 runtime_env.env_vars
    # 显式透传。REWARD_* 是 env fallback(config-first 之外的兜底);REWARD_JUDGE_MAX_TOKENS
    # 让 thinking judge 的 max_tokens 也能外部调。见 memory/reward-zero-two-causes.md。
    for _rk in (
        "TOKENHUB_API_KEY",
        "REWARD_API_BASE",
        "REWARD_MODEL",
        "REWARD_API_KEY",
        "REWARD_JUDGE_MAX_TOKENS",
    ):
        _rv = os.environ.get(_rk)
        if _rv is not None:
            _passthrough[_rk] = _rv
    # ★ NCCL cuMem 关闭,透传到所有 ray actor(含 lightllm 推理副本):lightllm 开
    # enable_torch_memory_saver(CUDA VMM/cuMem 劫持 cudaMalloc),与 NCCL 默认
    # NCCL_CUMEM_ENABLE=1(NCCL 也用 cuMem 给 P2P buffer 分配)冲突 → 部分 lightllm
    # 副本起服卡在 server-start-up(uvicorn 起不来,到不了 594)→ verl llm_server.py:521
    # 无超时 asyncio.gather 永久阻塞 → 16卡 rollout 从没开始就静默 hang(b1_16gpu 6 次
    # 复发,换节点仍撞;§45 定案)。verl 已给 vllm/sglang 设 =0(sglang_rollout.py:57 引
    # sgl #6723),lightllm 是遗漏项。关的是 CUMEM 不是 P2P,TP 内带宽保留、保持 2 机。
    # 可 CL_NCCL_CUMEM 覆盖(极少数场景需 =1 时)。见 doc/debug §45。
    _passthrough.setdefault("NCCL_CUMEM_ENABLE", os.environ.get("CL_NCCL_CUMEM", "0"))
    # ★ 诊断 env 透传到所有 ray actor(含 lightllm 推理子进程):CL_DIAG=1 时把
    # NCCL/日志级别调详细,worker 不继承 driver shell env,必须经 runtime_env 才到得了
    # lightllm 副本(起服 hang 排查,§48/§49)。仅在 shell 已 export(_train_impl.sh
    # CL_DIAG 分支)时透传,非诊断态不加、不影响正常日志量。
    for _dk in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "LIGHTLLM_LOG_LEVEL", "RAY_DEDUP_LOGS", "VERL_LOGGING_LEVEL"):
        _dv = os.environ.get(_dk)
        if _dv is not None:
            _passthrough[_dk] = _dv
    # ★ FileLogger 落盘路径透传到所有 ray actor:verl 原生 FileLogger 在 CLTaskRunnerV1
    # actor(Ray worker)里实例化,读 VERL_FILE_LOGGER_PATH 决定 metrics.jsonl 写哪。
    # _train_impl.sh 只 export 到 driver shell,worker 不继承 → 多机(nnodes>1)下
    # CLTaskRunnerV1 跑在别的节点,env 为空 → metrics 落到默认路径(非 logs/metrics/<exp>),
    # 表现为 logs/metrics/<exp>/ 空目录、看不到 reward 曲线(单机 nnodes=1 因 actor 与
    # driver 同机/local 恰好继承到才没暴露,2026-08-02 16卡 b1 定位)。必须经 runtime_env
    # 透传。resume/fold 逻辑(_train_impl.sh _fold_metrics)依赖它落到约定路径才生效。
    _flp = os.environ.get("VERL_FILE_LOGGER_PATH")
    if _flp:
        _passthrough["VERL_FILE_LOGGER_PATH"] = _flp
    # TEXT_MODEL_ONLY 控制 LightLLM Qwen3.5 的 infer_struct(0/1=原生 M-RoPE,2=标准 RoPE)。
    _tmo = os.environ.get("TEXT_MODEL_ONLY")
    if _tmo is not None:
        _passthrough["TEXT_MODEL_ONLY"] = _tmo
    if _passthrough:
        OmegaConf.update(
            cfg,
            "ray_kwargs.ray_init.runtime_env.env_vars",
            {**(OmegaConf.select(cfg, "ray_kwargs.ray_init.runtime_env.env_vars") or {}), **_passthrough},
            force_add=True,
        )
        print(f"[verl_runner] 透传 env 到 Ray worker: {list(_passthrough)}", flush=True)

    # verl 原生 run_ppo 负责 ray.init（含 transfer_queue env）+ 起 task runner actor。
    # 传入我们的 CLTaskRunnerV1 替代默认 TaskRunnerV1（注入 CL loss/buffer）。
    _ = ray  # ray import 触发 verl 的 ray 相关初始化路径一致性
    run_ppo(cfg, task_runner_class=CLTaskRunnerV1)


def _make_cl_task_runner_v1():
    """Build the ``@ray.remote`` CLTaskRunnerV1 class.

    复刻 verl ``TaskRunnerV1.run`` 三步(trainer.init → init_agent_loop_manager →
    fit)，在 init 后 / fit 前插入 CL 注入。因 verl ``TaskRunnerV1`` 已是 ``@ray.remote``
    actor（不能被继承再装饰），这里【复刻】而非子类化。延迟到函数内定义，避免模块
    import 时就依赖 ray（off-cluster / 单测友好）。
    """
    import ray

    @ray.remote
    class CLTaskRunnerV1:
        """CL 版 v1 TaskRunner：verl 原生 v1 流程 + CL loss/buffer 注入。"""

        def __init__(self):
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self):
            # 复刻 verl TaskRunnerV1.init_agent_loop_manager：按
            # rollout.agent.agent_loop_manager_class 选 manager（我们配
            # RemoteAgentLoopManager → 沙箱 harness 那套）。
            from verl.trainer.ppo.v1 import AgentLoopManagerTQ
            from verl.utils.import_utils import load_class_from_fqn

            fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            cls = load_class_from_fqn(fqn, "AgentLoopManager") if fqn else AgentLoopManagerTQ
            self.agent_loop_manager = cls.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                teacher_client=self.trainer.get_teacher_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

        def run(self, config):
            from pprint import pprint

            import transfer_queue as tq
            from verl.trainer.ppo.v1 import get_trainer_cls

            # 注册自定义 reward manager（@register("...")）—— import 触发注册，
            # 让 omni/observer reward 在 trainer 解析 reward_manager 前就位。
            try:
                import trainer.observer_reward_manager  # noqa: F401
            except Exception as exc:  # noqa: BLE001 -- registry absent off-cluster
                print(f"[cl] observer reward manager 未注册({exc})", flush=True)

            # 阶段 F：patch hook factory 认 FQN（让 ObserverDiffHook 能经 agent_loop_config
            # 的 harness.hooks 挂上；不改 verl 源码）。import 即安装 patch。
            try:
                import trainer.observer_hook_register  # noqa: F401
            except Exception as exc:  # noqa: BLE001 -- recipe_custom absent off-cluster
                print(f"[cl] observer hook factory patch 未安装({exc})", flush=True)

            trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)  # custom_sync
            config.transfer_queue.enable = True

            # val_files alias：v1 _init_dataloader(trainer_base.py:597)无条件为 val 建 dataset,
            # 不判空 → val_files=null 时 copy_to_local(None) 崩(assert src[-1],'NoneType')。
            # b1 baseline 不做在线验证(test_freq=-1/val_before_train=false),val 仅占位 →
            # 空/缺失时 alias 到 train_files(与 v0 CLTaskRunner 同逻辑)。保 config 的 null 原意。
            import os as _os

            _vf = config.data.get("val_files", None)
            _val_missing = bool(_vf) and isinstance(_vf, str) and not _os.path.exists(_vf)
            if not _vf or _val_missing:
                _reason = "空" if not _vf else f"文件不存在({_vf})"
                OmegaConf.update(config, "data.val_files", config.data.train_files, force_add=True)
                print(f"[cl] data.val_files {_reason} → alias 到 train_files(v1 硬要求 val dataloader;"
                      "在线验证已关,仅占位不实际验证)", flush=True)

            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)
            self.config = config

            # 9桶 buffer：b1(buffer.enabled=false/lambda=0) → None（hook 空转）；R 系列 → BucketReplayBuffer。
            from trainer.cl_main import build_buffer

            buffer = build_buffer(config)

            tq.init(config.transfer_queue)
            try:
                self.trainer = trainer_cls(config=config)
                self.trainer.init()
                # ── CL 注入点：init(建 actor_rollout_wg)之后、fit 之前 ──
                _inject_cl_into_v1_trainer(self.trainer, config, buffer)
                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
            finally:
                tq.close()

    return CLTaskRunnerV1


# 模块级惰性单例：首次 run_ppo 时构建（避免 import 期依赖 ray）。
class _CLTaskRunnerV1Proxy:
    """Lazy proxy so ``run_ppo(cfg, task_runner_class=CLTaskRunnerV1)`` works.

    ``run_ppo`` 对 task_runner_class 调 ``.remote()`` / ``.options(...).remote()``。
    这里在首次访问这两个属性时才真正构建 ray.remote 类。
    """

    _cls = None

    def _resolve(self):
        if _CLTaskRunnerV1Proxy._cls is None:
            _CLTaskRunnerV1Proxy._cls = _make_cl_task_runner_v1()
        return _CLTaskRunnerV1Proxy._cls

    def remote(self, *a, **k):
        return self._resolve().remote(*a, **k)

    def options(self, *a, **k):
        return self._resolve().options(*a, **k)


CLTaskRunnerV1 = _CLTaskRunnerV1Proxy()


# ── 评测版 TaskRunner（复刻 CLTaskRunnerV1 的 init，但不 fit，改跑 ClawEval）─────────

def _make_cl_task_runner_eval():
    """Build the ``@ray.remote`` CLTaskRunnerEval class（评测版）。

    复刻 CLTaskRunnerV1 的 init 三步(trainer.init → init_agent_loop_manager)，但跳过
    fit，改跑 ClawEval 任务 + 从 TQ 读 judge 四维打分。eval_tasks 经文件传递
    (config.cl.eval_tasks_file)，结果写 config.cl.eval_output。延迟到函数内定义，避免
    import 期依赖 ray。
    """
    import ray

    @ray.remote
    class CLTaskRunnerEval:
        def __init__(self):
            self.config = None
            self.trainer = None
            self.agent_loop_manager = None

        def init_agent_loop_manager(self):
            from verl.utils.import_utils import load_class_from_fqn

            fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            cls = load_class_from_fqn(fqn, "AgentLoopManager") if fqn else None
            if cls is None:
                raise RuntimeError("评测需要 agent_loop_manager_class（RemoteAgentLoopManager）")
            self.agent_loop_manager = cls.create(
                config=self.config,
                llm_client=self.trainer.get_llm_client(),
                teacher_client=self.trainer.get_teacher_client(),
                reward_loop_worker_handles=self.trainer.get_reward_handles(),
            )

        def run(self, config):
            """起 trainer + RemoteAgentLoopManager，跑 ClawEval，结果写文件。"""
            import json
            import os as _os
            from pprint import pprint

            import transfer_queue as tq
            from omegaconf import OmegaConf
            from verl.trainer.ppo.v1 import get_trainer_cls

            # 注册 reward manager（让 judge 打分生效，同训练）
            try:
                import trainer.observer_reward_manager  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                print(f"[cl-eval] observer reward manager 未注册({exc})", flush=True)
            try:
                import trainer.observer_hook_register  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                print(f"[cl-eval] observer hook patch 未安装({exc})", flush=True)

            trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
            config.transfer_queue.enable = True

            # val_files alias（v1 硬要求 val dataloader，同 CLTaskRunnerV1）
            _vf = config.data.get("val_files", None)
            if not _vf or (isinstance(_vf, str) and not _os.path.exists(_vf)):
                OmegaConf.update(config, "data.val_files", config.data.train_files, force_add=True)

            pprint(OmegaConf.to_container(config, resolve=True))
            OmegaConf.resolve(config)
            self.config = config

            # 读 eval 参数（eval_tasks 文件 + 输出路径 + num_runs，由 run_cl_eval 写入 config.cl）
            cl_cfg = config.get("cl", {}) or {}
            tasks_file = cl_cfg.get("eval_tasks_file")
            output = cl_cfg.get("eval_output")
            num_runs = int(cl_cfg.get("eval_num_runs", 3))
            with open(tasks_file) as f:
                eval_tasks = json.load(f)

            tq.init(config.transfer_queue)
            try:
                self.trainer = trainer_cls(config=config)
                self.trainer.init()
                self.init_agent_loop_manager()

                results = self._run_eval(eval_tasks, num_runs)
                with open(output, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"[cl-eval] TaskRunner 完成 {len(results)} 条 → {output}", flush=True)
            finally:
                tq.close()

        def _run_eval(self, eval_tasks: list[dict], num_runs: int) -> list[dict]:
            """构造 ClawEval prompts → generate_sequences → 从 TQ 读 reward 四维。"""
            from verl.utils import tensordict_utils as tu
            from verl.utils.tensordict_utils import list_of_dict_to_tensordict

            import uuid as _uuid

            rows = []
            for task in eval_tasks:
                p = task.get("prompt") or task.get("messages")
                raw = p if isinstance(p, list) else [{"role": "user", "content": p}]
                tid = task["task_id"]
                bucket = task.get("bucket") or task.get("category") or "Unlabeled"
                for i in range(num_runs):
                    # 评测行必须补齐训练行同款字段，否则 omni reward manager 读
                    # non_tensor_batch["reward_model"] 会 KeyError（session 全 abort）；
                    # extra_info.bucket/record_id 供 extract 归桶 + Pass^N 分组。
                    # ★ uid 必须【无下划线】：verl ReplayBuffer.sample 用 key.split("_")[0]
                    #   从轨迹 key 还原 prompt uid（replay_buffer.py:295），硬依赖 uid 无 `_`
                    #   （官方 uid=uuid4，只有连字符）。曾用 f"{tid}_run{i}"（tid 如
                    #   C01zh_mortgage_prepay 含多个 `_`）→ split("_")[0] 只取到 "C01zh"、
                    #   与 selected 里的完整 uid 匹配不上 → 返回 batch keys 恒空 → 585 成功却 0
                    #   结果（2026-08-28 第五层真根因，E23）。故改 uuid4；tid/run 走 extra_info。
                    rows.append({
                        "raw_prompt": raw,
                        "uid": _uuid.uuid4().hex,
                        "data_source": "claw_eval",
                        "reward_model": {
                            "ground_truth": "",
                            "style": "rule",
                            "reward_fn": {"_function_name": "trainer.model_reward_omni.compute_score"},
                        },
                        "extra_info": {
                            "record_id": tid,
                            "bucket": bucket,
                            "run_idx": i,
                        },
                    })

            prompts = list_of_dict_to_tensordict(rows)
            # global_steps / validate 是标量（整个 batch 一个），不是 per-sample，
            # 否则 session worker 里 int(global_steps) 会把 NonTensorStack 当 list 报错。
            tu.assign_non_tensor_data(prompts, "global_steps", 0)
            tu.assign_non_tensor_data(prompts, "validate", True)

            # ★ 预播种 val 分区（复刻官方 sync_trainer._validate:326-332）：generate_sequences
            # 【之前】把每个 uid 以 status=pending 写进 val 分区。缺这步则 replay_buffer.sample
            # (partition_id="val") 死等一个空分区——_has_enough_samples 永远 False → 无限刷
            # `pending:0 running:0 finished:0 failure:0` 空转不返回（2026-08-27 定位）。
            # RemoteAgentLoopManager.replay_buffer 恒 None（create 不传），故 running/failure 状态
            # 全靠 session worker 写回 TQ + 这里的 pending 播种，不是 manager 内部 add。
            import transfer_queue as tq

            uids = tu.get(prompts, "uid")
            uid_values = uids.tolist() if hasattr(uids, "tolist") else list(uids)
            seed_tags = [
                {"is_prompt": True, "status": "pending", "global_steps": 0}
                for _ in range(len(uid_values))
            ]
            tq.kv_batch_put(keys=[str(u) for u in uid_values], partition_id="val", tags=seed_tags)

            rollout_metrics = self.agent_loop_manager.generate_sequences(prompts) or {}

            # ★ 诊断（2026-08-27 第五层排查）：GS 返回的关键计数显式 print（logger.info 级会被
            # verl 日志级别/RAY_DEDUP 吞掉，看不到 num_success_outputs 就定位不了数据在哪环丢）。
            _diag = {k: rollout_metrics.get(k) for k in (
                "rollout/skipped_step", "rollout/num_success_outputs", "rollout/num_failed_uids",
                "rollout/num_success_sessions", "rollout/num_group_size_filtered_uids",
            )}
            print(f"[cl-eval][diag] generate_sequences 返回: {_diag} | 播种 uid 数={len(uid_values)}", flush=True)

            # ★ 全失败守卫（复刻官方 _validate:334-340）：num_success_outputs==0 时
            # generate_sequences 返回 rollout/skipped_step=1，val 分区无任何 terminal key →
            # 若仍进 sample 会死等。此时直接返回空结果（本 model_type 记 0 条），不阻塞后续 ckpt。
            if rollout_metrics.get("rollout/skipped_step", 0.0) > 0:
                print(
                    f"[cl-eval] ⚠️ 本批 rollout 全失败(skipped_step=1)，"
                    f"failure_code_counts={ {k: v for k, v in rollout_metrics.items() if 'failure_code' in k} }，"
                    f"跳过 sample，返回空结果",
                    flush=True,
                )
                return []

            # ★ sample 返回二元组 (KVBatchMeta, drop_metrics)（replay_buffer.py:300-301），
            #   必须解包——官方训练路径均为 `batch, _ = ...sample(...)`（trainer_base.py:470/925）。
            #   曾误写 `batch = ...sample(...)` → batch 是 tuple → getattr(tuple,"keys")=None →
            #   keys 数=0 → extract 抽 0 条 → 585 个成功任务却产出 0 结果（2026-08-28 第五层根因）。
            batch, _drop_metrics = self.trainer.replay_buffer.sample(
                global_steps=0, partition_id="val", batch_size=len(rows)
            )
            batch_keys = list(getattr(batch, "keys", None) or [])
            print(
                f"[cl-eval][diag] sample 返回 batch: keys 数={len(batch_keys)} "
                f"partition=val batch_size={len(rows)}",
                flush=True,
            )

            # ★ 结果提取【照官方 _validate 的 kv_batch_get 取字段】(trainer_base.py:960-995)，
            #   不再走 trajectory_adapter_v1.extract_trajectories_from_kvbatch(by_meta)。
            #   根因(2026-08-28 第六层 E25)：by_meta 依赖 KVBatchMeta.fields,但 sample 组装的
            #   batch fields=None → kv_batch_get_by_meta failed → 620 key 全取不到字段 → bucket
            #   None → skipped 620/620 → 0 结果。官方按 key 列表直接 kv_batch_get 取,不依赖
            #   fields 元数据。session worker 写回 key 格式 {uid}_{session}_{index}(worker.py:1021),
            #   field 含 rm_scores(总 reward,末 token) + extra_info(我们塞的 record_id/bucket/
            #   run_idx) + extra_fields.reward_extra_info(judge 四维)(worker.py:1083/1085/1091)。
            #   多输出 session 取每 session 最高 index 的最终输出(官方 933-950 逻辑)。
            results = []
            if batch_keys:
                # 每 session(uid_session)只留最高 index 的最终输出。
                session_max: dict[str, tuple[int, int]] = {}
                for pos, key in enumerate(batch_keys):
                    parts = key.rsplit("_", 2)
                    if len(parts) == 3:
                        skey = f"{parts[0]}_{parts[1]}"
                        idx = int(parts[2]) if parts[2].isdigit() else 0
                    else:
                        skey, idx = key, 0
                    if skey not in session_max or idx > session_max[skey][0]:
                        session_max[skey] = (idx, pos)
                final_keys = [batch_keys[pos] for _, (_, pos) in session_max.items()]

                data = tq.kv_batch_get(
                    keys=final_keys,
                    partition_id=batch.partition_id,
                    select_fields=["rm_scores", "extra_info", "extra_fields"],
                )

                import torch as _torch

                def _nontensor_col(v):
                    if v is None:
                        return [None] * len(final_keys)
                    return v.tolist() if hasattr(v, "tolist") else list(v)

                def _f(d, k, default):
                    v = d.get(k)
                    return float(v) if v is not None else float(default)

                # ★ rm_scores 是 NestedTensor（多轮 response 长度不齐），.tolist() 不支持；
                #   官方 _validate 用 .sum(dim=1) 归约成 [N] 标量再 tolist（trainer_base.py:970）。
                _rm = data.get("rm_scores") if hasattr(data, "get") else None
                if isinstance(_rm, _torch.Tensor) and getattr(_rm, "is_nested", False):
                    rm_scalars = _rm.sum(dim=1).tolist()
                elif isinstance(_rm, _torch.Tensor) and _rm.dim() >= 2:
                    rm_scalars = _rm.sum(dim=1).tolist()
                elif hasattr(_rm, "tolist"):
                    rm_scalars = _rm.tolist()
                else:
                    rm_scalars = [0.0] * len(final_keys)

                ei_col = _nontensor_col(data.get("extra_info") if hasattr(data, "get") else None)
                ef_col = _nontensor_col(data.get("extra_fields") if hasattr(data, "get") else None)

                for j in range(len(final_keys)):
                    ei = ei_col[j] if isinstance(ei_col[j], dict) else {}
                    ef = ef_col[j] if isinstance(ef_col[j], dict) else {}
                    rei = ef.get("reward_extra_info", {}) if isinstance(ef, dict) else {}
                    if not isinstance(rei, dict):
                        rei = {}
                    # judge 各分量（judge 打的，见 model_reward.compute_score 返回 dict）。
                    correctness = _f(rei, "correctness", 0.0)
                    trajectory = _f(rei, "trajectory", 0.0)
                    safety = _f(rei, "safety", 1.0)  # 缺省视为 safe(1)，与 judge 默认一致
                    # ★ 官方 ClawEval rubric：score = s_safety × (0.8·s_completion + 0.2·s_robustness)
                    #   映射 s_completion→correctness(0~1 完成质量)、s_robustness→trajectory(五维聚合)。
                    #   ≥0.75 算 pass（用户 2026-08-28 定，替代旧 task_done 阈值判定）。
                    score = safety * (0.8 * correctness + 0.2 * trajectory)
                    results.append({
                        "task_id": ei.get("record_id") or ei.get("task_id") or "",
                        "bucket": ei.get("bucket") or ei.get("category") or "",
                        "run_idx": ei.get("run_idx", -1),
                        "reward": score,
                        "task_done": 1.0 if score >= 0.75 else 0.0,
                        "correctness": correctness,
                        "trajectory": trajectory,
                        "safety": safety,
                    })
                tq.kv_clear(keys=batch_keys, partition_id=batch.partition_id)

            print(f"[cl-eval][diag] 提取 {len(results)} 条结果", flush=True)
            return results

    return CLTaskRunnerEval


def run_cl_eval(cfg: Any, eval_tasks: list[dict], num_runs: int) -> list[dict]:
    """起 Ray，复用 run_ppo 启动，但 task_runner_class = 评测版（跑 ClawEval）。"""
    import json
    import os
    import tempfile

    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo

    cfg = merge_verl_config(cfg)

    # eval_tasks 经临时文件传给 remote runner（ray.put 的对象 ref 在 remote 里不好取）。
    # ★ 必须写到 AFS 共享路径（非 /tmp/），否则跨节点 worker 读不到（多机 /tmp 不共享）。
    _eval_tmpdir = os.environ.get("CL_EVAL_TMPDIR", str(Path(__file__).resolve().parent.parent.parent / "eval" / ".tmp"))
    os.makedirs(_eval_tmpdir, exist_ok=True)
    _tmp_fd, tasks_file = tempfile.mkstemp(suffix=".json", dir=_eval_tmpdir)
    with os.fdopen(_tmp_fd, "w") as f:
        json.dump(eval_tasks, f, ensure_ascii=False)
    # ★ output 也是跨节点文件（worker 写、driver 读），同样放 AFS，不能放 /tmp。
    output = os.environ.get("CL_EVAL_OUTPUT", str(Path(_eval_tmpdir) / "cl_eval_result.json"))
    OmegaConf.update(cfg, "cl.eval_tasks_file", tasks_file, force_add=True)
    OmegaConf.update(cfg, "cl.eval_output", output, force_add=True)
    OmegaConf.update(cfg, "cl.eval_num_runs", num_runs, force_add=True)

    # env 透传：复用 run_cl_ppo 的透传项（评测 worker 同样不继承 driver shell env）。
    # 这里复刻关键项（PYTHONPATH/VERL_USE_EXTERNAL_MODULES/内存分配器/judge 凭证）。
    _passthrough = {}
    _pp = os.environ.get("PYTHONPATH")
    if _pp:
        _passthrough["PYTHONPATH"] = _pp
    _ext = os.environ.get("VERL_USE_EXTERNAL_MODULES")
    if _ext:
        _passthrough["VERL_USE_EXTERNAL_MODULES"] = _ext
    for _mk in ("LD_PRELOAD", "MALLOC_CONF", "MALLOC_ARENA_MAX", "MALLOC_TRIM_THRESHOLD_"):
        _mv = os.environ.get(_mk)
        if _mv is not None:
            _passthrough[_mk] = _mv
    for _rk in ("TOKENHUB_API_KEY", "REWARD_API_BASE", "REWARD_MODEL", "REWARD_API_KEY", "REWARD_JUDGE_MAX_TOKENS"):
        _rv = os.environ.get(_rk)
        if _rv is not None:
            _passthrough[_rk] = _rv
    _tmo = os.environ.get("TEXT_MODEL_ONLY")
    if _tmo is not None:
        _passthrough["TEXT_MODEL_ONLY"] = _tmo
    # 评测恒存完整 conversation（供官方 grader 评分用，见 trajectory_buffer.py）
    _passthrough["CL_STORE_TRAJECTORY_MESSAGES"] = "1"
    if _passthrough:
        OmegaConf.update(
            cfg,
            "ray_kwargs.ray_init.runtime_env.env_vars",
            {**(OmegaConf.select(cfg, "ray_kwargs.ray_init.runtime_env.env_vars") or {}), **_passthrough},
            force_add=True,
        )
        print(f"[cl-eval] 透传 env 到 Ray worker: {list(_passthrough)}", flush=True)

    task_runner_cls = _make_cl_task_runner_eval()
    run_ppo(cfg, task_runner_class=task_runner_cls)

    with open(output) as f:
        return json.load(f)


def build_trainer(cfg: Any, buffer: Any | None = None):
    """Construct RayPPOTrainer without ``init_workers`` (for tests / local wiring).

    Does not start Ray. Caller must call ``trainer.init_workers()``,
    ``inject_cl_loss()``, ``install_buffer_hooks()``, then ``trainer.fit()``.
    """
    from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.config import validate_config
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local

    cfg = merge_verl_config(cfg)
    runner = TaskRunner()
    actor_rollout_cls, ray_worker_group_cls = runner.add_actor_rollout_worker(cfg)
    runner.add_critic_worker(cfg)
    runner.add_reward_model_resource_pool(cfg)
    runner.add_teacher_model_resource_pool(cfg)
    runner.add_ref_policy_worker(cfg, actor_rollout_cls)
    validate_config(
        config=cfg,
        use_reference_policy=need_reference_policy(cfg),
        use_critic=need_critic(cfg),
    )
    local_path = copy_to_local(
        cfg.actor_rollout_ref.model.path,
        use_shm=cfg.actor_rollout_ref.model.get("use_shm", False),
    )
    trust_remote_code = cfg.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
    resource_pool_manager = runner.init_resource_pool_mgr(cfg)
    # See CLTaskRunner.run: verl requires a loadable non-empty val dataloader even
    # when in-loop validation is off; alias val_files -> train_files when empty OR
    # pointing at a missing file.
    import os as _os

    _vf = cfg.data.get("val_files", None)
    if not _vf or (isinstance(_vf, str) and not _os.path.exists(_vf)):
        OmegaConf.update(cfg, "data.val_files", cfg.data.train_files, force_add=True)
    train_dataset = create_rl_dataset(
        cfg.data.train_files,
        cfg.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=cfg.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        cfg.data.val_files,
        cfg.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=cfg.data.get("val_max_samples", -1),
    )
    train_sampler = create_rl_sampler(cfg.data, train_dataset)
    return RayPPOTrainer(
        config=cfg,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=runner.role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
    )
