"""阶段 D：9桶 buffer 在 verl v1 (custom_sync / KVBatchMeta) 下的 hook 适配.

见 plan swift-juggling-toast 阶段 D。用户已定：9桶 buffer 用【我们自己的】方案
（BucketReplayBuffer + 掺回放行），【不】子类化 v1 的 ReplayBuffer —— 后者是 online
off-policy 采样器（按 model version 新鲜度门控、选完即 kv_clear），跟我们"跨桶持久回放
防遗忘"是正交的两个维度。

与旧 install_buffer_hooks（trainer/verl_runner.py，hook DataProto 版 _update_actor）的差异：
v1 的 ``_update_actor(batch: KVBatchMeta, metrics)`` —— batch 是 transfer_queue 的元数据
句柄（keys/tags/partition_id），实际张量在 tq 里。所以：
  · PRE 掺回放行：把回放张量 ``kv_batch_put`` 写进 tq → ``KVBatchMeta.concat`` 合并 keys。
  · POST 抽 winner：``kv_batch_get_by_meta`` 从 tq 取 rollout 张量 → 按 task_id 挑 winner
    → buffer.add_trajectory。
b1 baseline（buffer.enabled=false / lambda_replay=0）不装此 hook。仅 R 系列触发。

⚠️ 本文件强依赖 transfer_queue / KVBatchMeta 运行时行为，本机无 GPU/tq server 无法端到端
验证；集群 R 系列跑前需实测（见文末 CLUSTER-TODO）。b1 不受影响。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# tq 字段：回放行与 rollout 行必须共享同一组 fields 才能 concat（KVBatchMeta.concat 校验
# fields 集合一致）。回放行补齐 rollout 的字段名，rollout 行补齐回放专属字段（镜像零值）。
_REPLAY_PARTITION = "train"


def install_buffer_hooks_v1(trainer: Any, buffer: Any | None, cfg: Any) -> None:
    """给 v1 custom_sync trainer 装 9桶 buffer hook（掺回放行 + 抽 winner 入库）.

    调用时机：CLTaskRunnerV1.run 里 ``trainer.init()`` 之后、``fit()`` 之前
    （与 inject_cl_loss 并列）。buffer is None（b1）时空转。
    """
    if buffer is None:
        return

    cl = cfg.get("cl", {}) or {}
    lambda_replay = float(cl.get("lambda_replay", 0.0))
    replay_batch_size = int(cl.get("replay_batch_size", 512))
    # replay_ratio: 新:旧 的比值（默认 5 → 每 5 条新轨迹配 1 条回放，占比恒定 1/6）。
    # >0 时回放量按【本 step 实际新轨迹数】动态算 ceil(new / ratio)，不再用固定 replay_batch_size，
    # 使回放随新数据等比缩放——新轨迹因沙箱失败缩水时回放同步缩小，占比不漂移（见 RunLog 2026-07-30）。
    # <=0 时退回旧行为（固定 replay_batch_size）。replay_batch_size 仍作为上限兜底。
    replay_ratio = float(cl.get("replay_ratio", 5.0))
    replay_warmup_size = int(cl.get("replay_warmup_size", 0))
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

    stats_logger = BufferStatsLogger(f"logs/buffer_stats/{exp_name}.jsonl")
    tokenizer = getattr(trainer, "tokenizer", None)
    original_update = trainer._update_actor  # v1 签名: (batch: KVBatchMeta, metrics: dict)

    def patched_update(batch, metrics):
        rl_batch = batch  # 抽 winner 用【掺入前】的 rollout-only batch（回放行不回灌 buffer）
        replay_rows: dict = {}
        replay_meta = None

        # ── 1. PRE：从 9桶 buffer 采旧桶 winner，掺进 tq + 合并 KVBatchMeta ──
        if lambda_replay > 0:
            # 回放量：replay_ratio>0 时按【本 step 实际新轨迹数】动态算 ceil(new/ratio)，
            # 新轨迹数 = 非 padding 行数（掺入前的 rl_batch）。否则退回固定 replay_batch_size。
            eff_replay = replay_batch_size
            if replay_ratio > 0:
                new_count = sum(
                    not tag.get("is_padding", False) for tag in getattr(rl_batch, "tags", []) or []
                )
                # ceil(new_count / ratio)，整数除法实现，避免 import math
                eff_replay = -(-new_count // int(replay_ratio)) if new_count > 0 else 0
                # replay_batch_size 作上限兜底（buffer 再大也不超它）
                if replay_batch_size > 0:
                    eff_replay = min(eff_replay, replay_batch_size)
            replay_rows = prepare_replay_rows(
                buffer, weighting, tokenizer, eff_replay, warmup_size=replay_warmup_size
            )
            if replay_rows:
                shuffle_seed = int(getattr(trainer, "global_steps", 0) or 0)
                batch, replay_meta = _append_replay_rows_v1(
                    batch, replay_rows, rl_batch, shuffle_seed=shuffle_seed
                )

        # ── 2. 原生 actor 更新（worker 从 tq 按 batch.keys 取张量训练）──
        result = original_update(batch, metrics)

        step = getattr(trainer, "global_steps", buffer._step)
        buffer.set_step(step)

        # ── 3. POST：forgetting_risk 回填（对刚回放的轨迹重算 current-policy logprob）──
        tids = replay_rows.get(REPLAY_TIDS_KEY) if replay_rows else None
        if tids and forgetting_update_freq > 0 and step % forgetting_update_freq == 0:
            means = compute_replay_current_logprobs(trainer, replay_rows)
            if means is not None:
                backfill_forgetting(buffer, tids, means)

        # ── 4. POST：抽 winner（每 task_id reward 最高）入 9桶 buffer ──
        _ingest_winners(rl_batch, buffer, exp_name, step)

        # ── 5. POST：buffer 动态证据（metrics + sidecar JSONL）──
        if stats_log_freq > 0 and step % stats_log_freq == 0:
            stats = buffer.stats()
            _merge_buffer_metrics(metrics, stats, flatten_buffer_stats)
            stats_logger.log(step, stats)

        # ── 6. POST：周期性 buffer 全量快照 ──
        if save_freq > 0 and step > 0 and step % save_freq == 0:
            snap = Path(f"buffer_dumps/{exp_name}-step-{step}.sqlite")
            snap.parent.mkdir(parents=True, exist_ok=True)
            buffer.dump(snap)

        # ── 7. 清理本 step 掺入的回放 keys（不让它们污染下一 step 的 tq 元数据）──
        if replay_meta is not None:
            _clear_replay_keys(replay_meta)

        return result

    trainer._update_actor = patched_update
    print("[cl] v1: 9桶 buffer hook 已装 (patched _update_actor, KVBatchMeta 适配)", flush=True)


def _merge_buffer_metrics(metrics: dict, stats: Any, flatten_fn) -> None:
    """把扁平化的 buffer stats 并进 v1 fit 的 metrics dict（wandb/file logger 可见）。

    v1 的 _update_actor(batch, metrics) 直接拿到 metrics 字典（不同于 v0 的 meta_info），
    所以这里直接 update 即可。"""
    if isinstance(metrics, dict):
        metrics.update(flatten_fn(stats))


def _ingest_winners(rl_batch: Any, buffer: Any, exp_name: str, step: int) -> None:
    """从 rollout batch 抽每 task_id 的 winner（reward 最高）入 9桶 buffer.

    v1: rl_batch 是 KVBatchMeta，需从 tq 取张量 + tags。用 v1 版 extractor。"""
    from data.cleaning import strip_zw
    from trainer.trajectory_adapter_v1 import extract_trajectories_from_kvbatch

    groups: dict[str, list[tuple[Any, str, dict]]] = {}
    for trajectory, bucket, meta in extract_trajectories_from_kvbatch(
        rl_batch, valid_buckets=getattr(buffer, "bucket_names", None)
    ):
        for msg in trajectory.get("messages", []):
            if isinstance(msg.get("content"), str):
                msg["content"] = strip_zw(msg["content"])
        tid = meta.get("task_id") or ""
        groups.setdefault(tid, []).append((trajectory, bucket, meta))

    winners = []
    for _tid, candidates in groups.items():
        best = max(candidates, key=lambda x: float(x[2].get("reward", 0) or 0))
        buffer.add_trajectory(*best)
        winners.append(best)

    _persist_winners(winners, exp_name, step)


def _append_replay_rows_v1(batch, replay_rows: dict, rl_batch, shuffle_seed: int = 0):
    """把回放张量写进 tq，返回 (合并后的 KVBatchMeta, 回放 KVBatchMeta).

    步骤：
      1. 回放行张量(TensorDict) 补齐 rollout 的字段名（is_replay/replay_mask/replay_weights
         + 镜像 old_log_probs/advantages 零值），rollout 行也已在 rollout 阶段带这些字段
         （或在此不需要——v1 rollout 产出的字段由 agent_loop 决定，见 CLUSTER-TODO）。
      2. 生成回放 keys（uid 前缀避免与 rollout key 冲突）。
      3. kv_batch_put 写进 tq(train 分区)。
      4. KVBatchMeta.concat([batch, replay_meta]) 合并 keys → 训练时 worker 取到回放行。

    ⚠️ CLUSTER-TODO：回放行的字段名/形状必须与 v1 rollout 写进 tq 的字段完全对齐
    （KVBatchMeta.concat 校验 fields 集合一致）。具体字段由 session_worker 写入决定，
    集群实测对齐。本机给出结构，字段清单标注在 _replay_tensordict。"""
    try:
        import transfer_queue as tq
        from tensordict import TensorDict
        from transfer_queue import KVBatchMeta
    except ImportError:
        return batch, None

    from trainer.replay_forward import REPLAY_TIDS_KEY

    tids = replay_rows.get(REPLAY_TIDS_KEY)
    fields_td = _replay_tensordict(replay_rows, batch)
    if fields_td is None or fields_td.batch_size[0] == 0:
        return batch, None

    n = fields_td.batch_size[0]
    base_step = int(shuffle_seed)
    replay_keys = [f"replay-{base_step}-{i}_{0}_{0}" for i in range(n)]
    tags = [{"is_padding": False, "is_replay": True, "status": "finished"} for _ in range(n)]

    tq.kv_batch_put(keys=replay_keys, partition_id=_REPLAY_PARTITION, fields=fields_td, tags=tags)
    replay_meta = KVBatchMeta(
        keys=replay_keys,
        tags=tags,
        partition_id=_REPLAY_PARTITION,
        fields=list(fields_td.keys()),
    )
    merged = KVBatchMeta.concat([batch, replay_meta])
    return merged, replay_meta


def _replay_tensordict(replay_rows: dict, batch):
    """把 prepare_replay_rows 产出的张量 dict 转成 tq 需要的 TensorDict.

    replay_rows 结构（build_replay_rows 产出）：prompts/responses/response_mask/
    replay_response_mask/replay_token_weights/is_replay + 镜像 old_log_probs/advantages。
    ⚠️ CLUSTER-TODO：与 v1 rollout 写 tq 的字段对齐（见 _append_replay_rows_v1 说明）。"""
    try:
        import torch
        from tensordict import TensorDict
    except ImportError:
        return None

    from trainer.replay_forward import REPLAY_TIDS_KEY

    rows = {k: v for k, v in replay_rows.items() if k != REPLAY_TIDS_KEY and hasattr(v, "shape")}
    if not rows:
        return None
    n = next(iter(rows.values())).shape[0]
    return TensorDict(rows, batch_size=n)


def _clear_replay_keys(replay_meta) -> None:
    """清掉本 step 掺入的回放 keys（下一 step 的 tq 元数据不含它们）。"""
    try:
        import transfer_queue as tq

        tq.kv_clear(keys=list(replay_meta.keys), partition_id=replay_meta.partition_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[cl] v1: 清理回放 keys 失败(不致命): {exc}", flush=True)


def _persist_winners(winners: list, exp_name: str, step: int) -> None:
    """Write winner trajectories as JSONL for offline analysis（与 v0 版一致）。"""
    if not winners:
        return
    import json

    out_dir = Path(f"rollouts/training/{exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"step-{step}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for traj, bucket, meta in winners:
            row = {
                "task_id": meta.get("task_id", ""),
                "bucket": bucket,
                "reward": meta.get("reward"),
                "messages": traj.get("messages", []),
                "step": step,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[persist] {len(winners)} winners → {out_file}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER-TODO（R 系列跑前集群实测）：
#  1. v1 rollout 写进 tq(train 分区)的字段清单 —— 回放行必须完全对齐（concat 校验）。
#     实测：训练时 dump 一条 rollout key 的 fields，据此补齐 _replay_tensordict。
#  2. 回放行 key 格式 "replay-{step}-{i}_{0}_{0}" 是否与 ReplayBuffer 的 uid_session_index
#     解析兼容（replay_buffer.py:203 用 key.split("_")[0] 取 uid；我们的 uid=replay-{step}-{i}）。
#  3. concat 后的 batch 进 _balance_batch / _compute_old_log_prob 是否对回放行(response_mask=0)
#     正确处理（回放行不该重算 advantage —— 已在 rollout 阶段前置，但 v1 流水线顺序需实测）。
#  4. extract_trajectories_from_kvbatch（trajectory_adapter_v1.py）能否从 tq 正确取回
#     messages/bucket/reward（v1 non_tensor 字段来源，见该文件 CLUSTER-TODO）。
# ─────────────────────────────────────────────────────────────────────────────
