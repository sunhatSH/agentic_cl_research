"""阶段 D 配套：从 v1 KVBatchMeta 抽轨迹入 9桶 buffer（trajectory_adapter 的 v1 版）.

旧 trajectory_adapter.extract_trajectories_from_batch 读 DataProto.non_tensor_batch /
batch(TensorDict)。v1 的 batch 是 KVBatchMeta（tq 元数据句柄），实际数据在 transfer_queue，
需 ``kv_batch_get_by_meta`` 取张量、从 tags 取元数据。本文件产出与旧版【相同的契约】：
返回 list[(trajectory, bucket, meta)]，meta 含 reward/task_id/bucket/response_token_ids/
original_logprobs/success_rate/messages —— 这样 buffer.add_trajectory 与 TokenWeighting 不变。

v1 字段来源（session_worker 写 tq，见 recipe_custom/agent/session_worker/worker.py:195）：
  · non_tensor: raw_prompt / data_source / reward_model / extra_info / agent_name
    - bucket 在 extra_info["bucket"]；task_id 在 extra_info["record_id"]
  · tensor: prompts / responses / response_mask / old_log_probs / rollout_log_probs
  · reward: rm_scores（标量）或 reward_info(dict)

⚠️ CLUSTER-TODO：v1 究竟把 extra_info/messages 以什么形状存进 tq（tag 还是 field）需实测。
本文件按"extra_info 进 tag、张量进 field"实现，集群 dump 一条 key 校准。b1 不触发（buffer 关）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 从 tq 取的张量字段（与 rollout 写入对齐；缺失字段 kv_batch_get 忽略）。
_TENSOR_FIELDS = ["prompts", "responses", "response_mask", "old_log_probs", "rollout_log_probs", "rm_scores"]


def _row_to_list(row) -> list[float] | None:
    if row is None:
        return None
    if hasattr(row, "tolist"):
        try:
            return [float(x) for x in row.tolist()]
        except (TypeError, ValueError):
            return None
    try:
        return [float(x) for x in row]
    except (TypeError, ValueError):
        return None


def _tag_get(tag: dict, *keys, default=None):
    """从 tag（可能嵌套 extra_info）按优先级取值。"""
    for k in keys:
        if k in tag:
            return tag[k]
    ei = tag.get("extra_info")
    if isinstance(ei, dict):
        for k in keys:
            if k in ei:
                return ei[k]
    return default


def extract_trajectories_from_kvbatch(
    batch: Any, default_bucket: str | None = None, valid_buckets: list[str] | None = None
) -> list[tuple[Any, str, dict]]:
    """从 KVBatchMeta 抽 (trajectory, bucket, meta)，契约同旧 extract_trajectories_from_batch.

    batch: v1 KVBatchMeta（keys/tags/partition_id）。返回可直接喂 buffer.add_trajectory。
    """
    try:
        import transfer_queue as tq
    except ImportError:
        return []

    keys = getattr(batch, "keys", None)
    tags = getattr(batch, "tags", None)
    if not keys:
        return []

    # 取张量（best-effort：只取存在的 field，避免 KeyError）。
    fields_present = set(getattr(batch, "fields", None) or [])
    want = [f for f in _TENSOR_FIELDS if not fields_present or f in fields_present]
    td = None
    if want:
        try:
            td = tq.kv_batch_get_by_meta(batch, select_fields=want)
        except Exception as exc:  # noqa: BLE001
            logger.warning("kv_batch_get_by_meta failed (%s); metadata-only extraction", exc)
            td = None

    def _tensor_row(name, i):
        if td is None or name not in td:
            return None
        try:
            return td[name][i]
        except (IndexError, KeyError, TypeError):
            return None

    out: list[tuple[Any, str, dict]] = []
    skipped = 0
    n = len(keys)
    for i in range(n):
        tag = tags[i] if tags and i < len(tags) else {}
        # 跳过回放行 / padding 行（它们不回灌 buffer）。
        if tag.get("is_replay") or tag.get("is_padding"):
            continue

        meta: dict[str, Any] = {}
        bucket = _tag_get(tag, "bucket", "category", default=default_bucket)
        task_id = _tag_get(tag, "task_id", "record_id", "id")
        if task_id is not None:
            meta["task_id"] = task_id
            meta["pattern_id"] = task_id

        # reward：rm_scores 是 [R] 张量，reward 放在最后有效 token 位（其余为 0），
        # sum 得到标量。回退到 tag 里的 reward/score 字段。
        rm = _tensor_row("rm_scores", i)
        if rm is not None:
            try:
                meta["reward"] = float(rm.detach().float().sum().item())
            except (TypeError, ValueError, RuntimeError):
                pass
        if "reward" not in meta:
            r = _tag_get(tag, "reward", "score")
            if r is not None:
                try:
                    meta["reward"] = float(r)
                except (TypeError, ValueError):
                    pass

        # priority 信号：rollout logprob → original_logprobs；组内通过率 → success_rate。
        lp = _tensor_row("rollout_log_probs", i)
        if lp is None:
            lp = _tensor_row("old_log_probs", i)
        lp_list = _row_to_list(lp)
        if lp_list is not None:
            meta["original_logprobs"] = lp_list
        sr = _tag_get(tag, "success_rate")
        if sr is not None:
            try:
                meta["success_rate"] = float(sr)
            except (TypeError, ValueError):
                pass

        # response token ids（TokenWeighting 用）：按 response_mask 去 padding。
        rids = _row_to_list(_tensor_row("responses", i))
        if rids is not None:
            rmask = _row_to_list(_tensor_row("response_mask", i))
            if rmask:
                rids = [int(t) for t, m in zip(rids, rmask, strict=False) if m]
            else:
                rids = [int(t) for t in rids]
            meta["response_token_ids"] = rids

        # trajectory messages：v1 从 tag/extra_info 取（session_worker 存的 message_history）。
        trajectory = _tag_get(tag, "messages", "message_history")
        if isinstance(trajectory, dict) and "messages" in trajectory:
            trajectory = trajectory["messages"]
        if not isinstance(trajectory, list):
            # 回退：raw_prompt + 解码 response（若 messages 未存）。
            trajectory = _tag_get(tag, "raw_prompt", default=[])
            if not isinstance(trajectory, list):
                trajectory = []

        # bucket 兜底：从 assistant 文本恢复 <task_domain>（同旧版）。
        if bucket is None:
            bucket = _domain_from_messages(trajectory, valid_buckets)
        if bucket is None:
            skipped += 1
            continue
        meta["bucket"] = bucket
        meta.setdefault("messages", trajectory if isinstance(trajectory, list) else [])
        out.append((trajectory, bucket, meta))

    if skipped:
        logger.warning(
            "extract_trajectories_from_kvbatch: skipped %d/%d trajectories with no "
            "resolvable bucket (NOT added to buffer, avoids quota distortion).",
            skipped,
            n,
        )
    return out


def _domain_from_messages(trajectory: Any, valid_buckets: list[str] | None) -> str | None:
    """复用旧版：从 assistant 文本恢复 LLM 标注的 <task_domain>。"""
    from trainer.trajectory_adapter import _domain_from_messages as _impl

    return _impl(trajectory, valid_buckets)
