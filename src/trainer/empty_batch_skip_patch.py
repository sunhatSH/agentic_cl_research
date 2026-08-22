"""Empty-batch skip guard for CustomPPOTrainerSync (E18).

问题（2026-08-21 实测崩溃）:
    verl/utils/seqlen_balancing.py:239
    assert len(seqlen_list) >= k_partitions → "number of items:[0] < k_partitions:[4]"
    调用链: sync_trainer._step_once → _balance_batch → get_seqlen_balanced_partitions。

根因:
    ``all_failed_policy: skip`` 只在生成阶段查 ``num_success_outputs == 0`` 就 skip 整步；
    但"部分 rollout 成功、经 group_size 组过滤后又全被丢"这条边界不触发那个 skip——
    batch 在 ``_step_once`` 里非空进入、materialize/过滤后变成 0 条，直接把空 batch 送进
    ``_balance_batch`` → k_partitions 划分 assert 崩，整训练进程挂掉（多机全崩）。
    典型触发: 瞬时故障（如 config 文件丢失）导致大量 session 失败，恢复瞬间少量成功但
    组不满 → 空 batch。也可能出现在任何"成功数 < 一个完整 group"的抖动步。

修法（不改 verl 源码，走 VERL_USE_EXTERNAL_MODULES 注入）:
    包 ``CustomPPOTrainerSync._balance_batch``：进函数先查 batch 是否为空（tags 为空 /
    全是 padding）。空则抛类型化 ``_EmptyBatchSkip``；再包 ``step`` 捕获它 → 记
    ``rollout/empty_batch_skip`` 指标并 return None，复用 fit() 里既有的 ``batch is None``
    → skip 分支（sync_trainer.py fit: "Skip training update ... because all rollouts failed"）。
    等价于把"空 batch"并入"all-failed skip"语义，训练跳过该步继续，而非崩。

关联: [[train-crash-hidden-bugs]] E18；skip 机制见 agent_loop_manager.all_failed_policy。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


class _EmptyBatchSkip(Exception):
    """Sentinel: batch 为空（rollout 全失败/全被组过滤），该步应 skip 而非崩。"""


def _batch_is_empty(batch) -> bool:
    """batch 无可训练样本：tags 为空，或所有 tag 都是 padding。"""
    tags = getattr(batch, "tags", None)
    if not tags:
        return True
    # 全 padding 也视为空（_balance_batch 会先 upsample，但若原始 0 条则 upsample 也救不回）
    return all(bool(t.get("is_padding", False)) for t in tags)


def _apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        from recipe_custom.trainer.v1.sync_trainer import CustomPPOTrainerSync
    except Exception as exc:  # pragma: no cover - 仅在 verl 环境可 import
        logger.warning("empty_batch_skip_patch: cannot import CustomPPOTrainerSync (%s); skip", exc)
        return

    if getattr(CustomPPOTrainerSync, "_cl_empty_batch_skip_patched", False):
        _PATCHED = True
        return

    _orig_balance = CustomPPOTrainerSync._balance_batch
    _orig_step = CustomPPOTrainerSync.step

    def _balance_batch(self, batch, metrics, *args, **kwargs):
        if _batch_is_empty(batch):
            n = len(getattr(batch, "tags", []) or [])
            logger.warning(
                "empty_batch_skip: batch has no trainable samples (tags=%s) at global_steps=%s; "
                "raising skip instead of crashing _balance_batch (all rollouts failed or filtered).",
                n,
                getattr(self, "global_steps", "?"),
            )
            raise _EmptyBatchSkip(f"empty batch at step {getattr(self, 'global_steps', '?')}")
        return _orig_balance(self, batch, metrics, *args, **kwargs)

    def step(self, metrics: dict, timing_raw: dict):
        try:
            return _orig_step(self, metrics, timing_raw)
        except _EmptyBatchSkip:
            # 复用 fit() 的 `batch is None` → skip 分支：记指标 + 返回 None，训练跳过该步。
            metrics["rollout/empty_batch_skip"] = 1.0
            metrics["rollout/skipped_step"] = 1.0
            logger.warning(
                "empty_batch_skip: skipping training update at global_steps=%s (empty batch).",
                getattr(self, "global_steps", "?"),
            )
            return None

    CustomPPOTrainerSync._balance_batch = _balance_batch
    CustomPPOTrainerSync.step = step
    CustomPPOTrainerSync._cl_empty_batch_skip_patched = True
    _PATCHED = True
    logger.info("empty_batch_skip_patch: patched CustomPPOTrainerSync._balance_batch + step (E18).")


_apply()
