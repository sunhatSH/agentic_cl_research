"""Offline tests for trajectory_adapter_v1 prompt recovery (no torch/tq needed).

v1 tq 实测 rollout 行的 ``prompts`` field 常取空 → 回放行缺 prompt → build_replay_rows
整条丢弃（回放空转）。修复：用 ``input_ids`` 减去 response 段还原 prompt。本模块测该纯函数。
"""

from __future__ import annotations

from trainer.trajectory_adapter_v1 import _recover_prompt_from_input_ids


def test_recover_prompt_normal():
    """input = cat([prompt, response]) → 减去 response 段得 prompt。"""
    assert _recover_prompt_from_input_ids([1, 2, 3, 4, 5], [4, 5]) == [1, 2, 3]


def test_recover_prompt_float_input_coerced_to_int():
    """tq 张量 tolist 出 float → 还原后转 int。"""
    assert _recover_prompt_from_input_ids([1.0, 2.0, 3.0], [3]) == [1, 2]


def test_recover_prompt_missing_input_ids():
    """input_ids 缺失/空 → None（无从还原）。"""
    assert _recover_prompt_from_input_ids(None, [1, 2]) is None
    assert _recover_prompt_from_input_ids([], [1, 2]) is None


def test_recover_prompt_missing_response():
    """response 缺失 → None（切不出 prompt 段边界）。"""
    assert _recover_prompt_from_input_ids([1, 2, 3], []) is None


def test_recover_prompt_no_prompt_segment():
    """input 长度 <= response（无 prompt 段）→ None，不返回空/负长度。"""
    assert _recover_prompt_from_input_ids([4, 5], [4, 5]) is None
    assert _recover_prompt_from_input_ids([5], [4, 5]) is None


def test_recover_prompt_single_token_prompt():
    """prompt 段只有 1 token 也能还原。"""
    assert _recover_prompt_from_input_ids([9, 4, 5], [4, 5]) == [9]
