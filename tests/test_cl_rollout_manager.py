"""Tests for the custom rollout manager (verl rollout -> our scheduler).

Off-cluster: ``trajectories_to_dataproto`` needs ``verl.DataProto``, so we inject
a minimal fake DataProto that records what was assembled; the pure assembly math
(left-pad prompts / right-pad responses / masks / position_ids) is asserted on
the recorded tensors. ``extract_queries_from_prompts`` is pure and tested directly.
The VerlRolloutGenerateFn bridge is tested with a mock async llm_client.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

import pytest

from rollout.session_pool import Trajectory

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

if HAS_TORCH:
    import numpy as np
    import torch


def _install_fake_verl_dataproto():
    """Ensure ``verl.DataProto`` exists and records tensors/non_tensors verbatim.

    Robust to another test having already registered a partial ``verl`` mock in
    sys.modules (e.g. test_cl_loss's losses mock without DataProto): we patch
    DataProto onto whatever ``verl`` module is present, real or fake.
    """
    if importlib.util.find_spec("verl") is not None and "verl" not in sys.modules:
        from verl import DataProto

        return DataProto
    # real verl already imported with a usable DataProto -> use it
    existing = sys.modules.get("verl")
    if existing is not None and getattr(existing, "DataProto", None) is not None:
        real = getattr(existing.DataProto, "from_dict", None)
        if real is not None and existing.__spec__ is not None and getattr(existing, "__file__", None):
            return existing.DataProto

    class _FakeDataProto:
        def __init__(self, batch=None, non_tensor_batch=None):
            self.batch = batch or {}
            self.non_tensor_batch = non_tensor_batch or {}

        @classmethod
        def from_dict(cls, tensors=None, non_tensors=None, **kw):
            return cls(batch=tensors, non_tensor_batch=non_tensors)

    verl = sys.modules.get("verl")
    if verl is None:
        verl = types.ModuleType("verl")
        verl.__spec__ = importlib.machinery.ModuleSpec("verl", loader=None)
        sys.modules["verl"] = verl
    verl.DataProto = _FakeDataProto
    return _FakeDataProto


def _traj(slot, resp_ids, logprobs=None, mask=None, bucket="Finance", sid="s0"):
    return Trajectory(
        slot_idx=slot,
        trajectory_id=f"{sid}-q0-s{slot}",
        messages=[{"role": "user", "content": "make a report"}, {"role": "assistant", "content": "done"}],
        response_token_ids=resp_ids,
        logprobs=logprobs or [],
        bucket=bucket,
        meta={"response_mask": mask or [1] * len(resp_ids), "session_id": sid},
    )


def test_assemble_shapes_and_padding():
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    trajs = [
        _traj(0, [10, 11, 12], logprobs=[-0.1, -0.2, -0.3]),
        _traj(1, [20], logprobs=[-0.5]),
    ]
    prompts = [[1, 2], [3, 4, 5]]  # different prompt lengths -> P=3
    dp = trajectories_to_dataproto(trajs, prompts, pad_token_id=0)

    P, R = 3, 3
    assert dp.batch["prompts"].shape == (2, P)
    assert dp.batch["responses"].shape == (2, R)
    assert dp.batch["input_ids"].shape == (2, P + R)
    # prompt is LEFT-padded: row0 prompt [1,2] -> [0,1,2]
    assert dp.batch["prompts"][0].tolist() == [0, 1, 2]
    assert dp.batch["prompts"][1].tolist() == [3, 4, 5]
    # response is RIGHT-padded: row1 [20] -> [20,0,0]
    assert dp.batch["responses"][1].tolist() == [20, 0, 0]
    # attention_mask: prompt-left + response-right real tokens
    assert dp.batch["attention_mask"][0].tolist() == [0, 1, 1, 1, 1, 1]  # pad,1,2 | 10,11,12
    assert dp.batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 0, 0]  # 3,4,5 | 20,pad,pad


def test_assemble_response_mask_and_logprobs():
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    # mask marks token 1 as observation (0)
    trajs = [_traj(0, [10, 99, 12], logprobs=[-0.1, 0.0, -0.3], mask=[1, 0, 1])]
    dp = trajectories_to_dataproto(trajs, [[1, 2]], pad_token_id=0)
    assert dp.batch["response_mask"][0].tolist() == [1, 0, 1]
    assert "rollout_log_probs" in dp.batch
    assert dp.batch["rollout_log_probs"][0].tolist() == pytest.approx([-0.1, 0.0, -0.3])


def test_assemble_position_ids_from_attention():
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    dp = trajectories_to_dataproto([_traj(0, [10, 11])], [[1, 2]], pad_token_id=0)
    # full seq all-real here (P=2,R=2) -> position_ids = 0,1,2,3
    assert dp.batch["position_ids"][0].tolist() == [0, 1, 2, 3]


def test_assemble_rm_scores_at_last_response_token():
    """rewards -> rm_scores [B,R], reward placed at last valid response token
    (verl reads batch['rm_scores'] as the training reward; use_rm=False)."""
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    # row0: 3 resp tokens, reward 0.7 -> at idx 2; row1: 1 resp token, reward 0.4 -> idx 0
    trajs = [_traj(0, [10, 11, 12]), _traj(1, [20])]
    dp = trajectories_to_dataproto(trajs, [[1], [2]], rewards=[0.7, 0.4])
    assert "rm_scores" in dp.batch
    rm = dp.batch["rm_scores"]
    assert rm.shape == (2, 3)  # R = max resp len = 3
    assert rm[0].tolist() == pytest.approx([0.0, 0.0, 0.7])  # last real token = idx 2
    assert rm[1].tolist() == pytest.approx([0.4, 0.0, 0.0])  # last real token = idx 0


def test_assemble_rm_scores_none_reward_and_empty_response_stay_zero():
    """reward=None (crashed slot / judge error) or empty response -> row all-zero
    (no reward signal, NaN-safe under GRPO std+epsilon)."""
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    good = _traj(0, [10, 11])
    none_reward = _traj(1, [20, 21])
    empty = _traj(2, [])  # crashed-slot placeholder (no response tokens)
    dp = trajectories_to_dataproto(
        [good, none_reward, empty], [[1], [2], [3]], rewards=[1.0, None, 0.9]
    )
    rm = dp.batch["rm_scores"]
    assert rm[0].tolist() == pytest.approx([0.0, 1.0])  # good: reward at last token
    assert rm[1].tolist() == pytest.approx([0.0, 0.0])  # None -> all zero
    assert rm[2].tolist() == pytest.approx([0.0, 0.0])  # empty response -> all zero


def test_assemble_no_rm_scores_when_rewards_absent():
    """Backward-compat: no rewards arg -> no rm_scores key (cold/offline paths)."""
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    dp = trajectories_to_dataproto([_traj(0, [10])], [[1]])
    assert "rm_scores" not in dp.batch


def test_assemble_non_tensor_messages_bucket_no_uid_by_default():
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    trajs = [_traj(0, [10], sid="sess7", bucket="SysOps")]
    dp = trajectories_to_dataproto(trajs, [[1]])
    assert dp.non_tensor_batch["bucket"][0] == "SysOps"
    assert dp.non_tensor_batch["messages"][0][0]["content"] == "make a report"
    # uid is NOT emitted by default: verl owns GRPO grouping via its own uid;
    # an invented uid would collide with verl's on union (union_numpy_dict
    # asserts conflicting keys are deep-equal).
    assert "uid" not in dp.non_tensor_batch
    # multi_modal_inputs is present as an empty dict per row (verl's fit()
    # iterates it unconditionally; text-only truth = no multi-modal input).
    assert list(dp.non_tensor_batch["multi_modal_inputs"]) == [{}]


def test_assemble_uid_only_when_explicit():
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    trajs = [_traj(0, [10], sid="sess7", bucket="SysOps")]
    dp = trajectories_to_dataproto(trajs, [[1]], uids=["g0"])
    assert dp.non_tensor_batch["uid"][0] == "g0"


def test_assemble_observer_report_carried():
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    trajs = [_traj(0, [10]), _traj(1, [11])]
    dp = trajectories_to_dataproto(
        trajs, [[1], [2]], observer_reports=["diff: wrote out.csv", ""]
    )
    assert dp.non_tensor_batch["observer_report"][0] == "diff: wrote out.csv"
    assert dp.non_tensor_batch["observer_report"][1] == ""
    # observer_report is a NEW key, never named extra_info (that belongs to the
    # dataset and would collide on union).
    assert "extra_info" not in dp.non_tensor_batch


def test_assemble_no_logprobs_when_absent():
    _install_fake_verl_dataproto()
    from trainer.cl_rollout_manager import trajectories_to_dataproto

    dp = trajectories_to_dataproto([_traj(0, [10, 11], logprobs=[])], [[1]])
    assert "rollout_log_probs" not in dp.batch


def test_extract_queries_from_raw_prompt():
    from trainer.cl_rollout_manager import extract_queries_from_prompts

    class _DP:
        non_tensor_batch = {
            "raw_prompt": np.array(
                [[{"role": "user", "content": "task A"}], [{"role": "user", "content": "task B"}]],
                dtype=object,
            )
        }
        batch: dict = {}

    qs = extract_queries_from_prompts(_DP(), tokenizer=None)
    assert qs == ["task A", "task B"]


def test_extract_queries_fallback_to_decode():
    from trainer.cl_rollout_manager import extract_queries_from_prompts

    class _Tok:
        def decode(self, ids):
            return "decoded:" + ",".join(map(str, ids))

    class _DP:
        non_tensor_batch: dict = {}
        batch = {
            "input_ids": torch.tensor([[0, 7, 8]]),
            "attention_mask": torch.tensor([[0, 1, 1]]),
        }

    qs = extract_queries_from_prompts(_DP(), tokenizer=_Tok())
    assert qs == ["decoded:7,8"]  # left-pad 0 dropped by attention_mask


def test_verl_generate_fn_bridges_async_client():
    from inference.generate import VerlRolloutGenerateFn

    class _TokenOutput:
        token_ids = [5, 6, 7]
        log_probs = [-0.1, -0.2, -0.3]

    class _Client:
        async def generate(self, request_id, *, prompt_ids, sampling_params):
            return _TokenOutput()

    class _Tok:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return [1, 2, 3]

        def decode(self, ids):
            return "txt:" + ",".join(map(str, ids))

    fn = VerlRolloutGenerateFn(_Client(), _Tok())
    step = fn([{"role": "user", "content": "hi"}])
    assert step.response_ids == [5, 6, 7]
    assert step.logprobs == [-0.1, -0.2, -0.3]
    assert step.text == "txt:5,6,7"
