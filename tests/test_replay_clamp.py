"""离线单测：replay 行截断契约 `_clamp_prompt_response`（纯函数，不依赖 torch）。

replay 截断必须和 rollout 一致（rollout winner 会进 buffer 成为 replay，同一条轨迹前后
长度上限须相同）。契约（照 rollout gateway convert_buffer_to_trajectory）：
  - response 截到 max_response_length；
  - prompt 全留，仅当 prompt+response 超 max_model_len 时从 HEAD 砍 prompt；
  - response 再截到 min(max_response_length, max_model_len - len(prompt))。
max_model_len=None 退回 legacy：prompt/response 各自截 max_response_length。
"""

from __future__ import annotations

from trainer.replay_forward import _clamp_prompt_response


def test_clamp_response_capped_at_max_response_length():
    """response 超 max_response_length → 截到它；prompt 短则全留。"""
    prompt = list(range(100))
    resp = list(range(1000))  # 1000 > 200
    p, r = _clamp_prompt_response(prompt, resp, max_response_length=200, max_model_len=131072)
    assert p == prompt  # prompt 全留
    assert len(r) == 200  # response 截到上限


def test_clamp_total_within_model_len_no_trim():
    """prompt+response 未超 max_model_len → 都不动（response 仍受自身上限）。"""
    prompt = list(range(50))
    resp = list(range(60))
    p, r = _clamp_prompt_response(prompt, resp, max_response_length=65536, max_model_len=131072)
    assert p == prompt and r == resp


def test_clamp_long_prompt_trims_prompt_head():
    """prompt+response 超 max_model_len → 照 rollout gateway 语义：prompt 优先，
    prompt_capacity = max_model_len - 1（只给 response 保底 1 slot），prompt 砍头留尾，
    response 拿剩余预算。总长 <= max_model_len。"""
    prompt = list(range(1000))  # 0..999
    resp = list(range(500, 800))  # len 300
    p, r = _clamp_prompt_response(prompt, resp, max_response_length=500, max_model_len=1000)
    assert len(p) + len(r) <= 1000
    # prompt_capacity = 1000 - 1 = 999；prompt 1000 > 999 → 砍头留尾到 999
    assert len(p) == 999
    assert p[-1] == 999  # 保留的是最新（尾部）prompt token
    # response 拿剩余 = 1000 - 999 = 1
    assert len(r) == 1


def test_clamp_response_squeezed_by_huge_prompt():
    """prompt 极长（接近 max_model_len）→ response 被压缩到剩余预算。"""
    prompt = list(range(990))
    resp = list(range(100))
    p, r = _clamp_prompt_response(prompt, resp, max_response_length=500, max_model_len=1000)
    # prompt_capacity = 1000 - 1 = 999 >= 990 → prompt 全留
    assert len(p) == 990
    # response 剩余预算 = 1000 - 990 = 10
    assert len(r) == 10


def test_clamp_legacy_none_model_len():
    """max_model_len=None → legacy：prompt/response 各自截 max_response_length，无总长耦合。"""
    prompt = list(range(300))
    resp = list(range(300))
    p, r = _clamp_prompt_response(prompt, resp, max_response_length=200, max_model_len=None)
    assert len(p) == 200 and len(r) == 200  # 各自独立截 200，总长 400 不受约束


def test_clamp_r0_realistic():
    """r0 真实配置：response 上限 65536、总上限 131072。"""
    prompt = list(range(8000))
    resp = list(range(70000))  # 超 65536
    p, r = _clamp_prompt_response(prompt, resp, max_response_length=65536, max_model_len=131072)
    assert len(r) == 65536  # response 截到 65536
    assert len(p) == 8000  # prompt 全留（8000+65536=73536 < 131072）
    assert len(p) + len(r) <= 131072
