"""E13 含图 trajectory 过滤：判定函数 _is_image_trajectory 单测（纯函数,不依赖 verl）。

见 trainer/image_trajectory_drop_patch.py。判定=并集：multi_modal_data 非空 OR
response_ids/response_mask 长度不等。含图轨迹判不可训练→session_worker 剔除→
min_group_success_ratio 组过滤接管。
"""

from __future__ import annotations

from trainer.image_trajectory_drop_patch import _is_image_trajectory


class _Traj:
    def __init__(self, mmd=None, rid=None, rmask=None):
        self.multi_modal_data = mmd
        self.response_ids = rid
        self.response_mask = rmask


def test_plain_text_not_image():
    # multi_modal_data 为 None/空 dict/空列表 + 长度相等 → 纯文本,不丢弃。
    assert _is_image_trajectory(_Traj(None, [1, 2, 3], [1, 1, 1])) is False
    assert _is_image_trajectory(_Traj({}, [1, 2, 3], [1, 1, 1])) is False
    assert _is_image_trajectory(_Traj({"images": [], "videos": []}, [1, 2], [1, 1])) is False


def test_multimodal_data_flags_image():
    # multi_modal_data 带非空 images/videos → 含图,丢弃。
    assert _is_image_trajectory(_Traj({"images": ["<png>"]}, [1, 2], [1, 1])) is True
    assert _is_image_trajectory(_Traj({"videos": ["<mp4>"]}, [1, 2], [1, 1])) is True


def test_length_mismatch_flags_image():
    # E13 直接症状：response_ids 与 response_mask 长度不等 → 双保险丢弃。
    assert _is_image_trajectory(_Traj(None, [1, 2, 3], [1, 1, 1, 1])) is True


def test_missing_fields_no_crash():
    # 字段缺失(异常轨迹)不崩,判不含图(交回原 trace_type 逻辑处理)。
    assert _is_image_trajectory(_Traj(None, None, None)) is False
