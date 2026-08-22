"""E13 端到端离线验证：强制截图 prompt 注入 + 含图轨迹被过滤。

不依赖 verl/GPU/沙箱,纯验证两段逻辑串起来的正确性：
1. CLAgentDataset 的 _inject_screenshot_request 把截图指令注入 prompt(CL_DEBUG_FORCE_SCREENSHOT)。
2. image_trajectory_drop_patch 的 _is_image_trajectory 把「截图产生的含图轨迹」判为丢弃。

集群真机验证见文末 CLUSTER-TODO：设 CL_DEBUG_FORCE_SCREENSHOT=1 跑 step1,应观察到
num_group_size_filtered_* 上涨 + ppo_loss 不崩(含图轨迹进不了训练)。
"""

from __future__ import annotations

import sys
import types

# cl_agent_dataset 顶层 import verl.utils.dataset.rl_dataset.RLHFDataset(本机无 verl)。
# 测试只验证 _inject_screenshot_request 纯函数,故临时 mock 最小 verl 桩让 import 通过,
# import 完【立即移除】桩,避免污染 sys.modules 让 test_verl_smoke(靠 import verl 失败来 skip)崩。
_STUBBED = []
if "verl" not in sys.modules:
    _verl = types.ModuleType("verl")
    _utils = types.ModuleType("verl.utils")
    _ds = types.ModuleType("verl.utils.dataset")
    _rl = types.ModuleType("verl.utils.dataset.rl_dataset")
    _rl.RLHFDataset = type("RLHFDataset", (), {})
    _ds.rl_dataset = _rl
    _utils.dataset = _ds
    _verl.utils = _utils
    for _name, _mod in [
        ("verl", _verl),
        ("verl.utils", _utils),
        ("verl.utils.dataset", _ds),
        ("verl.utils.dataset.rl_dataset", _rl),
    ]:
        sys.modules[_name] = _mod
        _STUBBED.append(_name)

from trainer.cl_agent_dataset import _inject_screenshot_request  # noqa: E402
from trainer.image_trajectory_drop_patch import _is_image_trajectory  # noqa: E402

# 移除 verl 桩：函数已绑定,后续测试不再需要;不清会让 test_verl_smoke 误以为 verl 已装。
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


class _Traj:
    def __init__(self, mmd=None, rid=None, rmask=None):
        self.multi_modal_data = mmd
        self.response_ids = rid
        self.response_mask = rmask


def test_inject_appends_to_last_user_str():
    msgs = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "读取 ./inputs 的数据"},
    ]
    out = _inject_screenshot_request(msgs)
    assert out[-1]["role"] == "user"
    assert "读取 ./inputs 的数据" in out[-1]["content"]
    assert "screenshot" in out[-1]["content"] or "截" in out[-1]["content"]


def test_inject_handles_structured_content():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    out = _inject_screenshot_request(msgs)
    assert isinstance(out[0]["content"], list)
    assert any(p.get("type") == "text" and "截" in p.get("text", "") for p in out[0]["content"])


def test_inject_no_user_appends_new():
    msgs = [{"role": "system", "content": "sys"}]
    out = _inject_screenshot_request(msgs)
    assert out[-1]["role"] == "user"


def test_inject_non_list_passthrough():
    assert _inject_screenshot_request(None) is None
    assert _inject_screenshot_request("") == ""


def test_screenshot_trajectory_gets_dropped():
    # 模拟：截图 prompt → agent 产图 → 轨迹带 multi_modal_data.images → 判丢弃。
    img_traj = _Traj(mmd={"images": ["<png bytes>"]}, rid=[1, 2, 3], rmask=[1, 1, 1])
    assert _is_image_trajectory(img_traj) is True
    # 对照：同任务纯文本完成(无图) → 保留。
    txt_traj = _Traj(mmd={}, rid=[1, 2, 3], rmask=[1, 1, 1])
    assert _is_image_trajectory(txt_traj) is False
