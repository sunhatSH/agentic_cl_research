"""CLAgentDataset — 标准 parquet RLHFDataset + fs-seed 输入文件注入.

阶段 E（见 plan swift-juggling-toast）。1795 个训练任务的 query 引用 ``./inputs`` 下的
输入文件（csv/xlsx/jsonl/sqlite），文件真身在 ``data/taskspecs_w3/<record_id>/files/``。

verl 原生 agent_loop 靠 dataset ``__getitem__`` 输出的 ``agent_assets`` 字段把文件注入
沙箱：session_worker（recipe_custom/agent/session_worker/worker.py:497）把 ``agent_assets``
放进 runner_kwargs → ``E2BAgentRunner.__call__`` 的 ``kwargs["agent_assets"]`` →
``write_agent_assets``（sandbox_setup.py:252，支持 ``type=dir`` 整树注入）→ 沙箱。

我们【不】走 recipe_custom 的 RLHFJSONDataset —— 它读 jsonl 走 TCSLoader→aoss_client
（腾讯 OSS，需 ~/aoss.conf），本项目保持本地 parquet、零外部依赖。故这里子类化 verl 标准
RLHFDataset，仅在 ``__getitem__`` 末尾按 record_id 补 ``agent_assets``。

用法（config）::

    data:
      custom_cls:
        path: trainer/cl_agent_dataset.py
        name: CLAgentDataset

agent_assets 格式（sandbox_setup.unique_asset_specs 契约）::

    {"files": [{"source": "<abs>/data/taskspecs_w3/<rid>/files", "sandbox": "inputs", "type": "dir"}]}

``sandbox="inputs"`` → 沙箱工作目录下的 ``./inputs``（与 query 里 "use the files in ./inputs" 对齐）。
"""

from __future__ import annotations

import os
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

# taskspecs 根目录：每个 <record_id>/ 下有 files/(输入) + answer_key.json + taskspec.yaml。
# 可用 env 覆盖（集群路径不同）；默认指向仓库内 data/taskspecs_w3。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKSPECS_ROOT = os.environ.get(
    "CL_TASKSPECS_ROOT",
    os.path.join(_REPO_ROOT, "data", "taskspecs_w3"),
)
# 沙箱内落地目录（query 文案统一用 ./inputs）。
SANDBOX_INPUTS_DIR = os.environ.get("CL_SANDBOX_INPUTS_DIR", "inputs")


def _record_id_of(row_dict: dict[str, Any]) -> str | None:
    """从行里取 record_id：优先 extra_info.record_id，回退顶层 record_id/id。"""
    ei = row_dict.get("extra_info") or {}
    rid = ei.get("record_id") or row_dict.get("record_id") or row_dict.get("id")
    return str(rid) if rid else None


def build_agent_assets(record_id: str | None, root: str = TASKSPECS_ROOT) -> dict[str, list[dict[str, str]]]:
    """按 record_id 定位 taskspecs_w3/<rid>/files 目录，产出 agent_assets(files/type=dir)。

    目录不存在（无输入文件的任务）→ 返回空 dict（runner 不注入）。整树注入用 type=dir，
    对应 write_agent_assets→write_local_dir 把整个 files/ 拷进沙箱 ./inputs。
    """
    if not record_id:
        return {}
    files_dir = os.path.join(root, record_id, "files")
    if not os.path.isdir(files_dir):
        return {}
    return {
        "files": [
            {
                "source": os.path.abspath(files_dir),
                "sandbox": SANDBOX_INPUTS_DIR,
                "type": "dir",
            }
        ]
    }


class CLAgentDataset(RLHFDataset):
    """标准 parquet RLHFDataset + 按 record_id 注入 taskspecs_w3/<rid>/files 到沙箱 ./inputs。"""

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        # super() 已填 raw_prompt/extra_info/index/tools_kwargs 等。这里补 agent_assets。
        record_id = _record_id_of(row_dict)
        assets = build_agent_assets(record_id)
        # 恒写该 key（无输入文件的 record 也写空 dict）——否则一个 gen-batch 里"有文件"
        # 和"无文件"的行 agent_assets 字段有无不一致，verl get_tensordict
        # (tensordict_utils.py) 会因非张量字段 batch 尺寸不一致抛
        # "AssertionError: Batch size of tensor agent_assets ... Expected N, got M"
        # （§37：4gpu step54 崩因）。空 dict 下游 unique_asset_specs/e2b runner
        # 均按 falsy 跳过注入（`if not agent_assets` / `if agent_assets`），语义无副作用。
        row_dict["agent_assets"] = assets if assets else {}
        return row_dict
