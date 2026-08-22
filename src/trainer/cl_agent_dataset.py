"""CLAgentDataset — 标准 parquet RLHFDataset + fs-seed 输入文件注入.

阶段 E（见 plan swift-juggling-toast）。1795 个训练任务的 query 引用 ``./inputs`` 下的
输入文件（csv/xlsx/jsonl/sqlite），文件真身在 ``datasources/taskspecs_w3/<record_id>/files/``。

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

    {"files": [{"source": "<abs>/datasources/taskspecs_w3/<rid>/files", "sandbox": "inputs", "type": "dir"}]}

``sandbox="inputs"`` → 沙箱工作目录下的 ``./inputs``（与 query 里 "use the files in ./inputs" 对齐）。
"""

from __future__ import annotations

import os
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

# taskspecs 根目录：每个 <record_id>/ 下有 files/(输入) + answer_key.json + taskspec.yaml。
# 可用 env 覆盖（集群路径不同）；默认指向仓库内 datasources/taskspecs_w3。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/trainer → repo
TASKSPECS_ROOT = os.environ.get(
    "CL_TASKSPECS_ROOT",
    os.path.join(_REPO_ROOT, "datasources", "taskspecs_w3"),
)
# generated_tasks_hermes 根：<D<N>>/<task_id>/inputs/ 存真实输入文件（Kaggle CSV/xlsx 等），
# <D<N>>/<task_id>/answer_key.json 是 judge ground truth。训练行 record_id 是 unk_*，与
# taskspecs_w3 目录名(D10_b10 式)不通 → 靠 extra_info.gen_task_id(如 D10_k982304_zh)定位。
# D<N> = task_id 下划线前缀。可 env 覆盖。
GEN_TASKS_ROOT = os.environ.get(
    "CL_GEN_TASKS_ROOT",
    os.path.join(_REPO_ROOT, "datasources", "generated_tasks_hermes"),
)
# 沙箱内落地目录（query 文案统一用 ./inputs）。
SANDBOX_INPUTS_DIR = os.environ.get("CL_SANDBOX_INPUTS_DIR", "inputs")

# ── F5-review：review 任务的 workspace 快照种子（2026-08-22）──────────────────────
# "读现有代码库"类 review 任务(约 900 个)要 review /home/user/workspace/ 下的 app.py、
# tests 等，文件是采集时 agent 的完整 workspace 快照，已由 scripts/data/copy_review_ws.py
# 拷进 datasources/review_ws/<branch>/<D>/<taskdir>/ws，并写 index.json(record_id→ws相对标识)。
# ws 是一个完整工作目录整体，整树注入沙箱 /home/user/workspace/(不拆 inputs/outputs)——
# 与 path_normalize 把 ws 路径归一到 /home/user/workspace/ 对齐。
REVIEW_WS_ROOT = os.environ.get(
    "CL_REVIEW_WS_ROOT",
    os.path.join(_REPO_ROOT, "datasources", "review_ws"),
)
SANDBOX_WORKSPACE_DIR = os.environ.get("CL_SANDBOX_WORKSPACE_DIR", "workspace")


def _load_review_ws_index() -> dict[str, str]:
    """读 record_id → ws 相对标识 索引（copy_review_ws.py 生成）。缺失返回空。"""
    idx = os.path.join(REVIEW_WS_ROOT, "index.json")
    if not os.path.isfile(idx):
        return {}
    try:
        import json as _json
        with open(idx, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:  # noqa: BLE001 -- 索引损坏不该崩训练，退化为无注入
        return {}


_REVIEW_WS_INDEX: dict[str, str] | None = None


def _review_ws_dir(record_id: str | None) -> str | None:
    """按 record_id 定位已拷进种子的 ws 目录（datasources/review_ws/<rel>/ws）。"""
    global _REVIEW_WS_INDEX
    if _REVIEW_WS_INDEX is None:
        _REVIEW_WS_INDEX = _load_review_ws_index()
    if not record_id:
        return None
    rel = _REVIEW_WS_INDEX.get(record_id)
    if not rel:
        return None
    ws = os.path.join(REVIEW_WS_ROOT, rel, "ws")
    return ws if os.path.isdir(ws) else None


def _gen_task_inputs_dir(gen_task_id: str | None) -> str | None:
    """按 gen_task_id 定位 generated_tasks_hermes/<D<N>>/<tid>/inputs 目录（存在且非空才返回）。"""
    if not gen_task_id:
        return None
    d_prefix = gen_task_id.split("_", 1)[0]  # D10_k982304_zh → D10
    inputs_dir = os.path.join(GEN_TASKS_ROOT, d_prefix, gen_task_id, "inputs")
    if os.path.isdir(inputs_dir) and os.listdir(inputs_dir):
        return inputs_dir
    return None

# ── DEBUG：强制产图（验证含图轨迹过滤 E13）。CL_DEBUG_FORCE_SCREENSHOT=1 时，只对
# 【第一个 sample】(item==0) 注入「用 PIL 生成一张图」指令，逼它产一张 PNG；其余行 prompt
# 正常。无头沙箱没有截图工具（grim/scrot 都不可用），截图指令 agent 根本执行不了，故改用
# PIL 生成图——这正是 E13 当初"agent 沙箱工具产 PNG"的真实场景。仅 debug 用，默认关。
_FORCE_SCREENSHOT = os.environ.get("CL_DEBUG_FORCE_SCREENSHOT", "0") == "1"
_SCREENSHOT_PROMPT = os.environ.get(
    "CL_DEBUG_SCREENSHOT_PROMPT",
    "请用 Python 的 PIL 库生成一张简单的图片（100x100 纯色方块），保存为 /tmp/test_image.png。"
    "然后用 file 工具的 read 功能直接读取 /tmp/test_image.png 这个文件的内容并返回。"
    "注意：用 file 的 read 读文件，不要用 vision_analyze 分析。只生成一张，不要重复。",
)

# ── DEBUG：直接塞图（验证含图轨迹过滤 E13，替代截图注入）。CL_DEBUG_FORCE_IMAGE=1 时，
# 对 item==0 往 raw_prompt 塞一张图（image_url），gateway MessageEncoder 提取 → trajectory
# 含图 → 验证 _filter_image_rows 能剔掉它。不依赖 agent 截图（截图会让 session 超时），
# agent 保留原任务正常 finalize，含图 trajectory 能进训练 batch。仅 debug 用，默认关。
_FORCE_IMAGE = os.environ.get("CL_DEBUG_FORCE_IMAGE", "0") == "1"
_IMAGE_URL = os.environ.get(
    "CL_DEBUG_IMAGE_URL",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
)


def _inject_image_into_prompt(row_dict: dict[str, Any]) -> None:
    """往 raw_prompt 第一个 user message 的 content 塞一张图（image_url）。

    保留原任务文本，只把 content 从 str 改成 [image, text] 多模态 block。这样 agent
    正常完成任务、正常 finalize，含图 trajectory 能进训练 batch，供 _filter_image_rows
    端到端验证。找不到 user message / content 非 str 非 list → 静默返回。
    """
    raw_prompt = row_dict.get("raw_prompt")
    if not isinstance(raw_prompt, list):
        return
    for msg in raw_prompt:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
                {"type": "text", "text": content},
            ]
        elif isinstance(content, list):
            content.insert(0, {"type": "image_url", "image_url": {"url": _IMAGE_URL}})
        return


def _record_id_of(row_dict: dict[str, Any]) -> str | None:
    """从行里取 record_id：优先 extra_info.record_id，回退顶层 record_id/id。"""
    ei = row_dict.get("extra_info") or {}
    rid = ei.get("record_id") or row_dict.get("record_id") or row_dict.get("id")
    return str(rid) if rid else None


def build_agent_assets(
    record_id: str | None,
    root: str = TASKSPECS_ROOT,
    gen_task_id: str | None = None,
) -> dict[str, list[dict[str, str]]]:
    """定位输入文件目录，产出 agent_assets(files/type=dir) 注入沙箱。

    优先级：
      1. gen_task_id → generated_tasks_hermes/<D<N>>/<tid>/inputs → 沙箱 ./inputs
         （训练集主路径，record_id 是 unk_* 与 taskspecs_w3 不通，靠 gen_task_id 定位）；
      2. 回退 record_id → taskspecs_w3/<rid>/files → 沙箱 ./inputs（冷启动/旧数据集用）；
      3. review-ws（F5-review）：record_id → datasources/review_ws/.../ws → 沙箱 workspace
         （"读现有代码库" review 任务的完整 workspace 快照，整树注入 /home/user/workspace）。
    (1)/(2) 与 (3) 可并存但训练里互斥（review 任务无 gen_task_id）。都不存在 → 空 dict。
    整树注入用 type=dir，对应 write_agent_assets→write_local_dir 把整个目录拷进沙箱。
    """
    files: list[dict[str, str]] = []

    # (1)/(2) 输入文件 → ./inputs
    files_dir = _gen_task_inputs_dir(gen_task_id)
    if files_dir is None and record_id:
        cand = os.path.join(root, record_id, "files")
        if os.path.isdir(cand):
            files_dir = cand
    if files_dir:
        files.append({
            "source": os.path.abspath(files_dir),
            "sandbox": SANDBOX_INPUTS_DIR,
            "type": "dir",
        })

    # (3) review-ws 完整 workspace 快照 → /home/user/workspace
    ws_dir = _review_ws_dir(record_id)
    if ws_dir:
        files.append({
            "source": os.path.abspath(ws_dir),
            "sandbox": SANDBOX_WORKSPACE_DIR,
            "type": "dir",
        })

    return {"files": files} if files else {}


class CLAgentDataset(RLHFDataset):
    """标准 parquet RLHFDataset + 按 record_id 注入 taskspecs_w3/<rid>/files 到沙箱 ./inputs。"""

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        # DEBUG：只对第一个 sample(item==0) 注入截图/塞图，其余行 prompt 保持原样。
        if item == 0 and "raw_prompt" in row_dict:
            if _FORCE_IMAGE:
                _inject_image_into_prompt(row_dict)
            elif _FORCE_SCREENSHOT:
                row_dict["raw_prompt"] = [{"role": "user", "content": _SCREENSHOT_PROMPT}]
        # super() 已填 raw_prompt/extra_info/index/tools_kwargs 等。这里补 agent_assets。
        record_id = _record_id_of(row_dict)
        gen_task_id = (row_dict.get("extra_info") or {}).get("gen_task_id") or None
        assets = build_agent_assets(record_id, gen_task_id=gen_task_id)
        # 恒写该 key（无输入文件的 record 也写空 dict）——否则一个 gen-batch 里"有文件"
        # 和"无文件"的行 agent_assets 字段有无不一致，verl get_tensordict
        # (tensordict_utils.py) 会因非张量字段 batch 尺寸不一致抛
        # "AssertionError: Batch size of tensor agent_assets ... Expected N, got M"
        # （§37：4gpu step54 崩因）。空 dict 下游 unique_asset_specs/e2b runner
        # 均按 falsy 跳过注入（`if not agent_assets` / `if agent_assets`），语义无副作用。
        row_dict["agent_assets"] = assets if assets else {}
        return row_dict
