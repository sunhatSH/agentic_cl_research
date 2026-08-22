"""ClawEval 官方 category → 项目 9 桶分类适配器。

把 ClawEval 的 38 个扁平 category 映射到项目的 9 个能力桶
（见 ``runs/_analysis/capability_buckets/buckets.json`` 与
``configs/base.yaml`` 的 ``bucket_names``）。映射口径与训练打标【同一套】，
保证训练桶与评测桶一致。

特殊处理：
    - 多模态 category（multimodal / video_* / doc_* / webpage_generation /
      web_dev）不归桶，映射为 ``None``——项目只评测纯文本任务。
    - ``user_agent``（多轮任务）不在静态映射表里：其首轮 prompt 内容跨桶，
      需用 LLM 按内容归桶。本模块不内置 LLM 调用，而是通过
      ``user_agent_classifier`` 回调把归桶策略交给调用方（依赖注入，
      避免评测/数据管线硬编码模型调用）。

典型用法::

    from eval.bucket_adapter import build_manifest_from_tasks

    manifest = build_manifest_from_tasks(
        "/path/to/claw-eval/tasks",
        output_path="eval/claweval_manifest.json",
    )

带 user_agent 归桶::

    def classify(prompt: str) -> str | None:
        ...  # 调 LLM，返回桶名
        return "finance"

    manifest = build_manifest_from_tasks(
        tasks_dir, user_agent_classifier=classify
    )
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import yaml

# --------------------------------------------------------------------------- #
# 桶定义
# --------------------------------------------------------------------------- #

#: 项目 9 个能力桶（顺序与 ``configs/base.yaml: bucket_names`` 一致）。
BUCKETS: tuple[str, ...] = (
    "workflow",
    "ops",
    "qa",
    "finance",
    "office",
    "communication",
    "safety",
    "coding",
    "research",
)

#: 多模态 category 集合——不归桶，返回 ``None``。
#: 模态不同、数据太少、目标不一致，项目不评测多模态任务。
MULTIMODAL_CATEGORIES: frozenset[str] = frozenset(
    {
        "multimodal",
        "multimodal_webpage",
        "video_qa",
        "video_search",
        "video_ocr",
        "video_webpage",
        "video_chart",
        "video_edit",
        "video_image",
        "doc_extraction",
        "doc_search",
        "webpage_generation",
        "web_dev",
    }
)

#: 多轮 ``user_agent`` category——不在静态映射表里，按首轮 prompt 内容归桶。
USER_AGENT_CATEGORY: str = "user_agent"

#: user_agent 归桶回调签名：(首轮 prompt) -> 桶名 | None。
UserAgentClassifier = Callable[[str], str | None]


def _build_category_to_bucket() -> dict[str, str | None]:
    """构建 官方 category -> 桶名 | None 的完整映射表。

    9 桶各自的 official_categories 映射到对应桶名；多模态 category 映射为
    ``None``；``user_agent`` 不放入表中（由 ``map_category_to_bucket`` 单独
    路由到 classifier）。
    """
    # 9 桶 → 官方 category（与 buckets.json 的 official_categories 一致）。
    bucket_to_cats: dict[str, tuple[str, ...]] = {
        "workflow": ("workflow", "productivity", "organization"),
        "ops": ("ops", "operations", "terminal", "file_ops"),
        "qa": ("what", "knowledge", "comprehension", "memory"),
        "finance": ("finance", "procurement"),
        "office": ("office_qa", "data_analysis"),
        "communication": ("communication", "content", "rewriting"),
        "safety": ("safety", "security", "compliance"),
        "coding": ("coding",),
        "research": ("research", "synthesis"),
    }
    mapping: dict[str, str | None] = {}
    for bucket, cats in bucket_to_cats.items():
        for cat in cats:
            mapping[cat] = bucket
    # 多模态 category 显式映射为 None（区别于"未知 category"）。
    for cat in MULTIMODAL_CATEGORIES:
        mapping[cat] = None
    return mapping


#: 完整的 38 category → 9 桶映射表（多模态 category 值为 ``None``）。
#:
#: key = ClawEval 官方 category，value = 桶名 / ``None``（多模态）。
#: ``user_agent`` 不在此表中——它是多轮任务，按首轮 prompt 内容归桶，
#: 由 ``map_task_to_bucket`` 经 ``user_agent_classifier`` 回调处理。
CATEGORY_TO_BUCKET: dict[str, str | None] = _build_category_to_bucket()


# --------------------------------------------------------------------------- #
# 单条映射
# --------------------------------------------------------------------------- #


def map_category_to_bucket(category: str) -> str | None:
    """单个 ClawEval category → 桶名。

    Args:
        category: ClawEval 官方 category 标签。

    Returns:
        桶名（9 桶之一）或 ``None``（多模态 category，不归桶）。

    Raises:
        ValueError: category 不在映射表里且不是多模态/``user_agent``——
            说明出现了未知的官方 category，应更新映射表而非静默忽略。
    """
    if category in CATEGORY_TO_BUCKET:
        return CATEGORY_TO_BUCKET[category]
    # user_agent 不在静态表里——单 category 级映射无法归桶（需首轮 prompt），
    # 这里返回 None，由 map_task_to_bucket 经 classifier 路由。
    if category == USER_AGENT_CATEGORY:
        return None
    raise ValueError(
        f"Unknown ClawEval category {category!r}; not in CATEGORY_TO_BUCKET, "
        f"not multimodal, not {USER_AGENT_CATEGORY!r}. "
        f"Update the mapping table if this is a new official category."
    )


def map_task_to_bucket(
    task: dict,
    user_agent_classifier: UserAgentClassifier | None = None,
) -> str | None:
    """单个 task record → 桶名。

    Args:
        task: task record，需含 ``category`` 字段；若 category 为
            ``user_agent``，需含 ``prompt``（首轮 prompt 文本）供 classifier
            归桶。
        user_agent_classifier: 可选回调，签名为 ``(prompt: str) -> str | None``。
            仅当 ``task['category'] == 'user_agent'`` 时调用。``None`` 时
            user_agent 任务返回 ``None``（未归桶）。

    Returns:
        桶名或 ``None``（多模态 / user_agent 且无 classifier）。

    Raises:
        ValueError: ``task`` 缺 ``category`` 字段，或 category 未知。
    """
    if "category" not in task:
        raise ValueError(f"task record missing 'category' field: {task!r}")
    category = task["category"]

    # user_agent：按首轮 prompt 内容归桶（依赖注入的回调）。
    if category == USER_AGENT_CATEGORY:
        if user_agent_classifier is None:
            return None
        prompt = task.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = str(prompt) if prompt is not None else ""
        bucket = user_agent_classifier(prompt)
        if bucket is not None and bucket not in BUCKETS:
            raise ValueError(
                f"user_agent_classifier returned unknown bucket {bucket!r}; "
                f"expected one of {BUCKETS} or None."
            )
        return bucket

    return map_category_to_bucket(category)


# --------------------------------------------------------------------------- #
# Manifest 构建
# --------------------------------------------------------------------------- #

#: manifest 每条记录必须包含的字段。
MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "task_id",
    "split",
    "modality",
    "bucket",
    "category",
    "prompt",
)
#: 合法 split 值（与 ``eval/run_eval.py: VALID_SPLITS`` 一致）。
VALID_SPLITS: tuple[str, ...] = ("General", "Multimodal", "Multi-turn")
#: 合法 modality 值（与 ``eval/run_eval.py: VALID_MODALITIES`` 一致）。
VALID_MODALITIES: tuple[str, ...] = ("text", "multimodal")


def _infer_split_modality(tags: list) -> tuple[str, str]:
    """从 task.yaml 的 tags 推断 (split, modality)。

    - tags 含 ``multimodal`` → split="Multimodal", modality="multimodal"
    - tags 含 ``user_agent`` → split="Multi-turn", modality="text"
    - 否则 → split="General", modality="text"

    multimodal 优先于 user_agent（多模态多轮任务归 Multimodal split，
    与项目"只评测纯文本"一致）。
    """
    tag_set = set(tags or [])
    if "multimodal" in tag_set:
        return "Multimodal", "multimodal"
    if "user_agent" in tag_set:
        return "Multi-turn", "text"
    return "General", "text"


def _extract_prompt(prompt_field) -> str:
    """从 task.yaml 的 prompt 字段提取首轮 prompt 文本。

    prompt 通常是 ``{text: ..., language: ...}`` dict；偶尔可能是裸字符串。
    """
    if isinstance(prompt_field, dict):
        text = prompt_field.get("text", "")
        return text if isinstance(text, str) else str(text) if text is not None else ""
    if prompt_field is None:
        return ""
    return str(prompt_field)


def build_manifest_from_tasks(
    tasks_dir: str,
    output_path: str | None = None,
    user_agent_classifier: UserAgentClassifier | None = None,
) -> list[dict]:
    """扫描 ClawEval tasks 目录，生成 manifest（JSON list）。

    每个 task 一个子目录，内含 ``task.yaml``。生成的 manifest 每条记录含::

        task_id, split, modality, bucket, category, prompt

    split/modality 由 tags 推断；bucket 由 category 映射；``user_agent``
    category 的 bucket 由 ``user_agent_classifier`` 决定（无 classifier 则
    标 ``None``）。

    Args:
        tasks_dir: ClawEval tasks 根目录（每个子目录一个 task.yaml）。
        output_path: 若给定，把 manifest 写到该路径（JSON list，UTF-8，
            ensure_ascii=False, indent=2）；若 ``None`` 只返回不落盘。
        user_agent_classifier: 可选回调，签名为
            ``(prompt: str) -> str | None``，用于给 user_agent 多轮任务归桶。

    Returns:
        manifest 记录列表（按 task_id 排序）。

    Raises:
        FileNotFoundError: ``tasks_dir`` 不存在或不是目录。
        ValueError: 某个 task.yaml 的 category 未知（非多模态/非 user_agent
            且不在映射表）。
    """
    root = Path(tasks_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"tasks_dir not found or not a directory: {tasks_dir}")

    records: list[dict] = []
    for task_yaml in sorted(root.glob("*/task.yaml")):
        try:
            data = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
        except yaml.YAMLError as e:  # pragma: no cover - 损坏的 task.yaml
            raise ValueError(f"Failed to parse {task_yaml}: {e}") from e
        if not isinstance(data, dict):
            continue

        task_id = data.get("task_id") or task_yaml.parent.name
        category = data.get("category", "")
        tags = data.get("tags") or []
        prompt = _extract_prompt(data.get("prompt"))

        split, modality = _infer_split_modality(tags)

        # 多模态 category → bucket=None；user_agent → 走 classifier；
        # 其余 → 静态映射。
        if category in MULTIMODAL_CATEGORIES:
            bucket: str | None = None
        elif category == USER_AGENT_CATEGORY:
            bucket = user_agent_classifier(prompt) if user_agent_classifier is not None else None
            if bucket is not None and bucket not in BUCKETS:
                raise ValueError(
                    f"user_agent_classifier returned unknown bucket {bucket!r} "
                    f"for task {task_id!r}; expected one of {BUCKETS} or None."
                )
        else:
            bucket = map_category_to_bucket(category)  # 可能 ValueError

        records.append(
            {
                "task_id": task_id,
                "split": split,
                "modality": modality,
                "bucket": bucket,
                "category": category,
                "prompt": prompt,
            }
        )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return records


# --------------------------------------------------------------------------- #
# Manifest 校验
# --------------------------------------------------------------------------- #


def validate_manifest(manifest: list[dict]) -> None:
    """校验 manifest 格式。

    检查：
        - 顶层是 list，每条是 dict；
        - 每条含 :data:`MANIFEST_REQUIRED_FIELDS` 全部字段；
        - ``split`` ∈ :data:`VALID_SPLITS`，``modality`` ∈ :data:`VALID_MODALITIES`；
        - ``bucket`` 若非 ``None`` 必须是 9 桶之一；
        - ``task_id`` 唯一。

    Args:
        manifest: ``build_manifest_from_tasks`` 的输出或等价结构。

    Raises:
        ValueError: 任一检查不通过，错误信息指明位置。
    """
    if not isinstance(manifest, list):
        raise ValueError(f"manifest must be a list, got {type(manifest).__name__}")

    seen_ids: set[str] = set()
    for i, rec in enumerate(manifest):
        where = f"manifest[{i}]"
        if not isinstance(rec, dict):
            raise ValueError(f"{where} must be an object, got {type(rec).__name__}")
        for field in MANIFEST_REQUIRED_FIELDS:
            if field not in rec:
                raise ValueError(f"{where} missing required field {field!r}")

        tid = rec["task_id"]
        if not isinstance(tid, str) or not tid:
            raise ValueError(f"{where}: task_id must be a non-empty string")
        if tid in seen_ids:
            raise ValueError(f"{where}: duplicate task_id {tid!r}")
        seen_ids.add(tid)

        if rec["split"] not in VALID_SPLITS:
            raise ValueError(f"{where}: split {rec['split']!r} not in {VALID_SPLITS}")
        if rec["modality"] not in VALID_MODALITIES:
            raise ValueError(f"{where}: modality {rec['modality']!r} not in {VALID_MODALITIES}")

        bucket = rec["bucket"]
        if bucket is not None and bucket not in BUCKETS:
            raise ValueError(f"{where}: bucket {bucket!r} not in {BUCKETS} and not None")

        if not isinstance(rec["category"], str):
            raise ValueError(f"{where}: category must be a string")
        if not isinstance(rec["prompt"], str):
            raise ValueError(f"{where}: prompt must be a string")
