"""Tests for eval.bucket_adapter (ClawEval category → 9 桶映射 + manifest 构建)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from eval.bucket_adapter import (
    BUCKETS,
    CATEGORY_TO_BUCKET,
    MULTIMODAL_CATEGORIES,
    USER_AGENT_CATEGORY,
    build_manifest_from_tasks,
    map_category_to_bucket,
    map_task_to_bucket,
    validate_manifest,
)

# --------------------------------------------------------------------------- #
# 静态映射表
# --------------------------------------------------------------------------- #

# 9 桶 → 官方 category（与 buckets.json 的 official_categories 一致）。
BUCKET_TO_CATS = {
    "workflow": ["workflow", "productivity", "organization"],
    "ops": ["ops", "operations", "terminal", "file_ops"],
    "qa": ["what", "knowledge", "comprehension", "memory"],
    "finance": ["finance", "procurement"],
    "office": ["office_qa", "data_analysis"],
    "communication": ["communication", "content", "rewriting"],
    "safety": ["safety", "security", "compliance"],
    "coding": ["coding"],
    "research": ["research", "synthesis"],
}


def test_category_to_bucket_maps_each_bucket_correctly():
    for bucket, cats in BUCKET_TO_CATS.items():
        for cat in cats:
            assert map_category_to_bucket(cat) == bucket, f"category {cat!r} should map to bucket {bucket!r}"


def test_category_to_bucket_table_covers_all_text_categories():
    # 表里 9 桶的官方 category 全部存在且映射正确。
    for bucket, cats in BUCKET_TO_CATS.items():
        for cat in cats:
            assert CATEGORY_TO_BUCKET[cat] == bucket


def test_multimodal_categories_return_none():
    for cat in MULTIMODAL_CATEGORIES:
        assert map_category_to_bucket(cat) is None
        assert CATEGORY_TO_BUCKET[cat] is None


def test_user_agent_category_not_in_static_table():
    # user_agent 不在静态映射表里（需按首轮 prompt 归桶）。
    assert USER_AGENT_CATEGORY not in CATEGORY_TO_BUCKET


def test_unknown_category_raises():
    with pytest.raises(ValueError, match="Unknown ClawEval category"):
        map_category_to_bucket("not_a_real_category")


# --------------------------------------------------------------------------- #
# map_task_to_bucket
# --------------------------------------------------------------------------- #


def test_map_task_uses_category_field():
    task = {"task_id": "T1", "category": "workflow"}
    assert map_task_to_bucket(task) == "workflow"


def test_map_task_multimodal_returns_none():
    task = {"task_id": "M1", "category": "video_qa"}
    assert map_task_to_bucket(task) is None


def test_map_task_user_agent_without_classifier_returns_none():
    task = {"task_id": "C1", "category": "user_agent", "prompt": "帮我算算房贷"}
    assert map_task_to_bucket(task) is None


def test_map_task_user_agent_with_classifier_calls_it():
    calls: list[str] = []

    def classifier(prompt: str) -> str | None:
        calls.append(prompt)
        return "finance"

    task = {"task_id": "C1", "category": "user_agent", "prompt": "帮我算算房贷"}
    assert map_task_to_bucket(task, user_agent_classifier=classifier) == "finance"
    assert calls == ["帮我算算房贷"]


def test_map_task_user_agent_classifier_returning_none_passes_through():
    def classifier(prompt: str) -> str | None:
        return None

    task = {"task_id": "C1", "category": "user_agent", "prompt": "xxx"}
    assert map_task_to_bucket(task, user_agent_classifier=classifier) is None


def test_map_task_user_agent_classifier_returning_unknown_bucket_raises():
    def classifier(prompt: str) -> str | None:
        return "not_a_bucket"

    task = {"task_id": "C1", "category": "user_agent", "prompt": "xxx"}
    with pytest.raises(ValueError, match="unknown bucket"):
        map_task_to_bucket(task, user_agent_classifier=classifier)


def test_map_task_missing_category_raises():
    with pytest.raises(ValueError, match="missing 'category'"):
        map_task_to_bucket({"task_id": "T1"})


# --------------------------------------------------------------------------- #
# build_manifest_from_tasks
# --------------------------------------------------------------------------- #


def _write_task(
    dirpath: Path,
    task_id: str,
    category: str,
    tags: list[str],
    prompt_text: str = "hello",
) -> None:
    """写一个最小 task.yaml 到 dirpath/task_id/task.yaml。"""
    task_dir = dirpath / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_id": task_id,
        "task_name": task_id,
        "category": category,
        "difficulty": "easy",
        "tags": tags,
        "prompt": {"text": prompt_text, "language": "zh"},
    }
    (task_dir / "task.yaml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def test_build_manifest_generates_correct_records(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir, "T001", "workflow", ["general"], "do workflow")
    _write_task(tasks_dir, "T002", "video_qa", ["multimodal"], "watch video")
    _write_task(tasks_dir, "C001", "user_agent", ["general", "user_agent"], "算房贷")
    _write_task(tasks_dir, "T003", "finance", ["general"], "算税")

    manifest = build_manifest_from_tasks(str(tasks_dir))

    by_id = {r["task_id"]: r for r in manifest}
    assert set(by_id) == {"T001", "T002", "C001", "T003"}

    # General text task → workflow bucket
    assert by_id["T001"]["split"] == "General"
    assert by_id["T001"]["modality"] == "text"
    assert by_id["T001"]["bucket"] == "workflow"
    assert by_id["T001"]["prompt"] == "do workflow"

    # Multimodal task → Multimodal split, bucket None
    assert by_id["T002"]["split"] == "Multimodal"
    assert by_id["T002"]["modality"] == "multimodal"
    assert by_id["T002"]["bucket"] is None

    # user_agent without classifier → Multi-turn, bucket None
    assert by_id["C001"]["split"] == "Multi-turn"
    assert by_id["C001"]["modality"] == "text"
    assert by_id["C001"]["bucket"] is None

    # finance → finance bucket
    assert by_id["T003"]["bucket"] == "finance"


def test_build_manifest_user_agent_classifier_used(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir, "C001", "user_agent", ["general", "user_agent"], "算房贷")

    def classifier(prompt: str) -> str | None:
        assert prompt == "算房贷"
        return "finance"

    manifest = build_manifest_from_tasks(str(tasks_dir), user_agent_classifier=classifier)
    assert len(manifest) == 1
    assert manifest[0]["bucket"] == "finance"


def test_build_manifest_writes_output_file(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir, "T001", "workflow", ["general"], "do workflow")
    out = tmp_path / "manifest.json"

    manifest = build_manifest_from_tasks(str(tasks_dir), output_path=str(out))

    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == manifest
    assert len(loaded) == 1
    assert loaded[0]["task_id"] == "T001"


def test_build_manifest_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest_from_tasks(str(tmp_path / "does_not_exist"))


def test_build_manifest_multimodal_priority_over_user_agent(tmp_path):
    # 一个 task 同时带 multimodal 和 user_agent tag → 归 Multimodal split
    # （multimodal 优先，与项目"只评测纯文本"一致）。
    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir, "M001", "multimodal_webpage", ["multimodal", "user_agent"], "x")
    manifest = build_manifest_from_tasks(str(tasks_dir))
    assert manifest[0]["split"] == "Multimodal"
    assert manifest[0]["modality"] == "multimodal"
    assert manifest[0]["bucket"] is None


# --------------------------------------------------------------------------- #
# validate_manifest
# --------------------------------------------------------------------------- #


def _good_record(task_id: str = "T1", **overrides) -> dict:
    rec = {
        "task_id": task_id,
        "split": "General",
        "modality": "text",
        "bucket": "workflow",
        "category": "workflow",
        "prompt": "do something",
    }
    rec.update(overrides)
    return rec


def test_validate_manifest_accepts_good_records():
    manifest = [_good_record("T1"), _good_record("T2", bucket=None, category="video_qa")]
    validate_manifest(manifest)  # no raise


def test_validate_manifest_rejects_non_list():
    with pytest.raises(ValueError, match="must be a list"):
        validate_manifest({"task_id": "T1"})  # type: ignore[arg-type]


def test_validate_manifest_rejects_missing_field():
    rec = _good_record()
    del rec["bucket"]
    with pytest.raises(ValueError, match="missing required field 'bucket'"):
        validate_manifest([rec])


def test_validate_manifest_rejects_bad_split():
    with pytest.raises(ValueError, match="split"):
        validate_manifest([_good_record(split="Nope")])


def test_validate_manifest_rejects_bad_modality():
    with pytest.raises(ValueError, match="modality"):
        validate_manifest([_good_record(modality="video")])


def test_validate_manifest_rejects_bad_bucket():
    with pytest.raises(ValueError, match="bucket"):
        validate_manifest([_good_record(bucket="not_a_bucket")])


def test_validate_manifest_rejects_duplicate_task_id():
    manifest = [_good_record("T1"), _good_record("T1")]
    with pytest.raises(ValueError, match="duplicate task_id"):
        validate_manifest(manifest)


def test_validate_manifest_rejects_non_dict_record():
    with pytest.raises(ValueError, match="must be an object"):
        validate_manifest(["not a dict"])  # type: ignore[list-item]


# --------------------------------------------------------------------------- #
# 端到端：build → validate 一致性
# --------------------------------------------------------------------------- #


def test_build_manifest_output_passes_validation(tmp_path):
    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir, "T001", "workflow", ["general"], "do workflow")
    _write_task(tasks_dir, "T002", "video_qa", ["multimodal"], "watch video")
    _write_task(tasks_dir, "C001", "user_agent", ["general", "user_agent"], "算房贷")

    manifest = build_manifest_from_tasks(str(tasks_dir))
    validate_manifest(manifest)  # no raise


def test_all_buckets_are_the_nine():
    assert BUCKETS == (
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
