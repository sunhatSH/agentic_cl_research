#!/usr/bin/env python3
"""Build sandbox filesystem seeds from taskspecs (datasources/taskspecs → docker/sandbox/fs-seeds).

每个 task 的 ``files/``（初始文件系统）→ ``fs-seeds/<task_id>/``，作为沙箱实例
启动时铺开的 seed（1 task ↔ 1 seed，1:1）。实例启动按 ``AGENTIC_CL_PERSONA=<task_id>``
由 ``bin/seed_workspace.sh`` 铺开（见 ``doc/sandbox/沙箱_Dockerfile制作方案.md``）。

**元数据过滤（防 reward 欺骗）**：拷贝时**排除**以下非业务文件——它们是 OS/编辑器
元数据或 OpenClaw 运行态残留（含历史会话轨迹/答案），绝不能进实例 filesystem
（否则模型可偷看历史答案 = reward 欺骗）。OpenClaw 运行时自己会创建所需目录。

排除清单：
  - OS/编辑器元数据：.DS_Store / Thumbs.db / desktop.ini / ~$*（Office 临时锁）
  - OpenClaw 运行态：.agents/ / .openclaw/（含 sessions/agents/state，历史答案）
  - git 占位：.gitkeep / .git
  - 其它隐藏文件：. 开头（保守排除，业务文件均在可见目录）

保留：reports/ / Desktop/ 等可见目录下的业务文件（md/csv/log/py/txt/sh 等），
**含任务设计的噪声层**（*_EMPTY.log / *_GARBLED.log / *_TIMEOUT.log 等——这些是
任务设计的一部分，agent 要学会在噪声中定位真实任务文件，不是元数据）。

输出：
  - 替换 docker/sandbox/fs-seeds/（删旧 3 手造 persona + manifest，写 14 task seed）
  - 重写 manifest.json（每个 task: id/dir/buckets/summary）
  - buckets 由 LLM 对 seed_query 分桶（复用 data_pipeline.classify）

用法：
  python scripts/data/build_fs_seeds.py [--taskspecs datasources/taskspecs] [--out docker/sandbox/fs-seeds] [--no-classify]
  --no-classify: 跳过 LLM 分桶（buckets 留空，离线/无网时用）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 元数据/运行态排除清单（绝不进实例 filesystem）。
_EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitkeep", ".git"}
_EXCLUDE_DIR_NAMES = {".agents", ".openclaw", ".git"}


def _is_excluded(path: Path) -> bool:
    """该文件/目录是否应排除（OS 元数据 / OpenClaw 运行态 / git 占位 / 隐藏）。"""
    name = path.name
    if name in _EXCLUDE_NAMES:
        return True
    # ~$ 开头的 Office 临时锁文件（~$tmp.lock 等）
    if name.startswith("~$"):
        return True
    # 隐藏文件/目录（. 开头）——业务文件均在可见目录，保守排除
    if name.startswith(".") and name not in (".", ".."):
        return True
    return False


def copy_filtered(src: Path, dst: Path) -> int:
    """把 src(files/) 拷到 dst(seed)，排除元数据/运行态。返回保留文件数。"""
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in src.rglob("*"):
        if item.is_dir():
            # 目录本身：若在排除清单内，整树跳过（rglob 仍会遍历其下，靠文件级 _is_excluded 兜底）
            if item.name in _EXCLUDE_DIR_NAMES or _is_excluded(item):
                continue
            continue
        # 文件：检查自身 + 所有父目录是否被排除
        if _is_excluded(item):
            continue
        if any(part in _EXCLUDE_DIR_NAMES or (part.startswith(".") and part not in (".", "..")) for part in item.relative_to(src).parts[:-1]):
            continue
        rel = item.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, out)
        count += 1
    return count


def load_taskspec(taskspec_yaml: Path) -> dict | None:
    """读 taskspec.yaml（用 yaml）。"""
    try:
        import yaml

        with open(taskspec_yaml, encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
        return spec if isinstance(spec, dict) else None
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] skip {taskspec_yaml}: {exc}", file=sys.stderr)
        return None


def classify_buckets(seed_queries: list[str]) -> list[str]:
    """LLM 对每个 seed_query 分桶（复用 data_pipeline.classify）。无网/失败留空。"""
    try:
        from data_pipeline.classify import classify_query, make_default_client
    except ImportError:
        return [""] * len(seed_queries)
    client = make_default_client()
    out = []
    for q in seed_queries:
        try:
            v = classify_query(q, client, max_tokens=256)
            out.append(v.get("bucket") or "")
        except Exception:  # noqa: BLE001
            out.append("")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--taskspecs", default="datasources/taskspecs")
    ap.add_argument("--out", default="docker/sandbox/fs-seeds")
    ap.add_argument("--no-classify", action="store_true", help="跳过 LLM 分桶（buckets 留空）")
    args = ap.parse_args()

    root = Path(args.taskspecs)
    out_dir = Path(args.out)
    if not root.is_dir():
        print(f"[build_fs_seeds] ERROR: taskspecs not found: {root}", file=sys.stderr)
        return 1

    # 1. 清空旧 fs-seeds（替换，不留手造 persona）
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = sorted(d for d in root.iterdir() if d.is_dir() and (d / "taskspec.yaml").exists() and (d / "files").is_dir())
    if not tasks:
        print(f"[build_fs_seeds] ERROR: no task dirs with taskspec.yaml+files under {root}", file=sys.stderr)
        return 1

    # 2. 收集 seed_query（分桶用）
    specs: list[tuple[str, dict]] = []
    for t in tasks:
        spec = load_taskspec(t / "taskspec.yaml")
        if spec is None:
            continue
        specs.append((t.name, spec))

    # 3. 分桶（可选）
    seed_queries = [s.get("seed_query", "") for _, s in specs]
    if args.no_classify:
        buckets = [""] * len(specs)
    else:
        print(f"[build_fs_seeds] classifying {len(specs)} seed_queries (LLM)...", file=sys.stderr)
        buckets = classify_buckets(seed_queries)

    # 4. 拷 seed（过滤元数据）+ 建 manifest
    #    manifest 同时输出 ``personas``（兼容旧 seed_workspace.sh，按 id 列表选）
    #    和 ``tasks``（带 bucket/task_family/difficulty/summary 元信息）。
    manifest = {
        "_comment": "taskspec seeds: 1 task = 1 seed dir, laid out at instance start by bin/seed_workspace.sh (AGENTIC_CL_PERSONA=<task_id>). Metadata/runtime excluded to prevent reward hacking.",
        "version": 2,
        "personas": [],  # 兼容 seed_workspace.sh 的 persona id 列表
        "tasks": [],
    }
    total_files = 0
    for (task_id, spec), bucket in zip(specs, buckets, strict=True):
        src = root / task_id / "files"
        dst = out_dir / task_id
        n = copy_filtered(src, dst)
        total_files += n
        summary = str(spec.get("seed_query", ""))[:120]
        bucket_list = [bucket] if bucket else []
        manifest["personas"].append(
            {"id": task_id, "dir": task_id, "buckets": bucket_list, "summary": summary}
        )
        manifest["tasks"].append(
            {
                "id": task_id,
                "dir": task_id,
                "bucket": bucket or None,
                "task_family": spec.get("task_family"),
                "difficulty": spec.get("difficulty"),
                "summary": summary,
            }
        )
        print(f"  {task_id}: {n} files (bucket={bucket or '—'})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[build_fs_seeds] {len(specs)} tasks, {total_files} files total -> {out_dir}")
    print(f"[build_fs_seeds] manifest -> {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
