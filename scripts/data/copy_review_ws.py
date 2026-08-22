#!/usr/bin/env python3
"""F5-review：把采集轨迹的 workspace 快照(ws)拷进项目种子，供 review 任务注入沙箱。

背景（2026-08-22，见 doc/tuning/Reward_归因方法与调优经验.md）：约 900 个"读现有代码库"
review 任务(train_cl 64 + train_exp2 73)要 review/verify /home/user/workspace/ 下的 app.py、
tests 等文件，但那些文件是采集时 agent 沙箱里的完整 workspace 快照，从未随数据集保存 →
沙箱里读不到 → task_done=0。文件真身在采集者目录：
  /mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory/<branch>/<D<N>>/<taskdir>/ws/

本脚本把【训练集实际用到的】ws 文件夹整体拷进 datasources/review_ws/<branch>/<D>/<taskdir>/，
排除运行时/二进制噪声(见 SKIP_*)。ws 是一个完整工作目录整体，不拆 inputs/outputs——
cl_agent_dataset 的 review-ws 分支按 extract_ws_dir 索引整树注入到沙箱 /home/user/workspace/。

用法:
  python3 scripts/data/copy_review_ws.py            # 干跑(列清单+体积)
  python3 scripts/data/copy_review_ws.py --apply    # 落地拷贝
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from path_normalize import extract_ws_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "datasources" / "labeled" / "new_trajectories_labeled.jsonl"
WS_SRC_ROOT = Path("/mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory")
DEST_ROOT = ROOT / "datasources" / "review_ws"

# 排除运行时/依赖/构建噪声——review 用不到，且撑大体积。
SKIP_DIR = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "node_modules",
            ".cache", "site-packages", "lib64", ".local", ".ipynb_checkpoints"}
SKIP_EXT = {".so", ".whl", ".pyc", ".dylib", ".a", ".o", ".egg-link"}
SKIP_NAME = {"python", "python3", "python3.11", "python3.8", "python3.10"}
MAX_FILE_BYTES = 5 * 1024 * 1024  # 单文件 >5MB 跳(大数据集 csv，review 用不到全量)


def _train_record_ids() -> set[str]:
    import pyarrow.parquet as pq
    ids: set[str] = set()
    for name in ("train_cl.parquet", "train_exp2.parquet"):
        p = ROOT / "datasets" / name
        if p.exists():
            for r in pq.read_table(str(p)).to_pylist():
                rid = (r.get("extra_info") or {}).get("record_id")
                if rid:
                    ids.add(rid)
    return ids


def _needed_ws_dirs() -> dict[str, str]:
    """训练集里 review 任务用到的 ws 相对标识 → 该标识对应的一组 record_id（去重）。"""
    need = _train_record_ids()
    ws_dirs: dict[str, list[str]] = {}
    for line in open(LABELED, encoding="utf-8"):
        d = json.loads(line)
        rid = d.get("record_id")
        if rid not in need:
            continue
        ws = extract_ws_dir(d.get("seed_query", ""))
        if not ws:
            continue
        src = WS_SRC_ROOT / ws
        if src.is_dir():
            ws_dirs.setdefault(ws, []).append(rid)
    return {k: v for k, v in ws_dirs.items()}


def _copy_one(ws_rel: str, apply: bool) -> tuple[int, int]:
    src = WS_SRC_ROOT / ws_rel
    dest = DEST_ROOT / ws_rel
    nfile = 0
    nbytes = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [x for x in dirs if x not in SKIP_DIR]
        rel = os.path.relpath(root, src)
        for f in files:
            if f in SKIP_NAME or os.path.splitext(f)[1].lower() in SKIP_EXT:
                continue
            sp = os.path.join(root, f)
            try:
                sz = os.path.getsize(sp)
            except OSError:
                continue
            if sz > MAX_FILE_BYTES:
                continue
            nfile += 1
            nbytes += sz
            if apply:
                dd = dest if rel == "." else dest / rel
                dd.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(sp, dd / f)
                except (PermissionError, OSError) as e:
                    # tongronglei 目录下个别文件属他人/无读权限 → 跳过，不中断整批
                    nfile -= 1
                    nbytes -= sz
                    print(f"  ⚠️ 跳过(读不了): {sp} ({type(e).__name__})", file=sys.stderr)
    return nfile, nbytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="落地拷贝（默认干跑）")
    args = ap.parse_args()

    ws_dirs = _needed_ws_dirs()
    total_files = 0
    total_bytes = 0
    for ws_rel in ws_dirs:
        nf, nb = _copy_one(ws_rel, args.apply)
        total_files += nf
        total_bytes += nb

    # 写 record_id → ws 相对标识 的索引（dataset 注入时读，避免每次 parse 193MB labeled）。
    index = {rid: ws_rel for ws_rel, rids in ws_dirs.items() for rid in rids}
    idx_path = DEST_ROOT / "index.json"
    if args.apply:
        DEST_ROOT.mkdir(parents=True, exist_ok=True)
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=0)

    rids = sum(len(v) for v in ws_dirs.values())
    mode = "已拷贝" if args.apply else "干跑(未写)"
    print(f"[{mode}] 唯一 ws 文件夹: {len(ws_dirs)}，覆盖 record_id: {rids}")
    print(f"          文件数: {total_files}，总大小: {total_bytes/1024/1024:.1f} MB")
    print(f"          目标根: {DEST_ROOT}")
    print(f"          索引: {idx_path}（{len(index)} 条 record_id→ws）")
    if not args.apply:
        print("确认无误后加 --apply 落地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
