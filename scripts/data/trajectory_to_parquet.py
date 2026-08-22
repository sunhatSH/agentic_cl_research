#!/usr/bin/env python3
"""trajectory JSONL -> verl rl_dataset parquet (Step 6 of the data pipeline).

把冷启动采集产出的 trajectory JSONL 转成 verl 能训练的 parquet。

Input  : trajectory JSONL，每行一条 trajectory（collect_cold / sandbox_grpo_collect 产出）。
         支持两种 schema（自动识别）：
           A) collect_cold 风格（Trajectory dataclass asdict）:
              {"slot_idx","trajectory_id","messages","reward","response_token_ids",
               "logprobs","bucket","meta",...}
           B) sandbox_grpo_collect 风格（SlotTrajectory asdict）:
              {"query_index","slot_idx","messages","answer","reward","advantage",
               "is_winner","error",...}
Output : train.parquet + val.parquet（verl rl_dataset 列）：
           prompt        list[{role,content}]  -- system + 首个 user（rollout 起点）
           data_source   str                    -- "agentic_cl"
           reward_model  {ground_truth}         -- 空（judge 在线打分）
           extra_info    {record_id, bucket, queries, reward, trajectory_id, ...}

关键设计（与 doc/训练与推理流程.md §2 路径 A 一致）：
  - parquet 只装 prompt（对话起点），不装 trajectory 的 assistant 回复/token/logprob。
    verl 拿 prompt → lightllm 在线 rollout → 产 trajectory → 入 buffer。
  - 但冷启动采集的 trajectory 已经有 messages（含 assistant 回复），这里取它的
    【system + 首个 user】作为 prompt，其余 messages 丢弃（verl 会重新 rollout）。
  - reward / bucket / trajectory_id 进 extra_info，供 buffer 预热 + 论文追踪用。
  - 冷启动 trajectory 的 reward 可能为 None（collect_cold 不打分）；None 时 extra_info
    不写 reward 字段，buffer 用默认 priority。

为什么不用 convert_dataset.py：
  - convert_dataset.py 吃的是【会话 JSONL】（record.messages 是完整多轮对话，含 assistant），
    它的 pick_prompt 取 system + 首个 user —— 这部分逻辑我们要复用。
  - 但它的输入 schema（{record_id, record:{messages, meta, tools}}）和 trajectory 的
    schema（{trajectory_id, messages, reward, bucket, ...}）不同，直接喂会报错。
  - 本脚本复用 convert_dataset 的 pick_prompt / bucket_hint 纯函数，适配 trajectory schema。

Usage:
    python scripts/data/trajectory_to_parquet.py \\
        --input datasources/mock/rollouts/cold/rollouts_cold.jsonl \\
        --out-dir datasets \\
        --val-fraction 0.05
    # 多个输入文件：
    python scripts/data/trajectory_to_parquet.py \\
        --input rollouts1.jsonl rollouts2.jsonl \\
        --out-dir datasets

Notes:
  - 缺 messages 的 trajectory 跳过。
  - prompt 里没有 user turn 的跳过（无 rollout 起点）。
  - bucket 缺失时用 bucket_hint(messages) 关键词投票兜底，再不行标 "unknown"
    （buffer 会 skip unknown，B12；冷启动数据应已由 collect 阶段打桶）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# 复用 convert_dataset.py 的纯函数（pick_prompt / bucket_hint / split_assignment）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.data.convert_dataset import bucket_hint, pick_prompt, split_assignment  # noqa: E402

DATA_SOURCE = "agentic_cl"


def _extract_messages(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """从 trajectory 里取 messages（两种 schema 都支持）。"""
    msgs = traj.get("messages")
    if isinstance(msgs, list):
        return msgs
    # 有些 schema 把 messages 嵌在 record 里
    inner = traj.get("record")
    if isinstance(inner, dict):
        msgs = inner.get("messages")
        if isinstance(msgs, list):
            return msgs
    return []


def _extract_record_id(traj: dict[str, Any]) -> str:
    """trajectory_id / record_id / task_id 任一可用。"""
    for k in ("trajectory_id", "record_id", "task_id"):
        v = traj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_bucket(traj: dict[str, Any], messages: list[dict[str, Any]]) -> str:
    """trajectory.bucket -> bucket_hint(messages) -> 'unknown'。"""
    b = traj.get("bucket")
    if isinstance(b, str) and b.strip():
        return b.strip()
    hint = bucket_hint(messages)
    return hint or "unknown"


def trajectory_to_row(traj: dict[str, Any]) -> dict[str, Any] | None:
    """一条 trajectory -> 一行 verl rl_dataset row，或 None 跳过。"""
    messages = _extract_messages(traj)
    if not messages:
        return None
    prompt = pick_prompt(messages)
    if not any(p["role"] == "user" for p in prompt):
        return None  # 无 user turn，无 rollout 起点

    record_id = _extract_record_id(traj)
    bucket = _extract_bucket(traj, messages)
    reward = traj.get("reward")
    # follow_ups 不在 trajectory 里（trajectory 是单 query 的）；queries 只含 seed
    seed_query = next((p["content"] for p in prompt if p["role"] == "user"), "")

    extra: dict[str, Any] = {
        "record_id": record_id,
        "bucket": bucket,
        "queries": json.dumps([seed_query], ensure_ascii=False),
        "num_user_turns": 1,
        "checkers": "[]",  # 冷启动 trajectory 无 checker（judge 在线打分）
    }
    if reward is not None:
        try:
            extra["reward"] = float(reward)
        except (TypeError, ValueError):
            pass
    # 保留 trajectory_id 供论文追踪 + buffer 去重
    tid = traj.get("trajectory_id")
    if isinstance(tid, str) and tid.strip():
        extra["trajectory_id"] = tid.strip()

    return {
        "prompt": prompt,
        "data_source": DATA_SOURCE,
        "reward_model": {"ground_truth": "", "style": "rule"},
        "extra_info": extra,
    }


def convert(input_paths: list[Path], out_dir: Path, val_fraction: float, limit: int | None) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"train": [], "val": []}
    stats = {
        "total": 0,
        "skipped": 0,
        "buckets": Counter(),
        "with_reward": 0,
    }

    for inp in input_paths:
        with inp.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and stats["total"] >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                stats["total"] += 1
                try:
                    traj = json.loads(line)
                except json.JSONDecodeError:
                    stats["skipped"] += 1
                    continue
                row = trajectory_to_row(traj)
                if row is None:
                    stats["skipped"] += 1
                    continue
                info = row["extra_info"]
                stats["buckets"][info["bucket"]] += 1
                if "reward" in info:
                    stats["with_reward"] += 1
                split = split_assignment(info["record_id"] or f"traj-{i}", val_fraction)
                rows[split].append(row)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(split: str) -> Path:
        data = rows[split]
        table = pa.Table.from_pylist(
            [
                {
                    "prompt": r["prompt"],
                    "data_source": r["data_source"],
                    "reward_model": r["reward_model"],
                    "extra_info": r["extra_info"],
                }
                for r in data
            ]
        )
        path = out_dir / f"{split}.parquet"
        pq.write_table(table, path)
        return path

    train_path = _write("train")
    val_path = _write("val")
    kept = len(rows["train"]) + len(rows["val"])
    stats["kept"] = kept
    stats["train"] = len(rows["train"])
    stats["val"] = len(rows["val"])
    stats["reward_coverage"] = (stats["with_reward"] / kept) if kept else 0.0
    stats["train_path"] = str(train_path)
    stats["val_path"] = str(val_path)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="trajectory JSONL 文件（可多个）；collect_cold / sandbox_grpo_collect 产出",
    )
    ap.add_argument("--out-dir", default="datasets", help="输出目录（train.parquet + val.parquet）")
    ap.add_argument("--val-fraction", type=float, default=0.05, help="验证集比例（按 record_id hash 分流）")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条 trajectory（调试用）")
    args = ap.parse_args()

    input_paths = [Path(p) for p in args.input]
    for p in input_paths:
        if not p.is_file():
            raise FileNotFoundError(f"input not found: {p}")

    stats = convert(input_paths, Path(args.out_dir), args.val_fraction, args.limit)
    print("=== trajectory_to_parquet stats ===")
    print(f"total read   : {stats['total']}")
    print(f"kept         : {stats['kept']}  (train {stats['train']} / val {stats['val']})")
    print(f"skipped      : {stats['skipped']}")
    print(f"reward coverage : {stats['reward_coverage']:.1%}  ({stats['with_reward']} trajectories)")
    print("bucket distribution:")
    for b, c in stats["buckets"].most_common():
        print(f"  {b:14s} {c}")
    print(f"-> {stats['train_path']}")
    print(f"-> {stats['val_path']}")
    print(f"  下一步: 在 configs/cluster.yaml 填 data.train_files / val_files 指向这两个 parquet")


if __name__ == "__main__":
    main()
