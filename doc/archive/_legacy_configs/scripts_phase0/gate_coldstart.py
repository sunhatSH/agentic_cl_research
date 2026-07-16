#!/usr/bin/env python3
"""Phase 0 轨1 — 冷启动阶段自身指标 gate (无 GPU)。

见 doc/Plan_冷启动数据来源消融.md §4.1。对一个已生成的 ratio-mixed cold-start
buffer 计算冷启动自身指标，并按硬 gate 判定该臂是否进入下游短RL (Stage 3)。

判定项 (可算的即判 gate；依赖 judge 的项冷数据未打分时标 N/A)：
  - 桶配额达标率        : 各桶 size >= q_min 的桶数 / 9   (gate: == 9/9)
  - tool-call 合法率     : assistant 消息里结构化 tool_calls 可解析占比 (gate: >= 0.90)
  - 轨迹多样性          : distinct_4 (>= 0.6) 且 self_bleu_4 (<= 0.5)
  - judge 有效分命中率   : 需采集期 judge 打分；冷 buffer 无 reward 时 N/A + 提示

退出码：0 = 通过全部可判 gate；1 = 有 gate 不达标 (该臂淘汰)；2 = 用法/加载错误。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from eval.metrics import trajectory_diversity
from replay_buffer.bucket import BucketReplayBuffer
from trainer.domain_tagging import DEFAULT_BUCKETS

# gate 阈值 (doc/Plan_冷启动数据来源消融.md §4.1)
GATE_QUOTA_BUCKETS = len(DEFAULT_BUCKETS)  # 必须 9/9
GATE_TOOLCALL_RATE = 0.90
GATE_DISTINCT4 = 0.60
GATE_SELFBLEU4 = 0.50


def _has_valid_tool_call(msg: dict) -> bool | None:
    """Is this assistant message's tool call structurally parseable?

    Returns True/False for assistant messages that attempt a tool call,
    None for messages that make no tool-call attempt (excluded from the rate).
    """
    if msg.get("role") != "assistant":
        return None
    # OpenAI tool-use: structured tool_calls with a function name + JSON args.
    tcs = msg.get("tool_calls")
    if tcs:
        for tc in tcs:
            fn = (tc or {}).get("function") or {}
            if not fn.get("name"):
                return False
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    return False
        return True
    return None  # no tool-call attempt in this message


def _tokenize(messages: list) -> list:
    """Crude whitespace token stream over assistant content (diversity input)."""
    toks: list[str] = []
    for m in messages:
        if m.get("role") == "assistant":
            toks.extend(str(m.get("content") or "").split())
    return toks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buffer", required=True, help="ratio-mixed cold-start buffer sqlite")
    ap.add_argument("--manifest", default=None, help="sidecar manifest.json (per-bucket source counts)")
    ap.add_argument("--q-min", type=int, default=2000, help="hard floor per bucket")
    args = ap.parse_args()

    bpath = Path(args.buffer)
    if not bpath.exists():
        print(f"[gate] buffer not found: {bpath}", flush=True)
        return 2

    buf = BucketReplayBuffer(total_capacity=25000)
    buf.load(bpath)
    stats = buf.stats()

    # --- 1) 桶配额达标率 ---
    ok_buckets = sum(1 for b in DEFAULT_BUCKETS if stats["per_bucket"].get(b, {}).get("size", 0) >= args.q_min)
    quota_ok = ok_buckets == GATE_QUOTA_BUCKETS

    # --- 遍历轨迹：tool-call 合法率 + 多样性 + judge 命中 ---
    store = buf.store
    tc_ok = tc_total = 0
    reward_scored = reward_total = 0
    per_query_tokens: list[list[list]] = []
    # group trajectories by bucket for a coarse "same-query" diversity proxy
    for bucket in DEFAULT_BUCKETS:
        bucket_streams: list[list] = []
        for tid in store.list_by_bucket(bucket):
            got = store.get(tid)
            if got is None:
                continue
            traj, meta = got
            msgs = traj.get("messages") if isinstance(traj, dict) else None
            msgs = msgs or []
            for m in msgs:
                v = _has_valid_tool_call(m)
                if v is not None:
                    tc_total += 1
                    tc_ok += int(v)
            bucket_streams.append(_tokenize(msgs))
            reward_total += 1
            if meta.get("reward") is not None:
                reward_scored += 1
        if len(bucket_streams) >= 2:
            per_query_tokens.append(bucket_streams)

    toolcall_rate = (tc_ok / tc_total) if tc_total else None
    div = trajectory_diversity(per_query_tokens) if per_query_tokens else {}
    judge_hit = (reward_scored / reward_total) if reward_total else 0.0

    # --- 判定 ---
    fails: list[str] = []
    if not quota_ok:
        fails.append(f"桶配额 {ok_buckets}/{GATE_QUOTA_BUCKETS} (需 9/9, q_min={args.q_min})")
    if toolcall_rate is not None and toolcall_rate < GATE_TOOLCALL_RATE:
        fails.append(f"tool-call 合法率 {toolcall_rate:.3f} < {GATE_TOOLCALL_RATE}")
    d4 = div.get("distinct_4")
    sb4 = div.get("self_bleu_4")
    if d4 is not None and d4 < GATE_DISTINCT4:
        fails.append(f"distinct_4 {d4:.3f} < {GATE_DISTINCT4}")
    if sb4 is not None and sb4 > GATE_SELFBLEU4:
        fails.append(f"self_bleu_4 {sb4:.3f} > {GATE_SELFBLEU4}")

    # --- 报告 ---
    print(f"[gate] buffer={bpath.name} total_size={stats['total_size']}")
    if args.manifest and Path(args.manifest).exists():
        mani = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        print(f"[gate] ratio_27b={mani.get('ratio_27b')} per_bucket_source={mani.get('per_bucket_source')}")
    print(f"[gate] 桶配额达标: {ok_buckets}/{GATE_QUOTA_BUCKETS}  ({'PASS' if quota_ok else 'FAIL'})")
    print(
        f"[gate] tool-call 合法率: "
        f"{'N/A (无 tool_calls 消息)' if toolcall_rate is None else f'{toolcall_rate:.3f}'}"
    )
    print(f"[gate] 多样性: distinct_4={d4} self_bleu_4={sb4}")
    if judge_hit == 0.0 and reward_total:
        print(
            "[gate] judge 有效分命中率: N/A — 冷 buffer 未打分 (reward=None)。"
            "此项须在采集期用 judge 单独测 (见 §4.1)。"
        )
    else:
        print(f"[gate] judge 有效分命中率: {judge_hit:.3f} (scored={reward_scored}/{reward_total})")

    if fails:
        print(f"[gate] ❌ FAIL — 该臂淘汰，不进短RL：{'; '.join(fails)}")
        return 1
    print("[gate] ✅ PASS — 该臂进入下游短RL (Stage 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
