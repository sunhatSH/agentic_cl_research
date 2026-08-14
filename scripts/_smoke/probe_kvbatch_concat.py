#!/usr/bin/env python3
"""上集群探测 KVBatchMeta.concat 字段对齐要求（R0 回放崩溃 §64）。

背景：R0 step 2 回放激活即崩 —— `KVBatchMeta.concat([batch, replay_meta])`
报 "Field names do not match"。concat 校验（已确认）：
    if chunk.fields is not None and set(chunk.fields) != base_fields_set:
        raise ValueError("Field names do not match for concatenation.")
即每个 chunk 字段集必须 == data[0].fields（rollout batch）。且合并结果用
base_fields —— 所以 3 个 replay 标记字段必须进 rollout 侧，否则 worker 不 fetch。

本脚本打印 API 结构 + 探测 kv_batch_put 对已存在 key 是【追加字段】还是
【整行替换】（决定怎么给 rollout keys 补零值字段）。

用法（集群，无需 GPU）：
    python scripts/_smoke/probe_kvbatch_concat.py
日志同时输出到 AFS：logs/probe/probe_kvbatch_concat.log
"""
from __future__ import annotations

import datetime
import inspect
import sys
from pathlib import Path


class _Tee:
    """同时写 stdout 和 AFS 日志文件。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def main() -> None:
    try:
        import transfer_queue as tq
        from transfer_queue import KVBatchMeta
    except ImportError:
        print("transfer_queue 未安装 —— 本脚本必须在集群跑。")
        return

    print("=" * 60)
    print("1. KVBatchMeta.concat 源码（字段校验 == 全等；结果用 base_fields）")
    print("=" * 60)
    try:
        print(inspect.getsource(KVBatchMeta.concat))
    except (TypeError, OSError) as exc:
        print(f"取源码失败: {exc}")

    print("=" * 60)
    print("2. KVBatchMeta 全部方法（找 add_field / update / select_fields 之类）")
    print("=" * 60)
    for name in sorted(dir(KVBatchMeta)):
        if not name.startswith("__"):
            print(f"  {name}")

    print("=" * 60)
    print("3. KVBatchMeta.__init__ 签名")
    print("=" * 60)
    try:
        print(inspect.signature(KVBatchMeta.__init__))
    except (TypeError, ValueError) as exc:
        print(f"取签名失败: {exc}")

    print("=" * 60)
    print("4. tq 模块级 API 签名")
    print("=" * 60)
    for fn in ("kv_batch_put", "kv_batch_get_by_meta", "kv_clear"):
        f = getattr(tq, fn, None)
        if f is not None:
            try:
                print(f"  tq.{fn}{inspect.signature(f)}")
            except (TypeError, ValueError):
                print(f"  tq.{fn}(...)")

    print("=" * 60)
    print("5. kv_batch_put 源码（看对已存在 key 是追加字段还是整行替换）")
    print("=" * 60)
    put = getattr(tq, "kv_batch_put", None)
    if put is not None:
        try:
            print(inspect.getsource(put))
        except (TypeError, OSError) as exc:
            print(f"取源码失败: {exc}（尝试看 tq_client.put）")

    print()
    print("下一步：据此在 cl_replay_hook_v1._append_replay_rows_v1 给 rollout batch")
    print("补 is_replay/replay_response_mask/replay_token_weights 三个零值字段，")
    print("使其与回放行字段集一致后再 concat（见 RunLog §64）。")


if __name__ == "__main__":
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs" / "probe"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "probe_kvbatch_concat.log"
    with open(log_path, "w", encoding="utf-8") as logf:
        orig = sys.stdout
        sys.stdout = _Tee(orig, logf)
        try:
            print(f"# probe run @ {datetime.datetime.now().isoformat()}")
            main()
        finally:
            sys.stdout = orig
    print(f"\n[probe] 日志已写入 AFS: {log_path}")
