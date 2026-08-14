#!/usr/bin/env python3
"""上集群探测 KVBatchMeta.concat 字段对齐要求（R0 回放崩溃 §64）。

背景：R0 step 2 回放激活即崩 —— `KVBatchMeta.concat([batch, replay_meta])`
报 "Field names do not match"。回放行带 3 个 rollout 没有的专属字段
（is_replay / replay_response_mask / replay_token_weights）。

本脚本在集群（有 transfer_queue）上跑，打印：
  1. concat 的字段校验源码（要两边完全一致？还是子集？）
  2. KVBatchMeta 是否有给现有 key 追加 field 的 API（决定怎么给 rollout 行补零值字段）
  3. kv_batch_put 覆盖已存在 key 时，是新增 field 还是整行替换

用法（集群）：python scripts/_smoke/probe_kvbatch_concat.py
"""
from __future__ import annotations

import inspect


def main() -> None:
    try:
        import transfer_queue as tq
        from transfer_queue import KVBatchMeta
    except ImportError:
        print("transfer_queue 未安装 —— 本脚本必须在集群跑。")
        return

    print("=" * 60)
    print("1. KVBatchMeta.concat 源码（看字段校验是全等还是子集）")
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
    print("3. KVBatchMeta.__init__ 签名（fields 参数怎么给）")
    print("=" * 60)
    try:
        print(inspect.signature(KVBatchMeta.__init__))
    except (TypeError, ValueError) as exc:
        print(f"取签名失败: {exc}")

    print("=" * 60)
    print("4. tq 模块级 API（kv_batch_put / kv_batch_get_by_meta 等签名）")
    print("=" * 60)
    for fn in ("kv_batch_put", "kv_batch_get_by_meta", "kv_clear"):
        f = getattr(tq, fn, None)
        if f is not None:
            try:
                print(f"  tq.{fn}{inspect.signature(f)}")
            except (TypeError, ValueError):
                print(f"  tq.{fn}(...)")

    print()
    print("下一步：据此在 cl_replay_hook_v1._append_replay_rows_v1 给 rollout batch")
    print("补 is_replay/replay_response_mask/replay_token_weights 三个零值字段，")
    print("使其与回放行字段集一致后再 concat（见 RunLog §64）。")


if __name__ == "__main__":
    main()
