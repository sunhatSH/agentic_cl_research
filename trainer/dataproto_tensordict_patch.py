"""DataProto → TensorDict 入口转换 —— verl 0.8.0 v1 引擎 worker 类型不匹配的系统性修复（零侵入，不改 verl 源码）。

问题（r0 4卡/16卡每 step 崩 ``AttributeError: 'DataProto' object has no attribute 'keys'/'shape'``）：
- verl 0.8.0 v1（custom_sync / transfer_queue / KVBatchMeta）里，``engine_workers.py`` 的
  ``infer_batch`` / ``train_batch`` / ``train_mini_batch`` 函数标注 ``data: TensorDict``，函数体
  用 TensorDict API（``data.keys()`` / ``data.shape[0]`` / ``tu.get`` / ``tu.pop`` /
  ``tu.assign_non_tensor`` / ``tu.make_iterator`` / ``maybe_fix_3d_position_ids``）。
- 但 r0 掺回放行（``trainer/cl_replay_hook_v1._append_replay_rows_v1`` 的 ``KVBatchMeta.concat``）
  后，分发给 worker 的 ``data`` 实际是 **DataProto**（不是函数标注的 TensorDict）。DataProto 没有
  ``.keys()`` / ``.shape``；``tu.assign_non_tensor`` / ``tu.pop`` / ``tu.make_iterator`` 都
  ``assert isinstance(tensor_dict, TensorDict)``。于是 r0 每 step 崩。b1（无 replay）分发的
  ``data`` 是 TensorDict，不崩——这正说明问题是「r0 路径让 data 变成了 DataProto」而非函数本身写错。

已确认崩溃点（``verl/workers/engine_workers.py``，dependencies/verl 与 workspace/verl 同结构）：
- ``infer_batch``：line 386 ``tu.pop(data, "no_lora_adapter", default=False)``（DataProto.pop 签名不兼容，
  把字符串 key 当 batch_keys 遍历字符 → ``assert 'n' in self.batch.keys()`` 崩）；line 398
  ``if key not in data.keys()``；line 399 ``tu.assign_non_tensor(data, ...)``。
- ``train_batch``：line 344 ``if key not in data.keys()``；line 345 ``tu.assign_non_tensor(data, ...)``。
- ``train_mini_batch``：line 243 ``maybe_fix_3d_position_ids(data)``；line 244 ``data.shape[0]``；
  line 264 ``tu.make_iterator(data, ...)``。

治本（一次修完，不打地鼠）：在 ``tqbridge``（``verl.utils.transferqueue_utils.tqbridge``）层，对每个
``@register`` 分发函数的 ``data`` 入参做 **DataProto → TensorDict** 无损转换
（``DataProto.to_tensordict()``，protocol.py:1102，batch→tensor + non_tensor_batch→NonTensorStack +
meta_info→non_tensor_dict，要求 tensordict≥0.10——集群 Dockerfile ``tensordict<=0.10.0`` 即装 0.10.0）。

为什么 patch ``tqbridge`` 而不是逐个 patch ``infer_batch``/``train_batch``/``train_mini_batch``：
1. ``tqbridge`` 是 verl v1 里所有 ``@register`` 分发函数（infer_batch/train_batch/train_mini_batch/
   compute_log_prob/update_actor/compute_ref_log_prob/...）共用的 KVBatchMeta↔TensorDict 桥接装饰器。
   当入参是 DataProto（非 BatchMeta/KVBatchMeta）时，tqbridge 的 ``_find_meta`` 返回 None，直接把
   DataProto 透传给函数体 → 函数体用 TensorDict API 崩。在 tqbridge 层统一把 DataProto 转 TensorDict，
   一处覆盖全部，且未来新增的 @register 函数也自动被覆盖。
2. tqbridge 由 ``verl.protocol → verl.utils.transferqueue_utils`` 在 ``import verl`` 时（__init__.py:21
   经 protocol 的 BatchMeta/KVBatchMeta import）就已加载，patch 它不需要提前 import 沉重的
   ``engine_workers``（避免循环 import + 拖慢每个 verl 进程启动）。
3. 时序正确：本模块在 ``verl/__init__.py:38-41``（VERL_USE_EXTERNAL_MODULES）时 patch
   ``transferqueue_utils.tqbridge``；此后 ``decorator.py:20`` 才 ``from verl.utils.transferqueue_utils
   import tqbridge``，拿到的是 patch 后的版本；``register``（decorator.py:425）再拿它包函数。故所有
   ``@register`` 方法在类定义时就带上了转换。

防御性保留 ``tu.pop`` 的 DataProto 兼容（原 tensordict_pop_patch 的打地鼠版）：tqbridge 转换后
``tu.pop`` 本不该再遇到 DataProto，但保留兜底覆盖「绕过 tqbridge 直接调 tu.pop 传 DataProto」的路径。
（注：``tu.get`` 对 DataProto 不崩——DataProto 走 ``__getitem__`` 序列协议 ``key not in data`` 恒 False →
返回 default；只有 ``tu.pop`` 因 ``DataProto.pop(batch_keys=...)`` 签名不兼容才硬崩。）

触发：经 ``VERL_USE_EXTERNAL_MODULES``（scripts/_train_impl.sh 追加本模块名）在【每个】verl 进程
（含 worker/lightllm 副本）import verl 时 import 本模块 → import 即 patch。仿
trainer/observer_hook_register.py。幂等。
"""

from __future__ import annotations

import functools
import inspect
import sys

_PATCHED = False
_DataProto = None


def _get_dataproto_cls():
    """惰性取 DataProto 类（import verl 时 protocol 已加载，这里缓存避免每次调用重复 import）。"""
    global _DataProto
    if _DataProto is None:
        from verl.protocol import DataProto

        _DataProto = DataProto
    return _DataProto


def _dataproto_to_tensordict(data):
    """DataProto → TensorDict 无损转换（protocol.py:1102）。要求 tensordict≥0.10（集群已满足）。"""
    return data.to_tensordict()


def _convert_dataproto_args(args, kwargs):
    """把 args/kwargs 里的 DataProto 就地转成 TensorDict（非 DataProto 原样保留）。"""
    DataProto = _get_dataproto_cls()
    args = tuple(_dataproto_to_tensordict(a) if isinstance(a, DataProto) else a for a in args)
    kwargs = {k: (_dataproto_to_tensordict(v) if isinstance(v, DataProto) else v) for k, v in kwargs.items()}
    return args, kwargs


def _patch_tqbridge() -> None:
    """Monkey-patch tqbridge：每个 @register 分发函数的 DataProto 入参统一转 TensorDict。"""
    from verl.utils import transferqueue_utils as tqu

    if getattr(tqu, "_cl_dataproto_tensordict_patched", False):
        return

    _orig_tqbridge = tqu.tqbridge

    def _patched_tqbridge(dispatch_mode=None):
        deco = _orig_tqbridge(dispatch_mode)

        def _new_deco(func):
            wrapped = deco(func)  # 原 tqbridge 返回的 inner / async_inner
            if inspect.iscoroutinefunction(wrapped):
                @functools.wraps(func)
                async def _conv(*args, **kwargs):
                    args, kwargs = _convert_dataproto_args(args, kwargs)
                    return await wrapped(*args, **kwargs)

                return _conv

            @functools.wraps(func)
            def _conv(*args, **kwargs):
                args, kwargs = _convert_dataproto_args(args, kwargs)
                return wrapped(*args, **kwargs)

            return _conv

        return _new_deco

    tqu.tqbridge = _patched_tqbridge
    tqu._cl_dataproto_tensordict_patched = True

    # 防御：若 decorator 已在本模块之前被 import（理论上不会，decorator 只在 engine_workers 等
    # 后续模块里被 import），其命名空间里的 tqbridge 引用仍是旧的，这里同步过去。
    deco_mod = sys.modules.get("verl.single_controller.base.decorator")
    if deco_mod is not None and getattr(deco_mod, "tqbridge", None) is not None:
        deco_mod.tqbridge = _patched_tqbridge


def _patch_tu_pop() -> None:
    """防御性兜底：tu.pop 的 DataProto 兼容（tqbridge 转换后本不该再遇到 DataProto）。

    原 tensordict_pop_patch 的 E8 打地鼠修复，保留作第二道防线。DataProto 下 tu.pop 的
    ``tensordict.pop(key, sentinel)`` 会走 ``DataProto.pop(batch_keys=...)`` 签名不兼容硬崩；
    这里手动查 batch/non_tensor_batch/meta_info 三处，key 不存在返回 default。
    """
    from verl.utils import tensordict_utils as tu

    if getattr(tu, "_cl_pop_dataproto_compat", False):
        return

    _orig_pop = tu.pop

    def _pop_dataproto_compat(tensordict, key, default=None):
        DataProto = _get_dataproto_cls()
        if isinstance(tensordict, DataProto):
            batch = tensordict.batch
            if batch is not None and key in batch:
                return _orig_pop(batch, key, default)
            ntb = tensordict.non_tensor_batch
            if ntb and key in ntb:
                return ntb.pop(key)
            mi = tensordict.meta_info
            if mi and key in mi:
                return mi.pop(key)
            return default
        return _orig_pop(tensordict, key, default)

    tu.pop = _pop_dataproto_compat
    tu._cl_pop_dataproto_compat = True


def install() -> None:
    """Monkey-patch（幂等）：tqbridge 入口 DataProto→TensorDict + tu.pop 兜底。"""
    global _PATCHED
    if _PATCHED:
        return

    try:
        _patch_tqbridge()
        _patch_tu_pop()
    except Exception as exc:  # noqa: BLE001 -- verl absent off-cluster
        print(f"[cl] DataProto→TensorDict patch 跳过（verl 不可用: {exc}）", flush=True)
        return

    _PATCHED = True
    print(
        "[cl] DataProto→TensorDict 入口转换 patch 已安装（tqbridge 统一转 + tu.pop 兜底，不改 verl 源码）",
        flush=True,
    )


# import 即安装（与 trainer/observer_hook_register 的"import 触发 patch"风格一致）。
install()
