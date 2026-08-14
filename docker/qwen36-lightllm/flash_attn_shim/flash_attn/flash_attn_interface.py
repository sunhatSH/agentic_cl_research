# Shim: flash_attn.flash_attn_interface —— 转发到真 FA3(flash_attn_3)。
#
# 背景(2026-07-29 集群定位)：镜像的 flash_attn_shim 只实现了 bert_padding,靠
# PYTHONPATH=/opt/flash_attn_shim 前置遮蔽了真 flash_attn。但 transformer_engine.pytorch
# (被 recipe_custom→megatron 链 import)需要 `from flash_attn.flash_attn_interface import
# flash_attn_func, flash_attn_varlen_func` —— shim 缺这个子模块 → import 崩 → 训练起步挂。
#
# 真 FA3 已装在 site-packages/flash_attn_3/(verl 官方约定目录,来自 Dao-AILab hopper),
# 只是路径名是 flash_attn_3 而非 flash_attn。这里把真 FA3 的接口【转发】过来,让
# flash_attn.flash_attn_interface 可 import,且拿到的是真 kernel(非假实现)。
# 若 flash_attn_3 也不可用,退回一个会明确报错的占位(绝不静默返回错误结果)。

try:
    # 真 FA3(hopper)的接口。函数名可能是 flash_attn_func/flash_attn_varlen_func,
    # 也可能 hopper 版命名不同 —— 尽量转发,缺的用占位。
    from flash_attn_3.flash_attn_interface import (  # type: ignore[import-not-found]
        flash_attn_func,
        flash_attn_varlen_func,
    )
    _FA3_OK = True
except Exception:  # noqa: BLE001 -- flash_attn_3 缺失/命名不同时退回占位
    _FA3_OK = False

    def _missing(*_args, **_kwargs):
        raise RuntimeError(
            "flash_attn.flash_attn_interface shim: 真 FA3(flash_attn_3) 不可用,"
            "且此调用无 torch fallback。若训练真的走到 flash-attn kernel(而非仅 import),"
            "需在镜像补齐 flash_attn_3 或换真 flash-attn。当前 fsdp2 路径通常只 import "
            "transformer_engine 不调用它,故 import 过关即可。"
        )

    flash_attn_func = _missing  # type: ignore[assignment]
    flash_attn_varlen_func = _missing  # type: ignore[assignment]

# TE backends 可能还 import 其它符号;用 __getattr__ 兜底:未定义的属性返回占位,
# 保证 `from flash_attn.flash_attn_interface import X` 永远能 import 成功(X 若真被
# 调用才报错),不让缺符号在 import 期把整个 TE/megatron 链炸掉。


def __getattr__(name):  # PEP 562 module-level __getattr__
    def _placeholder(*_args, **_kwargs):
        raise RuntimeError(
            f"flash_attn.flash_attn_interface shim: 符号 {name!r} 未由真 FA3 提供,"
            "且被实际调用。见本文件说明。"
        )

    return _placeholder
