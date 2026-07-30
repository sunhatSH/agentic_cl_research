"""让 verl 的 hook 加载器（create_hook）认 FQN —— 零侵入 monkey-patch，不改 verl 源码.

见 plan swift-juggling-toast 阶段 F。问题：recipe_custom 的 hook factory
（recipe_custom/agent/runners/hooks/factory.py::create_hook）写死只认内置名
``clawgym_sandbox_reward``，其他 name 直接 raise ValueError。我们的 ObserverDiffHook
（trainer/observer_hook.py）是项目自定义 hook，factory 不认识。

解法（与 verl 已有的自定义口一致：reward 的 _function_name / dataset 的 custom_cls /
buffer 的 custom_sampler）：monkey-patch ``create_hook``，当 spec.name 是一个 FQN
（含 "."，如 ``trainer.observer_hook.ObserverDiffHook``）时用 load_class_from_fqn 动态
加载并实例化；否则回退原 factory（内置名照旧）。patch 幂等、只包一层，不动 verl 文件。

触发：CLTaskRunnerV1.run 里 ``import trainer.observer_hook_register``（与 import
observer_reward_manager 同处）。import 即执行 patch。

hook 实例化约定：自定义 hook 的 __init__ 接受可选 ``settings``（与内置 ClawGymSandboxRewardHook
一致）；ObserverDiffHook 不需要 settings（__init__ 无参），故按签名探测：能传 settings 就传，
否则无参构造。
"""

from __future__ import annotations

import inspect

_PATCHED = False


def install() -> None:
    """Monkey-patch recipe_custom hook factory to accept FQN hook names. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from recipe_custom.agent.runners.hooks import factory as _factory
    except Exception as exc:  # noqa: BLE001 -- recipe_custom absent off-cluster
        print(f"[cl] observer hook factory patch 跳过（recipe_custom 不可用: {exc}）", flush=True)
        return

    from verl.utils.import_utils import load_class_from_fqn

    _orig_create_hook = _factory.create_hook

    def _create_hook_with_fqn(spec):
        # 非 dict / 无 name → 交回原实现。
        name = spec.get("name") if spec else None
        if not name or "." not in str(name):
            return _orig_create_hook(spec)

        # name 是 FQN（含 "."）：动态加载我们自己的 hook 类。
        hook_cls = load_class_from_fqn(str(name), "AgentRunHook")
        settings = dict(spec.get("settings") or {})
        # 兼容两类 __init__：接受 settings 的（内置风格）/ 无参的（ObserverDiffHook）。
        try:
            params = inspect.signature(hook_cls.__init__).parameters
            if "settings" in params:
                return hook_cls(settings=settings)
        except (TypeError, ValueError):
            pass
        return hook_cls()

    _factory.create_hook = _create_hook_with_fqn
    # create_hooks 在 factory 模块内直接调 create_hook（同模块名引用），patch 模块属性即生效。
    _PATCHED = True
    print("[cl] observer hook factory 已 patch（create_hook 支持 FQN，不改 verl 源码）", flush=True)


# import 即安装（与 observer_reward_manager 的"import 触发注册"风格一致）。
install()
