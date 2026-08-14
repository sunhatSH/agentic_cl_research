"""关闭 e2b SDK 的 HTTP/2，改用 HTTP/1.1 —— 从根上消除 GOAWAY（零侵入，不改 e2b/verl 源码）。

## 背景 / 为什么需要它

腾讯 AGS 入口网关对单条 HTTP/2 连接累计 ~1000 个 stream 后主动发 GOAWAY 回收连接。
e2b SDK（``e2b`` 2.30.0）构造 httpx transport 时 **默认 ``http2=True``**，且长连接复用
（``max_keepalive_connections=20``、``keepalive_expiry=300s``）——长任务 rollout 期间
envd（command/filesystem 流式）通道会撞到 GOAWAY，业务侧表现为 RemoteProtocolError /
ConnectionState.CLOSED，直接让一条 session 失败。

之前的兜底是在 verl 的 ``AgentSessionWorker._run_session_with_timeout`` 里对 GOAWAY 重采
一次——但那是**连接层的关注点塞进了训练框架**，位置不对，且 e2b SDK 当前版本没有暴露
关 HTTP/2 的 env 开关。HTTP/1.1 不存在“单连接 stream 上限”这回事（一连接一请求，连接回收
由 keepalive 自然管理），故切到 HTTP/1.1 = 从根上消除 GOAWAY，重采逻辑随之删除。

## 实现

e2b 的 transport 工厂（``e2b/api/client_async/__init__.py`` 与 ``client_sync/__init__.py``）
是 ``get_transport(config, http2=True)`` / ``get_envd_transport(config, http2=True)``，
``http2`` 是**函数默认参数**且调用方（``sandbox_async/main.py`` 等）从不显式传值。
故把这 4 个函数的 ``__defaults__`` 里的 ``http2`` 翻成 ``False`` 即可全局生效——
``sandbox_async/main.py`` 用 ``from ... import get_envd_transport as get_transport`` 引用的是
**同一个函数对象**，改 ``__defaults__`` 对所有别名一次到位。

幂等；``CL_E2B_DISABLE_HTTP2=0`` 可关（回退 SDK 默认 http2）；e2b 不可用时静默跳过。

## 触发点

经 ``VERL_USE_EXTERNAL_MODULES``（逗号分隔，``verl/__init__.py`` 在**每个** verl 进程
——含 AgentSessionWorker——``import verl`` 时消费）自动 import 本模块。见
``scripts/_train_impl.sh``。无需改 verl 源码。
"""

from __future__ import annotations

import os

_PATCHED = False


def _flip_http2_default(func) -> bool:
    """把带 ``http2: bool = True`` 关键字默认的工厂函数改成默认 False。返回是否改动。

    ``http2`` 是仅有一个默认值的位置/关键字参数（``(config, http2=True)``），
    ``__defaults__`` 是 ``(True,)``。稳妥起见按参数名定位，不假设位置。
    """
    import inspect

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    defaults = list(func.__defaults__ or ())
    if not defaults:
        return False
    # __defaults__ 对齐的是「带默认值的尾部参数」；找 http2 在其中的下标。
    defaulted = [p for p in params if p.default is not inspect.Parameter.empty]
    for idx, p in enumerate(defaulted):
        if p.name == "http2" and idx < len(defaults):
            if defaults[idx] is False:
                return False  # 已是 False
            defaults[idx] = False
            func.__defaults__ = tuple(defaults)
            return True
    return False


def install() -> None:
    """Monkey-patch e2b transport 工厂默认 ``http2=False``。幂等。"""
    global _PATCHED
    if _PATCHED:
        return
    if os.getenv("CL_E2B_DISABLE_HTTP2", "1") != "1":
        print("[cl] e2b HTTP/2 patch 跳过（CL_E2B_DISABLE_HTTP2 != 1）", flush=True)
        return

    try:
        from e2b.api import client_async, client_sync
    except Exception as exc:  # noqa: BLE001 -- e2b 不在环境里（off-cluster / 无沙箱）
        print(f"[cl] e2b HTTP/2 patch 跳过（e2b 不可用: {exc}）", flush=True)
        return

    flipped = []
    for mod, name in (
        (client_async, "get_transport"),
        (client_async, "get_envd_transport"),
        (client_sync, "get_transport"),
        (client_sync, "get_envd_transport"),
    ):
        func = getattr(mod, name, None)
        if func is not None and _flip_http2_default(func):
            flipped.append(f"{mod.__name__.rsplit('.', 1)[-1]}.{name}")

    _PATCHED = True
    if flipped:
        print(f"[cl] e2b 已切 HTTP/1.1（http2=False）: {', '.join(flipped)}", flush=True)
    else:
        # 已是 False / 签名变了 / 找不到——不致命，只是没生效。
        print("[cl] e2b HTTP/2 patch 未改动任何工厂（可能 SDK 版本变更，请核对）", flush=True)


# import 即安装（与 trainer/observer_hook_register 的“import 触发 patch”风格一致）。
install()
