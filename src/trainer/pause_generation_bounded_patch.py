"""pause_generation abort_all 有界等待 → 超时放行,根治 16卡 abort 死锁 —— 零侵入 monkey-patch,不改 LightLLM 源码.

问题(§59，16卡 step6 卡死根因)：
- lightllm 请求是跨进程 shm 对象，``ref_count`` = 持有它的进程数，正常结束各进程 put_back 降回 1
  才能被 recycle_resource_loop 回收。但存在**引擎侧 refcount 泄漏 bug**：部分请求 ref_count 卡在
  5 不降（``can release False refcount 5`` 刷数千次），recycle 因 ``can_release()`` 要求 ref_count==1
  永远回收不掉 → 僵尸请求占满 KV 池（token used ratio: 真实活跃仅 8.9%，含僵尸达 99.99%）。
- **致命化的关键**：每 step 边界 ``replica.abort_all_requests → async_lightllm_server.py:199
  pause_generation()`` → lightllm ``HttpServerManager.pause_generation``（manager.py:1013-1024）里是
  **无限 ``while True``**：``abort_request(abort_all=True)`` 内 ``_wait_for_abort_released`` 死等
  ``req_id_to_out_inf`` 变空，僵尸请求回收不了 → 集合永不空 → 60s 超时返回 False → while 无限重试
  → 整个 job 永久 hang（日志 ``abort request wait release timeout`` + ``still waiting`` 反复刷）。

**4卡对照（铁证）**：4卡 b1/k1/k2/k3 的 ``refcount 5`` 泄漏次数（5354/4992/5256/5331）和 16卡（5354）
**几乎一样**，但 ``abort 超时 = 0``，训练跑到 step 158-170。即**泄漏是良性共性 bug，是 abort 无限重试
把它在 16卡放大成致命死锁**（16卡 util=0.65 池小僵尸更快占满 + 每 step 边界必 pause）。

解法（不碰危险的 shm 强摘——lightllm 团队正连续 fix pause，强制释放会让别的进程访问已释放 shm 崩）：
把 ``pause_generation`` 的无限 ``while True`` 换成**有界等待**，超 CL_PAUSE_MAX_WAIT（默认 180s）仍未
清空则**放行**（生成已 pause，权重同步安全；僵尸请求留给 recycle_loop 后台继续清）。让 16卡的泄漏像
4卡一样**良性**：泄漏仍在但不再死锁。**治"致命化"不治"泄漏本身"**（泄漏根治需 lightllm 团队修，
最新 commit fix pause / fix auto ipc handle 正在这方向，可后续升级 LightLLM 分支）。

触发：经 ``VERL_USE_EXTERNAL_MODULES``（scripts/_train_impl.sh 追加本模块名）在【每个】verl 进程
（含 lightllm 副本进程）import verl 时 import 本模块 → import 即 patch。仿 trainer/observer_hook_register.py。
"""

from __future__ import annotations

import os

_PATCHED = False


def install() -> None:
    """Monkey-patch HttpServerManager.pause_generation 有界等待放行。Idempotent."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import asyncio

        from lightllm.server.httpserver.manager import HttpServerManager, logger
        from lightllm.server.io_struct import AbortReq
        from lightllm.server.core.objs.req import FinishStatus
    except Exception as exc:  # noqa: BLE001 -- lightllm absent off-cluster
        print(f"[cl] pause_generation 有界化 patch 跳过（lightllm 不可用: {exc}）", flush=True)
        return

    # 总时限:超过它 pause_generation 放行(不再无限死等僵尸回收)。默认 180s——覆盖正常
    # abort 收尾(4卡实测秒级),又不至于让死锁拖垮整 job。CL_PAUSE_MAX_WAIT 可覆盖。
    _max_wait = float(os.environ.get("CL_PAUSE_MAX_WAIT", "180"))

    async def _pause_generation_bounded(self, reject_new: bool = False):
        # 完全复刻原 manager.py:1013-1024 的上下文管理器结构,只把无限 while 改成有界。
        async with self._gen_pause.pause_and_abort_context(reject_new) as do_abort:
            if not do_abort:
                return
            waited = 0.0
            while waited < _max_wait:
                success, msg = await self.abort_request(AbortReq(request_id=None, abort_all=True))
                if success:
                    return
                logger.warning(f"pause_generation abort_all still waiting: {msg}")
                await asyncio.sleep(1.0)
                # abort_request 内 _wait_for_abort_released 每轮最多阻塞 ~60s(其自身 timeout),
                # 故实际每轮耗时 ~1+60s;waited 累加两者,让 deadline 语义反映真实墙钟。
                waited += 1.0 + 60.0

            # ── 超时放行：修复僵尸请求，使其能被 recycle_resource_loop 正常回收 ──
            # 问题：abort_all 超时后 zombie req 仍在 req_id_to_out_inf，阻塞整条释放链路：
            #   release_memory_occupation → assert len(req_id_to_out_inf)==0 → 500
            #   → 下个 step 的 on_step_end → resume_memory_occupation → timeout → 训练崩
            #
            # 解法：对每个 zombie 补齐 can_release() 的 4 个条件，然后 recycle_resource_loop
            # （每 0.02s 运行）自然走完 put_back→ref_count=0→release_req_index→shm 真释放。
            # 不直接调 put_back：各进程的 proc_private_get_state 不同，HTTP manager 只能减自己
            # 那一份；直接设 ref_count=1 再由 recycle loop 减 1→0 才是安全的单进程操作。
            zombie_count = len(self.req_id_to_out_inf)
            for req_status in list(self.req_id_to_out_inf.values()):
                if req_status is None:
                    continue
                for req in req_status.group_req_objs.shm_req_objs:
                    # (1) ref_count → 1：让 can_release 的 ref_count==1 检查通过。
                    #     recycle loop 会再调一次 put_back → 0 → 真释放 shm slot。
                    idx = req.index_in_shm_mem
                    with self.shm_req_manager.get_req_lock_by_index(idx):
                        req.ref_count = 1
                    # (2) finish_status → FINISHED_ABORTED：满足 is_finished() 条件
                    req.finish_status.status = FinishStatus.FINISHED_ABORTED
                    # (3) can_released_mark → True：各 worker 已通过 abort 标记退出，
                    #     不会再写 shm，安全释放。
                    req.can_released_mark = True
                    # (4) out_tokens_queue → 空：head == tail
                    req.out_tokens_queue.head = req.out_tokens_queue.tail

            logger.error(
                "pause_generation abort_all 超 %ss 未清空(疑 lightllm refcount 泄漏僵尸请求 "
                "ref_count 不降),已标记 %d 组僵尸为可释放,recycle_resource_loop 将回收 shm。见 §59。",
                _max_wait,
                zombie_count,
            )
            return

    HttpServerManager.pause_generation = _pause_generation_bounded
    _PATCHED = True
    print(
        f"[cl] pause_generation 有界化 patch 已安装（abort_all 超 {_max_wait}s 放行,不无限死等僵尸回收,"
        "根治 16卡 abort 死锁,不改 LightLLM 源码）",
        flush=True,
    )


# import 即安装（与 trainer/observer_hook_register 的"import 触发 patch"风格一致）。
install()
