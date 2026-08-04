# LightLLM `pause_generation` abort 死锁 —— 请教材料

> 面向 LightLLM 维护同事。分支:`LightLLM @ rl_verl_rebase_main`(commit `e336d827` 附近)。
> 用途:verl 0.8.0 RL 训练的 rollout 推理引擎(colocate 推理训练共卡)。
> 问题:同样 LightLLM 代码,别人多机没事,我们 16 卡 RL 训练卡死。

## 一、现象

16 卡(2 节点×8 卡,8 个 lightllm 副本,每副本单节点 TP2)RL 训练,step 6 **整个 job 永久 hang**。一个副本无限刷:
```
[manager.py:916] left req id 39432 can release False refcount 5   # 同几个 id 反复
[manager.py:856] abort request wait release timeout, abort_all=True, timeout=60.0s
[manager.py:1023] pause_generation abort_all still waiting: ...   # 无限重试
```
- 卡死副本 KV 池:真实活跃 **8.9%**,含 unrefed 僵尸 **99.99%**(僵尸请求占满,真实负载很轻)。
- 僵尸请求 `ref_count` **卡在 5 不降**(正常应降到 1 才回收)、`can_released_mark=False`。

## 二、死锁链(代码位置)

1. `pause_generation`(`manager.py:1013`)是**无限 `while True`**:`abort_request(abort_all=True)` 失败就 `sleep(1)` 重试,无上限。
2. `abort_request → _wait_for_abort_released`(`manager.py:830`)死等 `req_id_to_out_inf` 变空,每轮 60s 超时。
3. 回收在 `recycle_resource_loop`(`manager.py:872`),只回收 `Req.can_release()` 为真的请求;`can_release`(`req.py:366`)**硬要求 `ref_count==1`**。
4. 僵尸 `ref_count=5≠1` → `can_release()` 永远 False → 永不回收 → `req_id_to_out_inf` 永不空 → `_wait_for_abort_released` 永远超时 → **`while True` 无限重试 → 整 job hang**。

> 注:`can_release`(`req.py:371`)里 `if self.is_aborted and can_released_mark and ref_count_ok: return True` 这条**被注释掉了**(commit `b8cfd70c` 重构)——被 abort 的请求少了一条回收路径。

## 三、ref_count 语义(我方理解,请核对)

请求是跨进程 shm 对象,`ref_count`=持有它的进程数(`shm_req_manager.py:124 get +1` / `:136 put_back -1`)。get 者:httpserver/router/detok/infer/visual/audio/multi_level_kv。正常走完各进程 get→put_back 降回 1 才回收。**`ref_count=5` = 5 个持有者没 put_back**,我方未能定位是哪个进程漏了。

## 四、关键数据:4 卡【无真泄漏】,只有 16 卡有僵尸(决定性对照)

`manager.py:911` 每 120s 打印一次未释放请求快照。按 **同一 req id 被抓到几次** 区分"真卡住"vs"正常处理中":

| | 同一 req id 最大出现次数 | 判读 |
|---|---|---|
| **4 卡**(单机,跑到 step 170+ 不崩) | **≤4 次** | 请求短暂滞留就释放走,不断有新请求 → **无真泄漏** |
| **16 卡**(step6 卡死) | 同几个 id **308 次**(几小时不放) | **真僵尸,死锁** |

(`refcount 5` 原始次数两边都 ~5600,具迷惑性;按 req id 去重后真相相反。)

## 五、我们环境的特殊因素(疑触发条件)

1. **高频 `abort_all`(RL 特有)**:每 step 边界暂停推理做权重同步 → `pause_generation → abort_all`。本 run abort **3016 次**。纯推理 serving 不会反复全量 abort。
2. **外挂 gateway 转发**:请求经我方 `GatewayActor` 转发到 lightllm,`request_id` 复用 session_id、多轮 continuation。多一层 shm 引用持有者。
3. **TP2 多进程 + chunked prefill 恒开 + 长请求**(prompt 3-6 万 token):持有 ref_count 的进程多、持有时间长——直接关系"为什么 ref_count 到 5"。
4. **colocate KV 池小**(`mem_fraction=0.65`,给训练留显存):僵尸更快占满撞临界。

**已排除**:我方副本 `nnodes=1`(单节点 TP2),`is_multinode_tp=False`,**不走** commit `b737d2af` 的 multinode abort broadcast 路径 → 泄漏不在跨节点 abort。差异收窄到"16 卡高 abort 频率 × 小 KV 池,在单副本内触发 abort 释放竞态"。

## 六、想请教的问题

1. **`ref_count` 卡 5 不降,哪个进程 get 了没 put_back?** 尤其 abort / 多轮 continuation 场景有无已知泄漏路径?
2. **被 abort 的请求走哪条路进 `finished_reqs` 释放?**(我方最强嫌疑)`_get_classed_reqs`(`base_backend.py:847-868`)里请求进 finished_reqs 要 `filter_mark==True` 或 `finish_status.is_finished()`;`is_aborted=True` 但这两个都不满足的请求,是否卡在中间态、infer 侧 `_filter`(`infer_batch.py:266`)永不 put_back?
3. `can_release` 里 `is_aborted` 回收分支为何注释掉(`b8cfd70c`)?abort 请求现在靠什么回收?
4. 最近 commit(`fix pause` / `fix pause generation` / `add reject_all for pause_generation` / `fix auto ipc handle`)是否已修此泄漏?我方分支该升到哪个 commit?
5. `pause_generation` 无超时 `while True` 是有意设计吗?我方临时改成有界等待超时放行(见下),有无副作用?

## 七、我方临时规避(治标,待指正)

monkey-patch `pause_generation` 把无限 `while True` 改成有界:超 `CL_PAUSE_MAX_WAIT`(默认 180s)仍未清空则**放行**(此时已 PAUSED、推理已停、权重同步安全;僵尸留给 recycle_loop 后台清)。只让 16 卡像 4 卡一样不死锁,**未解决 refcount 泄漏本身**。

## 附:关键行号(我方分支)
- `manager.py:1013` pause_generation(无限 while)· `:830` _wait_for_abort_released(60s)· `:872` recycle_resource_loop · `:818` abort(只标 is_aborted)
- `core/objs/req.py:366` can_release(ref_count==1 + 注释掉的 aborted 分支)
- `core/objs/shm_req_manager.py:124/136` get(+1)/put_back(-1)
- `router/model_infer/mode_backend/base_backend.py:788` _get_classed_reqs · `infer_batch.py:266` _filter(put_back)
- `detokenization/manager.py:162` 置 can_released_mark=True
