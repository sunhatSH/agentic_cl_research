# 16 卡 baseline 起服 hang — 交接总结(仅现象,2026-08-01)

> **✅ 2026-08-02 已解决**：真根因 = **8 个 lightllm 副本只调度起 7 个,第 8 个(replica_rank=7)的 HTTP server actor(`num_cpus=1`)在 driver 节点抢不到 free CPU 名额 → Ray 静默 PENDING → verl init_hybrid 的 asyncio.gather 永等 → 整 job hang**。修复 = `ray start` 加 `--num-cpus=<nproc>`(scripts/_train_impl.sh,把 Ray 从 cgroup 保守估值≈16 顶到真实逻辑核)。真机验证:8 副本全起、update_weights=24、进 Training Progress。**下方 §52 的"双 infer_loop 竞态"归因是误判(Ray dedup 日志假象),已被 RunLog §55 推翻。以下现象记录保留作调试史。**

> 供新会话排查。**只描述现象与已知事实,不含根因推测。** 详细排查历程见 `doc/archive/RunLog.md` §36–§52、`doc/debug/Training_Debug_2026-07-24.md` §44–§47。

## 一、核心现象

`qwen35_9b_b1_16gpu`(2 节点×8 卡,TP=2,8 个 lightllm 副本)**反复起服 hang,无法进入训练**,已复发 ~7 次(每次换节点):

- 8 个 lightllm 推理副本起服基本完成(`Capture cudagraph success`、`server start up ok`)。
- 之后进程**静默/自旋**:`Training Progress=0`、沙箱请求=0、`update_weights_from_ipc 200`=0,即**从未进入 rollout / 训练循环**。
- **GPU 利用率 = 0%,GPU 显存占住不放**(监控图:显存起服后钉在高位,util 归 0)。
- 进程不退出(mtime 持续更新是因为在刷 NCCL 日志,不是在计算)。

## 二、关键日志特征(开 CL_DIAG=1 后可见,当前 run 17:00:04Z 启动)

- NCCL 疯刷:`Broadcast: opCount 0 ... count 1 datatype 2 op 0 root 0`,**243183 次**,`opCount` **恒为 0**(集合从不推进)。
- 该 broadcast 由 `LightLLMHttpServer`(推理副本)发出;master-0 与 worker-0 两节点、每副本 TP 组的 rank[0] 和 rank[1] **都在高频刷**(各线程 ~4400+ 次),**无 rank 缺席**。
- **无** OOM、**无** Traceback、**无** `Cuda failure`、**无** ActorDied/RayActorError。
- 对应 lightllm 代码:`base_backend.py:247 node_broadcast_tensor=torch.tensor([0],dtype=int32)` + `:633 broadcast(..., group=node_nccl_group)`(serve loop `_try_read_new_reqs` 每 tick 的控制同步);`node_nccl_group=create_new_group_for_current_node("nccl")`(:248);`:284-287` 无条件起 2 个 infer_loop 线程。

## 三、两节点监控曲线差异(用户观察,已核实)

- **k1 实验**(同为 16 卡,也起服失败):两节点 **GPU 显存几乎重合**(master≈worker≈75%)。
- **b1 baseline**:两节点 **GPU 显存有固定差距**(master≈72% / worker≈55%,差 ~17%);内存也是 master 略高。
- CPU/内存/GPU-util 曲线两实验形态类似(起服尖峰后 util 归 0、显存钉住)。

## 四、已尝试的修复及结果(现象层面)

| 改动 | 位置 | 结果(现象) |
|------|------|------|
| `NCCL_CUMEM_ENABLE=0` | verl_runner.py passthrough + _train_impl.sh | 已透传生效;P2P `Cuda failure` 从有→归 0。仍 hang。 |
| 模型预热到 `/dev/shm` node-local | _train_impl.sh `_prewarm_model` | 生效(40s 完成,model.path 指向 /dev/shm)。仍 hang。 |
| `gpu_memory_utilization` 0.75→0.65 | 18 份 16卡 config | 生效(KV 池缩小)。仍 hang。 |
| `running_max_req_size`/`graph_max_batch_size` 256→64 | 18 份 16卡 config | 生效。仍 hang。 |
| `disable_symm_mem_allreduce=true` | 18 份 16卡 config | 生效(StartArgs 确认、`SymmMemAllreduce enabled`=0)。**该 count=1 broadcast 仍刷、仍 hang。** |
| `_try_read_new_reqs` 加进程级 `threading.Lock` | LightLLM `base_backend.py:253/626` | .pyc 已含锁符号(加载生效)。**broadcast 仍刷 24万次、仍 hang。** |

> 即:上述 6 项均已确认落地生效,但 **b1 16卡 hang 依旧**。

## 五、对照事实

- **4 卡 baseline** `qwen35_9b_b1_4gpu`(单节点×4,TP=2,**2 副本**):**正常训练到 step 12+**,reward 健康(0.3~0.67 波动),无此 hang。
- **k1 16卡**:曾用**旧配置**(无 NCCL_CUMEM=0、无预热)跑到 step 2 后死于 **GPU 显存 CUDA OOM**(prefill,§36/§42);现用新配置也起服失败(与 b1 同类现象)。
- 差异维度:4卡=单节点/2副本(通),16卡=2节点/8副本(不通)。

## 六、诊断开关(排查用)

- `CL_DIAG=1`(_train_impl.sh):开 `NCCL_DEBUG=INFO` + `NCCL_DEBUG_SUBSYS=INIT,COLL,P2P` + `RAY_DEDUP_LOGS=0` + lightllm/verl debug 日志,均经 verl_runner passthrough 进 Ray worker。**不开则 Ray 日志去重(`[repeated Nx]`)会把 per-rank 现象折叠成假象(如"1/8 副本到 594"实为去重假象,真实 7-8 个都到)。**
- 未验证的诊断手段:`py-spy dump` 卡住的 lightllm rank pid(看栈)、driver pid(看是否卡在 `asyncio.gather`);节点内 `ray list actors`(确认 8 个 LightLLMHttpServer actor 是否都 ALIVE)、`nvidia-smi`、`lspci -vvv|grep acsctl`(P2P/ACS 拓扑)。

## 七、当前未提交改动(git,本 session)

- 主项目 `agentic_cl_research`:18 份 `configs/run/*_9b_16gpu.yaml`(util/并发/disable_symm_mem)、`configs/run/b1_9b_4gpu.yaml`、`scripts/_train_impl.sh`(预热+CL_DIAG+NCCL_CUMEM)、`trainer/verl_runner.py`(passthrough)、`trainer/cl_agent_dataset.py`(agent_assets)、`doc/archive/RunLog.md`、`doc/debug/*`。**均未 commit。**
- LightLLM 仓库:`base_backend.py`(加锁 patch,未生效于解决问题,**未 commit**)。

## 八、日志位置

- b1 当前 run:`logs/experiments/qwen35_9b_b1_16gpu/train.log`(带 CL_DIAG,含 NCCL INFO)。
- 历史归档:`logs/experiments/qwen35_9b_b1_16gpu/archive/`。
- k1:`logs/experiments/qwen35_9b_k1_16gpu/train.log`。
- 4卡对照:`logs/experiments/qwen35_9b_b1_4gpu/train.log`(step12+,健康)。
