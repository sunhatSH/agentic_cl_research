# 训练排障总表 — 精简无重复版(截至 2026-08-18)

> 本文合并 `doc/debug/` 下四份排障记录,去重后按**类别**归并。每条 = 现象/根因 + 最终解决办法(已被后续推翻的旧假说不再列为独立条目,仅在需要时以「⚠️ 曾误判」标注)。
> 详细逐条历程仍在源文件,本表只保留**结论**:
> - `Training_Debug_2026-07-24.md`(§1–§50 全量历程,最权威)
> - `16gpu_hang_handoff_2026-08-01.md`(16 卡 hang 现象快照,根因见本表 D1)
> - `16—migrate-4.md`(verl 原生 agent_loop 迁移问题)
> - `OOM_求助_GPT.md`(colocate OOM 求助稿,结论并入本表 B 类)
>
> **16 卡训练已在另一台机器修复(根因 = Ray 少调度 1 个 lightllm 副本,见 D1),现可正常训练。**

---

## A. 启动 / 集群拉起

| # | 问题(现象 + 根因) | 解决办法 |
|---|---|---|
| A1 | torch.distributed rendezvous URL 缺参 / re-rendezvous 报错;Ray head 启动竞态;verl 无视已有 Ray 集群 | 规范 rendezvous URL + 参数;master/worker 启动加同步栅栏;4 卡单机 config 去掉 `ray_init.address: auto`(单机无已起 Ray),16 卡才保留 |
| A2 | lightllm 起 server 崩 `OSError:98 Address already in use`:recipe `async_lightllm_server` 三端口(pd_master 1212 / httpmanager 12345 / router_gloo 20001)argparse 写死 default,16 卡 8 副本/节点撞端口 | 三端口改 `get_free_port` 动态分配(verl recipe 侧) |
| A3 | 脚本目录重组后路径断裂;SwanLab 认证失败直接崩训练 | 修路径;SwanLab key 只从 `.env`/env 取,缺失打 WARNING 不崩;AFS 日志全覆盖 + 自动续训 |
| A4 | `import verl` 崩 `transformer_engine has no attribute 'pytorch'`:镜像 flash_attn 是残缺 shim(只有 bert_padding),megatron→TE 要 `flash_attn.flash_attn_interface` | 给 shim 补 `flash_attn_interface.py`(转发真 FA3 + `__getattr__` 兜底);`_train_impl.sh` PYTHONPATH 前置 shim 目录 |
| A5 | 数据加载崩 `assert src[-1]` NoneType:v1 `_init_dataloader` 无条件建 val dataset,而 config `val_files: null` | `CLTaskRunnerV1.run` 里 val 空/缺失时 alias 到 `train_files` |
| A6 | 4卡单机 `ray.init(address='auto')` 崩 `Could not find any running Ray instance`:`scripts/_train_impl.sh` 单机分支(NNODES≤1)从不 `ray start --head`,直接 `_run_single`;verl `run_ppo` 写死 `address='auto'`(只有多机分支才 ray start,单机路径漏覆盖) | 单机分支补 `ray start --head`(复用 §55 的 `--num-cpus $(nproc)` 名额)+ 捕获训练退出码(脚本退出码=训练码,防假成功)+ `ray stop --force` 清理 |

## B. 显存 / OOM(colocate,16 卡 2 节点 × 8,H800 80G)

| # | 问题(现象 + 根因) | 解决办法 |
|---|---|---|
| B1 | `update_actor` CUDA OOM:长序列激活峰值 + micro-batch 塞满 + FSDP all-gather;训练进程 72.7G + lightllm 常驻权重 6.4G ≈ 撑满单卡,余量仅 ~0.07G。**主因是物理挤满(两进程一卡),碎片(reserved-unallocated 仅 147M)只是最后一根稻草** | micro-batch 2→1;`ppo_max_token_len_per_gpu` 65536→32768→20480;param/optimizer offload;**`expandable_segments:True` 撤销**(进程级全局,与 lightllm `torch_memory_saver` 硬互斥,开了 lightllm 启动即 raise) |
| B2 | `max_response_length=53886` 误设:RL parquet 只有 prompt 无 response,54K 偏大 6–8× | 统一降到 16384 |
| B3 | 超长序列(319663 / 峰值 119586)撞 assert:`response_length` 只是**单次生成** max_tokens,多轮 ReAct 累加后整条只按 `max_model_len` 截,**轨迹级无闸门** | (a) 单条工具输出 `max_tool_response_length=16384` 截;(b) 缩 `max_model_len` 131072→73728,gateway `response_capacity` 自然压到 ~65k(零改码);单次生成与整条闸门解耦(保 response 真实) |
| B4 | prefill CUDA OOM → hung(`24 MiB, 24.75 MiB free` = KV 池已 ~99.97% 满):**256 并发 × 多轮回填 prompt**(实测 prompt_token_num 到 122131)≫ KV 池,非单条超长 | **降并发第一位**:`running_max_req_size`/`graph_max_batch_size` 256→64(按单副本 = train_batch×n ÷ 副本数 × 2 倍余量,**不是全局总量**);`gpu_memory_utilization` 0.75→0.65。峰值 = 并发 × 单条长度,两者缺一不可 |

## C. verl 契约 / 迁移(自写 rollout → 原生 agent_loop)

| # | 问题(现象 + 根因) | 解决办法 |
|---|---|---|
| C1 | rollout ×8 契约冲突(导致 6 次 0-checkpoint):verl 已按 `rollout.n=8` 复制 gen_batch,自写 rollout 每行又跑 8-slot → ×64 行数对不上 | 改 per-row 每行 1 条 rollout,n=8 交给 verl uid 分组(`cc0de24`) |
| C2 | reward 从头没接上:自写 rollout 从不写 `rm_scores` → verl `KeyError` | 内联算好的 reward 写进 `rm_scores` 末位有效 token(`b3bb485`) |
| C3 | verl 严格 dataclass 拒自定义 config 键(§19/§21/§27 撞 3 次):塞它不认的键 → `TypeError` | 并发参数落 `cl.rollout.sessions_per_step`;`max_turns`/`model_type` 移出严格段;`load_config` 全量校验 |
| C4 | `ppo_kl`/`clipfrac` 全 0:rollout 没请求 logprob → PPO clip 失效 | rollout 请求 logprob(`4d478ef`) |
| C5 | `compute_log_prob` 崩 triton CE assert(维度 512 vs 378):`model_type` 放进 `override_config` → GDN 层丢 `cu_seqlens` | **`model_type` 挪到 config 顶层**(只改一个 key 位置,不改码,`6434fe3`) |
| C6 | 自写 rollout 反复撞 verl 契约(绕开原生机制的代价) | **大迁移**到 verl 原生 agent_loop(`main_ppo` + `use_v1` + `custom_sync` + `RemoteAgentLoopManager`),只保留 CL 注入(loss + buffer hook + observer);19 份 9B + 27B config 全量对齐,`load_config` 21 份 0 失败 |
| C7 | observer hook 崩 `Unknown post-run hook`:driver 端 monkey-patch factory 传不到独立 ray actor(AgentSessionWorker) | (a) `hooks/factory.py` 内建 FQN 分支;(b) `verl_runner.py` 把 PYTHONPATH 透传进 `runtime_env.env_vars` |

## D. 16 卡起服 hang(反复复发 ~7 次)—— ✅ 已定案 + 真机验证解决

| # | 问题(现象 + 根因) | 解决办法 |
|---|---|---|
| **D1** | **8 个 lightllm 副本只调度起 7 个**:第 8 个(replica_rank=7)HTTP server actor(`num_cpus=1`,不进 PG)在 driver 节点抢不到 free CPU 名额 → **Ray 静默 PENDING** → verl `llm_server.py:521` init_hybrid `asyncio.gather` 永等 → 整 job hang。现象层面:8 副本"起服完成"但 GPU util=0/显存钉高位、`Training Progress=0`、NCCL 刷 `Broadcast opCount=0`(7 副本空转,非死锁)。**元凶 = Ray 按 cgroup quota(≈16)估 num_cpus 偏保守** | ✅ **方案① 一发命中**:`ray start --head/--address` 加 `--num-cpus=$(nproc)`(`_train_impl.sh`,`CL_RAY_NUM_CPUS` 可覆盖)。真机验证:8 副本全起、`replica_rank={0..7}`、`update_weights=24`、进 Training Progress、reward 0.6 健康 |
| — | **⚠️ 曾误判(全部推翻,勿再当死因)**:①cuMem×NCCL P2P 冲突;②AFS tokenizer 并发加载 + 无超时 gather;③双 infer_loop 线程竞态 / SymmMem。**这些都是 `RAY_DEDUP_LOGS=1` 日志折叠假象**(把 per-rank 现象折叠成"1/8 副本到 594"之类假象) | **排查铁律:排 hang 必先 `RAY_DEDUP_LOGS=0`,先数 actor 是否起全,再谈竞态** |
| D2 | 以下修复**保留不回退**(各治各坎、非 hang 本因):`NCCL_CUMEM_ENABLE=0`(P2P Cuda failure 归 0)、模型预热到 `/dev/shm` node-local、`gpu_memory_utilization` 0.65、`disable_symm_mem_allreduce=true`、`running_max_req_size=64` | 已透传生效,保留 |
| D3 | (可选加固,未做,优先级低)verl `llm_server.py:521` init_hybrid gather 无超时,§47 的超时加在 recipe 另一层拦不到;若再缺副本仍会静默 hang | recipe 侧包一层超时暴露缺席 replica_rank(治标安全网) |

## E. 长跑稳定性 / rollout / reward

| # | 问题(现象 + 根因) | 解决办法 |
|---|---|---|
| E1 | 50+ 步后节点 OOM:`GatewayActor` 长跑内存暴涨 = **glibc arena 碎片**(非 session 泄漏),128 核放大 | `LD_PRELOAD` jemalloc + `MALLOC_CONF`,透传进 Ray `runtime_env`(`6b2d574`) |
| E2 | 超时 session 不释放:`except Exception` 抓不到 `CancelledError`(BaseException) | 合并 verl 官方 commit `0d5a988`(无条件 finally + shield abort) |
| E3 | GOAWAY 偶发 `RemoteProtocolError: ConnectionTerminated`:e2b SDK 默认 HTTP/2,腾讯 AGS 网关单连接 ~1000 stream 后回收;`HTTPX_DISABLE_HTTP2` 无效(e2b 不读) | patch e2b 全部 4 条 transport factory 强制 `http2=False` 走 HTTP/1.1(连接层根治);连接池调大(`E2B_MAX_KEEPALIVE=1000/MAX=2000`);worker `_run_session_with_timeout` GOAWAY 重采一次兜底。实测 900 并发:执行耗时(秒级)≫ 建连开销(毫秒级),HTTP/1.1 不亏但 GOAWAY 重采省数十秒 |
| E4 | reward 从 step1 恒 0(两根因):①`SUFY_API_KEY` 没进 Ray worker → `sk-local` → sufy 401 → 静默判 0;②judge=deepseek-v4-flash(thinking),`max_tokens=4096` 被 reasoning 吃光 → 截断 → `judge_error=1` | ①`_passthrough` 加 `SUFY_API_KEY`+`REWARD_*`;②judge `max_tokens` 4096→16384(env 可调)(`4baf17c`) |
| E5 | `b1_4gpu` step54 崩 `AssertionError: agent_assets batch 4 vs 2`:`cl_agent_dataset.py` `if assets:` 条件写 key,gen-batch 混合有/无输入文件的 record → batch 尺寸断言崩 | `__getitem__` **恒写 key**:无文件时 `agent_assets={}`(下游对空值容忍),batch 内每行字段一致 |
| E6 | metrics 空目录(16 卡):`VERL_FILE_LOGGER_PATH` 只 export 到 driver shell,FileLogger 在 CLTaskRunnerV1(Ray worker)实例化不继承 → 多机 fallback 到 `agentic-cl/{exp}.jsonl`(4 卡单机同机侥幸继承故没暴露) | `verl_runner.py` `_passthrough` 加 `VERL_FILE_LOGGER_PATH`(下次重启生效) |
| E7 | **step17 崩(hang 修好后新崩溃)**:纯文本训练混进含图请求打崩 Qwen3.5 多模态 M-RoPE。agent 沙箱工具产出 PNG → Hermes 拼进 chat → gateway 传 image_data → `Qwen35InferStateInfo` 无条件继承 qwen2_vl M-RoPE(`start_idx=None`)→ `RuntimeError: Could not infer dtype of NoneType` → 副本 infer_loop 全崩 → TP 组残缺 → 死锁。**M-RoPE 架构固有,`disable_vision` 管不到**;本项目按设计禁多模态(9 桶去多模态、parquet 无图字段) | 新建 `trainer/gateway_image_drop_patch.py`:patch `GatewayActor._handle_chat_completions`(含图 → 投毒 + 400,图不进 lightllm)+ `SessionManager.finalize_session`(投毒 session 产空轨迹踢出训练);走 `VERL_USE_EXTERNAL_MODULES` 不改 verl;22 config `remote_agent` 加 `all_failed_policy: skip` + `min_group_success_ratio=0.5`。**待上机重启验证越过 step17** |
| E8 | r0 每步 `infer_batch` 崩 `AssertionError: assert key in self.batch.keys()`(no_lora_adapter):verl 0.8.0 v1 `infer_batch`(engine_workers.py:386)无条件 `tu.pop(data,"no_lora_adapter",default=False)`,函数标注 `data: TensorDict`,但 v1 架构(KVBatchMeta+transfer_queue lazy dispatch)下 worker 实际收到 **DataProto**;`tensordict_utils.pop` 里 `tensordict.pop(key,sentinel)` 对 DataProto 变成 `DataProto.pop(batch_keys=...)`(签名不兼容)硬 assert。无 lora(ref_in_actor=False)时该 key 从不在 batch → 本应走 default 返回 False,但 default 语义被踩爆。r0 掺回放行后 batch 从 TensorDict 变 DataProto 才触发(b1 无 replay 不崩) | 新增 `trainer/tensordict_pop_patch.py`:monkey-patch `tensordict_utils.pop`,检测 DataProto 时手动查 `batch`/`non_tensor_batch`/`meta_info` 三处,key 不存在返回 default;走 `VERL_USE_EXTERNAL_MODULES` 不改 verl |
| E9 | reward 恒 0 **新根因**(补充 E4):judge 请求带非法 thinking 字段 → tokenhub luna 400。`trainer/model_reward.py` `_call_once` 给 judge 塞 `thinking`/`enable_thinking`/`reasoning_effort`/`chat_template_kwargs` 4 个字段,其中 3 个是 vLLM/deepseek 私有字段,tokenhub gpt-5.6-luna 走 OpenAI-compatible 不认 → 400 "Unknown parameter" → judge_error=1 → discard=1 → reward 全 0。逐字段隔离实测:`thinking`/`enable_thinking`/`chat_template_kwargs` → 400,`reasoning_effort` → 200 | 删掉 3 个非法字段,只留 `reasoning_effort="medium"`(luna 认);本地实测 judge 恢复正常返回 verdict |

| E10 | r0 每步崩 `AttributeError: 'DataProto' object has no attribute 'keys'/'shape'`(E8 同类根因的完整版):verl 0.8.0 v1 `infer_batch`/`train_batch`/`train_mini_batch`(engine_workers.py)函数体全用 TensorDict API(`data.keys()`:398/344、`data.shape[0]`:244、`tu.assign_non_tensor`:399/345、`tu.make_iterator`:264、`maybe_fix_3d_position_ids`:243),但 r0 掺回放行后 data 实际收到 **DataProto**。E8 只打了 `tu.pop`(386) 一个地鼠,其余各点仍会逐次崩 | 新建 `trainer/dataproto_tensordict_patch.py`:在 `tqbridge`(`verl.utils.transferqueue_utils`)层统一把 `@register` 分发函数的 DataProto 入参 `to_tensordict()` 转 TensorDict,一处覆盖全部崩溃点(含未来新增 @register 函数,不打地鼠);原 E8 的 `tu.pop` 兼容并入本模块作兜底(`_patch_tu_pop`),旧 `tensordict_pop_patch.py` 已删。走 `VERL_USE_EXTERNAL_MODULES` 不改 verl。**待集群重跑 r0 验证:data.keys()/no_lora_adapter 崩溃 0 次、训练过 step2、reward~0.3** |

| E11 | **replay_loss ≈ 2.4e-8(应 O(0.1~1)),几乎为 0**:`replay_token_weights` 在 `weighting._clip_and_normalize` 里**按 batch 内 Σw=1 归一化**(doc `CL_Design.md` §L_replay 权重 "保证 batch 内 Σw 归一";64 条 replay × ~2 万 response token → 每 token 权重 ~7e-7),但 `select_replay_rows` 用 **mean 聚合**(分母 = `mask.sum()` = token 数 ~1e6)。于是把 Σw=1 的权重 mass(~1)再摊到 1e6 个 token → loss = mean(-logπ)·7e-7 ≈ 2e-8,λ₃·loss ≈ 1e-8 对训练**零贡献**(L_replay 实际失效)。非 v1 字段丢失:is_replay 存活(replay_empty=0.92<1)、RL loss/entropy 正常(forward 健康)、mask.sum()>0(loss 非精确 0)。`replay_empty≈0.92` 由 `shuffle:false` 回放行拼 batch 尾部、只落最后 1 个 mini-batch 解释(已知预期,非本 bug) | `trainer/replay_forward.py::select_replay_rows` 分母从 `mask.sum()`(token 数)改为 `Σ(w·mask)`(权重 mass,clamp≥1e-12)→ 得 -logπ 的**加权平均** O(1),与 doc "E[-logπ·w] + Σw=1" 自洽;`token_weights=None` 分支仍用 `mask.sum()`(无权重时退 unweighted mean)。加 `CL_REPLAY_DEBUG=1` 诊断 dump(默认关,`compute_replay_loss`/`select_replay_rows` 各一段:is_replay/mask/log_probs/weights 的 sum·mean·abs_mean·denom),上集群前设 env 确认 Σw≈1 且 log_probs 非零即可定位。单测 `test_select_replay_rows_is_differentiable_and_weighted` 权重改 Σw=1 契约 |

## F. Observer / Reward 审查(反 reward-hacking)

| # | 问题 | 解决办法 |
|---|---|---|
| F1 | observer 探针 `MAX_FILES=200` 截断 → 假 diff;`max_depth=5` 漏深目录;`.` 开头缓存文件/`*.log`/`*.pid`/`node-compile-cache` 进 diff 噪声 | MAX_FILES→500、depth→7、过滤所有 `.` 开头目录+文件、`_is_runtime_file` 加 `*.log`/`*.pid`、SKIP 名单扩充 |
| F2 | observer 报告 `str(ObservationReport)` 难读;judge prompt 无交叉核对指令;轨迹不是完整 JSON | `_format_changes` 三段式(BEFORE/AFTER 内容);rubric 加 `## MANDATORY cross-check` 段;轨迹以完整 OpenAI messages JSON 呈现 |
| F3 | observer 空 report 武断判 0:`rollout/simulated_session.py` `_score_all_slots` 里 `rep is None or rep.is_empty()` 直接 gated 判 0,连 judge 都不调。有些任务不改系统状态(QA/纯对话/只读),observer 无 FS/sys diff,但 agent 可能回答得好 → 整组 GRPO advantage 归零(桶级静默坍缩) | 空 report 时用 `_last_assistant_reply(t.messages)` 判断 agent 是否真有回复:有则构造兜底 report(把回复 fold 进 final)交给 judge;只有真没产出(空轨迹/崩溃/timeout)才判 0 |

## G. 遗留 / 待办(不阻塞)

- k 系列 4 卡非致命噪声:多轮 ReAct 累积 prompt 撑爆 `max_req_total_len=262144`(`3x万 + 65536 > 262144`),单请求被拒非崩溃,训练照推进;会污染被拒轨迹 reward。缩 `max_assistant_turns` 或加轨迹级闸门(未做)。
- Hermes 900s 超时:部分 agent 任务超时,靠 hook partial output 打分兜底(4 卡日志可见 `wall-clock timeout 1000s`,属正常兜底路径)。
- fla/causal-conv1d torch fallback(GDN kernel 慢,CUDA13 编不出);FlashInferAllReduce disabled(无害 warning)。
- E7 的 step17 修复待上机重启验证;C4 的 512 崩溃是否随迁移消失待验证。
- R 系列 buffer 端到端验证;冷启动完整轨迹(`cold_start/train.parquet` 已含 messages 列)等 SFT/replay 阶段启用。
- 冷启动 warmup 数据补 floor(2026-08-18):coding 29→30、research 52→54 各差 1 条,差的都是 `status=error` 空壳轨迹(messages=[],api_calls=0)。897(引用沙箱不存在的 `@web/index.html`)/1354(引用本地代码库)是脏种子,重采必然稳定失败;1419(自包含)可救。方案:从 seed 池 `_archive/queries_cold.jsonl` 换 2 条干净 query 替换 897/1354 + 复用 1419,requery 重采补到 floor。**数据质量不做额外淘汰**——防遗忘目标下 rollout 自然产生的脏数据无所谓,只需达 floor 让 replay 能采到旧桶。
