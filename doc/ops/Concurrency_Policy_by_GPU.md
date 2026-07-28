# 按 GPU 规模的并发/显存策略表(与训练逻辑解耦)

> **定位**:这是一张**压力参数查找表**,按集群 GPU 规模索引。
> 训练**逻辑/语义**(实验是 B1 还是 K2、λ、桶、lr、batch 数学)在 `configs/run/*.yaml` 和
> `configs/base.yaml`;**压力/吞吐参数**(rollout 并发、KV 池、显存 util、微批、序列上限)在这里定义。
> 换 GPU 规模时**只查这张表填 config 的压力字段,不碰训练逻辑**。
>
> 为什么解耦:并发/显存是**硬件规模的函数**(卡数、单卡显存、TP/DP/SP 布局),随集群变;
> 而"训哪个实验、用什么 CL 参数"是**科研逻辑**,与硬件无关。混在一起 → 换机器要改一堆
> 语义 config、容易改错(见 debug doc 反复撞的 config 事故)。分开后:换规模 = 查表改 5 个数。

---

## 参数职责划分(哪些归这张表管)

| 类别 | 字段 | 归属 |
|------|------|------|
| **压力/吞吐(本表管)** | `cl.rollout.sessions_per_step`(rollout 并发) | 本表 |
| | `rollout.gpu_memory_utilization`(KV 池上限) | 本表 |
| | `engine_kwargs.lightllm.running_max_req_size` / `graph_max_batch_size` | 本表 |
| | `actor.ppo_micro_batch_size_per_gpu`(训练微批) | 本表 |
| | `actor.ppo_max_token_len_per_gpu`(单卡前向 token 上限) | 本表 |
| | `PYTORCH_CUDA_ALLOC_CONF`(碎片治理,env) | 本表 |
| **并行布局(本表管)** | `nnodes` / `n_gpus_per_node` / TP / SP / DP | 本表 |
| **训练逻辑(不在本表)** | 实验语义:`cl.lambda_replay` / `buffer.*` / `use_kl_loss` / `kl_loss_coef` | `configs/run/*.yaml` |
| | 优化:`lr` / warmup / scheduler | `configs/base.yaml` + run |
| | Batch 数学:`train_batch_size` / `ppo_mini_batch_size` / `rollout.n` | run(随规模微调但属逻辑) |
| | 数据 / 桶 / 权重方案 | `base.yaml` |

---

## 策略表

### 16 GPU(2 节点 × 8 卡,H800 80GB)—— **当前生产配置**

| 参数 | 值 | 依据 |
|------|-----|------|
| 并行布局 | DP=8, SP=2, TP=2 | 16÷SP2=DP8;推理 TP=2 |
| **sessions_per_step(rollout 并发)** | **512** | =train_batch64×n8,一次采完一 step;实测 rollout 显存 85%、内存 60%,有余量且稳 |
| **gpu_memory_utilization** | **0.7** | rollout 阶段 lightllm KV 池上限;训练时 lightllm sleep 让出,不与训练争 |
| running_max_req_size | 512 | lightllm 一次并发上限,须 ≥ sessions_per_step |
| graph_max_batch_size | 512 | 配合 512 并发 |
| **ppo_micro_batch_size_per_gpu** | **1** | micro=2 时 update_actor 激活峰值 ~76GB → OOM;降 1 减半 |
| **ppo_max_token_len_per_gpu** | **32768** | 65536→32768,进一步压激活 |
| max_response_length | 53886 | claude opus 轨迹 p90(逻辑/数据决定,列此仅供显存核算) |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True | 治边界碎片 OOM(想分 2MB 却只剩 2.5MB) |

**压测记录(2026-07-28)**:
- rollout 阶段:内存 60% / 显存 85% → 健康,并发 512 不必再提(85% 已有余量,再提风险大于收益)。
- 训练阶段:512 并发下 rollout 侧稳;瓶颈在 **update_actor 显存**——micro=1 + max_token 32768 + expandable_segments 三管齐下治 OOM。
- 边界:训练进程 72.71GB + lightllm 残留 6.39GB(TP2 模型权重常驻,util 调低也降不掉)≈ 79.1GB / 79.18GB,**余量仅 0.07GB**。故碎片治理是关键;若仍 OOM,后备:max_token 32768→24576。

### 64 GPU(8 节点 × 8 卡,H800/H100 80GB)—— **待压测,占位**

> 上集群后按实测填。以下为**推断起点**,非定论。规模上去后 DP 变大、单卡 shard 更小、
> 激活占比相对升高;rollout 并发可随总卡数线性放大,但受 sandbox 后端(e2b)吞吐 + judge
> serve QPS 约束,不一定线性。

| 参数 | 推断起点 | 待验证点 |
|------|---------|---------|
| 并行布局 | DP=32, SP=2, TP=2(或 fully-async 40+24 分离) | colocate 64 vs 分离 40+24,见 CL_Design §GPU |
| sessions_per_step | 从 512 起,按 sandbox/judge 吞吐上调 | e2b 后端并发上限 + judge QPS 才是真瓶颈,不是 GPU |
| gpu_memory_utilization | 0.7 起 | shard 更小 → 训练余量更大,或可提 util |
| ppo_micro_batch_size_per_gpu | 先 1,shard 变小后试 2 | DP=32 时单卡 param shard 更小,激活余量可能够 micro=2 |
| ppo_max_token_len_per_gpu | 先 32768,余量够再回 65536 | |

**64 卡压测 checklist(上集群做)**:
1. 先跑 1 step,看 rollout 显存/内存占比 → 定 util 与并发上限。
2. 看 update_actor 峰值显存 → 定 micro 与 max_token。
3. 测 sandbox(e2b)并发吞吐 + judge serve QPS → 这才是 rollout 并发的真上限。
4. 结果回填本表 64 GPU 行,把"推断起点"改成"实测值"。

---

## 使用方式(换规模时的操作)

1. 查本表对应 GPU 规模行,拿到压力参数。
2. 填进 `configs/run/<exp>_<规模>.yaml` 的对应字段(见上"参数职责划分"的字段名)。
3. env 层参数(`PYTORCH_CUDA_ALLOC_CONF`)已在 `scripts/_train_impl.sh`,一般不用动。
4. **训练逻辑字段(λ/buffer/kl/lr/batch 语义)照抄实验定义,不随规模变。**

> 相关文档:硬件显存核算 `doc/ops/9B_16GPU_Config_2026-07-24.md`;
> 崩溃排查 `doc/debug/16GPU_Training_Debug_2026-07-24.md`;
> GPU 部署/精度 `doc/source/CL_Design.md` §GPU 资源分配。
