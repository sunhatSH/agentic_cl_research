# 求助:verl + LightLLM colocate 训练 update_actor OOM,碎片治理与 torch_memory_saver 互斥

## 一句话问题
16 卡(2 节点 × 8,H800 80GB)colocate 跑 verl 0.8.0 GRPO,rollout 阶段正常,进
`update_actor` 时 CUDA OOM。想用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 治碎片,
但它与 LightLLM 的 `torch_memory_saver` 硬互斥(后者是 colocate 下训练时让推理引擎让出显存的
机制)。**求:在不牺牲 torch_memory_saver 的前提下,怎么根治这个 OOM?**

---

## 环境
- 硬件:H800 80GB × 16(2 节点 × 8)。
- 框架:verl 0.8.0(不 fork,外挂 set_loss_fn + 自定义 AgentLoopManager)+ LightLLM(rollout 后端,async,TP=2)+ FSDP。
- 模型:Qwen3.5-9B,bf16。混合结构:24 层 GatedDeltaNet(线性注意力)+ 8 层 Full Attention。
  因这个混合结构,`use_remove_padding` 未开(依赖 flash-attn varlen,与 GDN 的 causal-conv1d 不兼容)。
  attn_implementation=sdpa,use_fused_kernels=false。
- 部署模式:**colocate**(训练与 rollout 同卡分时复用,靠 torch_memory_saver 让 LightLLM 在训练阶段 sleep 让出显存)。

## 并行 / batch
- 布局:DP=8, SP(ulysses)=2, TP(推理)=2。
- train_batch=64 query × rollout.n=8 = 512 轨迹/step;ppo_mini=64;**ppo_micro_batch_size_per_gpu=1**。
- max_prompt=2048,max_response=53886(长序列,claude opus 轨迹 p90)。
- **ppo_max_token_len_per_gpu=32768**(曾 65536)。
- FSDP:param_offload=true, optimizer_offload=true。
- **enable_gradient_checkpointing=true(verl 默认已开)**,enable_activation_offload=false。
- rollout: gpu_memory_utilization=0.7, enable_torch_memory_saver=true,
  running_max_req_size=512, graph_max_batch_size=512。

## OOM 报错原文(崩在 update_actor)
```
ray::WorkerDict.actor_rollout_update_actor()
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB.
GPU 0 has a total capacity of 79.18 GiB of which 2.50 MiB is free.
  Including non-PyTorch memory, this process has 72.71 GiB memory in use.
  Process 3703 has 6.39 GiB memory in use.
  Of the allocated memory 65.47 GiB is allocated by PyTorch,
  and 147.92 MiB is reserved by PyTorch but unallocated.
If reserved but unallocated memory is large try setting
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
```

显存拼图(单卡 79.18GB):
- 训练进程 72.71GB = PyTorch allocated 65.47 + reserved-unallocated 0.147 + **non-PyTorch ~7.1GB**(CUDA context / NCCL / 可能含同进程 LightLLM 运行时)。
- Process 3703(LightLLM 残留)6.39GB —— 这是 **TP2 下 9B 模型权重常驻**;即便 torch_memory_saver 让它 sleep 让出 KV 池,**权重本身不释放**,`gpu_memory_utilization` 调低也降不掉这 6.39GB。
- free 仅 2.5MB。**训练进程 72.71 + LightLLM 6.39 ≈ 79.1 / 79.18,物理余量仅 ~0.07GB。**

## 我试过 / 判断
1. **micro_batch 2→1**:激活峰值减半,曾从"确定性 OOM"缓解,但长序列下仍触顶。
2. **max_token 65536→32768→24576**:降单次前向 token 峰值。**我认为这是治标**——只是"要得少一点、可能刚好躲过 0.07GB 的缝",没动"两进程挤一卡"的根;且 gradient_checkpointing 已开,激活压缩空间有限。
3. **expandable_segments:True**:唯一直接治碎片的手段,但**当场把训练搞崩**——LightLLM 启动即
   `RuntimeError: TorchMemorySaver is disabled for the current process because expandable_segments is not supported yet`
   (torch_memory_saver/entrypoint.py:140 _sanity_checks 主动 raise)。因 `PYTORCH_CUDA_ALLOC_CONF`
   是进程级全局,colocate 下训练与 LightLLM 共享同一 CUDA 分配器,无法只给训练开、不给 LightLLM 开。
4. 观察:**reserved-but-unallocated 仅 147MB**(不大)→ 说明**主因不是碎片,而是物理快满**,碎片只是压垮的最后一根稻草。

## 具体想请教的
1. **根因判断**:这是"碎片"还是"物理不够(两进程挤一卡余量 0.07GB)"为主?我倾向后者,对吗?
2. **torch_memory_saver 让 KV 池 sleep 后,为何还残留 6.39GB?** 那是模型权重吗?colocate 下能否让 LightLLM
   训练阶段**连权重一起 offload/释放**(sleep level 2 之类),把这 6.39GB 也让出来?LightLLM 有没有这种更彻底的 sleep?
3. **expandable_segments 与 torch_memory_saver 二选一**:
   - 有没有办法让二者共存(某个版本已支持?环境变量粒度控制?)?
   - 若必须二选一:**放弃 torch_memory_saver、改用其他方式给训练腾显存**(比如干脆分离部署:推理独占若干卡、训练独占其余卡,不 colocate)——在 16 卡规模下值得吗?
4. **不碰 CUDA 全局分配器的前提下**,还有哪些手段能实打实降训练峰值到 <72GB:
   - enable_activation_offload=true(激活也 offload 到 CPU)?代价多大?
   - FSDP 的 reshard_after_forward / forward_prefetch / limit_all_gathers 调参?
   - SP 从 2 提到 4/8(更细序列切分,降单卡激活),但 16 卡下 DP 会掉到 4/2?
   - ppo_max_token_len 继续降到 16384?
5. **分离部署 vs colocate**:CL_Design 里正式方案(64 卡)本就是 Fully-Async 分离 40+24。
   16 卡是否也该直接分离(如 12 训练 + 4 推理),从根上消除"两进程挤一卡"?分离后各自独占,
   expandable_segments 也能自由开。代价是卡数少时利用率下降——16 卡分离划算吗?

## 附:相关文件
- 配置:configs/run/b1_9b_16gpu.yaml、configs/base.yaml
- 显存核算:doc/ops/9B_16GPU_Config_2026-07-24.md
- 完整崩溃史:doc/debug/16GPU_Training_Debug_2026-07-24.md(§20 首次 OOM、§22 本次碎片/memsaver 互斥)
