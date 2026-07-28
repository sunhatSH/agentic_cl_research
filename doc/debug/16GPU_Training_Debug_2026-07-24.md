# 16GPU 训练启动排障记录

2026-07-24，B1 baseline（Qwen3.5-9B，16 卡），8 次提交才跑通。记录遇到的问题和解决方案。

## 故障总览

| # | 症状 | 根因 | 修复 |
|---|------|------|------|
| 1 | `rank parameter missing` | rendezvous URL 缺参数 | 加 `?rank=X&world_size=Y` |
| 2 | `available 8 < desired 16` | Ray head 竞态 | head 先启再做 barrier |
| 3 | `Unable to perform re-rendezvous` | torch 新版拒重复 call | 换 `init_process_group` |
| 4 | `available 8 < desired 16`（又） | verl `address: local` 无视集群 | 改 `auto` |
| 5 | swanlab KeyFileError + E2B 缺 key | 目录重组路径断裂 | `..` → `../..` + `BASH_SOURCE` |
| 6 | swanlab 认证失败崩训练 | key 未加载但 config 含 swanlab | CLI 覆盖兜底 |

---

## 1. torch.distributed.rendezvous URL 缺参数

**症状：** 副节点启动时报 `ValueError: rank parameter missing`，容器退出

**根因：** `_train_impl.sh:185` 的 python one-liner 只传 `tcp://host:port`，PyTorch 2.x 的 `_tcp_rendezvous_handler` 需要 `?rank=X&world_size=Y` 参数。同时脚本头是 `set -uo pipefail`（缺 `-e`），失败被静默吞掉，两个节点跳过 barrier 各跑各的。

**修复：** URL 加 query 参数 + `|| exit 1` 显式中止

---

## 2. Ray head 启动竞态

**症状：** `available GPUs 8 < desired 16`，副节点容器被 kill

**根因：**
```
barrier 释放 → [主] ray start --head（耗时 3-5s）
             → [副] ray start --address（立即，head 未就绪）
```

**修复：** 主节点先启动 Ray head，再做 barrier，最后副节点连接。barrier 释放时 head 已监听 6379。

---

## 3. torch.distributed.rendezvous "re-rendezvous" 错误

**症状：** `RuntimeError: Unable to perform re-rendezvous using tcp:// method`，主副节点同时失败

**根因：** PyTorch 新版 `rendezvous()` 内部用全局 dict 缓存 TCPStore。同地址的后续调用直接抛异常。

**修复：** 弃用 `dist.rendezvous(url)`，改用：
```python
dist.init_process_group('gloo', init_method=f'tcp://{addr}:{port}', rank=rank, world_size=ws)
dist.barrier()
dist.destroy_process_group()
```

---

## 4. verl 无视已有 Ray 集群

**症状：** 多机同步成功、FSDP 16 路联通，但 verl 仍报 `available 8 < desired 16`

**根因：** `b1_9b_16gpu.yaml` 中 `ray_init.address: local` 让 `ray.init()` 每次都新建本地 Ray 实例，无视 `ray start --head` + `ray start --address` 搭好的多机集群。

**修复：** `address: local` → `address: auto`

---

## 5. 脚本目录重组路径断裂

**背景：** 将 `scripts/` 下 64 个文件分类到 7 个子目录（env/sandbox/collect/data/pipeline/analysis/serve）

**症状：**
- swanlab 报 `api key not configured`
- E2B 报 `E2B_API_KEY and E2B_DOMAIN must be set`

**根因：** `load_training_env.sh` 和 `load_tencent_env.sh` 用 `dirname $0/..` 解析项目根。从 `scripts/` 下移到 `scripts/env/` 后，`..` 从项目根跳成了 `scripts/`，读不到 `.env` 和 `docker/sandbox/tencent.env`。

**修复：**
- `..` → `../..`（13 个脚本）
- `load_tencent_env.sh` 额外：`$0` → `BASH_SOURCE[0]`（被 source 时 `$0` 是调用方路径）

---

## 6. SwanLab 认证失败崩训练

**症状：** `swanlab.error.KeyFileError`，训练退出

**根因：** #5 导致 key 未加载，但 config 里仍配了 `logger: [console, swanlab]`。swanlab 初始化时调 API 认证，失败不降级直接崩。

**修复：** `_run_single` 加保护——key 缺失时 CLI 注入 `trainer.logger=[console]` 覆盖 yaml，不崩训练。

---

## 系统改进

### AFS 日志全覆盖
```bash
exec > >(tee -a "$_LOGDIR/train.log") 2>&1
```
所有 shell 阶段输出落盘，不再依赖拿不到的容器 stdout。日志头带 rank：
```
[train_cl] === 2026-07-24T13:03:53Z host=xxx-master-0 rank=0/2 pid=1 ===
```

### 自动续训
`_run_single` 启动前检测 `ckpts/<exp>/global_step_*`，存在则自动 `--resume-from`。

### 脚本精简
- 删 21 个冗余文件（phase dirs、train_*gpu wrappers、旧 experiments、启动脚本）
- `train.sh` 统一入口：`--config`（单实验）、`--phase N`（批）、`--all`
- 多机同步从一行 python one-liner → 三段式 heredoc

---

## 最终结果

`bash scripts/train.sh 16gpu --config configs/run/b1_9b_16gpu.yaml`

16 卡训练跑通：FSDP 联通、LightLLM rollout 正常、SwanLab 上报就绪。

---

# 第二阶段：启动跑通之后的崩溃排查（2026-07-27 ~ 07-28）

07-24 解决的是「多机能不能启动」；启动之后 B1(baseline)/K1-K3(KL) 反复失败，暴露了 rollout 契约、reward 链、verl 严格 config 解析、数值健壮性、显存等一连串更深的问题。多数是逐个撞出来的，逐条源码核实后修复。**共性教训见文末。**

## 故障总览（第二阶段）

| # | 症状 | 根因 | 修复 | commit |
|---|------|------|------|--------|
| 7 | 训练常崩 + reward 全 0 静默 | 见 f30652b 一组隐藏 bug | 见下 §7 | f30652b |
| 8 | 验证阶段 `too many dimensions 'str'` | chat_template 返回文本非 ids | _safe_tokenize str→encode + int 校验 | 95c6907 |
| 9 | **rollout ×8 契约冲突（根因级）** | verl 已 ×n,我们每行又跑 8-slot=×n² | per-row 每行 1 条 rollout | cc0de24 |
| 10 | `Unknown reward manager: cl_observer` | reward 跑独立进程,@register 不生效 | source: importlib 直接加载 | 3502939 |
| 11 | `KeyError: 'timing'` | 返回 DataProto 缺 meta_info["timing"] | 补空 timing dict | 5042f71 |
| 12 | `KeyError: 'multi_modal_inputs'` | 纯文本 rollout 未设该 non_tensor | 补每行空 dict | 0f696f7 |
| 13 | 空 messages `IndexError` + 假成功 | 失败 slot 空轨迹 + 脚本不看退出码 | 空则回退 query;脚本查 rc | aa9ede3 |
| 14 | `NoneType has no len()` | val_files=null,verl 强建 val | 空/缺失→别名 train_files | f6e4daa |
| 15 | 一个坏沙箱崩整步 | crashed session 返回 [] 少一行 | 返回占位轨迹保行数 | 01591a1 |
| 16 | **reward 阶段 `KeyError: rm_scores`（根因级）** | 自定义 rollout 从不写 rm_scores | 内联 t.reward 写入 rm_scores | b3bb485 |
| 17 | NaN 传进 verl loss（潜在） | clamp/logprob 不挡 NaN | math.isnan/isfinite 检测 | 56d2ef1,21d0a58 |
| 18 | `FileNotFoundError: val.parquet` | 兜底只挡空、不挡缺失文件 | 扩到"文件不存在"+全配置 null | cdeaa59 |
| 19 | 开 KL 实验 `FSDPActorConfig got 'path'` | ref 段残留 path/model 键 | 删 ref.path/ref.model.path | 76c61d5 |
| 20 | 训练 OOM（update_actor） | 长序列+micro=2 激活峰值 ~76GB | micro 2→1 + max_token 减半 | 96a5b5d |
| 21 | `AgentLoopConfig got 'sessions_per_step'` | 自定义 key 塞进 verl 严格段 | 改用合法字段 num_workers | 5dda210 |

> 另有 lr/并发/util 的调优（非崩溃）：lr 加 warmup 定 2e-6（b35987c,63f00f8）；rollout 并发 64→256→512（778e07b,96a5b5d,5dda210 via num_workers）；gpu_mem_util 0.4→0.55→0.6→0.7（96a5b5d,9a78134）。

---

## §9 rollout ×8 契约冲突（最根本，导致 6 次 0-checkpoint）

**症状：** 6 次启动全崩、0 checkpoint。修完前置崩点后必然撞此。

**根因：** verl 0.8.0 rollout 契约 = **verl 自己按 `rollout.n=8` 复制 gen_batch（interleave）再交给 `generate_sequences`，并期望「进多少行返多少行」**（union 断言等行数）；GRPO 分组由 verl 自己的 uid（数据集 uid×8）完成。旧代码对**每个输入行**又跑 8-slot pool → verl×8 + 我们×8 = **×64**，行数彻底对不上必崩。

**修复：** `generate_sequences` 改为**每输入行 1 条 rollout**（`slots=1`），原序返回等行数。8 路 GRPO 依然成立——只是「分组」从 session 内改由 verl uid 做。并发 = num_workers（每 session 1 条轨迹）。

**衍生（同一改造顺带修）：** 不发 uid（发了 union 冲突）；空 messages 回退 query；crashed session 返回占位轨迹保行数；补 meta_info["timing"] / multi_modal_inputs 空 dict（verl fit 无条件读的字段）。

---

## §16 reward 从头没接上（rm_scores）

**症状：** rollout 采样成功后，reward 阶段 `KeyError: 'rm_scores'`。

**根因：** verl `use_rm=false`（judge 外部 serve）时**假定 rollout 已把 reward 写进 `batch["rm_scores"]`**（默认 AgentLoopManager 在 _postprocess 写），直接 `extract_reward(batch["rm_scores"])`。我们自定义 rollout 替换了整条路径、**从不写 rm_scores** → 崩。且发现 observer→ObserverRewardManager→compute_score 那条 verl reward-manager 路径**是死代码**（reward 走 rm_scores，不调 manager）。

**修复：** reward 其实已在 rollout 内联算好（`_score_all_slots` → `score_followup`（observer diff + judge）→ `t.reward`）；`trajectories_to_dataproto` 把 t.reward 写进 `rm_scores[B,R]`（最后有效 response token 处，照 verl agent_loop.py:933 放法）。observer 仍只观察不打分，但其取证经此喂进训练 judge。

---

## §19 / §21 verl 严格 dataclass 拒绝自定义 config 键（同一类，撞了两次）

verl 用严格 dataclass 解析 config 各段（actor→FSDPActorConfig、agent→AgentLoopConfig 等），**段里多一个它不认的 key 就 `TypeError: __init__() got unexpected keyword`**。撞过两次：

- **§19**：base.yaml 残留扁平 `ref.path: 27B`（旧基座）+ run config `ref.model.path` → 开 KL 的实验建 ref policy 时 FSDPActorConfig 收到非法 `path` 崩（baseline 无 KL 不建 ref 故不崩）。修：删 ref.path/ref.model.path，ref 用共享 `actor_rollout_ref.model.path`。
- **§21**：把 rollout 并发写成 `agent.sessions_per_step`（我方自定义 key）→ AgentLoopConfig 崩。修：改用合法字段 `agent.num_workers`，代码读它。

---

## §20 训练 OOM（update_actor）

**症状：** rollout 成功、进 `_update_actor → update_actor` 时 `CUDA OOM`，训练进程单卡 ~63GB + 要 13GB → 爆 80GB。**崩在训练阶段不是 rollout。**

**根因：** 长序列（max_response 53886）+ micro=2 的激活峰值 + FSDP all-gather + log_prob 计算。（注：`param/optimizer_offload` 已开，训练态 offload 到 CPU；但激活值不 offload，是长序列大头。）

**修复：** `ppo_micro_batch_size_per_gpu 2→1`（激活峰值减半）+ `ppo_max_token_len_per_gpu 65536→32768`。**未开 use_remove_padding**：它依赖 flash-attn varlen，与本模型 sdpa + 混合 GatedDeltaNet 结构不兼容（GDN 的 causal-conv1d/flash-linear-attn 走 torch fallback），风险高。

**显存分时复用（重要背景）：** `enable_torch_memory_saver=true` 让 verl 在 rollout 完成后 `sleep_replicas()` 让 lightllm 让出显存（残留 ~4-7GB），训练阶段近乎独占整卡。所以 `gpu_memory_utilization`（0.7）只是 rollout 阶段 lightllm 的 KV 池上限，不与训练争显存——这是能把 util 提到 0.7 而训练不 OOM 的前提。

---

## 共性教训（第二阶段最该记住的）

1. **verl 严格 dataclass**：凡我方自定义 config 参数，**不能塞进 verl 严格解析的段**（actor/ref/agent/model...），否则 `unexpected keyword` 崩。复用 verl 合法字段（如 num_workers），或放在 verl 不解析的位置。撞了 3 次（ref.path、sessions_per_step、以及 extra_info 那类）。
2. **verl rollout 契约**：verl 自己按 n 复制 gen_batch，自定义 `generate_sequences` 必须「进多少行返多少行」、且返回 verl fit 无条件读的字段（rm_scores/timing/multi_modal_inputs）。GRPO 分组交给 verl uid，不要在 session 内自己 ×n。
3. **NaN 绕过一切比较**：`nan<0`/`nan>1`/`nan==0` 全 False → clamp 和零保护都挡不住。外部来源（judge/推理/GPU 数值）进张量/概率运算前必须 `math.isfinite` 显式检测。
4. **脚本要如实报退出码**：崩溃被脚本打成「训练结束」+rc 0 会持续误导（`aa9ede3` 修）。任何"成功"都要能对上退出码。
5. **显存分层看**：rollout 阶段(lightllm) vs 训练阶段(FSDP)分时复用，OOM 要看是哪个阶段；激活值(不 offload)是长序列训练的真正大头。


## §21 后续修正:并发参数落位 num_workers → cl.rollout.sessions_per_step（语义正确）

§21 初版把并发借用 verl 合法字段 `agent.num_workers`——**能跑但语义不对**:num_workers
在 verl 是"创建几个 AgentLoopWorker Ray actor"(agent_loop.py:1048),我们重写了
generate_sequences、根本不建那些 actor,只是借它的数值当线程并发数;且 num_workers 还是
验证路径的 pad divisor(ray_trainer.py:592),512 会让验证 batch pad 到 512 倍(验证虽关
但是潜在雷)。

**最终修法(语义干净)**:并发参数放进 `cl:` 顶层段(verl 从不 .get() cl 段,零校验、零副作用),
用回正确的字段名 `cl.rollout.sessions_per_step`。manager 的 `self.config` 是全量 config
(agent_loop.py:217),故 `self.config.cl.rollout.sessions_per_step` 可达。agent 段恢复干净
(无自定义 key,num_workers 回默认)。这印证共性教训 1 的正解:**自定义参数放 verl 不解析的
顶层段(cl:),而非借用 verl 段的字段**。


## §22 update_actor 边界碎片 OOM(2026-07-28,512 并发 + micro=1 下仍崩)

**症状:** 05:37 启动的 baseline,穿过 rollout(内存 60% / 显存 85%,健康),进 `update_actor`
时 `CUDA OOM: Tried to allocate 2.00 MiB. GPU 0 total 79.18 GiB, of which 2.50 MiB is free`。
崩在 `actor_rollout_update_actor()`,训练阶段,rc=1。

**根因(与 §20 的"绝对不够"不同,这次是"够但碎"):**
- PyTorch 已 allocated 65.47GB、**reserved-but-unallocated 仅 147MB** → 想分 2MB 都分不出,典型碎片化。
- 显存拼图:训练进程 72.71GB + lightllm 残留进程(Process 3703)6.39GB = **79.1GB / 79.18GB,余量仅 0.07GB**。
  lightllm 那 6.39GB 是 **TP2 下 9B 模型权重常驻**——即使 `enable_torch_memory_saver` 让它 sleep 让出
  KV 池,**权重本身不释放**,`gpu_memory_utilization` 调低也降不掉这块。所以训练态可用显存被压到极限,
  任何碎片都会触发边界 OOM。

**修复:** `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(`scripts/_train_impl.sh`,PyTorch
OOM 报错本身即建议)。可扩展段分配器回收 reserved 碎片,消除"总量够、分不出小块"的边界 OOM。
`${VAR:-default}` 形式保留可被外部覆盖。**后备**(若仍 OOM):`ppo_max_token_len_per_gpu` 32768→24576。

**❌ 修复反转(2026-07-28,同日):** 上面的 `expandable_segments` 方案**当场把训练搞崩了**,已撤销。
lightllm 启动即报 `RuntimeError: TorchMemorySaver is disabled for the current process because
expandable_segments is not supported yet` → lightllm 起不来 → 整训练 rc=1。
根因:`PYTORCH_CUDA_ALLOC_CONF` 是**进程级全局**,会连带影响同机的 lightllm 进程;而 lightllm 依赖
`torch_memory_saver`(训练时 sleep 让出 ~6GB 显存的机制,是 util 0.7 不 OOM 的前提),二者**互斥**。
不能为治训练侧碎片而牺牲推理侧的 memory saver。
**正解(改用不碰 lightllm 的手段):** 降 `ppo_max_token_len_per_gpu` 32768→24576,减小单卡前向激活
峰值腾出那 0.07GB 边界余量——纯训练侧参数,不影响 lightllm。`_train_impl.sh` 里明确注释禁用
expandable_segments。**教训**:全局 CUDA 分配器 env 会波及 colocate 的推理引擎,改它前先想清楚同机
还有谁在用 CUDA;能用局部(config 字段)解决的别动全局 env。

**教训补充:** OOM 要分清"绝对不够"(§20,靠降 micro/max_token)vs"够但碎"(§22,靠分配器策略)。
报错里 `free` 极小 + `reserved-but-unallocated` 极小 + 需求极小(2MB)= 碎片,不是缺量;此时降 batch
收效甚微,该换分配器。另:lightllm 残留权重(TP shard)是训练态一块拿不掉的固定占用,做显存核算要算进去。


## §23 max_response_length 从 53886→16384 + k3 大块 OOM(2026-07-28)

**触发:** k3(kl_strong)07:12 启动,穿过 rollout,进 `update_actor` OOM,rc=1。

**k3 OOM 画像(与 b1 §22 的碎片型不同,是"大块需求型"):**
```
Tried to allocate 11.31 GiB. GPU 0 total 79.32, of which 11.28 GiB free.
Process(训练) 61.25GB;Process(lightllm) 6.70GB;PyTorch allocated 53.22GB,reserved 169MB。
```
- 想一次分配 **11.31GB**,而 free 只有 **11.28GB** —— 差一点点、真·大块不够,不是碎片(reserved 才 169MB)。
- k3 有 KL → 建 ref policy(π_ref),比 b1 多一份 ref 前向 + all-gather,显存需求更大,更早触顶。
- 那个 11.31GB 大块与序列长度强相关(激活 ∝ seq_len)。

**根因(更上游):`max_response_length=53886`(≈54K)设错了。**
- train.parquet 是 RL 数据,**只有 prompt、无 response**(response 是 rollout 在线生成)。故 53886 不是
  训练数据实测值,号称"claude opus 轨迹 p90",来源不明。
- 实测手上冷启动轨迹(rollouts/cold_start*.jsonl):response **p90≈2.5-7K token、max≈9K**;
  早期 RunLog 亦写过"max_response 起步 8K / 8192"。**54K 严重偏大(6-8×)。**
- 54K 作为生成 max_tokens 上限 → 放任 temp=1.0 早期 RL 的重复退化跑飞到几万 token → 长序列激活
  炸显存,正是 §20/§22/§23 一系列 OOM 的上游推手。

**修复:** 18 个 `configs/run/*_16gpu.yaml` 的 `max_response_length` 统一 53886→**16384**。
- 16K 覆盖真实 max≈9K 有近一倍余量,**真实分布零截断**(不影响训练 loss);
- 只砍跑飞的超长生成(本就该砍),是"防跑飞保险";
- 对大块型 OOM(k3):若崩溃大块来自长序列激活,cap 降到 16K 使该块需求缩到约 1/3,11.28GB 余量够。

**关于"降 cap 是否治 OOM"的准确说法:** 训练张量宽度 `R=max(batch 实际生成长度)`(动态,见
cl_rollout_manager.py:136),**不按 cap 静态预留**。故:模型实际不超长时,cap 高低不直接影响激活;
只有当模型确实生成到接近 cap 的超长序列时,降 cap 才直接削峰。因此 16K 对 b1(碎片型)是"保险",
对 k3(大块型、且大块疑似长序列激活)更可能是"直接解药"。真实效果需集群重跑验证。

一、size/长度类配置全盘点(b1_9b_16gpu.yaml)

  配置: max_prompt_length
  值: 2048 
  限制什么: 单条 prompt 长度
  超了怎么处理: 丢弃(filter_overlong_prompts=true,整条剔除不训)      
  ────────────────────────────────────────
  配置: max_response_length
  值: 16384
  限制什么: ①单次生成上限(sampling max_tokens) ②rollout 整条 response 总长闸门(我新加)
  超了怎么处理: ①单次截断停止生成 ②整条截断(当轮后停止后续轮)
  ────────────────────────────────────────
  配置: max_total_response_tokens(代码,=max_response_length)
  值: 16384
  限制什么: 多轮 ReAct 整条累积 response
  超了怎么处理: 截断(停后续轮,当轮完整保留)
  ────────────────────────────────────────
  配置: ppo_max_token_len_per_gpu
  值: 40960
  限制什么: dynamic_bsz 每 micro-batch token 打包上限
  超了怎么处理: 不截不丢——超了 assert 崩(须≥最坏序列,已保证)
  ────────────────────────────────────────
  配置: log_prob_max_token_len(=ppo×2)
  值: 81920
  限制什么: log_prob 阶段打包上限
  超了怎么处理: 同上
  ────────────────────────────────────────
  配置: max_model_len
  值: 65536
  限制什么: lightllm 引擎单请求上下文上限
  超了怎么处理: 引擎层截断
  ────────────────────────────────────────
  配置: max_num_batched_tokens / batch_max_tokens
  值: 8192
  限制什么: lightllm 单批 prefill token 数
  超了怎么处理: 引擎调度(分批,不丢)
  ────────────────────────────────────────
  配置: running_max_req_size / graph_max_batch_size
  值: 512
  限制什么: 推理并发请求数
  超了怎么处理: 排队(不丢)
  ────────────────────────────────────────
  配置: train_batch_size / gen_batch_size
  值: 64
  限制什么: 每 step query 数
  超了怎么处理: 无所谓超
  ────────────────────────────────────────
  配置: ppo_mini_batch_size
  值: 64
  限制什么: 每次参数更新的样本数
  超了怎么处理: —
  ────────────────────────────────────────
  配置: sessions_per_step
  值: 512
  限制什么: rollout 并发轨迹数
  超了怎么处理: —

  截断 vs 丢弃小结:
  - 丢弃:只有 max_prompt_length(超长 prompt 整条剔除)。
  - 截断:max_response_length(单次+整条)、max_model_len(引擎)。
  - 崩(assert):ppo/log_prob_max_token_len 若 < 最坏序列。已设 40960 ≥ 34816,安全。

## §25 update_actor OOM(dynamic_bsz 按 40960 塞满 micro-batch,2026-07-28)

**症状:** §24 的 assert 修好后(dynamic_bsz+16K+40960+总长闸门),双机 b1 穿过打包,进
`actor_rollout_update_actor()` CUDA OOM。训练进程吃 72.54GB(PyTorch allocated 70.04)+
lightllm 6.70GB ≈ 79.24/79.32,爆卡。

**根因:** `ppo_max_token_len_per_gpu=40960` 太大。dynamic_bsz 下它=单 micro-batch token 上限,
打包时真把 micro 塞到接近 40960 token → 单次前向激活 ~10.7GB → 加模型/梯度/lightllm 残留爆 80GB。
(先前误判"40960 是天花板不吃满显存"——错,dynamic_bsz 会尽量塞满预算。)

**关键认知(用户指出):** 不该"抬预算迁就最坏序列 34816",而应"压最坏序列本身,让小预算够用"。
verl 的 rearrange_micro_batches 把整条序列作不可分割单位打包(assert max_token_len>=max_seq_len),
原生不支持把长序列切到多 micro。故正解=降最坏序列,不是绕 assert。34816=prompt2048+闸门16384+
末轮16384,三者都是自设上限,压它们即可。

**修复:** max_response_length 16384→8192(同时管单次生成+整条闸门)+ ppo_max_token_len 40960→20480。
新最坏=2048+8192+8192=18432<20480,不撞 assert;单 micro 激活 10.7GB→5.4GB。真实 response p90
才 2.5-7K,8192 覆盖绝大多数不损训练。**预算 20480 < 原最坏 34816 仍正常训练——靠压最坏序列本身。**


## §26 单次生成与整条闸门解耦(保 response 真实,2026-07-28)

**需求(用户):** response 长度尽量不砍以保数据真实,用别的手段省显存,不降 DP。

**分析:** 降并发对训练激活无效(只影响 rollout 阶段);"单条多段"verl dynamic_bsz 原生
做不到(整条序列必进一个 micro)。真正杠杆是 SP(会降 DP)或解耦单次/整条。

**方案(解耦):** max_response_length 一值原本同时管【单次生成】和【整条闸门】。拆开:
- 整条闸门 = data.max_response_length = 16384(数据保真,整条可到 16K)
- 单次生成 = cl.rollout.max_single_gen_tokens = 8192(实测单次 p90 才 2.5-7K,够)
- 最坏序列 = prompt2048 + 闸门16384 + 末轮单次8192 = 26624(而非全16384的34816)
- ppo_max_token_len = 28672(≥26624)。SP=2/DP=8 不降。

**代码:** cl_rollout_manager 加 _max_single_gen_tokens()(读 cl.rollout.max_single_gen_tokens,
缺省回退 response_length),单次生成 max_tokens 用它;整条闸门仍用 _max_total_response_tokens()。

**显存:** budget 20480→28672(+40%),step1 实测 53.6GB → 估 ~65-70GB,需真实验证。
