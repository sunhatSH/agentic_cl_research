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


## §27 max_turns 塞进 verl 严格段崩(同 §21 类,2026-07-28)

**症状:** 真实训练启动即 `TypeError: MultiTurnConfig.__init__() got an unexpected keyword argument 'max_turns'`。

**根因:** 为把 max_turns 16→8,在 config 的 rollout.multi_turn 段加了 `max_turns: 8`。但该段被 verl
解析为 MultiTurnConfig(严格 dataclass),只认 max_assistant_turns/max_user_turns 等,不认自造的
max_turns → 构造时 unexpected keyword 崩。第三次撞"自定义 key 塞进 verl 严格段"(§19 ref.path、
§21 sessions_per_step、§27 max_turns)。

**修复:** 复用 verl 合法字段 max_assistant_turns=8(不再自造 key);代码 cl_rollout_manager 改为优先读
multi_turn.max_assistant_turns、回退 max_turns、再回退 16。共性教训1 再次印证。


## §28 超长序列(319663)撞 assert + SP2→4 + util0.6(2026-07-28~29)

**症状:** 真实训练跑到 step 16(2 小时)崩:
`AssertionError: max_token_len=40960 and max_seq_len=tensor(319663)`,崩在 compute_log_prob。

**根因:** 某次 sandbox tool 输出巨大(cat 大文件 / ls -R / 循环打印),`obs_ids` 一次 extend
就几万 token,整条 response 冲到 31 万。§24 的总长闸门是"当轮后"检查,挡不住"单次 tool
输出就 31 万";且各 step response_length/max 普遍超 8192 闸门(step14 达 32037)。→ 撞 verl
rearrange_micro_batches 的 assert,整个训练 rc=1 崩。

**用户核心要求:** 正式训练【不该被调试 assert 崩】——超过就丢弃,不是 assert。assert 是
调试用的,正式跑该丢弃超长。

**修复(三层截断,让 verl 永远收不到超标序列 → assert 结构上不触发):**
1. collect.py: 单次 tool 输出截断 max_obs_tokens=4096(治巨型 observation)。
2. collect.py: 轨迹返回前【最终硬截断】整条 response 到 max_total_response_tokens(逐轮检查
   的兜底,最后一轮完整保留仍可能略超)。
3. cl_rollout_manager.trajectories_to_dataproto: 进 verl 前【再硬截断】每条 resp_ids/masks/
   logprobs 到 max_response_tokens(=8192),同步截断保持对齐。这是最后一道防线。
单测验证:5万token 巨型 tool 输出 + 319663 超长,两层都截到 8192,verl 收不到超标。

**并行度调整(用户观察训练/推理都 95%):**
- 推理 gpu_memory_utilization 0.7→0.6(降 rollout 阶段 KV 池)。
- 训练 SP(ulysses)2→4 → DP 8→4(单卡序列切4段,训练激活减半)。16头%4=0、train_batch32%DP4=0。


## §29 ppo_kl/clipfrac 全 0:rollout 没请求 logprob(2026-07-29)

**现象:** 首次真实训练 16 step,actor/ppo_kl 和 actor/pg_clipfrac 全程恒 0。日志每步
"256/256 dropping rollout_log_probs → verl recompute old_log_prob"(批级 drop 17 次)。

**根因(源码确认,非推断):** verl lightllm client `generate()` 从 `sampling_params.pop("logprobs",
False)` 读是否返回 logprob(async_lightllm_server.py:220),默认 False。我们自定义 rollout 的
sampling_params 只有 {temperature, max_tokens},没传 logprobs → server 返回 log_probs=None →
generate.py 拿到 [] → 每步空 → cl_rollout_manager 的 _mismatched 检测到 len(logprob)!=len(resp)
→ drop 整批 → verl 用当前策略重算 old_log_prob → old==new → ratio=exp(0)=1 → ppo_kl=0、
clipfrac=0。PPO 的信任域裁剪【实际失效】,退化成 vanilla policy gradient(不崩,reward 仍涨,
但失去 PPO 稳定性保护,是隐患)。
注:config 的 rollout.calculate_log_probs:true 是 verl 训练侧开关,不影响我们自定义 rollout 的请求。

**修复:** cl_rollout_manager sampling_params 加 "logprobs": True。server 里 token_id 与 logprob
同循环 append(async_lightllm_server.py:277-278),长度天然对齐,不会引入 mismatch。零风险。

**审查副产:** 本 session 的三层截断(§28)经单测确认未引入对齐漏洞——所有场景 resp_ids==
response_mask(不等才会崩训练);reward 写 rm_scores[rlen-1] 用截断后 rlen,位置自动跟随不越界。
logprob 错位是本节根因(预先存在),非截断引入。


## §30 大迁移:自写 rollout → verl 原生 agent_loop(recipe_custom, 2026-07-29)

**动机:** §22-§29 一连串崩溃(超长 assert、logprob 缺失致 ppo_kl=0、OOM)的病根都在自写 rollout
(cl_rollout_manager+collect.py)产出的数据质量。verl 原生/recipe_custom 的 agent_loop 全处理好了。

**关键认知(推翻 §20):** Qwen3.5-9B 混合 GatedDeltaNet 结构【能】用 use_remove_padding+
flash_attention_3+dynamic_bsz。verl 已为它实现 GDN 变长(packed)forward
(verl/models/transformers/qwen3_5.py + recipe_custom/models/transformers/qwen3_5.py),GDN 层
用 cu_seqlens+_packed_chunk_gated_delta_rule,full-attn 层走 FA3 varlen。前提是开
VERL_USE_EXTERNAL_MODULES=recipe_custom.bootstrap + model_type=custom_language_model。这就是
参考脚本(debug_rl_qwen35_9b.sh,同为9B)能 max_response=65536 而我们只能 8192 的原因——我们
之前退回了 sdpa+关 remove_padding。

**迁移(RayPPOTrainerV1 桥接,CL 注入不改):**
- VERL_DIR → dependencies/verl(GitLab dev-0.8.0)。
- trainer 换 recipe_custom.ray_trainer_v1.RayPPOTrainerV1(继承标准 RayPPOTrainer,用 DataProto,
  init_workers 把 rollout 换成原生 agent_loop RolloutManager)。CL loss(set_loss_fn)+buffer hook
  (_update_actor patch)照旧。
- 引擎:remove_padding+fused_kernels+flash_attn3+custom_language_model+fsdp2+dynamic_bsz;
  SP=4→DP=4;长度 8196/65536/131072;ppo_max_token=131072/4=32768;rollout_correction.rollout_is=token。
- 沙箱:exps/agent_loop_config.yaml 的 hermes_agent runner(template=agentic-cl-sandbox 自己的镜像);
  项目本就用 hermes(collect.py make_hermes_agent_fn),与 recipe_custom HermesHarness 同源。
- 数据:不转格式。标准 RLHFDataset 消费现有 parquet;default_agent_loop=hermes_agent 兜底 agent_name;
  add_reward_fn.py 给 reward_model 补 reward_fn。
- reward:保留 omni 壳,实际 judge 仍是项目 model_reward.py(端点/模型不变),model_reward_omni.py 适配
  omni 调用约定;reward_fn={"_function_name":"trainer.model_reward_omni.compute_score"}。

**超长处理(禁 assert,§28 诉求):** verl 原生 max_tool_response_length=16384 截工具输出;
remove_padding+dynamic_bsz 变长打包,ppo_max_token 只需 ≥ 单卡最长(131072/4=32768),不撞
seqlen_balancing assert。

**b1 层面 CL 状态:** buffer.enabled=false+lambda_replay=0 → buffer/replay 本就不运行(遗忘下界);
CL loss 走 no_replay 分支。R 系列(buffer.enabled=true)跑前需适配 extract_trajectories_from_batch
——原生 agent_loop non_tensor_batch 无 messages/bucket 字段。

**待集群验证:** 镜像能否被 HermesHarness 驱动;e2b 凭证/template;端到端 ppo_kl 非0/reward有差异/
不 OOM 不 assert。commit f72d3de(阶段1)/1b1a7e4(阶段2)/27d5e45(阶段3)。

## §31 compute_log_prob 崩 triton CE assert:model_type 放错层 → GDN 层丢 cu_seqlens(2026-07-30)

**现象**:4 卡 9B debug(`configs/run/b1_9b_4gpu.yaml`)step 0 的 `_compute_old_log_prob` 崩,4 rank 全崩。
`logs/experiments/qwen35_9b_b1_4gpu/train.log` 栈:
```
sync_trainer.py:275 _compute_old_log_prob
→ engine/fsdp/transformer_impl.py:1451 forward_step → self.module(...)
→ models/transformers/dense_common.py:189 forward_with_triton_backend → linear_cross_entropy(...)
→ utils/kernel/kernels.py:582
   assert hidden.shape[0] == labels.shape[0] and hidden.shape[1] == weight.shape[1]
AssertionError
```

**断的是哪条 assert(先说清,避免误解)**:两个条件——
- `shape[1]` = hidden_size(隐藏层宽度)。模型结构固定,lm_head.weight 的维度,前向怎么走都不变 →
  `hidden.shape[1]==weight.shape[1]` **永远成立**。
- `shape[0]` = token 数量(打平后的序列长度维)。**真正断的是这条**:模型前向吐出的 hidden token 数
  ≠ 引擎按 SP 切好的 labels token 数。
> 即**不是隐藏层维度对不上,而是序列 token 数对不齐**。

**根因(逐行 grep 确认,非猜测)**:`model_type: custom_language_model` 被写进了
`actor_rollout_ref.model.override_config`,应放在 `model:` **顶层**(对齐参考脚本
`recipe_custom/scripts/e2b_agent/debug_rl_qwen35_9b.sh` 的 `+actor_rollout_ref.model.model_type=`)。
两处语义完全不同:

| | 顶层 `model.model_type`(对) | `override_config.model_type`(错) |
|---|---|---|
| 作用 | **引擎选择键**:`engine_workers.py:128 EngineRegistry.new(model_type=...)` 选中 `CustomFSDPEngineWithLMHead`(GDN 变长+路由重放);引擎 `__init__` 再把 hf_config.model_type 复位回 `language_model` | 经 `utils/model.py::update_model_config` 直接改 **HF config.model_type** → `custom_language_model` |
| monkey_patch 分派 | `model.config.model_type`=真实 `qwen3_5` → `monkey_patch.py:270` 命中 → 用 `qwen3_5.py::forward_with_triton_backend` | 匹配不到 `qwen3_5` → 落 `else` 通用分支 `dense_common.py::forward_with_triton_backend` |
| 是否 thread `cu_seqlens` 进模型 | **传**:`if cu_seqlens is not None: kwargs["cu_seqlens"]=cu_seqlens; self.model(input_ids,**kwargs)` | **不传**:`forward_base_model` 签名里根本没有 cu_seqlens(grep 确认 dense_common 全文不含 cu_seqlens) |

**为什么丢 cu_seqlens 就崩(混合结构 + SP 的记账)**:
- SP 切分是引擎层 `prepare_model_inputs` 做的,对 `input_ids`(`:1106`)和 labels
  (`input_ids_rmpad_rolled`,`:1112`)**两个都** `ulysses_pad_and_slice_inputs` 切成 `total_nnz/SP + pad`;
  fused 路径 `:1211` 把已切好的 local labels 作 `shift_labels` 传入。
  → **两条前向路径拿到的 labels 都是 local 长度**,不存在"labels 全长"。
- Qwen3.5 是**混合结构**(GDN 线性注意力层 + full-attn 层):
  - full-attn 层靠 ulysses monkey-patch(all-gather 序列→变长 FA→scatter 回 local),两条路都打了 patch;
  - **GDN 层 `qwen3_5_gated_delta_net_forward`(`qwen3_5.py:192-337`)必须拿到 `cu_seqlens`**
    才能识别"本片 local seq_len = global total_nnz / SP"(`:210-226` 看到 local≠global 就建
    `_build_fla_cp_context` 按 local 产出),做正确的变长 packed + SP 分片记账。
- 通用 `dense_common` 前向不给模型喂 `cu_seqlens` → GDN 层对"变长打包 + SP 分片"记账错乱 →
  产出的 hidden token 数与引擎切好的 local labels 数对不上 → `hidden.shape[0] != labels.shape[0]` 崩。

**日志实证**:
- `train.log:216` `override_config` 里确实含 `model_type: custom_language_model`;
- `train.log:1046` `Using Triton backend ... Qwen3_5ForConditionalGeneration`(patch 生效但走了通用分支);
- `grep "enable_routing_replay in CustomFSDPEngineWithLMHead"` = **0 次** → 证明
  `CustomFSDPEngineWithLMHead` 根本没被选中(退化成 base `FSDPEngineWithLMHead`,路由重放也一并丢失)。
  → **"size 不匹配"与"自定义引擎没生效"是同一个错配的两个后果**。

**修复**:`configs/run/b1_9b_4gpu.yaml` + `b1_9b_16gpu.yaml` 把 `model_type: custom_language_model`
从 `override_config` 提到 `model:` 顶层,`override_config` 只留 `attn_implementation: flash_attention_3`。
`trainer/cl_main.py::load_config` 用 OmegaConf.merge 且 `model_type` 是 `HFModelConfig` 合法字段
(默认 `language_model`),加顶层键无 struct 冲突。改后 `load_config` 校验两份 config:
`model.model_type=custom_language_model`、`override_config={attn_implementation:...}` ✓。

**未实测**:没打印两个 `shape[0]` 的具体数值(差几倍未验),但"断的是 shape[0] 不是 shape[1]"由 assert
定义直接确定(hidden_size 结构上不可变);cu_seqlens 缺失→GDN 记账错→长度不齐这条链来自逐行代码。

**状态**:config 已改已校验;重跑 4 卡 debug 待执行(同代码路径,结论适用 16 卡)。参考 RunLog 2026-07-30 两条(原记录 + 订正)。

### §31 补充:为什么"只改一个配置 key 的位置"就能修(不改代码)

本修复**没动任何代码**,只把 `model_type: custom_language_model` 从 `override_config` 挪到 `model:` 顶层。
为什么挪个位置就修好?因为 `model_type` 这个词在 verl 里被**两套完全独立的机制**读取,放在哪一层决定喂给谁。

**改动 diff:**
```yaml
# ❌ 改之前（崩）
actor_rollout_ref:
  model:
    path: .../Qwen3.5-9B
    override_config:
      attn_implementation: flash_attention_3
      model_type: custom_language_model   # ← 埋在 override_config 里

# ✅ 改之后（对）
actor_rollout_ref:
  model:
    path: .../Qwen3.5-9B
    model_type: custom_language_model      # ← 提到 model 顶层
    override_config:
      attn_implementation: flash_attention_3
```

**两个消费者,取值来源不同:**

① **顶层 `model.model_type` = 引擎选择键(我要的)**
`model:` 顶层对应 verl 的 `HFModelConfig` dataclass,`model_type` 是其合法字段(默认 `language_model`)。
`engine_workers.py:128 EngineRegistry.new(model_type=config.model_type, ...)` 拿它查引擎注册表 →
值为 `custom_language_model` → 选中 `CustomFSDPEngineWithLMHead`(GDN 变长打包 + 路由重放)。
该引擎 `__init__` 关键一步:把 `hf_config.model_type` **复位回 `language_model`**,所以模型"真实身份"
没被污染,后续 monkey_patch 仍认出它是真 `qwen3_5`,走对 SP-aware 前向。

② **`override_config.model_type` = 直接改 HF 模型 config(污染,导致崩)**
`override_config` 是透传字典,`utils/model.py::update_model_config` 把里面每个 key 直接盖到 HuggingFace
model config 对象上。放进 `model_type` 就等于:`HF config.model_type: "qwen3_5" → 被强改成 "custom_language_model"`。
连锁后果:
- `monkey_patch.py:270` 按 `model.config.model_type` 分派前向,读到 `custom_language_model` 匹配不到
  `qwen3_5` → 落通用 `dense_common` 分支(不喂 GDN 层 cu_seqlens)→ 最终 assert 崩。
- 同时顶层 `model.model_type` 仍是默认 `language_model` → 引擎也选错(base 引擎,非自定义引擎)。
→ **一个 key 放错层同时坏两件事:引擎没选对 + 前向函数没选对。**

**比喻:** `model.model_type` 像点菜时告诉服务员"要哪套餐"(选引擎/选前向),厨房用什么食材(模型真实结构
`qwen3_5`)照旧;`override_config.model_type` 像跑进厨房把食材标签改了——把"qwen3_5 这块肉"标签改成
"custom_language_model",厨房照标签用错处理流程(通用前向),混合结构的 GDN 层被错误对待。

**为什么改配置就够、不用改代码:** verl 和参考脚本本就这么设计——`debug_rl_qwen35_9b.sh`(同 9B,已验证跑通)
用的就是顶层 `+actor_rollout_ref.model.model_type=custom_language_model`。之前的 config 只是翻成 yaml 时
放错了嵌套层级。这**不是用配置 workaround 绕 bug,而是把配置改回它本该在的位置**,让框架既定的
引擎选择链 + monkey_patch 分派链正常工作;自定义引擎 + qwen3_5 前向的代码路径本身是好的,只是之前没被走到。

## §32 全量对齐:19 份 9B + 27B/64GPU 迁到新路线(2026-07-30)

§31 修好 b1_9b_{4,16}gpu 后,configs/run 下其余配置仍在旧路线(strategy=fsdp + 自写
cl_rollout_manager.AgentLoopManager + 无 use_v1/transfer_queue/remove_padding),与已迁两份及
启动脚本 `_train_impl.sh`(已导 VERL_USE_EXTERNAL_MODULES=recipe_custom.bootstrap)不自洽。全量对齐:

- **17 份 k*/r*_9b_16gpu**:以 b1_9b_16gpu 为模板用一次性生成器重写,机制层逐字一致,只留各自
  CL/KL 语义(KL 系数、lambda_replay、buffer/weighting、experiment_name)。
- **cluster.yaml**(27B/64GPU 共用引擎 overlay):迁新路线(model_type 顶层 + impl_backend=triton +
  remove_padding + flash_attn3;fsdp2;RemoteAgentLoopManager + custom.remote_agent;omni reward;
  use_v1 custom_sync + transfer_queue)。b1.yaml/r4.yaml 自动继承。Qwen3.6-27B 经 config.json 确认
  同为 qwen3_5 混合 GDN(full_attention_interval=4),§31 修复同样适用。
- **暂不动**:b1_9b.yaml/b1_9b_8gpu.yaml(非启动路径本地测试残留,用户指示);b1_8b.yaml(8B 另模型)。
- **校验**:load_config 全量 21 份全 OK(model_type 顶层 / override_config 只剩 attn / fsdp2 /
  fused+triton / use_v1 / RemoteAgentLoopManager),0 失败。真机重跑待执行。
