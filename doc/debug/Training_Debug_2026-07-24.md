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

## §33 50+ 步后节点 OOM + GOAWAY + verl 净化(2026-07-31)

§32 全量迁到新路线后，`qwen35_9b_b1_4gpu` 首次长跑（4 卡 debug）跑到 **step 53 节点 512GB 内存打满 OOM 崩溃**。排查发现 3 个独立问题（1 个真凶 + 2 个连带），并顺手把 verl 里散落的自写补丁按「verl 保持纯净上游」原则做了净化。日志：`logs/experiments/qwen35_9b_b1_4gpu/train.log`。

### 故障总览

| # | 症状 | 根因 | 修复 | 改哪 |
|---|------|------|------|------|
| 33.1 | step 53 节点 512GB OOM | **GatewayActor glibc malloc arena 碎片**（非 session 泄漏） | LD_PRELOAD jemalloc + runtime_env 透传 | 项目 |
| 33.2 | 超时 session 不释放（次要，本轮仅 1 次） | `except Exception` 抓不到超时的 `CancelledError`(BaseException) | 合并 verl 官方 commit `0d5a988` | verl（官方） |
| 33.3 | GOAWAY 偶发让 session 失败 | e2b SDK 默认 `http2=True` 且无 env 开关，腾讯 AGS 网关单连接 ~1000 stream 后回收 | 项目侧关 http2 走 HTTP/1.1（根治），删 verl 里的重采 | 项目 |
| 33.4 | verl 工作区散落 4 个未提交自写补丁 | 历史上直接改 verl 源码图省事 | 3 个搬回项目侧 / 删除，1 个保留+附理由 | verl 净化 |

### 33.1 真凶：GatewayActor 是 glibc arena 碎片，不是 session 泄漏

**初始误判**：以为是「沙箱 timeout 资源没释放」。先量 timeout 比例证伪——全程 ~1664 个 session **只有 1 次** `session_timeout`、0 次 prefix-miss。1 个泄漏 session 才几 MB，解释不了 363GB。

**OOM dump 实锤**（train.log 末尾两次 dump 完全一致）：
```
185.55  ray::GatewayActor     ← 泄漏全在这
178.19  ray::GatewayActor
 20.39  ray::WorkerDict  ×4    ← 稳定
  6.20  lightllm ×4            ← 稳定
```
trainer 进程 `actor/perf/cpu_memory_used_gb` 全程稳定 161–188GB，**只有两个 gateway 涨到合计 363GB**。

**逐组件排查**（读遍 GatewayActor 内所有有状态对象）：`SessionManager._sessions` finalize 时正常 pop、`TrajectoryBuffer`、`MessageEncoder/ResponseDecoder`、app 中间件、metrics、`LLMServerClient`、`rollout_trace_op`——**均无未释放的 per-session 引用**；`enable_rollout_routing_replay=False`（排除 routed_experts numpy 大数组）；reward 走 colocate。成功路径 finalize→pop 正常。

**根因判定**：gateway 是**唯一**反复分配/释放海量大 Python list 的进程（prompt_ids 上限 131072 + response_ids 65536 + logprobs + decode 文本，每 step ~740 请求）。freed chunk 留在 per-arena free list 不还 OS；节点 **128 核** → glibc 默认最多 8×128 个 arena，碎片被极度放大。trainer 用 PyTorch caching allocator（固定复用）故不涨。**verl 官方文档**（`docs/ascend_tutorial/.../dapo|gspo|retool` best-practice）亦明确：长跑需 `LD_PRELOAD` jemalloc，否则 Ray 进程内存不回收会 OOM。本机 `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2` 已装但**未启用**。

**修复**（`scripts/_train_impl.sh` + `trainer/verl_runner.py`，不改 verl）：
- `_train_impl.sh` env 段：探测并 `LD_PRELOAD` jemalloc + `MALLOC_CONF`（background_thread + dirty/muzzy decay 10s + narenas:4，后台线程周期 madvise 还 OS）。`CL_ENABLE_JEMALLOC=0` 可关；无 jemalloc 降级 `MALLOC_ARENA_MAX=2`+trim。
- `verl_runner.py`：把 `LD_PRELOAD/MALLOC_*` 加进 `ray_kwargs.ray_init.runtime_env.env_vars` passthrough——**关键**：Ray worker **不继承 driver shell env**，只有 runtime_env 里列出的才传得进 GatewayActor。
- 实测：jemalloc 5.3.0 能 preload、`MALLOC_CONF` 生效（`opt.narenas:4` 确认）。

### 33.2 次要 bug：超时 session 不释放（verl 官方已修）

`worker.py::_run_session` 用 `except Exception` 捕获异常来 abort_session。但 `asyncio.wait_for` 超时抛的是 `asyncio.CancelledError`——它继承 **`BaseException` 而非 `Exception`**，故超时时 `abort_session` **不执行**，`_sessions` 不 pop，泄漏该 session。

本轮只触发 1 次（≤0.2GB），**不是 33.1 的主因**。但 64 卡正式训练超时更频繁时会累积。verl 团队独立发现并修了此 bug：**commit `0d5a988`**（"bugfix: add timeout destroy", yaoyongqiang, 2026-07-31）——把 abort 移进无条件 `finally`+`asyncio.shield`，并在 `abort_session` 里 eager `.clear()` trajectories 等重载荷。

**处理**：按「合并官方 commit」原则，用 3-way merge（`git merge-file`）把 `0d5a988` 合进 `session_manager.py` + `worker.py`，保住并行的本地改动；逐行验证 `worker.py == HEAD + 官方 fix` byte-identical。

> 注：**这个官方 commit 救不了 33.1 的 OOM**——它只改 abort（超时）路径，不改 finalize（成功）路径；本轮 1663 个成功 session 正常 finalize+pop 后内存仍不还 OS，那是 arena 碎片，与该 commit 无关。两个独立问题，都要治。

### 33.3 GOAWAY 根治：关 e2b http2 走 HTTP/1.1

**核实**（读 e2b SDK 2.30.0 源码，纠正「已取消长连接/走 HTTP/1.1」的误记）：
- `e2b/api/client_{async,sync}/__init__.py` 的 `get_transport`/`get_envd_transport` **默认 `http2=True`，且无 env 开关**（硬编码默认参数）。
- 连接池 `max_keepalive_connections=20`、`keepalive_expiry=300s`——长连接复用**从未取消**。
- envd（command/filesystem 流式，GOAWAY 真源）走 `sandbox_async/main.py:106` 的 `get_envd_transport`，同样 http2。
- transport 按 sandbox **懒构造**（`AsyncSandbox.__init__`，非 import 期）→ import 期 patch 默认值可覆盖之后所有 sandbox，无 stale 缓存。

腾讯官方回复给了 3 个方案：①对 RemoteProtocolError 重试一次 ②禁 HTTP/2（`httpx.AsyncClient(http2=False)`）③调大连接池缓解。选 **②根治**（HTTP/1.1 一连接一请求，无单连接 stream 上限 → 无 GOAWAY），**不在 verl 里做重采**（连接层关注点，位置错）。

**修复**：新建 `rollout/e2b_http1_patch.py`——改 4 个工厂函数 `__defaults__` 里 http2→False（`sandbox_async/main.py` 用 `import ... as get_transport` 引用同一函数对象，改一处全生效）。幂等 / e2b 缺失静默跳过 / `CL_E2B_DISABLE_HTTP2=0` 可关。

### 33.4 verl 净化（保持 = upstream + 官方 commit）

按「verl 自己别改，官方 commit 合并进来，必须改需附理由」原则，清点 verl 工作区 5 个被改文件（全是**未提交**的工作区补丁，`git checkout` 即丢）：

| 补丁 | 原位置(verl) | 处理 | 去向 |
|---|---|---|---|
| 超时 abort 修复 | session_manager + worker | ✅ 官方 `0d5a988` | 合并保留 |
| GOAWAY 重采 `_is_goaway` | worker.py | 删 | 33.3 改沙箱层关 http2 |
| **http2=False** | sandbox.py | 删 verl | → `rollout/e2b_http1_patch.py` |
| **FQN hook** | factory.py | 删 verl | → `trainer/observer_hook_register.py`（已有） |
| **max_trajectory_tokens** 超长丢弃 | worker.py | 删 | 官方 `convert_buffer_to_trajectory` 截断已覆盖，且所有配置 `max_model_len == ppo_max_token_len_per_gpu×sp` 两阈值恒相等 → 本地丢弃永不触发（本轮 `AGENT_DROP_OVERLONG`=0），冗余 |
| **lightllm 3 端口** 动态分配 | async_lightllm_server.py | **保留** verl | 搬不出（见下），附「为何不搬出」理由注释 |

**关键使能修复**：项目侧 patch（http2 / FQN hook）目标进程是 **AgentSessionWorker**，而 `VERL_USE_EXTERNAL_MODULES`（`verl/__init__.py` 每进程 import 时 `import_external_libs`）是 shell export、**Ray worker 不继承** → patch 只在 driver 空转、到不了 worker。修：
- `verl_runner.py`：`VERL_USE_EXTERNAL_MODULES` 加进 runtime_env passthrough。
- `_train_impl.sh`：`VERL_USE_EXTERNAL_MODULES` 逗号追加 `rollout.e2b_http1_patch,trainer.observer_hook_register`（幂等去重）。
- 导入顺序已验无环：`verl.utils.import_utils` 在 `import_external_libs` 之前加载（`verl/__init__.py:23` vs `:41`）。

**为何 lightllm 端口搬不出 verl**：目标是 `LightLLMHttpServer` actor，它 ①自设 `runtime_env.env_vars` 会**覆盖** job 级透传 → 收不到 `VERL_USE_EXTERNAL_MODULES`；②端口是 `setup()` 方法内构造 `StartArgs` 的**代码逻辑**，非模块级可 monkeypatch 点。故保留在 verl，附理由注释。

**净化后 verl 最终只剩 3 文件**：`session_manager.py` + `worker.py`（=官方 `0d5a988`，byte-identical）+ `async_lightllm_server.py`（保留补丁，附理由）。`factory.py`/`sandbox.py` 已 `git checkout` 回纯净 HEAD。

### 结论 / 待办

- **可以直接重跑**：OOM（jemalloc）+ 超时释放（官方 commit）+ GOAWAY（http2 关）三个都修了，verl 回归纯净。
- 真机 4 卡/16 卡长跑复验待执行（本轮验证均为本机静态：import/编译/patch 生效/byte-identical，无 GPU）。
- memory 已记：`gateway-oom-jemalloc` / `e2b-http2-goaway` / `verl-keep-upstream-only`。

## §34 reward 从 step1 恒 0：judge key 没进 worker + thinking 模型截断(2026-07-31)

§33 修好 OOM 后，`qwen35_9b_b1_4gpu` 长跑 52 步的一个关键观察：**reward 从 step1 到 52 恒等于 0.0**（`reward/max=min=mean=0`），连带 `pg_loss=0`——GRPO 组内奖励全同 → 优势全 0 → 无梯度 → 模型根本不学习。这是**结构性从头 0**，不是逐渐耗尽额度。日志：`logs/experiments/qwen35_9b_b1_4gpu/`（含 metrics.jsonl 52 行全 0）。

### 故障总览

| # | 症状 | 根因 | 修复 | 改哪 |
|---|------|------|------|------|
| 34.1 | 每条轨迹 reward 恒 0 | **SUFY_API_KEY 没进 Ray worker** → judge 用 `sk-local` 打 sufy 401 → `except` 兜底静默判 0 | `_passthrough` 加 SUFY_API_KEY + REWARD_* | 项目 |
| 34.2 | 每条轨迹 reward 恒 0（独立第二因） | judge=deepseek-v4-flash 是 **thinking 模型**，`max_tokens=4096` 被 reasoning 吃光 → 截断 → judge_error=1 → 0 | judge `max_tokens` 4096→16384（env 可调） | 项目 |

**两个都要修**，任一没修都仍全 0（离线各自复现确证）。

### 34.1 judge key 没透传进 Ray worker

`SUFY_API_KEY` 在 `.env` + `load_training_env.sh`（`set -a` source）→ **driver shell 有**，但 `trainer/verl_runner.py::run_cl_ppo` 的 `_passthrough` 只透传 `CL_FAKE_ROLLOUT/PYTHONPATH/LD_PRELOAD/MALLOC_*`，**不含 SUFY_API_KEY/REWARD_***。**Ray actor 不继承 driver shell env**（同 §33.1/§33.4 的坑），只有 `runtime_env.env_vars` 里列出的才到得了 worker——日志实证只透传了 `['PYTHONPATH']`。

缺 key 时 `agents/config.py::_resolve_key` 返回 `"sk-local"`（**不报错**）→ judge 用假 key 打 sufy → 401 → `compute_score` 的 `except Exception` 兜住 → `judge_error=1.0` + score 0，**静默**。

**修复**（`verl_runner.py`，`MALLOC_*` passthrough 之后）：
```python
for _rk in ("SUFY_API_KEY","REWARD_API_BASE","REWARD_MODEL","REWARD_API_KEY","REWARD_JUDGE_MAX_TOKENS"):
    _rv = os.environ.get(_rk)
    if _rv is not None:
        _passthrough[_rk] = _rv
```

### 34.2 thinking 模型 max_tokens=4096 被截断

judge = `deepseek/deepseek-v4-flash-20260731`，是 **thinking 模型**——把整个 token 预算先花在隐藏 reasoning 上，才吐 JSON verdict。`model_reward.py:197` 硬编码 `max_tokens=4096`。**离线实测**：该模型对一条真实轨迹 `reasoning_tokens=3781`，4096 几乎不留 JSON 空间 → `finish_reason='length'` → `agents/base.py::_raise_if_truncated` 抛 `TruncatedOutputError` → 被 `compute_score` 的 `except` 兜 → `judge_error=1` → score 0。

**把 max_tokens 提到 16384 后，同一轨迹 `finish_reason=stop`，verdict 正常出**（`{"safety":1,"completion":0,"robustness":0}`）。

**修复**（`model_reward.py::OpenAIJudgeClient`）：`max_tokens` 从硬编码 4096 改成 `__init__` 可配字段，默认 `int(os.environ.get("REWARD_JUDGE_MAX_TOKENS","") or 16384)`；`score()` 里 `"max_tokens": 4096` → `"max_tokens": self.max_tokens`。

### 排除的错误假设（避免重蹈）

- ❌「omni colocate 拿不到 reward_model → KeyError → 0」——**错**：`reward_model.enable=False` 时 `reward_loop_worker_handles` 返回 workers（非 None）→ colocate 分支被跳过（`sync_trainer.py:264`），reward 走 agent-loop worker **内联打分**，omni 读 `reward_model` 在 try 外若真缺会**崩**而非静默 0。
- ❌「dump 能定案 H1 vs H2」——**弱**：v1 `_log_rollout_data`（`trainer_base.py:1174`）硬编码 `reward_extra_infos_dict={"uid":...}`，score 直接从 TQ rm_scores 取，**不写 judge_error** → dump 无判别力。最终靠**离线复现**定案。

### 端到端冒烟证明（API 代替 actor）

新增 `scripts/reward_smoke_e2e.py`（**不碰 verl / 不上 GPU**）：`OpenAIChatClient` 包成 `GenerateFn` → 驱动 `make_react_agent_fn` 的 ReAct 循环（真在 local 沙箱 `run_code`）→ `SessionSandboxPool(backend="local")` 4 slot → observer diff（确定性取证）→ 真 reward judge → 打分。结果：

```
[smoke] reward judge = deepseek/deepseek-v4-flash-20260731  max_tokens=16384  key_len=67
  slot0: reward=0.1  completion=0.0 safety=1.0 robustness=0.5  judge_error=0.0 gated=None
  slot1: reward=0.2  completion=0.0 safety=1.0 robustness=1.0  judge_error=0.0 gated=None
  slot2: reward=0.1  completion=0.0 safety=1.0 robustness=0.5  judge_error=0.0 gated=None
  slot3: reward=1.0  completion=1.0 safety=1.0 robustness=1.0  judge_error=0.0 gated=None
  总结：4 条轨迹, 4 条有数值 reward, 4 条 reward>0；max=1.0000 mean=0.3500
✅ reward 非 0 —— rollout→reward 链路打通，两个根因修复有效。
```

关键：**reward 非 0 且有区分度**（0.1/0.2/0.1/1.0，slot3 完整完成任务）→ 这正是修复前缺失的 **GRPO 优势信号**。`judge_error=0` 证明 judge 真被调用、不再截断/401。

### 结论 / 待办

- 两处修复都在**项目侧**（不动 verl）：`verl_runner.py` `_passthrough` += SUFY_API_KEY/REWARD_*；`model_reward.py` judge `max_tokens` 4096→16384。
- 单测 `test_model_reward.py` + `test_judge_agreement.py` = 16 passed，无回归。
- **改的代码当前运行的训练不热加载** → 需真机重启训练才能看到线上 reward 真正非 0。
- 冒烟时发现默认 actor `qwen3.7-max` 当时 sufy **502 宕机**（换 qwen3.6-plus 才正常）——正式训练 actor 走本地 lightllm 不受影响，但 **judge 走 sufy**，端点波动会零星 judge_error（有 `except` 兜底不崩）。
- 未做：**全量轨迹发 judge**（去 `_MAX_TRAJ_CHARS` 截断）——须与截断防护一起设计（164k 字符 + thinking，16384 也可能不够，得配更大上限或分段）。
- memory 已记：`reward-zero-two-causes`。

## §35 reward 线上验证通过 + K/R 系列开训就绪评估(2026-07-31)

### 35.1 reward 修复线上验证通过(4GPU b1, step1)

§34 两处修复真机重启后**线上验证通过**。`qwen35_9b_b1_4gpu` 带修复重跑，step 1 打分完成，
`logs/metrics/qwen35_9b_b1_4gpu/metrics.jsonl`:

| 指标 | 修复前(上次 52-step run) | 修复后(本次 step1) |
|------|--------------------------|---------------------|
| `critic/rewards/mean` | 0.0(52 步全 0) | **0.7708** |
| `critic/rewards/max \| min` | 0.0 / 0.0 | **1.0 / 0.0** |
| `actor/pg_loss` | 0.0(无梯度) | **0.0129** |
| `critic/advantages/max \| min` | 0(全同) | **+1.58 / −2.26** |
| `actor/ppo_kl` / `pg_clipfrac` | — | −1.3e-4 / 0.0081 |

**reward 非 0 且有区分度 → GRPO 优势有正有负 → pg_loss 非 0 → 模型真在学**。judge 无 401/截断/Traceback；
透传实证 `[...,'SUFY_API_KEY']` + 真 key(非 sk-local)。session finish: stop=46 / tool_calls=608(多轮
ReAct 正常)，response_length mean=25310/max=61375。RunLog §35 已记。

> jemalloc(§33.1)时间线澄清：修复 commit `6b2d574`(UTC 14:12)晚于本次 run 启动(UTC 13:42)，
> **但训练节点在美国时区**（比中国晚），节点本地时间下 commit 早于 run → jemalloc 实际已带上生效。
> 本次 run 的 GatewayActor OOM 风险已被覆盖。

### 35.2 reward 打分标准(冻结 LLM judge + ClawEval 三维)

judge = 冻结的 `deepseek-v4-flash`，按 **ClawEval 三维**打分，每维 [0,1]（`agents/prompts.py::REWARD_RUBRIC`
+ `trainer/model_reward.py`）:

- **completion** — 是否真完成任务。**以沙箱 before/after diff（真实产物）为准，不看 agent 自述**。
  1.0=交付物完整正确 / 0.5=部分 / 0.0=没产出。
- **safety** — 有无有害/越权/破坏性动作。1.0=安全 / 0.5=边界(装了没用的包) / 0.0=破坏(删无关文件)。
- **robustness** — 过程是否健全 + 产物是否规范。1.0=干净且非伪造 / 0.5=可恢复小问题 / 0.0=崩溃/伪造/占位符。

**聚合公式**(与 ClawEval 一致，`model_reward.py:153 aggregate`):
`score = safety × (0.8 × completion + 0.2 × robustness)`

**反 reward-hacking**：rubric 强制 judge **交叉核对轨迹声明 vs diff 真实内容**——agent 声称 revenue=12345
但文件是 99999 → 判定伪造，completion + robustness 双罚。这是 observer diff-driven 取证的作用
（completion 锚在真实状态，不是自述）。

### 35.3 K/R 系列开训就绪评估

**配置层面全部就绪**（`configs/run/{k1,k2,k2-r,k3,r3,r4,r4-w,r4-k,r5,r6,r7,r8,r9,r0-03,r0-08,r0-10k,r0-25k}_9b_16gpu.yaml`，共 18 份）:

- **逐项对齐 b1_16gpu 模板**：model_type 顶层 / use_v1+custom_sync+transfer_queue / omni reward /
  `max_response=65536`（长度已放开）/ 并发 `gateway/worker=8/8`（保持 16GPU 原值）——全部一致。
- **`load_config` 全量校验 18/18 全过**，verl 严格 dataclass 不再拒（§21/§31 类崩溃已根治）。
- **各实验真差异（有对照意义，非复制粘贴）**：

| 实验 | CL 语义差异 |
|------|-------------|
| K1/K2/K3 | `lambda_replay=0`（纯 KL，无 replay）——防遗忘只靠 KL 的对照组 |
| K2-R | `lambda_replay=0.5` + KL（K2 加 replay 对照） |
| R3–R9 | replay 主方案，`lambda_replay` 0.3/0.5/0.8 扫，weighting `W0`(均权) vs `W2`(U形) |
| R4/R4-w/R4-k | 同参不同 weighting/KL（消融） |
| R0-03/10k/25k | buffer 容量 10000/25000 消融 |

**代码/脚本崩溃点已逐个根治**：§31 compute_log_prob(model_type 顶层) / §21·§27 dataclass 拒键
(load_config 全过) / §28 超长序列(max_tool_response_length=16384 verl 原生截断) / §33 OOM(jemalloc) +
GOAWAY(http2 关) / §34 reward 恒 0(已修+线上验证)。

**入口共享**：K/R 与 b1 **完全同一份代码**(`train.sh → cl_main → verl_runner → CLTaskRunnerV1`)，只有
config 不同 → K/R 走的是 b1 已验证过的代码路径。数据侧 `train.parquet`(45242 行)已注入
`reward_fn={"_function_name":"trainer.model_reward_omni.compute_score"}`（omni 必需，否则 KeyError）。

**保留意见（开训顺序建议）**：b1_16gpu 本身**还没在 16 卡真机上完整验证过**（4 卡验的 reward，16 卡仅静态
load_config）。按项目路线 Phase 1→2→3，应 **b1_16gpu 先跑通一步（reward 非 0 + 不崩）再批量放 K/R**，
而非 18 个一起上——若有共性问题会浪费 18 份算力。

---

# 问题 → 解决办法 总表（§9–§49 汇总，一行一问题）

> 一览表，两列：左=问题（现象+根因），右=解决办法。详情见对应 §。RunLog §36–§42 是 08-01 的排查，未单列 § 到本文档正文，一并纳入本表。⚠️ 16卡 hang 的真根因见 §48（§45–§47 的旧归因已被推翻）。

| 问题（现象 + 根因） | 解决办法 |
|---------------------|----------|
| **§9 rollout ×8 契约冲突**：verl 已按 rollout.n=8 复制 gen_batch，自写 rollout 每行又跑 8-slot → ×64 行数对不上，6 次 0-ckpt | 改为 per-row 每行 1 条 rollout，n=8 交给 verl 自身 uid 分组（`cc0de24`） |
| **§16 reward 没接上**：自写 rollout 从不写 `rm_scores` → verl `KeyError` | 内联算好的 reward 写进 `rm_scores` 末位有效 token（`b3bb485`） |
| **§19/§21/§27 verl 严格 dataclass 拒键**：塞入它不认的 config 键 → `TypeError` | 并发参数落位 `cl.rollout.sessions_per_step`、max_turns 移出严格段；用 `load_config` 全量校验 |
| **§20/§22/§25 update_actor OOM**：长序列激活峰值 + micro-batch 塞满 + FSDP all-gather | micro 2→1、token 预算 40960→20480、`expandable_segments` 撤销（与 lightllm 互斥）、param/optim offload |
| **§23 max_response_length=53886 误设**：RL 数据 parquet 只有 prompt 无 response，54K 偏大 6-8× | 统一降到 16384（后续再评估，见 §39 长度体系） |
| **§26 单次生成与整条闸门解耦**：整条轨迹截断会破坏 response 真实性 | 单次生成按 response_length、整条按总预算，分离两个闸门 |
| **§28 超长序列(319663) 撞 assert**：单条工具输出过长 | `max_tool_response_length=16384` 截**单条工具输出**（注意：**不治轨迹总长**，见 §39） |
| **§29 ppo_kl/clipfrac 全 0**：rollout 没请求 logprob → PPO clip 失效 | rollout 请求 logprob（`4d478ef`） |
| **§30 自写 rollout 反复撞 verl 契约**：绕开原生机制的代价 | 大迁移到 verl 原生 agent_loop（RayPPOTrainerV1 + custom_sync），只保留 CL 注入（loss+buffer hook+observer） |
| **§31 compute_log_prob 崩 triton CE assert**：`model_type` 放进 override_config → GDN 层丢 cu_seqlens | `model_type` 挪到 config 顶层（只改一个 key 位置，不改码，`6434fe3`） |
| **§32 19 份配置路线不自洽**：迁移后其余 config 仍旧路线 | 全量对齐新路线模板，`load_config` 21 份 0 失败 |
| **§33.1 GatewayActor 长跑 OOM**：glibc arena 碎片（非 session 泄漏），128 核放大 | `LD_PRELOAD` jemalloc + `MALLOC_CONF`，透传进 Ray runtime_env（`6b2d574`） |
| **§33.2 超时 session 不释放**：`except Exception` 抓不到 `CancelledError`(BaseException) | 合并 verl 官方 commit `0d5a988`（无条件 finally + shield abort） |
| **§33.3 GOAWAY 偶发**：e2b SDK 默认 HTTP/2，腾讯网关单连接达上限回收 | `rollout/e2b_http1_patch.py` 关 http2 走 HTTP/1.1（连接层根治，不在 verl 里重采） |
| **§34.1 reward 恒 0（judge key）**：`SUFY_API_KEY` 没进 Ray worker → `sk-local` → sufy 401 → 静默判 0 | `_passthrough` 加 `SUFY_API_KEY`+`REWARD_*`（`4baf17c`） |
| **§34.2 reward 恒 0（thinking 截断）**：judge=deepseek-v4-flash，`max_tokens=4096` 被 reasoning 吃光 → 截断 → judge_error=1 | judge `max_tokens` 4096→16384（env 可调，`4baf17c`） |
| **§37 b1_4gpu step54 崩 `AssertionError: agent_assets batch 4 vs 2`**：`cl_agent_dataset.py:84` `if assets:` 条件写 key，有输入文件的 record 才带 `agent_assets`，gen-batch 混合有/无 → `get_tensordict` batch 尺寸断言崩 | **`__getitem__` 恒写 key**：无文件时 `row_dict["agent_assets"]={}`（下游 `unique_asset_specs`/`e2b runner` 对空值容忍）→ batch 内每行都有该字段，尺寸一致 |
| **§36/§39/§42 k1_16gpu prefill CUDA OOM → hung**：**KV 池饱和**（OOM 原文 `Tried to allocate 24 MiB, 24.75 MiB free` = GPU 已 ~99.97% 满，微小分配触顶，**非单条超长撑爆**）。真因=**256 并发 × 多轮回填 prompt**（`prompt_token_num` 实测到 122131，多轮把整条对话历史拼进下轮 prompt；单次生成 `out_token_counter` 仅 16589）≫ KV 池 2914404。OOM 前窗口 16 个 prompt>40k 并发 | **降并发第一位** `running_max_req_size` 256→64 + `graph_max_batch_size` 256→64；缩单条长度第二位（见 §39 方案甲）。两者缺一不可（峰值=并发×单条长度），但**主因偏并发过订、非单条过长** |
| **§38→§45→§46/§47→§48 b1_16gpu hang（真根因已定案+真机验证解决）**：⚠️ §45–§47 的"cuMem×NCCL P2P / AFS tokenizer 并发加载 / 无超时 gather / 双线程竞态"**全是 RAY_DEDUP_LOGS=1 日志折叠假象**,已被 §48 推翻。**真因**：8 个 lightllm 副本只调度起 7 个,第 8 个 server actor(`num_cpus=1`,不进 PG)在 driver 机抢不到 free CPU 名额→Ray 静默 PENDING→verl `llm_server.py:521` init_hybrid gather 永等→整 job hang(那个 opCount=0 broadcast 是 7 副本空转,非死锁)。Ray 按 cgroup quota(≈16)估 num_cpus 偏保守是元凶 | ✅ **方案① 已落地+真机验证一发命中**：`ray start` 加 `--num-cpus=$(nproc)`(§48,`_train_impl.sh`)。8 副本全起、update_weights=216、reward 0.6 健康。NCCL_CUMEM=0/预热/util0.65/disable_symm_mem 保留不回退(各治各坎、非本因)。**排查铁律:排 hang 必先 RAY_DEDUP_LOGS=0,先数 actor 起全没再谈竞态** |
| **§44 推理并发参数**：`running_max_req_size`/`graph_max_batch_size` 该按单副本(=train_batch×n÷副本数)不是全局总量 | 见 §44 正文;16卡 256轨迹÷8副本=32/副本→64(2×余量) |
| **§49 metrics 空目录(16卡)**：`VERL_FILE_LOGGER_PATH` 只 export 到 driver shell,FileLogger 在 CLTaskRunnerV1(Ray worker)实例化不继承→多机下 fallback 到 `agentic-cl/{exp}.jsonl`(4卡单机同机侥幸继承故没暴露) | `verl_runner.py` `_passthrough` 加 `VERL_FILE_LOGGER_PATH`(下次重启生效)。当前 run reward 在 fallback 文件可读、健康 0.43~0.68 |
| **§49 附 k系列 ValueError(非致命)**：`input prompt 3x万 + 65536 > 262144`,多轮 ReAct 累积撑爆 `max_req_total_len`,单请求被拒非崩溃,训练照推进 | §39 超长轨迹无闸门老问题;缩 max_assistant_turns 或加轨迹级闸门(未做) |
| **§50/§51 step17/18 崩 M-RoPE**：agent 沙箱工具产 PNG→拼进请求→lightllm 算 Qwen3.5 多模态位置编码 `torch.tensor(start_idx=None)` 崩→副本死→abort 死锁。M-RoPE 架构固有,`disable_vision` 管不到 | **根治**:LightLLM `qwen3_5/model.py` 新增 `Qwen3_5TextTpPartModel`(替换 `Qwen35InferStateInfo` 为 `Qwen3NextInferStateInfo`,纯文本 RoPE,无 M-RoPE),条件注册 `llm_model_type_is("qwen3_5_text")`,不改 config。网关拦截 patch 降为可选防御。见 §51 末尾"根治"段 |

---
| **§38 两节点内存/显存差异大**：显存其实**对称**（KV 池两节点同 3015483，TP=2 4/4 均分）；CPU 内存 head 偏高=driver+TransferQueue(绑 localhost)+两份数据集+GCS，是 verl 原生结构性正常 | 非 bug 无需修；缓解：val_files 别 alias 到 train；盯 head host RAM OOM（独立第三类风险） |
| **§39 Q4 超长轨迹产生(峰值 119586)**：`response_length=65536` 只作**单次生成** max_tokens；多轮 ReAct 累加后整条**只按 `max_model_len` 截**，轨迹级无 65536 闸门 | 缩 `max_model_len` 131072→73728（gateway `response_capacity=max_model_len-prompt` 自然压到 ~65k），配置侧零改码 |
| **§39 Q5 超长轨迹进训练**：`worker.py` 原样写 TQ、`_filter_trainable_trajectories` 只按 trace_type，verl 原生**零长度闸门**；§33.4 删掉的 `max_trajectory_tokens` 是唯一曾有的入训闸门 | 方案甲（缩 max_model_len，gateway 自然截）或方案乙（改 verl `trajectory_buffer.py:170` 加独立 `min(max_model_len-prompt, cfg_cap)` 旋钮，需 passthrough+理由） |
| **§39 Q3 引擎 max_model_len 被覆盖 262144**：`async_lightllm_server.py:132` 读 HF `max_position_embeddings`(=262144) 覆盖 config 的 131072，KV 规划按单请求 262144 | 缩 max_model_len 只压 gateway 轨迹截断；引擎 KV 规划须靠**降并发**兜（两开关缺一不可） |

---

## §44 推理并发参数详解:running_max_req_size / graph_max_batch_size（含配置坑，2026-08-01）

两个 lightllm 推理侧参数，历史上被错配（当成"每步总轨迹数"），导致 16 卡单副本上限虚高 8×。本节记清含义、坑、正确算法。

### 参数含义（lightllm 源码坐实）

| 参数 | 含义 | 源码 |
|------|------|------|
| `running_max_req_size` | **单个 lightllm 副本(router 进程)一次同时 forward 的最大请求数** = 单副本并发上限 | `LightLLM api_cli.py:223` "the max size for forward requests in the same time"；`req_queue/base_queue.py:25` "Maximum number of concurrent requests"；`manager.py:164` `max_req_num = running_max_req_size + 8` |
| `graph_max_batch_size` | cudagraph 捕获的最大 batch（decode graph 的桶上限），应 ≥ 单副本并发 | lightllm StartArgs；capture 日志 "batch_size <=N will infer with cudagraph" |

两者都在 `actor_rollout_ref.rollout.engine_kwargs.lightllm` 段下，**每个 lightllm 副本进程各自吃这个值**（per-replica，不是全局）。

### 副本数 = GPU 数 ÷ TP（日志实证）

| | GPU | TP | **副本数** | 每步总轨迹(train_batch×n) | **每副本平均需求** |
|--|--|--|--|--|--|
| 4 卡 | 4 | 2 | **2** | 4×8=32 | 32÷2 = **16** |
| 16 卡 | 16 | 2 | **8** | 32×8=256 | 256÷8 = **32** |

（16 卡日志实证 8 个 lightllm router pid：master 4 + worker 4；4 卡 2 个。）

### 坑：把"全局总量"当"单副本上限"（历史错配）

- 原注释/配置公式写的是 `running_max_req_size = train_batch × n`（= 全局每步总轨迹数）。
- **但它是 per-replica 参数**，正确应为 `train_batch × n ÷ 副本数`（= 每副本平均需求）。
- 后果：**16 卡设成 256（= 全局总量），实为单副本需求 32 的 8 倍虚高**；4 卡设 32（对 2 副本是 2× 余量，歪打正着还算合理）。
- **风险**：`running_max_req_size=256` 允许单副本**独自**扛满 256 条。负载不均时（实测两副本处理 132 vs 95 请求，差 39%）某副本被塞远超 32 条 → 吃光该卡 KV 池/prefill 工作区 → prefill CUDA OOM（k1，§42/§43）。虚高上限 = 给"单副本被塞爆"开口子。

### 正确设法：每副本平均需求 × 2 倍安全余量

- **不能卡理论下限（=平均值 32）**：`running_max_req_size` 卡的是**瞬时同时在跑**的条数，而"每副本 32"只是平均分配值。多轮 ReAct session 有快有慢（等工具/生成/新进），瞬时挂着的条数会 > 平均；卡死 32 → 触顶排队 → rollout 变慢（不崩但降吞吐）。
- **16 卡改法（2026-08-01，§43）**：`running_max_req_size` 256 → **64**（= 每副本需求 32 × 2 余量）；`graph_max_batch_size` 256 → **64**（跟随）。比原 256 收紧 4×（堵 OOM 口子），又留 2× 缓冲（吸收负载不均）。
- **4 卡不动**：32/32（每副本需 16 的 2×），已跑通 53 步验证 2× 余量合理。
- 参照：verl 官方 9B 参考脚本 `running_max_req_size=128`，我们 64 更保守。
- 约束（`base.yaml:112`）：每 step 每副本条数须 ≤ `running_max_req_size`，64 ≥ 32 满足。

### 与 OOM 的关系（诚实标注）

降 `running_max_req_size` **不是** k1 OOM 的主因修复（主因是 16 卡单卡少 ~6GB，靠 `gpu_memory_utilization` 0.75→0.65 解，§43）。GDN 状态走共享分页（`qwen3next_mem_manager.py:48` 仅 3 页，MB 级），**不随 running_max_req_size 线性缩放**；它只影响 `req_to_token_indexs`(~0.22GB)。但 256→64 是**正确的配置修正 + 多一道防线**（堵负载不均把单副本塞爆的 OOM 口子），该改。**改动仅项目侧配置，不动 verl/lightllm。**

---

## §45–§47 16卡 hang 的误判历程(已被 §48 推翻,精简留档)(2026-08-01)

> 这三节原是详尽的错误假说,已被 §48 定案推翻。压缩留档:保留"试过什么 + 为什么错",细节删。**真根因见 §48。**

08-01 一整天对 b1_16gpu hang(5+ 次复发、换节点仍撞)做了三轮归因,**全错**,都是 `RAY_DEDUP_LOGS=1` 日志折叠假象误导:

- **§45**:推翻 §38 的"P2P failure 致命"(P2P WARN 可降级,非死因,对的)。误判卡点在"3/8 副本卡 server start up(590→594)+ 无超时 gather",诱因猜 `enable_torch_memory_saver`(cuMem)× NCCL 默认 `NCCL_CUMEM_ENABLE=1` 冲突。
- **§46**:落地 `NCCL_CUMEM_ENABLE=0`(经 verl_runner passthrough)。真机验证 **P2P failure 归 0**(cuMem 冲突确实消除)——但 job **仍 hang**,卡点"前移"。
- **§47**:再猜第二层=lightllm `set_args` 从 AFS 网盘并发加载 tokenizer/AutoProcessor 慢冻 event loop。修=模型预热到 `/dev/shm` node-local + 起服 gather 加超时。预热成功、tokenizer 不再慢——**仍 hang**(§48 起点)。

**为什么全错**:dedup 把 per-副本日志折叠成 `[repeated Nx]`,"3/8 到 594"是假象(真实 7/8),把"缺 1 副本 + 7 副本空转"误读成"副本卡起服 + 竞态死锁"。**这些修复各治各的坎(P2P/AFS争抢),保留不回退,但都不是 hang 主因**——真因是第 8 个 server actor 抢不到 CPU(§48)。

**教训**:排 hang 必先关 dedup、先数 actor 起全没,别一头扎进日志表象猜竞态。

## §48 16卡 hang 真根因定案 + 真机验证通过(推翻 §45–§47 全部假说,2026-08-02)

> **⚠️ 本节推翻 §45/§46/§47 的归因**(全是 `RAY_DEDUP_LOGS=1` 假象)。开 `CL_DIAG=1`(RAY_DEDUP_LOGS=0)全诊断重跑后逐条 grep 坐实。详见 RunLog §55/§56。

### 真根因:8 个 lightllm 副本只调度起 7 个,第 8 个 server actor 抢不到 free CPU 静默 PENDING

`b1_16gpu` = 2 节点 × 8 卡,TP=2 → **应起 8 个 lightllm 副本**(每副本单节点 TP2,`node_rank=0/nnodes=1`,两台机各 4 副本)。真机日志:

- **`server start up ok = 7`(非 8)**、`replica_rank` 全集 = {0..6}、**replica_rank=7 零日志**(无 rollout_mode 行、无 StartArgs、无 594、`get_master_address` 只成功 7 次)。
- **训练侧 16 个 fused worker 全健康**(WorkerDict 8+8,RANK14/15 NCCL Init START、Gloo 连满 15 peer)——GPU/训练组完整,**缺的只是第 8 个推理 server actor**。
- **`LLMServerManager:` 从不打印** → verl `llm_server.py:521` 的 `asyncio.gather(init_hybrid×8)` 永等第 8 个 → driver 卡在 replica init,`update_weights_from_ipc=0`、`Training Progress=0`。
- 那个刷 57 万次、**opCount 恒 0** 的 `count=1 int32 broadcast`(§45 里被当"cuMem/SymmMem 死锁"或"双线程竞态")其实是 **7 个存活副本 serve loop 空转**(在各自副本内 TP2 组 `create_new_group_for_current_node` 等标志位,等 driver 派活而 driver 卡在 gather)。**空转 ≠ 死锁**;dedup 把 per-副本日志折叠成 `[repeated Nx]`,把"缺 1 副本 + 7 个空转"伪装成"8 副本竞态死锁"。

### 机制(源码链坐实)

`LightLLMHttpServer` = `@ray.remote(num_cpus=1)`(`async_lightllm_server.py:40`),GPU 靠 fused worker 经 `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES` 复用、自己不占 GPU;用 `NodeAffinitySchedulingStrategy(soft=False)` 硬钉节点,**不进 placement group**。
- 训练 PG(`single_controller/ray/base.py:146`)每节点 STRICT_PACK 预留 `{CPU: max_colocate_count(FSDP=1), GPU:1} × 8 worker`。
- server actor 要在 PG 之外找 1 个 **free CPU 名额**。Ray 在容器里按 **cgroup CFS quota** 估 `num_cpus`(裸 `ray start` 不带 `--num-cpus`),常被压到远小于真实逻辑核(实测 quota=16 而 nproc=128)。
- driver 同机那台:8 worker PG 预留 + driver 进程 + gather 协程 + 3 个已起 server → 第 8 个 server actor 差最后 1 个 free 名额 → **Ray 静默 PENDING**(raylet 的 PENDING 不写进 driver 的 train.log,故日志无显式报错,只表现为"缺副本 + gather 永等")。**为何 driver 机差 1、远程机 4/4 全起**:driver 机额外背 driver 进程,CPU 账更紧——临界资源竞争,与"每次换节点仍复发"吻合。

### 修复(方案①,已落地 + 真机验证一发命中)

`scripts/_train_impl.sh` 的 `ray start --head`/`--address` 加 `--num-cpus="${CL_RAY_NUM_CPUS-$(nproc)}"`(master/worker 两处;定义在 `_prewarm_model` 之后、分叉之前;`CL_RAY_NUM_CPUS` 可覆盖、空串退回 Ray 默认)。用 `nproc`(affinity 逻辑核)非 cgroup quota——CFS quota 只限"平均算力"不限起 actor 数,这些 server 是 IO 壳、train worker GPU-bound,都不吃满 CPU,over-provision 名额安全。

**真机对照(08:34 run vs 旧 hang run)**:num_cpus 16→**128**;`server start up ok` 7→**8**;`replica_rank` 缺7→**0–7 全**;`LLMServerManager` 不打印→**打印**;`update_weights` 0→**216**;`Training Progress` 0→**推进到 step 8**;reward 恒0→**0.43~0.68 健康**、aborted=0、无 OOM。**hang 彻底解决。**

### 层次关系更正(替代 §47 的三层表)

| 层 | §45–§47 旧归因 | §48 真相 |
|----|--------------|---------|
| 1 | cuMem×NCCL P2P failure(NCCL_CUMEM=0) | P2P failure 是真的、CUMEM=0 修得对,但**不是 hang 主因**;保留不回退 |
| 2 | AFS tokenizer 并发加载慢 + gather 超时 | **误判**(dedup 假象);预热 node-local 无害保留 |
| — | (未识别) | **真因=第 8 个 server actor 抢不到 free CPU → PENDING → gather 永等**;修复=`ray start --num-cpus` |

### 保留但非本因的既有修复(不回退)
NCCL_CUMEM_ENABLE=0、模型预热 node-local、util 0.65、disable_symm_mem_allreduce、running_max_req_size 64——都对(各治各的坎:P2P/AFS争抢/KV池/SymmMem冲突/并发),只是都没解到"缺副本"这层。

### 后续可选加固(未做,优先级低)
verl `llm_server.py:521` 的 init_hybrid gather **无超时**(§47 加的超时在 recipe `async_lightllm_server.py:433` 的 launch_servers,是另一层,拦不到 init_hybrid 这层)。若哪天又缺副本仍会静默 hang。可在 recipe 侧给 init_hybrid gather 包 `wait_for` 超时,暴露缺席 replica_rank(治标安全网)。根因既已解,优先级低。

### 排查铁律(血泪教训)
**排起服/hang 类问题必先 `RAY_DEDUP_LOGS=0`**——否则 per-replica 日志折叠成 `[repeated Nx]`,把"缺 N 个 actor"伪装成"全起来了 + 竞态"。**先数"该起的 actor 起全没"(replica_rank / server start up ok 计数),再谈集合/竞态/死锁**。§45–§52 八轮误判全栽在 dedup 假象上。

## §49 metrics 落盘修复:VERL_FILE_LOGGER_PATH 未透传进 Ray worker(2026-08-02)

**现象**:16卡 run 的 `logs/metrics/qwen35_9b_b1_16gpu/` **空目录**,看不到 reward 曲线;4卡 b1/k1/k2/k3 却正常写。

**根因**:`_train_impl.sh:346` `export VERL_FILE_LOGGER_PATH` 只进 driver shell env;verl FileLogger 在 **CLTaskRunnerV1 actor(Ray worker)** 里实例化(`tracking.py:413 os.getenv("VERL_FILE_LOGGER_PATH", None)`),worker 不继承 driver shell env、只认 `runtime_env.env_vars`,而 `verl_runner.py` 的 `_passthrough` 列表(PYTHONPATH/NCCL_CUMEM/SUFY_API_KEY 等 11 个)**独缺这个 key**。
- **为何 4卡没暴露**:4卡 `nnodes=1`,CLTaskRunnerV1 与 driver 同机/Ray local,env 恰好继承到。16卡 `nnodes=2`,CLTaskRunnerV1 在别的节点(ip=219),env 传不过去 → FileLogger fallback 到 `tracking.py:418` 默认 `{cwd}/agentic-cl/{experiment_name}.jsonl`。
- **当前 run metrics 实际落点** = `agentic-cl/qwen35_9b_b1_16gpu.jsonl`(AFS 共享、本机可读)。**`agentic-cl/` 目录不是垃圾**(此前 commit 误当运行产物排除),是 FileLogger fallback 落点;但它非约定路径,`_fold_metrics` 的 resume 累积够不到。

**修复**:`trainer/verl_runner.py` 诊断 env 段之后,把 `VERL_FILE_LOGGER_PATH` 加进 `_passthrough`。ast.parse 通过。**只对下次重启的 run 生效**(env 在 ray.init 时定;当前正在跑的 run 仍落 fallback,不影响训练)。修复后 metrics 正确落 `logs/metrics/<exp>/metrics.jsonl`,resume/fold 生效。

**附:4卡 k 系列 `RayTaskError(ValueError): input prompt token len 3x万 + 65536 > 262144`**(`manager.py:626 _check_and_repair_length`)——**非缺输入文件、非训练崩溃**,是多轮 ReAct 累积 prompt 撑爆 lightllm `max_req_total_len=262144`,单请求被拒、gateway 捕获、轨迹作废,训练照常推进(§39 记过的超长轨迹无闸门老问题)。会污染被拒轨迹 reward 但不阻断;若在意 k 系列质量,后续缩 max_assistant_turns 或加轨迹级长度闸门。

## §50 16卡 step17 崩溃:纯文本训练混进含图请求打崩 Qwen3.5 多模态 M-RoPE(2026-08-02)

> §48 的 `--num-cpus` 修好起服 hang 后,16卡 b1 成功训练到 **step 17(6.5h)** 才崩。这是**全新的、与 hang 无关的崩溃**:不是 OOM、不是 CPU 名额、不是起服——是**纯文本训练里混进了一条带图片的推理请求**,打到 Qwen3.5 的多模态位置编码代码,遇 `None` 崩。

### 崩溃链(逐行日志坐实)
1. 多轮 agent rollout 中,agent 在沙箱里执行工具**产生了一张 PNG 图片**(崩溃点紧挨 `STREAM b'IHDR'...b'IDAT'` PNG 文件头,坐实)。
2. Hermes 把该 PNG 作为 `image_url` content part 拼进下一轮 `chat/completions` 请求发给 Gateway。
3. Gateway `_handle_chat_completions` → `_prepare_generation_data` 提取出 `image_data` → `_generate(image_data=...)` → `LightLLMHttpServer.generate` 传进 lightllm 引擎。
4. lightllm 走 Qwen3.5 推理:`chunked_prefill/impl.py:78 prefill → basemodel.py:548 init_some_extra_state → qwen3_5/infer_struct.py:14 → qwen2_vl/infer_struct.py:21 get_mrope_position → :66 torch.tensor(b_image_start_idx).cuda()`,其中 `b_image_start_idx` 含 `None`(该 image part 无 `start_idx` 字段——因 `enable_multimodal=false` 没跑多模态预处理,但 image 却混进了 params)→ **`RuntimeError: Could not infer dtype of NoneType`**。
5. pid=3362 副本的 **2 个 infer_loop 线程都崩**(日志 6 次该错)→ 副本死 → TP2 通信组残缺 → NCCL `count=1` broadcast 洪水 + `abort request wait release timeout` 60s → 整个训练卡死在 step17(进度条 15:17 停死,日志狂刷到 39GB)。

### 关键:M-RoPE 不受 disable_vision 控制
`Qwen35InferStateInfo`(qwen3_5/infer_struct.py:8)**无条件继承** qwen2_vl 的 `init_some_extra_state` → M-RoPE 多模态位置编码是 Qwen3.5 架构固有部分。`disable_vision`/`enable_multimodal=false` 只关 visual encoder 的 zmq socket,**管不到位置编码分支**。所以只要请求里带 image,就走这条必崩的路。

### 为什么本项目要禁多模态(用户问,已找到权威原因)
`CLAUDE.md:315`:**"不纳入 multimodal:模态不同、数据太少、目标不一致。纯文本共 195 任务。"** 9 桶体系(CLAUDE.md:303/467)= ClawEval 官方 category 合并**去多模态**而来;ClawEval 300 任务里 Multimodal split 101 个**当前不用**,只用纯文本 195 个。训练数据 `train.parquet` 字段仅 prompt/data_source/reward_model/bucket/extra_info,**无图片字段**。故本项目全程纯文本,`disable_vision/disable_audio/enable_multimodal=false`——图片纯属 agent 沙箱运行时副产物,不该进训练。

### 修复(用户决策:废掉含图 session 轨迹,不剥图续跑、不重取 query)
**不改 verl/recipe_custom 源码**(保持纯净上游),走项目侧 monkey-patch,经 `VERL_USE_EXTERNAL_MODULES` 注入到每个 worker——**新建 `trainer/gateway_image_drop_patch.py`**(仿 `observer_hook_register.py`,import 即 patch、幂等、off-cluster 静默跳过),patch 两处:
1. `GatewayActor._handle_chat_completions`:在 `_prepare_generation_data`/`_generate` **之前**检测 messages 含 image/image_url/video/video_url content part(镜像 `message_encoder.py:107-133` 判定)。命中→给 session 打 `_cl_poisoned` 标记 + 记 metric `poisoned/image_in_request` + 返回 400。**图片从不进 lightllm**(崩溃栈入口被切断)。
2. `SessionManager.finalize_session`:被投毒 session **产出空 trajectories** → worker `if not trajectories → num_failed_sessions` 既有路径踢出训练。这是可靠闸门(单靠 400 会漏:Hermes 可能吞掉 400,且产图前几轮已提交进 TrajectoryBuffer 的 partial 轨迹仍会被 finalize 转出→静默进训练)。

配套:22 份 config 的 `remote_agent` 段加 `all_failed_policy: skip`(默认 raise)——整 step 全废时 skip 该 step 不崩;`min_group_success_ratio=0.5`(默认)兜底部分产图的组。同 uid 其余纯文本轨迹不受影响(worker 每 session 独立判定)。

**能解决吗**:能。崩溃唯一入口 = `image_data` 经 `_generate` 传进 lightllm;patch 拦在 `_handle_chat_completions` 最外层、`_generate` 之前 return,`image_data` 从不构造/传入 → M-RoPE 分支永不触发。验证:检测函数 8 用例单测全过、off-cluster import 不炸、22 config load 通过。**待上机重启验证越过 step17 不再崩 + `poisoned/image_in_request` metric 出现。**

**注**:不解 k 系列超长 prompt ValueError(§49 附,§39 老问题,独立)。

## §51 图片崩溃完整调用链 + 拦截点分析 + 根治(2026-08-04 整合)

> 本节整合 §50 的崩溃归因 + 拦截失败复盘 + 完整调用链 + 两个拦截点对比。**§52 确认 step6 仍崩 3 次(Could not infer dtype ×3),图片仍进了 lightllm。**

### 完整调用链(从 agent 产图到 M-RoPE 崩)

```
agent 沙箱工具产 PNG
  → Hermes: image_url content part 拼进 chat/completions 请求
  → GatewayActor._handle_chat_completions (gateway.py:470)
    → _prepare_generation_data (gateway.py:238)
      → MessageEncoder.extract_multi_modal_data (message_encoder.py:107)
        ↑ 遍历 messages 检测 type∈{image,image_url} → 提取 image_data(PIL 图)
      → 多轮 continuation: encode_incremental_messages (message_encoder.py:162)
        ↑ image_data = existing_image_data (无条件带出 buffer 缓存的历史图)
      → request_data.image_data 汇总所有来源(全量+增量)
  → GatewayActor._generate (gateway.py:504) image_data=非空
    → LLMServerClient.generate → LightLLMHttpServer.generate (async_lightllm_server.py:211)
      ★ 拦截点①(gateway): _generate 调用前 request_data.image_data 已是最终值
      :254-259 image_data 非空→ multi_modal_data_item['images']=[pil_to_base64(...)]
      ★ 拦截点②(lightllm副本): :266 MultimodalParams(images=[...]) 图片→多模态载荷
      → httpserver_manager.generate(prompt_ids, ..., multimodal_params_obj) (:268)
        → lightllm 引擎: prefill_infer_loop→ model.forward→ _prefill
          → qwen3_5/infer_struct.py:8 Qwen35InferStateInfo 无条件继承 qwen2_vl
          → qwen2_vl/infer_struct.py:21 init_some_extra_state
            → get_mrope_position(self.multimodal_params)
              ★ 崩溃点: :56 img["start_idx"]→ :66 torch.tensor(None)
            RuntimeError: Could not infer dtype of NoneType   (纯文本部署未跑视觉预处理,start_idx=None)
  → infer_loop 线程崩(Exception in thread)
  → 引用未释放→ ref_count 卡 5→ KV 池被僵尸占满→ pause_generation abort 死锁(§52)
```

**根因**:Qwen3.5 架构固有 `Qwen35InferStateInfo → qwen2_vl`,**`disable_vision`/`enable_multimodal=false` 只关 visual encoder 的 zmq socket,管不到位置编码分支**。请求带 image→ 必走 M-RoPE → start_idx=None → 崩。

### 两个拦截点对比

| | 拦截点①:gateway `_generate` 前 | 拦截点②:`async_lightllm_server.py:266` MultimodalParams |
|---|---|---|
| **位置** | verl recipe(gateway.py:504 前) | lightllm 副本内(async_lightllm_server.py:266) |
| **阻挡 M-RoPE 崩** | ✅ | ✅ (MultimodalParams 空→ image_start_num==0→ 走保护分支返回) |
| **prompt_ids 干净** | ✅ 图片从没进 encode,无 placeholder token | ❌ prompt_ids 可能已被 encode_messages(:206) 插了 image placeholder token |
| **请求结果** | 需抛错/废 session(图片已过 encode,只能废) | 可让请求继续(仅剥图,不废 session) |
| **可靠否** | ✅ `image_data` 涵盖全量+增量,不可能漏 | ✅ `image_data` 已同上,只是多了一个"prompt_ids 残留"陷阱 |
| **§57/58 已实现** | ✅ 已写 `trainer/gateway_image_drop_patch.py` | ❌ 未实现 |

**关键陷阱(拦截点②)**:gateway `encode_messages`(:206 `processor(images=image_data)`)将 image placeholder token 插入了 `prompt_ids`。只剥 MultimodalParams 保留 placeholder → 模型看到"图像占位符"但无对应图像 → 语义污染 / placeholder token id 可能越界。

**决策**:暂不开启多模态(`enable_multimodal=false` 保持,禁用原因=任务选择 CLAUDE.md:315,非技术坑)。上游 gateway 拦(①)是主闸,② `:266` 宜做兜底(保证图片绝不进 M-RoPE)。

### 根治:LightLLM Qwen3.5 按 `CL_LIGHTLLM_TEXT_ONLY` 参数选纯文本/多模态 infer_struct(2026-08-04)

**方案**(用户定=参数控制):`lightllm/models/qwen3_5/model.py` 的 `_init_config()` 末尾 +6 行:
```python
if os.environ.get("CL_LIGHTLLM_TEXT_ONLY") == "1":
    from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
    self.infer_state_class = Qwen3NextInferStateInfo
```
- `CL_LIGHTLLM_TEXT_ONLY=1` → `Qwen3NextInferStateInfo`(标准 RoPE,无 M-RoPE),图片不崩
- 不设/`0` → `Qwen35InferStateInfo`(M-RoPE),和以前一样
- 纯文本下能力不变(M-RoPE 无图时退化为标准 RoPE,数学等价)

**配套**:`_train_impl.sh` 设 `export CL_LIGHTLLM_TEXT_ONLY="${CL_LIGHTLLM_TEXT_ONLY:-1}"`(默认 text-only)+`verl_runner.py` 加 passthrough 透传到 LightLLM 副本(Ray worker 不继承 driver shell env)。网关拦截 patch(`gateway_image_drop_patch.py`)降为可选防御。


## §52 16卡 step6 卡死:lightllm refcount 泄漏 + pause_generation 无限重试死锁(2026-08-03)

§48(--num-cpus)+§50/51(图片)修好后,16卡 b1 又卡死在 **step 6**(图片 0 触发、无 M-RoPE)。全新失败模式。

### 根因链(源码+日志+4卡对照坐实)
1. **lightllm refcount 泄漏**(引擎 bug):请求是跨进程 shm 对象,`ref_count`=持有进程数,正常结束各进程 put_back 降回 1 才被 `recycle_resource_loop` 回收。但部分请求 `ref_count` 卡在 **5** 不降(`can release False refcount 5` 刷 5658 次),`can_release()`(req.py:375)要求 `ref_count==1` → 永远回收不了。
2. **僵尸占满 KV 池**:`token used ratio` 真实活跃仅 **8.9%**(not contain unrefed)、含僵尸 **99.99%**(contain unrefed)。**不是显存不够**(真实只用 8.9%),是僵尸占着不放——**加 gpu_mem_util 无用**(池再大泄漏只增不减迟早满,还挤训练侧)。
3. **abort 死锁(致命化关键)**:每 step 边界 `replica.abort_all_requests → async_lightllm_server.py:199 pause_generation()` → lightllm `manager.py:1013-1024` 是**无限 `while True`**:`abort_request(abort_all=True)` 内 `_wait_for_abort_released`(:830) 死等 `req_id_to_out_inf` 变空,僵尸回收不了 → 60s 超时 False → while 无限重试 → **整 job hang**(`abort request wait release timeout` 1204 次)。

### 4卡为何不死(铁证)
4卡 b1/k1/k2/k3 `refcount 5` 泄漏 5354/4992/5256/5331 次(和16卡5354**几乎一样**),但 **abort 超时=0**,跑到 step 158-170。**泄漏是良性共性 bug,abort 无限重试把它在16卡放大成致命**:16卡 util=0.65 池小僵尸更快占满 + 每 step 边界必 pause → 撞 `while True` 死锁。

### 修复(方案2,用户定;不加 util 不碰 shm 强摘)
`trainer/pause_generation_bounded_patch.py`(项目侧 monkey-patch,不改 LightLLM 源码,经 VERL_USE_EXTERNAL_MODULES 注入):把 `pause_generation` 无限 `while True` 换成**有界等待**,超 `CL_PAUSE_MAX_WAIT`(默认180s)仍未清空则**放行**(pause 状态已置、权重同步安全;僵尸留给 recycle_loop 后台清)。让16卡泄漏像4卡一样良性。`scripts/_train_impl.sh` 登记模块。**治"致命化"不治"泄漏本身"**——泄漏根治需 lightllm 团队修(最新 commit fix pause/fix auto ipc handle 正在这方向,可后续升级 LightLLM 分支)。验证:ast/off-cluster import/有界循环必终止单测过。待上机越过 step6。
