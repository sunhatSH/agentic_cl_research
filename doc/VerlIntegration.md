# verl 集成与使用指导

本文档评估 CL 训练方案在 [verl](https://github.com/volcengine/verl) 框架上的实现路径，回答"是否需要 fork verl 修改源码"以及"Replay Buffer 如何接入"两个核心问题，并给出推荐的工程结构。

> 本文结论基于阅读 verl 上游源码（`main` 分支）得出，关键引用包含文件路径与行号。本仓库无可执行代码，所有讨论为设计层面的实现路径选型。

---

## 目录

- [核心结论](#核心结论)
- [一、是否需要 fork verl](#一是否需要-fork-verl)
- [二、Replay Buffer 接入方式](#二replay-buffer-接入方式)
- [三、推荐工程结构](#三推荐工程结构)
- [四、风险点与下一步验证](#四风险点与下一步验证)
- [附录：verl 关键源码索引](#附录verl-关键源码索引)

---

## 核心结论

| 问题 | 结论 |
|---|---|
| **是否需要 fork verl 修改源码？** | **多数情况不需要。** verl 提供清晰的 loss 注入扩展点（`register_policy_loss` 装饰器 + `engine.train_batch(loss_function=...)` + `actor.set_loss_fn()` API），自定义 CL Loss 走外挂注入即可。 |
| **Replay Buffer 是否需要改 verl 源码？** | **完全不需要。** verl 自身没有 experience replay buffer 的概念（只有 MoE router replay 这种无关概念），Buffer 是一片空白领地，纯 Python 模块即可，通过 Sampler / loss 内调用 / 自定义训练入口三种方式任选其一接入。 |
| **唯一可能 fork 的场景** | $L_{replay}$ 在同一 step 内对 replay batch 单独前向一次时，FSDP / 梯度累积兼容性需实测验证。若不通过才考虑 fork —— 与 Buffer 设计无关。 |

**推荐路线：pip 安装 verl + 自己写代码（Buffer 全套外挂 + 自定义 `cl_loss` 注入），不 fork。**

---

## 一、是否需要 fork verl

### 1.1 verl 暴露的 loss 扩展点

verl 在三个层级提供 loss 注入接口，自定义 loss 完全无需修改源码：

#### 扩展点 1：`register_policy_loss` 装饰器（注册表式）

文件：`verl/trainer/ppo/core_algos.py`

```python
PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,           # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}

def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    ...
```

用法：自定义 policy loss 加 `@register_policy_loss("cl_grpo")` 装饰器，配置文件里写 `actor.policy_loss.loss_mode=cl_grpo` 即可生效。

**局限**：此签名只接受 `(old_log_prob, log_prob, advantages, response_mask, ...)`，**没有 replay batch 的入口**。仅靠这个扩展点不足以实现 $L_{replay}$，需配合下一层。

#### 扩展点 2：`engine.train_batch(data, loss_function=...)`（顶层 loss 替换）

文件：`verl/workers/engine/base.py:99` 与 `:113`

```python
def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
    """
    Args:
        data: The input data for the forward pass, typically containing tensors and metadata.
        loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
    """

def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
    """Perform a training step on a batch of data."""
```

loss function **作为参数传入** engine，签名为：

```python
def my_loss(config, model_output, data, dp_group=None) -> tuple[torch.Tensor, dict]:
    ...
```

`data: TensorDict` 是开放容器——你可以在 trainer 层把 replay batch 的 token、log_prob、token weight、bucket id 等任意字段打包进 `data`，loss function 内部自取。

#### 扩展点 3：`actor.set_loss_fn(loss_fn)` API

文件：`verl/workers/engine_workers.py:584-587`

```python
self.loss_fn = partial(ppo_loss, config=actor_config)
self.actor = TrainingWorker(config=actor_training_config)
self.actor.reset()
self.actor.set_loss_fn(self.loss_fn)
```

worker 层提供了显式的 `set_loss_fn` 方法，外部 loss 直接注入即可。

### 1.2 CL 方案各项需求的实现路径

| 方案需求 | 实现方式 | 是否需要 fork |
|---|---|---|
| **CL Loss 总公式** $L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}$ | 写 `cl_loss(config, model_output, data, dp_group)` 函数，内部自由组合 RL/KL/replay/entropy 四项；通过 `actor.set_loss_fn()` 注入 | ❌ 不需要 |
| **$L_{replay}$ 监督回放** | trainer 层从 buffer 采 replay batch，把 token / mask / w 拼进 `data` 传下去；loss 函数里用 `data["replay_log_probs"]`、`data["replay_token_weights"]` 计算 $-\log\pi(a\|s) \cdot w$ | ❌ 不需要 |
| **Token 级权重 $w_t^{(i)}$**（priority × $\gamma^{\text{block}}$ × $\beta_{\text{final}}$ + clip + normalize） | trainer 层（rollout 后、forward 前）算好 `replay_token_weights` 张量，塞进 `data`；loss 函数内 element-wise 相乘 | ❌ 不需要 |
| **Reverse KL，$\pi_{ref}$ 切 $\pi_0$ / $\pi_{t-1}$** | verl 的 `kl_penalty(...)` 接受 `kl_penalty=...` 参数，自带多种 KL 类型；$\pi_{ref}$ 切换走 `actor_rollout_ref.ref` 的 ckpt 路径配置 | ❌ 不需要（最多改 yaml） |
| **Entropy bonus $\lambda_4=0.001$ 全程开启** | `actor.entropy_coeff` 配置项 | ❌ 不需要 |
| **Trajectory Diversity / Output Entropy 监控** | 自己加 metric，loss 函数返回的 `metrics` dict 自动上报 | ❌ 不需要 |
| **Replay Buffer 7 桶 + Priority + 两级采样 + 淘汰** | 完全独立的 Python 模块，与 verl 解耦 | ❌ 不需要 |

---

## 二、Replay Buffer 接入方式

### 2.1 verl 自身有 replay buffer 概念吗？

**没有。** 在 verl 仓库中搜索 `buffer` / `replay` / `memory` 相关文件，全部为无关概念：

```
tests/special_standalone/test_memory_buffers.py    # CUDA memory pool 测试
tests/utils/test_shared_memory.py                  # 共享内存
tests/utils/veomni/test_router_replay_on_cpu.py    # MoE router replay
verl/utils/megatron/memory.py                      # GPU 显存
verl/utils/megatron/router_replay_patch.py         # MoE router replay
verl/utils/megatron/router_replay_utils.py         # MoE router replay
verl/utils/memory_utils.py                         # CUDA memory utility
verl/utils/profiler/torch_memory_profile.py        # 显存 profiler
verl/utils/veomni/router_replay.py                 # MoE router replay
```

**没有任何 experience replay 抽象**。这意味着 verl 既不强制使用某种 buffer 实现，也不提供现成 buffer 实现。Buffer 是空白领地，自由设计。

### 2.2 三种接入方式（按改动量从小到大）

#### 方式 A：作为 Sampler / Dataloader 接入（**零源码改动**，推荐）

`RayPPOTrainer.__init__` 显式接受外部 sampler：

文件：`verl/trainer/ppo/ray_trainer.py:296-323`

```python
def __init__(
    self,
    ...
    train_sampler: Optional[Sampler] = None,
    ...
):
    """
    Args:
        ...
        train_sampler (Optional[Sampler], optional): Sampler for the training dataset.
    """
```

主循环（`ray_trainer.py:1423`）：

```python
for batch_dict in self.train_dataloader:
    ...
    gen_batch = self._get_gen_batch(batch)
    ...
    combined_gen_output = self.async_rollout_manager.generate_sequences(...)
```

dataloader 吐什么它就吃什么。把 7 桶 Buffer 包装成 `torch.utils.data.Sampler`（或自定义 dataloader），让每个 batch 既包含**新任务 query** 也包含 **buffer 采样出的旧轨迹 prompt**，外部从 `RayPPOTrainer(train_sampler=YourBucketSampler(...))` 注入即可。

**适用场景**：希望 buffer 数据走完整 rollout → advantage → loss 链路。

#### 方式 B：在 cl_loss 内部直接用 buffer（**零源码改动**）

旧轨迹已经存好 token + log_prob，不需要再 rollout 一遍。让 buffer 完全绕开 dataloader——

在你的 `cl_loss(config, model_output, data, dp_group)` 函数里，直接从 buffer 单例 `bucket_sampler.sample(batch_size)` 拿一批旧数据，前向计算 $-\log\pi(a\|s) \cdot w$，加到 loss 上返回。

**buffer 是普通 Python 对象，verl 完全不需要知道它存在。**

**适用场景**：本方案推荐路径——$L_{replay}$ 是监督回放（SFT-style），不走 advantage 计算，与 RL 数据路径独立。

#### 方式 C：自定义训练入口（**零源码改动**）

verl 提供了 `recipe/` 目录用于自定义训练流程；`main_ppo.py` 入口也可被自定义入口替换。写一个 `cl_main.py`，在每个 step 之后调 `buffer.update(new_trajectories)`、之前调 `buffer.sample()`——所有 buffer 操作在 verl 之外。

**适用场景**：需要"replay 频率 ≠ 新任务频率"等非对齐时序时使用。本方案 $L_{replay}$ 与 $L_{rl}$ 每 step 同步，无需此方式。

### 2.3 推荐组合：方式 A + 方式 B 结合

- **新轨迹写入 buffer**：每 step 收尾从 `actor_rollout_wg` 输出取 trajectory，写入 buffer（可在 sampler 的 `on_step_end` 回调里做，或 monkey-patch `update_actor`）。
- **$L_{replay}$ 取数**：在 `cl_loss` 内部从 buffer 采样，独立前向计算监督 loss。
- **新任务 query**：走 verl 默认 dataloader（如需 buffer 影响新任务采样比例，再启用方式 A 的 Sampler）。

---

## 三、推荐工程结构（当前项目快照）

```
agentic_cl_research/
├── replay_buffer/           # 纯 Python，与 verl 解耦
│   ├── bucket.py            # 7 桶结构 + quota 分配
│   ├── priority.py          # 4 信号 priority 融合
│   ├── eviction.py          # 桶内淘汰
│   ├── sampler.py           # 两级采样
│   ├── weighting.py         # W0/W2 U 形权重
│   └── store.py             # 内存 + SQLite 快照
├── trainer/                 # 基于 verl，零源码改动
│   ├── cl_main.py           # 入口：load_config → build_buffer → run_cl_ppo
│   ├── verl_runner.py       # CLTaskRunner: init_workers → inject_cl_loss → install_buffer_hooks → fit
│   ├── verl_async_runner.py # Fully-Async 分离训练 scaffold
│   ├── cl_loss.py           # make_cl_loss() + compute_replay_loss
│   ├── replay_forward.py    # 携梯度 replay batch 构建
│   ├── replay_batch.py      # driver 侧 replay batch 准备
│   ├── trajectory_adapter.py# verl rollout batch → buffer 轨迹
│   ├── domain_tagging.py    # LLM domain → 7 桶路由
│   ├── model_reward.py      # LLM judge reward（外部冻结 judge）
│   ├── replay_metrics.py    # buffer 动态 / forgetting 指标
│   └── cl_rollout_manager.py# 自定义 rollout（AgentLoopManager）
├── agents/                  # UserSim 三 agent（与 verl 解耦）
│   ├── observer.py / questioner.py / reward.py
│   ├── personas.py / personas.json（42 人设）
│   └── prompts.py / schema.py / base.py
├── rollout/                 # 采样侧，与 verl 解耦
│   ├── sandbox_client.py    # E2B 沙箱客户端
│   ├── sandbox_env.py       # 环境变量注入
│   ├── session_pool.py      # 会话级沙箱编排
│   ├── scheduler.py         # 16×8 + winner-sync
│   ├── simulated_session.py # UserSim 算法
│   └── collect.py           # 轨迹采集
├── inference/               # 单步生成边界
│   └── generate.py           # VerlRolloutGenerateFn
├── configs/                 # 21 个实验 + 3 层继承
│   ├── base.yaml            # 共享默认值
│   ├── cluster.yaml         # 64 卡集群引擎层
│   ├── run/{b1,r4}.yaml     # 集群可运行配置
│   └── phase<N>/*.yaml      # 各 phase 实验定义
├── eval/                    # ClawEval 评测
├── scripts/                 # 35 个启动脚本
│   ├── train.sh / eval.sh   # 通用入口
│   ├── start_train.sh / run_phases.sh  # 集群训练
│   ├── collect_cold.py / collect_rollout.py / prepare_queries.py / convert_dataset.py # 数据
│   ├── serve_reward_model.sh / mock_judge.py / calibrate_judge.py  # Judge
│   ├── build_sandbox_image.sh / sandbox_smoke.py / validate_sandbox_dockerfile.sh # 沙箱
│   ├── warmup_buffer.py / clean_buffer.py / clean_queries.py  # 工具
│   └── phase<N>/run.sh      # Phase 快捷方式
├── docker/sandbox/          # OpenClaw 沙箱镜像
├── skills/                  # 5 篇工程规范
├── tests/                   # ~200 单测
├── paper/                   # 论文产出
├── bin/                     # 数据管道工具
│
│ ─── 运行时产物（gitignored）───
├── ckpts/ / buffer_dumps/ / logs/ / wandb/ / eval/results/
```

> ⚠️ 本结构与当前仓库实际保持一致。`scripts/train.sh` 的正确用法为 `bash scripts/train.sh configs/phase1/b1.yaml`（非 `configs/b1.yaml`）。

### 模型与 checkpoint 路径约定

| 类别 | 存放位置 | 配置字段 |
|------|----------|----------|
| 预训练基底模型 | 仓库外（共享 NFS / HuggingFace cache） | `model.name`、`model.tokenizer`。**当前固定为 Qwen3.6-27B**（HF: `Qwen/Qwen3.6-27B`，已在 `configs/base.yaml` 中默认） |
| 参考策略 $\pi_{ref}$ | 仓库外或 `ckpts/` | `actor_rollout_ref.ref.path`（$\pi_0$ → 基底路径，$\pi_{t-1}$ → 上阶段 ckpt） |
| 训练 checkpoint | `ckpts/<实验名>-step-<N>/` | trainer 按 `save_freq` 自动写入 |
| Buffer 快照 | `buffer_dumps/` | 可选持久化，训练中断后恢复 buffer 状态 |

> verl 通过 pip 安装（`pip install verl`），不在仓库目录内、不 fork。

### 3.1 `cl_main.py` 骨架示例

```python
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from replay_buffer import BucketReplayBuffer
from trainer.cl_loss import make_cl_loss

# 1. 实例化 buffer
buffer = BucketReplayBuffer(
    num_buckets=7,
    total_capacity=25000,
    q_min=2000,
    bucket_names=["Workflow", "SysOps", "Dialogue", "Finance",
                  "Communication", "Knowledge", "OfficeQA"],
    bucket_task_counts=[54, 52, 38, 18, 12, 11, 10],
)

# 2. 构造自定义 loss（闭包捕获 buffer）
cl_loss_fn = make_cl_loss(
    buffer=buffer,
    lambda_kl=0.05,
    lambda_replay=0.5,
    lambda_ent=0.001,
    kl_type="reverse",
    pi_ref="pi_0",  # or "pi_{t-1}"
    use_token_weighting=True,  # W2 方案
)

# 3. 构造 trainer
trainer = RayPPOTrainer(config=cfg, ...)

# 4. 注入 loss（走 set_loss_fn 链路）
#    具体注入点取决于 verl 版本，参考 engine_workers.py:587 的链路
trainer.actor_rollout_wg.set_loss_fn(cl_loss_fn)

# 5. 训练
trainer.fit()
```

### 3.2 `cl_loss.py` 骨架示例

```python
def make_cl_loss(buffer, lambda_kl, lambda_replay, lambda_ent, **kw):
    def cl_loss(config, model_output, data, dp_group=None):
        # ---- L_rl: 复用 verl 的 ppo_loss 主体 ----
        from verl.workers.utils.losses import ppo_loss
        rl_loss, metrics = ppo_loss(config, model_output, data, dp_group)

        # ---- L_replay: 从 buffer 采样 + 单独前向 + token-level w ----
        replay_batch = buffer.sample(batch_size=...)
        replay_token_w = buffer.compute_token_weights(replay_batch)  # W2 方案
        # 在 data 里夹带 replay 数据，或在此处独立调用 model.forward()
        replay_loss = compute_replay_loss(model_output, replay_batch, replay_token_w)

        # ---- L_kl: verl 已支持，read from data["ref_log_prob"] ----
        # （已在 ppo_loss 中处理；reverse KL 通过 config.kl_loss_type 配置）

        # ---- L_ent: verl 已支持 entropy_coeff ----
        # （已在 ppo_loss 中处理）

        total_loss = rl_loss + lambda_replay * replay_loss
        metrics["actor/replay_loss"] = replay_loss.detach()
        return total_loss, metrics

    return cl_loss
```

### 3.3 buffer 写入时机

新 rollout 写入 buffer 有两种实现：

**实现 1**：buffer 暴露 `on_step_end(rollout_output)` 回调，在 sampler 内部调用（适合方式 A）。

**实现 2**：monkey-patch 包一层 `update_actor` / `generate_sequences`，前后插入 buffer 操作（适合方式 B）：

```python
original_update = trainer.actor_rollout_wg.update_actor
def patched_update(batch_td):
    result = original_update(batch_td)
    buffer.add_trajectories(batch_td)  # 写入新轨迹
    return result
trainer.actor_rollout_wg.update_actor = patched_update
```

---

## 四、风险点与下一步验证

### 4.1 已识别的唯一风险

**$L_{replay}$ 在同一 step 内对 replay batch 单独前向一次时的 FSDP / 梯度累积兼容性。**

verl 的 `engine.train_batch` 一次只接一个 batch。要在同一个 step 里同时算 RL 前向 + replay 前向，有两条路：

- **路 A**：在 `cl_loss` 函数内部用 `engine` 或 `model` 再前向一次 replay batch（`data` 里夹带 replay 的 input_ids），把两部分 loss 加起来返回。
- **路 B**：在 trainer 层先用 `engine.infer_batch(replay_data)` 拿 log_prob → 当作"目标"塞进 `data` → loss 函数里只做 `-log_prob * w` 的标量运算。要求 replay 的 log_prob 是当下策略的。

两条路**都不需要 fork**，但路 A 在分布式 / FSDP / 梯度累积下的细节需实测确认（同一 step 多次前向是否被 engine 的梯度累积逻辑正确处理）。

### 4.2 推荐的最小验证步骤

在投入完整实现前，先做一次最小可行性验证：

1. 本地装 verl（`pip install verl` 或源码安装）。
2. 写一个 toy `cl_loss(config, model_output, data, dp_group)`：
   ```python
   def toy_cl_loss(config, model_output, data, dp_group=None):
       rl_loss, metrics = ppo_loss(config, model_output, data, dp_group)
       dummy_replay_loss = (model_output["log_probs"].mean()) * 0.0  # 占位
       return rl_loss + 0.1 * dummy_replay_loss, metrics
   ```
3. 通过 `actor.set_loss_fn()` 注入。
4. 在 8×H100 上跑 1-2 个 step，观察：
   - backward 是否正常完成
   - grad_norm 是否合理
   - loss 是否下降
   - FSDP shard / 梯度累积是否报错
5. **通过则走外挂路线，不通过再考虑 fork。**

### 4.3 fork 的触发条件

仅在以下情况考虑 fork verl：

- 上述 4.2 验证不通过，且找不到外挂级 workaround
- 需要修改 advantage 计算或 GRPO 流程内部逻辑（**本 CL 方案无此需求**）
- 需要修改 rollout 引擎或分布式调度（**本方案无此需求**）

若必须 fork，改动范围应严格限制在：
- `verl/workers/utils/losses.py`（加自定义 loss 入口）
- `verl/workers/engine/fsdp/transformer_impl.py`（如需调整 forward-backward 时序）

**不要动** rollout / GRPO advantage / Ray 分布式调度，否则同步 upstream 极痛苦。

---

## 附录：verl 关键源码索引

> 基于 [volcengine/verl](https://github.com/volcengine/verl) `main` 分支。具体行号随版本可能漂移，使用时以 `git blame` 为准。

| 功能 | 文件 | 关键位置 |
|---|---|---|
| Policy loss 注册表 | `verl/trainer/ppo/core_algos.py` | `register_policy_loss` 装饰器 + `POLICY_LOSS_REGISTRY` |
| Loss 函数标准签名 | `verl/workers/utils/losses.py` | `ppo_loss(config, model_output, data, dp_group)` |
| Engine 顶层接口 | `verl/workers/engine/base.py` | `train_batch(data, loss_function)` (L99, L113) |
| Loss 注入 API | `verl/workers/engine_workers.py` | `ActorRolloutRefWorker.set_loss_fn()` → `TrainingWorker.set_loss_fn()` (v0.8.0) |
| Trainer 主循环 | `verl/trainer/ppo/ray_trainer.py` | `RayPPOTrainer.fit()` 中 `for batch_dict in self.train_dataloader` |
| 推荐入口 (v0.8.0) | `verl/trainer/main_ppo.py` | `run_ppo()` + `TaskRunner`；本仓库用 `trainer/verl_runner.CLTaskRunner` |
| 自定义 sampler 入口 | `verl/trainer/ppo/ray_trainer.py` | `__init__(train_sampler: Optional[Sampler])` (L307) |
| KL penalty | `verl/trainer/ppo/core_algos.py` | `kl_penalty(logprob, ref_logprob, kl_penalty=...)` |
| Advantage estimator 注册 | `verl/trainer/ppo/core_algos.py` | `register_adv_est` + `ADV_ESTIMATOR_REGISTRY` |
| 自定义训练 recipe 目录 | `verl/recipe/` | 用户自定义训练流程示例 |

### 配置项速查

| 配置路径 | 含义 | CL 方案对应 |
|---|---|---|
| `actor.policy_loss.loss_mode` | policy loss 注册名 | 设为 `cl_grpo`（或自定义名） |
| `actor.entropy_coeff` | $\lambda_4$ | 固定 0.001 |
| `actor.use_kl_loss` | 是否启用 KL loss | true |
| `actor.kl_loss_coef` | $\lambda_2$ | Phase 2 扫描 0.01 / 0.05 / 0.10 |
| `actor.kl_loss_type` | KL 类型 | reverse |
| `actor_rollout_ref.ref.*` | $\pi_{ref}$ ckpt | $\pi_0$ 或 $\pi_{t-1}$ 路径 |
| `actor_rollout_ref.rollout.n` | traj/query | 8 |
| `data.train_batch_size`, `data.gen_batch_size` | batch 配置 | 1024 (S1) / 4096 (S2) |
| `algorithm.adv_estimator` | advantage 估计 | grpo |
