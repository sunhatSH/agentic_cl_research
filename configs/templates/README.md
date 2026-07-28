# 配置分层模板(参考用,不接入启动链)

> **这是纯参考模板**,不被任何 `scripts/train.sh` 引用。现网仍用 `configs/run/*.yaml`(扁平写法)。
> 本目录示范一种**正交分层**思路:把一份训练配置拆成三层互不交叉的文件,
> 换实验 / 换 GPU 规格 / 换模型 size 时各自独立替换,不互相牵连。
> 想正式采用时,把某个 `run/<exp>.yaml` 改成下方 `defaults` 组合即可(改前务必做
> "effective config 逐字段不变"验证,见文末)。

---

## 三层职责(正交,互不交叉)

| 层 | 文件 | 装什么(随谁变) | 举例字段 |
|---|---|---|---|
| **① 逻辑层** | `configs/base.yaml`(现有) | 与硬件、实验都无关的科研/工程逻辑 | reward manager/端点/`custom_reward_function`、`rollout.temperature`、打分细则(代码里)、`algorithm.adv_estimator`、数据 schema、CL 权重方案(weighting)、桶定义 |
| **② 压力/硬件层** | `templates/hardware/<GPU>_<模型>.yaml` | 随 **GPU 规格 × 模型 size** 变 | `nnodes`/`n_gpus_per_node`、TP/SP/DP、`gpu_memory_utilization`、`ppo_micro_batch_size_per_gpu`、`ppo_max_token_len_per_gpu`、`sessions_per_step`(并发)、`running_max_req_size`、模型 `path`、fsdp offload、lr(按模型 size) |
| **③ 实验语义层** | `templates/experiment/<exp>.yaml` | 随**实验**变 | `experiment_name`、`cl.lambda_replay`、`buffer.enabled`、`use_kl_loss`、`kl_loss_coef`、`total_training_steps`、`save_freq` |

组合方式(OmegaConf `defaults` 列表,后者覆盖前者):

```yaml
defaults:
  - ../../_generated_ppo_trainer          # verl 全量默认(集群 Hydra 基底)
  - ../../base                            # ① 逻辑层
  - ../hardware/16GPU_9BQwen3.5           # ② 压力/硬件层
# ③ 实验语义写在本文件 body
```

**换维度只换一行**:
- 换 GPU/模型:`16GPU_9BQwen3.5` → `64GPU_27BQwen3.6`,其余不动。
- 换实验:复制实验骨架改 `cl.*` / `experiment_name`,硬件层不动。

---

## 本目录文件

```
templates/
├── README.md                          # 本文件
├── hardware/
│   ├── 16GPU_9BQwen3.5.yaml           # 当前生产规格(实测压测值)
│   ├── 32GPU_9BQwen3.5.yaml           # 推断起点,待压测
│   └── 64GPU_27BQwen3.6.yaml          # 27B 生产规格,待压测
└── experiment/
    ├── _baseline.yaml                 # B1 语义骨架(无 CL)
    ├── _kl.yaml                       # K* 语义骨架(开 KL)
    └── _replay.yaml                   # R* 语义骨架(开 buffer/replay)
```

硬件层各规格的压测依据与 checklist 见 `doc/ops/Concurrency_Policy_by_GPU.md`。

---

## 采用某模板时的验证(必做)

分层重构必须**行为不变**。把 `run/<exp>.yaml` 改成 defaults 组合后,对比 effective config:

```python
from trainer.cl_main import load_config
from omegaconf import OmegaConf
a = OmegaConf.to_container(load_config("configs/run/<exp>.yaml"), resolve=False)
# 与重构前快照逐字段 diff,必须 0 处差异才算等价
```

`load_config`(`trainer/cl_main.py:61`)按 `defaults` 顺序 `OmegaConf.merge`,后引用的文件覆盖前者;
本文件 body 覆盖所有 defaults。这正是三层能干净叠加的机制。
