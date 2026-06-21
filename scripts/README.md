# Scripts — 分类索引

35 个脚本按功能分为以下几类，方便快速定位。

## 训练启动

| 脚本 | 说明 |
|------|------|
| [`train.sh`](train.sh) | 单实验入口，`bash scripts/train.sh configs/phase3/r4.yaml` |
| [`start_train.sh`](start_train.sh) | 集群训练启动（64 卡 = 8×8），不依赖 run_phases |
| [`run_phases.sh`](run_phases.sh) | 按顺序跑多个实验（默认 b1 + r4），rank0 起 ray head |
| [`launch_8node.sh`](launch_8node.sh) | 8 节点冷启动采集（ssh 到各节点） |
| [`_node_worker.sh`](_node_worker.sh) | `launch_8node.sh` 的每节点 worker |
| [`load_training_env.sh`](load_training_env.sh) | 导入训练凭证（.env → 环境变量） |

### Phase run 快捷方式

| 脚本 | 说明 |
|------|------|
| `phase1/run.sh` | Phase 1 全部实验 |
| `phase2/run.sh` | Phase 2 全部实验（`--only k2` 选单个） |
| `phase3/run.sh` | Phase 3 全部实验（`--only r4` 选单个） |
| `phase4/run.sh` | Phase 4 全部实验（`--only c1` 选单个） |
| `phase5/run.sh` | Phase 5 全部实验（`--only s1` 选单个） |
| `phase6/` | Phase 6（占位，后续填充） |

## 沙箱操作

| 脚本 | 说明 |
|------|------|
| [`build_sandbox_image.sh`](build_sandbox_image.sh) | 构建 Agent Runtime 沙箱镜像 |
| [`push_sandbox_image.sh`](push_sandbox_image.sh) | 推送沙箱镜像到腾讯 CCR |
| [`create_sandbox_tool.sh`](create_sandbox_tool.sh) | 从 JSON 创建沙箱 Tool |
| [`create_sandbox_via_api.sh`](create_sandbox_via_api.sh) | 通过 tccli 创建 Tool + API key + 测试实例 |
| [`validate_sandbox_dockerfile.sh`](validate_sandbox_dockerfile.sh) | 静态校验 Dockerfile（无需 docker daemon） |
| [`print_sandbox_runtime_env.sh`](print_sandbox_runtime_env.sh) | 打印合并后的运行时环境变量 JSON |
| [`load_tencent_env.sh`](load_tencent_env.sh) | 导入腾讯沙箱凭证 |
| [`sandbox_smoke.py`](sandbox_smoke.py) | 沙箱冒烟（execute + M 采样 + winner 固化 + domain→bucket） |

## 数据采集与处理

| 脚本 | 说明 |
|------|------|
| [`collect_cold.sh`](collect_cold.sh) | 冷启动数据采集启动器 |
| [`collect_cold.py`](collect_cold.py) | 冷启动采集实现 |
| [`collect_rollout.sh`](collect_rollout.sh) | 多轮 user-sim rollout 采集启动器 |
| [`collect_rollout.py`](collect_rollout.py) | 多轮 rollout 采集实现 |
| [`prepare_queries.py`](prepare_queries.py) | query 提取 / 格式转换 |
| [`convert_dataset.py`](convert_dataset.py) | jsonl → verl parquet 格式转换 |
| [`clean_queries.py`](clean_queries.py) | query 清洗去重 |
| [`clean_buffer.py`](clean_buffer.py) | Buffer 快照清理 |
| [`warmup_buffer.py`](warmup_buffer.py) | Buffer 预热（冷启动数据 → replay buffer） |

## 评测与 Judge

| 脚本 | 说明 |
|------|------|
| [`eval.sh`](eval.sh) | ClawEval 评测入口 |
| [`calibrate_judge.py`](calibrate_judge.py) | Judge 校准 |
| [`mock_judge.py`](mock_judge.py) | Mock judge（固定满分，用于本地开发调试） |
| [`serve_reward_model.sh`](serve_reward_model.sh) | 启动外部冻结 judge vLLM 服务 |

## 推理 / vLLM

| 脚本 | 说明 |
|------|------|
| [`vllm_serve_patched.py`](vllm_serve_patched.py) | 补丁版 vLLM 服务 |
| [`test_vllm_infer.py`](test_vllm_infer.py) | vLLM 推理测试 |
| `vllm_patch/sitecustomize.py` | vLLM 启动补丁 |

## 注意

- `scripts/phase<N>/` 中的 `run.sh` 通过相对路径调用 `scripts/train.sh`，因此执行时必须在项目根目录。
- 集群配置（环境变量、AFS 路径、凭证注入）详见 `run_phases.sh` 和 `start_train.sh`，非集群本地运行只需 `train.sh` + 对应 phase config。
