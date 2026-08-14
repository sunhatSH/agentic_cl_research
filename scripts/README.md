# Scripts

## 训练

```bash
bash scripts/train.sh 16gpu --config configs/run/b1_9b_16gpu.yaml    # 单实验
bash scripts/train.sh 16gpu --phase 1                                 # Phase 1 全部实验
bash scripts/train.sh 64gpu --phase 2 --only k2                      # 只跑 k2
bash scripts/train.sh 64gpu --all                                     # 所有 Phase
```

| 脚本 | 说明 |
|------|------|
| `train.sh` | 统一入口：拓扑 + 实验选择 + 参数覆盖 |
| `_train_impl.sh` | 实现：多机同步 + verl 启动 |
| `_sensecore_env.sh` | SenseCore 平台变量映射 |
| `check_train_env.sh` | 环境依赖自检 |
| `load_training_env.sh` | 训练凭证 |
| `load_tencent_env.sh` | 沙箱凭证 |
| `dev_env.sh` | 开发环境 PYTHONPATH |

## 沙箱

| 脚本 | 说明 |
|------|------|
| `build_sandbox_image.sh` | 构建镜像 |
| `push_sandbox_image.sh` | 推送镜像 |
| `create_sandbox_tool.sh` | 创建 Tool |
| `create_sandbox_via_api.sh` | API 创建 Tool + 实例 |
| `validate_sandbox_dockerfile.sh` | Dockerfile 校验 |
| `print_sandbox_runtime_env.sh` | 运行时 env |

## 采集

| 脚本 | 说明 |
|------|------|
| `collect_cold.sh` / `collect_cold.py` | 冷采集（无 agent） |
| `collect_rollout.sh` / `collect_rollout.py` | 多轮采集（observer+questioner） |
| `collect_smoke.sh` / `collect_floor.sh` | smoke / floor 采集 |
| `run_cold_start.py` | 冷启动采集 runner |
| `sandbox_grpo_collect.py` | GRPO 采集 |
| `sandbox_smoke.py` | 沙箱冒烟测试 |

## 数据管道

| 脚本 | 说明 |
|------|------|
| `queries_to_parquet.py` | queries → parquet |
| `labeled_to_parquet.py` | labeled data → parquet |
| `merge_bucket_parquet.py` | 合并桶 parquet |
| `label_buckets.py` / `label_capability.py` | 能力打标 |
| `convert_dataset.py` / `trajectory_to_parquet.py` | 格式转换 |
| `prepare_queries.py` / `taskspec_to_queries.py` | query 提取 |
| `build_fs_seeds.py` / `build_topup_queries.py` | 种子/补采构建 |
| `build_eval_manifest.py` | 评测 manifest |
| `clean_queries.py` / `clean_buffer.py` | 数据清洗 |

## Pipeline

| 脚本 | 说明 |
|------|------|
| `cold_start_pipeline.sh` | 全链路冷启动 |
| `run_cold_pipeline.sh` | 简化冷启动 |
| `run_data_pipeline.sh` | 数据 pipeline |
| `run_w3_pipeline.sh` | W3 采集 pipeline |
| `qc_cold_start.sh` | QC 质检 pipeline |
| `chain_buffill.sh` | buffer 补齐链 |
| `filter_and_borrow.sh` | 过滤 + 借任务 |

## 评测 / 分析 / 测试

| 脚本 | 说明 |
|------|------|
| `eval.sh` | 评测入口 |
| `mock_judge.py` | Mock judge |
| `serve_reward_model.sh` | Judge vLLM 服务 |
| `calibrate_judge.py` | Judge 校准 |
| `agents_harness.py` | Agent 端到端测试 |
| `verify_endpoints.py` / `verify_binary_extraction.py` | 端点/二进制验证 |
| `qc_trajectory.py` / `qc_trajectories.py` | 轨迹 QC |
| `capability_bucket_discovery.py` / `claweval_bucket_discovery.py` | 桶发现 |
| `score_bucket_coords.py` | 桶坐标评分 |
| `phase_summary.py` | Phase 汇总 |
| `analyze_observer_health.py` | Observer 诊断 |
| `warmup_buffer.py` | Buffer 预热 |
