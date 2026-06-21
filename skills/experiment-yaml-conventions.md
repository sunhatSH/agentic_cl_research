# Skill: 实验 YAML 规范（OmegaConf 继承 + verl Hydra key path）

## 适用场景
- 一组消融实验共享大部分配置、只在少数字段上分叉（本项目 20 个实验）。
- 配置最终要喂给 verl（Hydra schema），但实验作者只想写「与 base 的差异」。

## 核心步骤
1. **一个 `base.yaml` + 每实验只写 override**：实验 yaml 顶部 `defaults: [../base]`，其余只列变化字段。`trainer/cl_main.load_config` 用 `OmegaConf.merge(base, exp)` 解析继承。
2. **所有字段放在 verl 的规范 key path 下**，这样 merge 到 verl `ppo_trainer` defaults 时才会生效：
   - 模型：`actor_rollout_ref.model.path`
   - actor：`actor_rollout_ref.actor.{entropy_coeff,use_kl_loss,kl_loss_coef,kl_loss_type}`
   - 参考策略：`actor_rollout_ref.ref.path`
   - rollout：`actor_rollout_ref.rollout.n`
   - 数据：`data.{train_files,val_files,train_batch_size}`
   - trainer：`trainer.total_training_steps`（不是 `total_steps`）
3. **CL 私有段独立**：`cl:` / `cl.buffer:` / `cl.weighting:` 是本项目自有段，只被 `cl_main`/`cl_loss` 读取，verl 忽略。
4. **未决值用 `???`（OmegaConf MISSING）**：Phase 4 网格待定的 `λ` 等用 `???`，校验时用 `OmegaConf.is_missing` 跳过解析。
5. **全量校验测试**：一个参数化测试 load 所有 20 个 yaml，断言 key path 正确、无死配置、scheme 合法、buffer 可构建。

## 关键约束
- **不要写顶层 `actor:` / `model:`**——必须在 `actor_rollout_ref.*` 下，否则 verl 读不到。
- **不要引用未注册的 `policy_loss.loss_mode`**（如 `cl_grpo`）：loss 走 `set_loss_fn` 注入，留 verl 默认 `vanilla` 即可，写死的自定义 mode 是 dead config。
- **OmegaConf 列表是「替换」不是「合并」**：override `bucket_names` 会整体替换 base 的列表（单桶塌缩依赖这一点）。
- **traj/query = 8**：`actor_rollout_ref.rollout.n` 在 `base.yaml`；Phase 5 另扩 `data.train_batch_size`（query 数）。
- **运行时产物 gitignore**：ckpts/wandb/logs/datasets 不提交。

## 代码锚点
- `configs/base.yaml`（overlay）、`configs/phase{1..5}/*.yaml`（20 个）。
- `trainer/cl_main.py`：`load_config`、`build_buffer`。
- 校验对照：`.venv/lib/.../verl/trainer/config/ppo_trainer.yaml` 与 `actor/actor.yaml`。
- 测试：`tests/test_configs.py`（`test_found_all_twenty_configs`、`test_config_loads_and_matches_verl_schema`、`test_buffer_builds_when_fully_specified`）。
