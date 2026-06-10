# Skill: ClawEval 评测与遗忘度量

## 适用场景
- 用一个固定基准（ClawEval）在多个 checkpoint 上评测，并量化**持续学习的遗忘程度**与**新任务性能**。
- 评测数据（manifest）由他人提供、尚未到位，但需要先把**接口与度量**定形，让 harness 可在 fake 数据上单测。

## 核心步骤
1. **冻结 manifest 接口**：manifest 是 JSON task 列表，必填 `task_id` / `split` / `modality`，可选 `bucket`/`category`/`prompt`/`messages`。`load_tasks` 加载即校验，默认只保留 `modality=="text"`（195 子集）。
2. **rollout 与评分解耦**：`rollout_one_task(ckpt, task)` 是唯一与真实模型耦合的点；harness 其余部分（聚合、度量）可用 fake rollout 单测。
3. **Pass^N 聚合**：每任务跑 N 次独立 rollout（默认 3），`passed_all = all(run.passed)`，子分数（safety/completion/robustness/reward）跨 run 取平均。评分公式 `reward = safety·(0.8·completion + 0.2·robustness)`。
4. **遗忘度量**：按 `task_id` join 当前与 baseline 的 reward，`old_task_forgetting = mean(max(0, base − cur))`；`new_task_performance`；`cl_score(new_perf, forgetting, α)`。
5. **结果落盘**：`eval/results/<run_id>/{per_task.json,summary.json}`。

## 关键约束
- **manifest 校验前置**：split ∈ {General, Multimodal, Multi-turn}，modality ∈ {text, multimodal}；非法记录立即 `ValueError`（带索引定位）。
- **当前只评纯文本 195 任务**：默认 `text_only=True`；要含 multimodal 显式传 `--include-multimodal`。
- **不要把 C-series 当 General 证据**：C-series 是 Multi-turn split，不能作为 General split 的 24-category 依据（领域映射仅作直觉参考，真实 category 需从 HF 数据集统计）。
- **Pass^N 是「全过才算过」**：任一 run 失败即 `passed_all=False`，不要用「多数通过」。

## 代码锚点
- `eval/run_eval.py`：`REQUIRED_MANIFEST_FIELDS`、`validate_task_record`、`load_tasks`、`score_task`、`evaluate`、`aggregate`、`rollout_one_task`（待接 verl 推理）。
- `eval/metrics.py`：`new_task_performance`、`old_task_forgetting`、`cl_score`、`output_entropy`、`trajectory_diversity`。
- 度量基准说明：`doc/ClawEval_Metadata.md`（含 General category 证据缺口标注）。
- 测试：`tests/test_run_eval.py`（fake manifest + fake rollout）、`tests/test_metrics.py`。
