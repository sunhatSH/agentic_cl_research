# vllm 版本升级与 Qwen3.6-27B 兼容性解决记录

> 日期：2026-06-16
> 环境：4×H800 (80GB)，CUDA 12.9，Driver 550.90.07

---

## 问题

Qwen3.6-27B 无法在 vllm 0.13.0 上加载，导致 verl 的 rollout（采样）环节完全不可用。

## 根因

Qwen3.6-27B 是**混合架构模型**（hybrid linear attention + full attention）：

- 48 层 `linear_attention`（Mamba2 风格）
- 16 层 `full_attention`（标准 softmax attention，每 4 层一个）
- 配置字段：`layer_types`，`full_attention_interval: 4`
- HuggingFace architecture：`Qwen3_5ForConditionalGeneration`

vllm 0.13.0 **不支持** `Qwen3_5ForConditionalGeneration`，也不认识 `linear_attention` 层。

## 尝试过的方案（全部失败）

### 方案 1：修改 config.json 的 architectures → Qwen3ForCausalLM

- **失败原因**：Qwen3ForCausalLM 不认识 `linear_attention` 层，只认标准 `self_attn`（q_proj/k_proj/v_proj/o_proj）。模型结构完全不同。

### 方案 2：TransformersForCausalLM 通用路径

- **失败原因**：权重 key 带有 `model.language_model.` 前缀（多模态封装），通用路径无法自动映射。

### 方案 3：patch Qwen3ForCausalLM.load_weights 做 key remap

- **失败原因**：vllm Worker 进程是 spawn 出来的，主进程的 monkey-patch 不会传播到子进程。改用 fork 模式后，仍然因为模型结构不兼容（linear_attention vs self_attn）导致大量权重无法加载。

### 方案 4：修改 model.safetensors.index.json + remap key

- **未完成**：意识到即使 key 映射成功，模型结构仍然不兼容（linear_attention 层没有对应的 PyTorch Module）。

## 最终方案：升级 vllm 到 0.19.0

Qwen 官方文档明确要求 `vllm>=0.19.0`。vllm 0.19.0 的 ModelRegistry 直接支持 `Qwen3_5ForConditionalGeneration`。

### 安装过程

```bash
# 1. 创建独立 venv（避免污染系统 /opt/conda 环境）
uv venv /mnt/afs_toolcall/sunhao4/envs/vllm019_venv --python /opt/conda/bin/python --clear

# 2. 安装 vllm 0.19.0（全依赖，自动选 CUDA 12.8 兼容的 torch 2.10）
uv pip install vllm==0.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --python /mnt/afs_toolcall/sunhao4/envs/vllm019_venv/bin/python

# 3. 安装 verl（不带 vllm extra，避免降级 vllm）
uv pip install -e /mnt/afs_toolcall/sunhao4/verl \
  --python /mnt/afs_toolcall/sunhao4/envs/vllm019_venv/bin/python \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 强制恢复 vllm 0.19.0（verl[vllm] 会把它降级到 0.12.0）
uv pip install vllm==0.19.0 --reinstall-package vllm \
  --python /mnt/afs_toolcall/sunhao4/envs/vllm019_venv/bin/python \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### ⚠️ 关键踩坑：verl[vllm] 会降级 vllm

verl 的 `setup.py` 声明 `VLLM_REQUIRES = ["vllm>=0.8.5,<=0.12.0"]`。如果用 `pip install -e ".[vllm]"` 安装 verl，pip 会把 vllm 从 0.19 降级到 0.12。

**解决**：不传 `[vllm]` extra，只装 `pip install -e .`，然后手动确保 vllm 0.19 已安装。

### 最终版本

| 组件 | 版本 |
|------|------|
| vllm | 0.19.0 |
| verl | 0.9.0.dev (main 分支, commit 5a38699c) |
| torch | 2.10.0+cu128 |
| transformers | (vllm 0.19 自带) |
| ray | 2.55.1 |
| Python | 3.11.14 |
| venv 位置 | `/mnt/afs_toolcall/sunhao4/envs/vllm019_venv/` |

### vllm 0.19 API 变化（vs 0.13）

| 0.13 | 0.19 |
|------|------|
| `--disable-log-requests` | `--no-enable-log-requests` |
| `from vllm.utils import FlexibleArgumentParser` | `from vllm.utils.argparse_utils import FlexibleArgumentParser` |
| 不支持 `Qwen3_5ForConditionalGeneration` | ✅ 支持 |

## CUDA 版本兼容性

| vllm 版本 | 需要 CUDA | 当前环境 (Driver 550.90.07, CUDA 12.9) |
|-----------|-----------|----------------------------------------|
| 0.13.0 | CUDA 12.x | ✅ 兼容 |
| 0.18.1 | CUDA 12.x | ⚠️ C extension 与 torch 版本冲突（--no-deps 安装） |
| 0.19.0 | CUDA 12.x | ✅ 兼容（torch 2.10.0+cu128） |
| 0.20.2 | CUDA 13 | ❌ 不兼容（libcudart.so.13 not found） |

## 清理的残留文件

- `/mnt/afs_toolcall/sunhao4/models/Qwen3.6-27B-text/` — 之前 patch config 的目录（config.json 被改过，权重是 symlink），不再需要
- `/mnt/afs_toolcall/sunhao4/envs/vllm013/` — vllm 0.13 overlay，不再需要
- `/mnt/afs_toolcall/sunhao4/envs/vllm018/` — vllm 0.18 尝试安装，失败
- `/mnt/afs_toolcall/sunhao4/envs/vllm020/` — vllm 0.20.2 尝试安装，CUDA 不兼容
- `/mnt/afs_toolcall/sunhao4/agentic_cl_research/scripts/vllm_serve_qwen35.py` — key remap patch 脚本，不再需要
- `/mnt/afs_toolcall/sunhao4/agentic_cl_research/scripts/_vllm_qwen35_patch.py` — Worker patch 模块，不再需要

## 遇到的额外问题

### 僵尸 GPU context

**症状**：vllm Worker 进程异常退出后，`nvidia-smi` 仍显示已死进程占用 GPU 显存（75GB/卡）。`/proc/<pid>` 目录不存在，`kill -9` 无效，`nvidia-smi --gpu-reset` 返回 "Not Supported"（即使有 sudo）。唯一恢复方式：**重启机器**。

**根因**：NVIDIA 驱动不会自动回收已退出进程的 CUDA context。vllm 的 `multiproc_executor` spawn Worker 后，如果 Worker 在模型加载阶段 crash（例如 architecture 不支持），CUDA context 就泄漏了。反复尝试不同配置会累积泄漏。

**预防**：
- 启动 vllm 前先用 `python -c "from vllm.model_executor.models import ModelRegistry; ..."` 确认架构支持
- 不要连续尝试会 crash 的 vllm 配置
- 启动脚本必须 `trap cleanup EXIT`，cleanup 里 kill 整个进程组
- 从 tmux 退出 vllm 时先 Ctrl+C 优雅退出，不要直接 `tmux kill-session`

### vllm 0.19 multiprocessing spawn 必须 if __name__ == '__main__'

**症状**：`RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase`

**根因**：vllm 0.19 的 Worker 通过 `multiprocessing.spawn` 启动，子进程会重新 import 主脚本。如果 LLM 初始化代码不在 `if __name__ == "__main__"` 守卫下，子进程会递归创建 LLM，导致死循环。

**解决**：所有调用 `vllm.LLM` 或 `vllm.entrypoints` 的脚本必须用 `if __name__ == "__main__":` 包裹。

### vllm 0.19 强制 spawn，忽略 VLLM_WORKER_MULTIPROC_METHOD

**日志**：`We must use the spawn multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. Reasons: CUDA is initialized`

vllm 0.19 在 CUDA 已初始化的情况下强制使用 spawn。这是与 0.13 的关键区别——0.13 支持 fork，0.19 不再支持。

## 后续

- vllm 0.19 + Qwen3.6-27B 推理验证进行中
- 验证通过后跑通采集流程
- verl 全栈 smoke 依赖推理验证结果
