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
