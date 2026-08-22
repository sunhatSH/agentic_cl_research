# Reward 归因方法与调优经验（agentic CL）

> **本文重点：我是怎么一步步发现问题的**——从"reward 为什么不涨"这一个问题出发，靠**查得分 → 得分归类 → 低分归因 → 证伪 → 真机验证 → 判超参**的方法链，把一个笼统的"效果不好"拆成 4 个可定位、可修的具体问题。这是**调优方法论**文档（不是 bug 记录，结论条目另见 `doc/debug/Bug_Fix_精简总表.md` F7/F8）。
>
> 起点实验：cl2r_baseline（r0，CLEAR baseline）。日期：2026-08-22。
> 数据源：`rollouts/training/cl2r_baseline/rollout_status-*.jsonl` + `logs/harness/train/step-*/` + `logs/experiments/cl2r_baseline/train.log` + `datasets/train_cl.parquet`。

---

## 一、发现问题的完整步骤（方法链）

### 前置：先搞清楚数据落在哪、训练到底读哪个文件

归因的第一件事不是看指标，是**确认原料**。四类落盘数据：

| 数据 | 路径 | 有什么 |
|------|------|--------|
| 每步 rollout 打分 | `rollouts/training/cl2r_baseline/rollout_status-<step>.jsonl` | `{task_id, bucket, rollouts:[{status, reward, task_done, correctness, trajectory, safety}]}`（**无 messages 全文**） |
| 轨迹全文+结果 | `logs/harness/train/step-<N>/session-*/` | `hermes.log`（agent 全程）+ `meta.json`（outcome/elapsed_s/error_code/instruction）|
| 训练动态 | `logs/experiments/cl2r_baseline/train.log` | verl 每步 `critic/rewards/mean`、`actor/ppo_kl`、`grad_norm`、`entropy`（被 lightllm 日志淹没，要 grep）|
| 难度/桶标注 | `datasets/train_cl.parquet`（=`train_cl.jsonl`）| `extra_info.difficulty`（STRING '4'~'7'）、`bucket`、`record_id`（← 用它关联 task_id）|

> ⚠️ **第一个坑，也是最重要的教训**：一开始我用错数据源 `train_aligned.parquet`（只匹配上 346/1279 task），据此算出"难题 reward 更高"的**错误结论**。后来查训练 config `configs/exp1/cl2r_base.yaml` 的 `data.train_files` 才发现训练真正读的是 `train_cl.parquet`（100% 覆盖 1279 task），换对以后结论直接反转。**归因前必须确认训练读的到底是哪个文件。**

### 步骤 1｜查得分趋势：先判断"下降"是不是真的

从 `train.log` grep 出 20 步 `critic/rewards/mean`，做线性拟合。
- 结果：**0.49±0.03 抖动，斜率 −0.0013/step**（首 0.505→尾 0.493，区间 [0.449, 0.517]）。
- 判断：这是 GRPO 单桶 512 轨迹/步的**正常采样噪声**，不是系统性下降。20 步太短，谈趋势为时过早。
- 方法要点：**先量化，别被单点低值带偏**（step18/19 的 0.449 只是局部低点）。

### 步骤 2｜得分归类：把 reward 分档，拆成子维度看谁在拉分

reward 是复合的：`task_done ? 0.4·correct+0.4·traj+0.2 : 0.4·traj`，再 ×safety。所以先按 reward 分三档，看四个子维度的结构差异：

| reward 档 | task_done率 | correctness | trajectory | safety |
|-----------|-------------|-------------|-----------|--------|
| 高 >0.8 | 100% | 0.904 | 0.813 | 1.000 |
| 中 0.4–0.6 | 98% | 0.279 | 0.536 | 1.000 |
| 低 <0.15 | **4%** | 0.101 | 0.273 | 0.819 |

- 结论：**reward 由 task_done 门控决定**——低分档几乎全是 task_done=0（门控直接砍到底）。锁定 task_done 为突破口。

### 步骤 3｜难度交叉：验证"每步难度均衡"+"难题是否真更低"

- 难度 vs reward：d4=0.601(done 66.6%) / d5=0.585 / d6=0.425(done 44.4%) → **难题 reward 更低**（推翻前置那个错误记忆）。
- 每步难度构成：逐 step 打印，恒为 d4:21 / d5:3~4 / d6:39~40（均值 5.28）→ **每步难度均衡=已实现**（`build_train.py::_balanced_order` 生效），难度漂移被钉死，**不是波动源**。
- 方法要点：把"每步配比"逐 step 打出来一眼看均衡；难度×reward 交叉能证伪主观记忆。

### 步骤 4｜低分归因：读轨迹全文，定性 judge 判对还是判错

流程：rollout_status 挑低分 task_id → 关联 `train_cl.jsonl` 拿 instruction → 到 `logs/harness/train/step-*/` 按 instruction 前缀匹配 session → 读 `hermes.log` 全文 + `meta.json`。

抽 10 条低分轨迹**逐条对比【指令要求的产出】vs【agent 实际做了什么】**：
- **77%**：产出了内容但没达标（要求写 JSON 文件到指定路径，agent 只在对话打印 markdown 摘要 / 没写文件 / 不符 schema）→ judge 判 0 **判对**。
- **14%**：环境缺输入文件（要审查的代码库 / CSV 没预置），agent 只能反问或 BLOCKED → judge 判对，**但根因是数据/环境**。
- **8%**：纯反问 → **要分情况**（见步骤 6 派生问题）。
- 结论：**judge 没系统性判严**，低分主体是 agent 真没达标 + 环境缺文件。**不能靠松 judge 抬分**（那是 reward hacking）。

### 步骤 5｜关键证伪：是"工具 bug"还是"agent 没做"？

低分 1042 条**全部 status=success**（没崩没超时）但 task_done=0。用户提出关键怀疑："是不是沙箱不能写盘 / 没权限 / 路径不存在就拒绝写？" 证伪方法：
1. **反证**：看高分轨迹能否成功写文件 → unk_60001 明确写出 `./outputs/xxx.txt`（7511 bytes）→ **沙箱能写盘**，排除普遍性权限问题。
2. **按证据分类低分**：写成功 21% / **写失败报错 6%** / 根本没调写工具 72%（只打印或反问）。
3. **精读那 6% 的报错** → `[write_file] Failed to write file: //.hermes-tmp.505: Permission denied`、`/workspace/.hermes-tmp: No such file` → **锁定 Hermes `_atomic_write` 临时文件父目录 bug**。
- 方法要点：用"高分能做到 ⇒ 无普遍性障碍"证伪整体假设，再对失败样本按证据分类，把"工具 bug（6%）"从"agent 行为（72%）"里剥离出来。

### 步骤 6｜真机复现 + 验证修复（CPU 机即可，不需 GPU）

`scripts/verify_write_fix.py`：起真实 e2b 沙箱，照抄 Hermes `_atomic_write` 的 shell 逻辑，对不同目标路径测【修复前】vs【修复后】。只测文件系统行为，无需模型/GPU。
- 复现：`/home/user/outputs/x.json`、`/workspace/x.py`、`/x.py` 修复前全失败。
- 探针发现：沙箱用户 `user`(uid1000)、默认 cwd=/home/user、根 `/` 普通用户不可写。
- 方法要点：**能真机验证的绝不停留在推测**。沙箱验证不吃 GPU，CPU 机就能复现工具 bug、验修复。

### 步骤 7｜判定：prompt 问题 vs 超参问题

判据两头看——(a) judge 信号有无区分度（prompt 侧）、(b) policy 有没有在动（超参侧）：
- **prompt 侧健康**：trajectory 0.27→0.81、correctness 0.10→0.90 分得开，`cl/reward_std` 稳定 0.28（组内有对比信号，GRPO 有梯度可用）。
- **超参侧异常**：`actor/ppo_kl≈−0.0006` 20 步纹丝不动、`pg_clipfrac` 0.004 恒定、`lr=2e-6 constant + mini_batch=64 每步单更` → **policy 基本没离开初始点，模型还没开始学**。
- 结论：**reward 信号本身是对的，不涨的决定因素是 lr 太小（超参），不是 judge prompt。**

---

## 二、由归因定位的 4 个问题及处置

| 问题 | 性质 | 处置 | 状态 |
|------|------|------|------|
| **write_file 临时文件父目录 bug** | 工具（Hermes v2026.6.5 锁死不可改） | 见下方三层修复（数据+init+hook） | ✅ 真机验证 |
| **数据引用根目录 `/workspace`** | 数据 | `scripts/data/rewrite_workspace_path.py` 改写 parquet+jsonl（1661 行） | ✅ 已落地 |
| **judge 一律把反问判 0** | reward prompt | `REWARD_RUBRIC` 加"反问分情况"：judge 判必要性，必要的首次澄清→task_done=1 | ✅ 57 单测过 |
| **lr 太小 / mini_batch 单更 → policy 没动** | 超参 | 待调：lr 2e-6→5e-6/1e-5、拆 mini_batch，跑到 100 步看 ppo_kl 是否离 0 | ⏳ 待集群 |

### write_file bug 的三层修复（不碰 Hermes/verl 源码）

Hermes `_atomic_write` 把临时文件 `.hermes-tmp` 建在【目标父目录】，父目录不存在/不可写就崩。分三层根治，覆盖"根路径 + 根目录 + 深层子目录"：

1. **数据层**（治本，去非必要 sudo）：`scripts/data/rewrite_workspace_path.py` 把指令里 `/workspace`→`/home/user/workspace`（可写），正则后瞻含中文标点、排除 `/workspaces` 复数别词；同步改 `train_cl.parquet`（训练读）+`.jsonl`，prompt+extra_info 两处，**1661 行、0 残留、3 处 /workspaces 保留**。
2. **沙箱层**（`configs/exps/agent_loop_config.yaml` 的 `init_command`，官方注入点）：`mkdir -p /home/user/workspace /home/user/outputs; cd /home/user`——建常用根 + cd 回可写区，**无 sudo**。
3. **hook 层**（`src/trainer/mkdir_deliverable_hook.py::MkdirDeliverableHook`）：per-task 建深层子目录。复用 verl `AgentRunHook.prepare(sandbox, ctx, state)`（agent 命令**前**跑、持 sandbox+instruction），从**本任务指令**解析 `/home/user/{workspace,outputs}/...` 的父目录 `mkdir -p` 一层层建——每任务平均只建 ~1 个自己引用的目录，**零污染**（不给每个沙箱硬建 29 个无关空目录）。注册在 hooks 段（现有 factory patch 认 FQN）。9 单测 + 真机验证。

**真机验证覆盖**：workspace/outputs 根 ✅、深层子目录（`tests/`、`tcs_wave_a/frontend/js/views/`）✅；只剩 agent 主动写根 `/xxx`（数据不引用的坏路径）不兜，交 RL 收敛。

### 「读不到」的归因：不是文件缺失，是文件没拷进来（F9 / F5 延续）

约 900 个 review 任务要读 `/home/user/workspace/` 下的 `app.py`/`test_app.py` 等，沙箱里读不到。**第一反应容易误判为"数据缺失、无解"**——我一度就是这么判的。纠正的关键是**不轻易下缺失结论，用证据核对**：

1. **追路径来源**：review 任务 seed_query 里嵌着原始绝对路径 `/mnt/afs_toolcall/tongronglei/workspace/subagent_trajectory/<branch>/<D>/<taskdir>/ws/...`——这是采集者的工作区。
2. **核对文件是否真在**：`extract_ws_dir` 抽出 ws 标识，逐个核对 review 任务引用的文件（`TaskForge/module_a.py`、`inputs/app.py` 等）→ **在 tongronglei ws 里 100% 存在**。所以**文件不缺失**，只是没随数据集保存、没注入沙箱。
3. **顺带发现路径归错**：旧 F4 把这批 ws 路径**错误**归一到 `./outputs/`（应是 workspace），导致 query 路径和文件位置双重错位。

**教训（写给未来的自己）**：碰到"沙箱读不到文件"，先问"文件到底在不在某处"，别直接判缺失。seed_query / source_file 里的原始路径就是线索；文件在采集者目录、只是没拷进项目，是很常见的情况。查证"文件存在但没接上" vs "文件真没了"，是决定"能修"还是"无解"的分水岭。

修法（数据/prompt 重构 + 补齐文件，四步，不碰 Hermes/verl）：
1. `path_normalize` 加 `_REVIEW_WS`（ws 完整路径→`/home/user/workspace/`，置于通用 `.../ws/→./outputs` 之前）+ `extract_ws_dir()`（归一化前抽 ws 标识）+ bare `/workspace`→workspace（并入 F8 事后规则，重建时自动生效）。
2. `scripts/data/copy_review_ws.py`：从 tongronglei 拷训练集用到的 89 个 ws（137 任务，316MB，排除运行时/二进制/>5MB）进 `datasources/review_ws/`，写 `index.json`（record_id→ws）。
3. `cl_agent_dataset.build_agent_assets` 加 review-ws 分支：按 record_id 从 index 定位 ws，整树 `type=dir` 注入沙箱 `workspace`（=/home/user/workspace）。
4. `build_train.py` 透传 `extra_info.ws_dir`；`scripts/data/refresh_train_paths.py` 增量修 parquet（不全量重建、保 step 结构）：85 行 query 路径修正、67 行补 ws_dir、0 残留 bare /workspace。

**真机 e2e**：ws 注入沙箱后 agent 按 query 路径 `head /home/user/workspace/app.py` 读到真实代码 ✅。闭环：query 指 workspace ↔ 注入铺 ws 到 workspace ↔ ws 含被 review 的文件。

---

## 三、可复用的调优经验（跨实验）

1. **先确认数据源**：归因前查 config `data.train_files` 到底读哪个 parquet（本次栽在 train_aligned vs train_cl，算出反向结论）。
2. **reward 是复合的，先拆维度**：task_done 门控 / correctness / trajectory / safety，锁定谁在拉分再深入。
3. **趋势要量化**：算斜率，别被单点误导；短 run（20 步）谈趋势为时过早。
4. **judge 判对 ≠ 没问题**：低分主体常是"数据坏 + 模型没学会"，不是 judge 苛刻。**别靠松 judge 抬分**（reward hacking）。
5. **用反证证伪整体假设**：怀疑"沙箱不能写"？看高分能不能写——能写就排除普遍性障碍，再对失败样本按证据分类。
6. **能真机验证就别推测**：沙箱验证不吃 GPU，CPU 机即可复现工具 bug、验修复。
7. **prompt vs 超参判据**：judge 有区分度（分档拉得开、reward_std 不塌）说明信号 OK；`ppo_kl≈0` 说明 policy 没动 → 超参问题。
8. **不用非必要 sudo**：能从数据层解决（改路径）就不在沙箱里提权——非必要 sudo 是坏味道。
9. **反问要分情况**：第一次面对客观缺失/真歧义的反问是**必要**的，不该判 0（否则教模型"宁可瞎编也不问"）；信息齐全仍反问才算失败。
10. **"读不到"先查文件在不在，别急着判缺失**：seed_query/source_file 里的原始路径是线索；文件常在采集者目录、只是没拷进项目。"存在但没接上"能修，"真没了"才无解——先核对再定性。

