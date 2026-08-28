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

### 「读不到」的二次归因：不是 900 个，是 1667 个；且 research d4-6 干净池=0（F9 收尾，2026-08-23）

F9 第一轮只修了"有显式 ws 路径"的 67 个（train_cl 内）。但全量核验发现**坏行远不止这些**——用 `_file_ok`（query 引用 `/home/user/workspace` 或 `./inputs` 读文件，但文件源不存在）扫全表，**1667 行缺源**（coding 750 + research 917）。这批 query 里只有 `/workspace/xxx`（无 ws 绝对路径、无 D_xxx id），`extract_ws_dir`/`extract_gen_task_id` 都抽不出关联线索。

**定位过程（怎么查出来的）**：
1. **先确认"文件到底在不在"**：抽 research d4-6 的 query，引用 `research_brief.md`/`app.py` 等 → 在 tongronglei 的 raw ws 里找 → **文件在**（不缺失）。但 raw 目录是 parent 级（299 个），child 轨迹（34068 行）无任何 id 字段关联到它属于哪个 raw → **无法确定性映射**。
2. **试从轨迹重建文件**（方案 B）：用 seed_query 文本匹配轨迹会话 → 从 tool 输出拼文件 → 判完整性。跑出 514 个完整、866 不完整、282 锚不到。**但这是不可信的**——定位靠文本匹配（非确定性 id），内容没核验。**废弃**（删 `rebuild_review_ws_from_traj.py` + `_rebuilt/`）。
3. **查 id 溯源机制**：`extract_new_queries.py` 里 `rid = metadata.session_id or metadata.request_id or f"unk_{行号}"`。非 unk_ 的题（47835 个）有真实 session_id；unk_ 的题（71928 个）轨迹只有 messages+tools、**无任何元数据** → id 不可溯源是采集时就造成的，不是我没找到。
4. **查"能溯源的题怎么找到文件的"**：发现**全部靠 query 里的路径文本匹配**，不是靠 id。unk_ 题靠 query 里 `E:\hermes\runtime\bigtasks\D10\D10_k982304_zh\inputs\` 抽 gen_task_id；非 unk_ 题靠 `/workspace/user/.cache/cttap/D10_xxx`（但 `extract_gen_task_id` 正则只认 bigtasks/winruns，抽不出 cttap → 非 unk_ 反而找不到文件）。

**关键转折（用户点破）**：research d4-6 的 842 个"引用 workspace"题，精确判定后**真需底稿=0**——它们是产出型（"gather facts for a section in research_brief.md ... Do not edit files. Return JSON"），引用路径只是上下文/产出目标，**不需要预置文件**。我的 `_file_ok` 把"引用 workspace"一刀切判成读文件，误判了产出型。

**但用户最终口径**：即便产出型不需要文件，**缺源的也要换掉**（id 不对、路径不合理的都换），保证"文件能找到"。research d4-6 干净池=0（它们本身就是缺源的那批），所以 research 只能用 d7。

**最终修法（替换，不重建）**：`scripts/data/replace_bad_review_tasks.py`：
- 坏行判定：`_file_ok`（query 读文件但文件源不存在）= 1667 个。
- 替换池：`new_trajectories_labeled.jsonl` 同桶 + 不在保留好行 + unique + **文件齐全**（不引入新缺源，加了 `seed2traj_taskspecs/<rid>/files` 第三查找分支）。
- 优先级：非 unk_ d4-6 → 非 unk_ d7 → unk_ d4-6 → unk_ d7（尽可能用可溯源 id）。
- 结果：coding 750（31 个非 unk_，难度 d4 为主）、research 917（12 个非 unk_，全 d7，因 research d4-6 干净池=0）。
- **替换后 0 缺源、0 重复、12800 行结构不变**。

**build_agent_assets 三条查找分支**（让文件能找到）：
1. `gen_task_id` → `generated_tasks_hermes/<D>/<gid>/inputs`（数据分析任务，query 里 D_xxx 路径）
2. `record_id` → `taskspecs_w3/<rid>/files`（旧 taskspecs）
3. `record_id` → `seed2traj_taskspecs/<rid>/files`（dirty.bak 的 s_hash_tN 任务，本轮新增）

**妥协（用户口径"实验，过拟合无所谓，认了"）**：
- research 917 个全 d7（d4-6 干净池=0，难度偏难）。
- 可溯源 id 仅 81/12800（research 可溯源题只有 28，不够；用户禁用 dirty.bak 补）。
- 核心目标"0 缺源、文件能找到"达成。

**最终验证**：12800 行全量核验——9805 不引用路径（产出型，无需文件）+ 2995 引用路径且文件源存在 + **0 缺源**。读文件的题走三条查找分支都能定位到文件。

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
11. **id 溯源 vs 路径文本匹配**：现在"能找到文件"靠的是 query 里的路径文本匹配（抽 gen_task_id/ws_dir），不是靠 record_id。unk_ 行号 id 不起任何作用——id 和文件定位脱节。要"id 能溯源"得有真实 session_id，但采集时没存元数据的题就做不到。这是数据采集阶段的问题，不是重构能补的。
12. **产出型 vs 读现有要分清**：query 引用 `/workspace/xxx.md` 不一定是读现有文件——可能是产出目标路径（"写到这个文件"）或上下文（"为 brief 的某一节收集资料，Do not edit"）。`_file_ok` 一刀切判"引用 workspace = 读文件"会误判产出型。精确判定要看动词（read/inspect/update vs produce/return json/do not edit）。
13. **替换池要加"文件齐全"约束**：换进来的题自己不能又缺源，否则替换后仍有坏行（第一次干跑 952 仍缺源就是这个）。替换池必须预筛 `_file_ok`，保证 0 缺源。
14. **数据源要找对**：用户说"四万多数据源在 datasources 下"，我反复找错（taskspecs_labeled.jsonl 4941、dirty.bak 9321、new_trajectories_labeled 119763）。最终定位：`new_trajectories_labeled.jsonl` 里非 unk_ 的 47835 个就是"四万多可溯源"，但 research 桶可溯源仅 28。**别再乱猜，直接问用户具体文件名**。
15. **妥协要明确记录**：research d4-6 干净池=0、可溯源 id 不够——这些是硬卡点。用户口径"实验，过拟合无所谓，认了"：接受 research 全 d7、大部分 unk_ id，保"0 缺源、文件能找到"核心目标。

---

## 四、web 工具缺失压低 reward（2026-08-27，评测归因）

**现象**：评测日志里模型大量调 undefined function，`web_search` 97 次、`web` 86 次、`web_fetch` 2 次、`x_search`/`google_search` 8 次。检索类任务（research/qa 桶尤甚）拿不到网页内容 → 做不完 → task_done/correctness 掉 → reward 被系统性压低。**这是继 F4–F9 数据/文件坑之后，又一条"环境能力缺失压 reward"的归因**（不是模型不会，是工具根本不可用）。

**根因**：hermes 的 `web` toolset 原生就该暴露 `web_search` / `web_extract`，但需要一个 web search **provider backend** 才真正注册可用。本项目原来只把 serper/jina 当 `/opt/tools` 裸脚本（靠 terminal 调），`web.search_backend` 为空、无 provider → 这俩函数 undefined → 模型一调就失败。

**修复（需重建镜像才生效）**：`docker/sandbox/hermes_plugins/web/serper/` 写一个 `SerperWebProvider`（search 走 google.serper.dev、extract 走 r.jina.ai），Dockerfile COPY 进 `/home/user/.hermes/plugins/`，`hermes.config.yaml` 启用 `plugins.enabled=[web-serper]` + `web.search_backend/extract_backend=serper`；并把 `web_fetch` 注册成 `web_extract` 的别名（模型习惯叫 web_fetch）。

**⚠️ 未决隐患（check_fn gate）**：hermes 内置 `web_search`/`web_extract` 的 `check_fn=check_web_api_key`，而后者**硬编码只认 exa/tavily/firecrawl/... 8 个 backend、不认 serper、无 registry fallback**（`tools/web_tools.py:852`）→ 返回 False → `registry.py:417` 据此把工具**从模型可见列表剔除**（`continue`）。所以光注册 provider 可能不够：dispatch 时能跑（explicit config wins），但工具对模型不可见。**待办**：用 `register_tool(override=True)` 重注册这俩工具 + 换成认 serper 的自定义 check_fn。或先构建后实测 `hermes tools` 可见性再定。

**判 undefined 三分类的方法（写给未来的自己）**：
- **真能力缺口**（要修）：hermes 有这功能但没接上 backend → web_search/web_extract。
- **命名错**（模型能力问题，训练收敛，不用改环境）：hermes 有正确名，模型叫错——`read_file`✓/`ls`✗、`python`✗→`terminal`/`execute_code`、`skill_list`✗→`skills_list`。判定靠对照 `registry.register(name=...)` 的官方名清单。
- **纯幻觉**（模型编的，hermes 根本没有）：`feishu_doc`/`calendar`/`kanban`/`rss`/`molecular_finder`/`work_order_list` 等——无视。
- **排除"全局 gate"的证据法**：若某真实工具（如 read_file 16 次 undefined）被怀疑是 check_fn 门禁挂了，先看它 undefined 是否**均匀分布在所有 session**（全局 gate）还是**集中在少数 GatewayActor pid**（模型幻觉）。read_file/search_files 集中在 3 个 pid + 日志无 "Tool xxx unavailable (check failed)" → 判定是幻觉，file toolset 本身是通的。别看到真实工具名 undefined 就断定环境挂了。

**训练也一样缺 web（2026-08-28 核对）**：训练与评测**共用同一沙箱镜像 + `agent_loop_config.yaml` + `hermes.config.yaml`**，所以 web 工具缺失训练 rollout 同样撞。实测训练日志 `cl2r_baseline`/`cl2r_kl` 的 undefined：`web` 2228 / `web_search` 2140 / `web_fetch` 132 / `x_search` 117 / `google_search` 25 / `websearch` 17 / `browser` 153（browser toolset 未启用，属另一类）。**除 web 外无其它真能力缺口**——`read_file`/`write_file`/`search_files`（55/52/142 次）虽是真实工具名，但 undefined 分散在 6 个 GatewayActor pid、每个仅 1-4 次，且训练 reward 正常出（1420 次）、无 "check failed" gate 日志 → 判定为**模型幻觉**（base 没训过这套工具，乱猜名字），file toolset 本身可用。其余 `python`/`run`/`ls`/`delete_file`/`bash`/`echo`/`think` 等均为命名错或幻觉（hermes 官方是 `terminal`/`execute_code`/`read_file`/`write_file`/`patch`）。**结论：训练唯一的环境级工具缺口就是 web（search+fetch），与评测同源同修——serper provider 镜像修好后两边同时受益**；research/qa 桶 reward 被 web 缺失系统性压低，修后应回升。训练不出现评测的"0 条结果/sample 死等/session 墙钟"——那些是 eval `_run_eval` 专属路径，训练走 train 分区正常出 reward。


---

## 五、GT 分派不全 + LH 悬空 executable 引用（2026-08-27，构建 3200×2 后全量核验暴露）

三桶数据里 coding 桶有 D类/SWE/LH 三种 answer_key 形态。核验发现两处会**直接影响 reward 的 correctness 判断**：

### 5.1 GT 静默丢失 1089/6400 行 → correctness 无参照退化为开放判分
**为什么影响 reward**：`_load_ground_truth` 是 judge 判 correctness 的**唯一 GT 入口**。原来只有 SWE(rubric)/D(checks) 两分支，导致：
- **177 LH 全灭**（type=longhorizon 落 D 分支、checks 空 → 返回空）；
- **797 D类产出型**（`_s` from-scratch，GT 在 rubric 不在 checks）→ 空；
- **checks 键名碎片化**（name/value、description/expected、纯字符串、`[q,v]` 二元组…）多种没被认 → 空。

空 GT 时 judge 收不到"已知正确答案/验收标准"，correctness 只能**凭模型主观开放判**——这正是我们一直要避免的 reward hacking 风险面：无参照时 judge 容易被"看起来完成了"的漂亮轨迹骗高分。**1089 行（17%）的 correctness 信号原本是虚的**，修复后全部锚定到真实 GT（checks 命中率 / rubric 满足率）。

### 5.2 LH 悬空 executable 引用 → 误导 agent + 干扰 task_done 判断
**为什么影响 reward**：177 个 LH 的 query 都有 longhorizonCoding 格式标配一句"运行预编译 `test*_executable` 观察行为"+"**不要用 python，直接跑 executable**"。但该 executable 不存在（Python 源码已内嵌 query、files/ 也没有）。影响两头：
- **agent 侧**：被指令带去跑不存在的文件 → 报错/绕路 → 轨迹变脏、可能误判任务做不下去；
- **judge 侧**：judge 看到 query 要求"运行 executable"却没运行 → 可能据此压 task_done（"没按要求执行"），而实际上 agent 从内嵌源码完成迁移才是对的。

这是"query 文案与真实环境不一致"压 reward 的又一例（同 F4 Windows 路径、F9 缺文件一脉）——**环境/指令层面的噪声，不是模型能力问题**。修复后 query 指向内嵌源码，agent 和 judge 的依据一致，task_done/correctness 不再被悬空引用干扰。

**核验方法（写给未来的自己）**：无 files/ 的任务不能默认"自包含就没事"——用 AI（luna）逐条判 query 是否真不依赖外部文件。974 个无 files/ 里 AI 精准揪出 5 个 LH 有悬空 executable 引用（其余 LH 因源码内嵌更明显被判自包含,但其实全 177 个都有这句,AI 只标了最显眼的）→ 反查确认是格式标配 → 全量 177 一起修，而非只补 AI 标出的 5 个。**AI 核验抓典型，人再顺藤查是否全体同病。**

技术修复动作见 `doc/debug/Bug_Fix_精简总表.md` F14 + F15。
