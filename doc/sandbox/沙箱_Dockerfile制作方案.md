# 沙箱 Dockerfile 制作方案

> **状态**：方案已定（2026-06-25），待落地。本文件聚焦"怎么把 task 的初始文件系统（`files/`）做成可实例化的沙箱"。  
> **前置确认**：workspace ↔ query 保持 **1:1**（1 task = 1 `files/` = 1 seed_query）。  
> **数据源**：`data/taskspecs/<task_id>/{taskspec.yaml, files/}`（sample105 已弃）。  
> **当前阶段**：**简化版**——只需 1 个母版镜像即可。后续可能改进以模拟环境多样性（见 §6）。

---

## 1. 已定方案：1 母版镜像 + COPY 全部 seed + 启动时 env 选

**核心**：1 个母版 Dockerfile（装运行环境 + 全部 task 的 `files/`）→ build 成 1 个镜像 → 实例化多个容器，每个容器用 env 指定铺哪个 task 的文件结构。

```
1 个母版 Dockerfile
  ├─ 装运行环境（OpenClaw + 常用依赖，见 §3）
  └─ COPY fs-seeds/ → /opt/agentic-cl/fs-seeds/   ← 全部 task 的 files/ 打包进镜像
       ├─ <task_id_1>/  (= 该 task 的 files/ 内容)
       ├─ <task_id_2>/
       └─ ...（100 个 task = 100 个 seed 目录）

实例启动（seed_workspace.sh）：
  AGENTIC_CL_PERSONA=<task_id>
    → cp -a /opt/.../fs-seeds/<task_id>/ /root/workspace
  每个容器从同一镜像 fork、用 env 选自己的文件结构铺开
```

**为什么不用"文件内容内联进 Dockerfile"**（`RUN cat > file`）：
- task 的 `files/` 含二进制（Thumbs.db）、特殊字符文件名（`~$tmp.lock` 的 `$`、中文、空格）——内联转义不可维护。
- 100 个 task × 1000+ 文件，内联进 Dockerfile 不可行。
- 数据用 `COPY` 进镜像、运行环境用 `RUN` 装——职责分离，Dockerfile 干净。

**对应关系**（符合 1:1）：
- 1 task ↔ 1 `files/` ↔ 1 seed 目录（`fs-seeds/<task_id>/`）
- 1 个母版镜像含全部 seed（不是每 task 一个镜像）
- 1 镜像 → N 容器，每容器 1 个 seed（env 选）
- 1 seed_query → 8 容器（GRPO 8 路，从同一 seed fork）

---

## 2. 数据源结构（taskspecs）

```
data/taskspecs/<task_id>/
├── taskspec.yaml   ← 任务声明（seed_query/hidden_goal/verifier/user_profile/...）
└── files/          ← 初始 workspace（rollout 起点，平展文件系统）
    ├── reports/ 或 Desktop/  （agent 要处理的真实文件 + 噪声文件）
    ├── Thumbs.db / ~$tmp.lock / .DS_Store  （噪声层，原样保留——贴合真实使用场景）
    └── ...
```

- 14 个 task（首批），每 task `files/` 25–257 个文件、总量 <1MB。
- `files/` 直接作为 seed 内容（**原样保留，含噪声文件**——Thumbs.db 等是真实用户机器会有的，agent 要学会在噪声中定位任务文件，贴合使用场景）。
- `seed_query`（taskspec.yaml）= 该 task 的首 query，1:1 对应。

---

## 3. 母版镜像依赖（只装常用，不全装）

**原则**：母版只装**常用、稳定的依赖**；任务特定的依赖 **agent 运行时自己装**（`pip install` / `apt install`）——这贴合真实使用场景（用户不会预装所有可能用到的包，agent 该会自己装）。

**装进母版的（常用基础）**：
- 系统工具：`curl wget git jq ripgrep unzip zip less procps`
- 文档处理：`poppler-utils`(pdf) `pandoc` `fonts-noto-cjk`(中文)
- Python 基础 + 常用数据处理：`pandas openpyxl`(xlsx) `python-docx` `python-pptx`（OfficeQA/Finance 常用）
- OpenClaw runtime（agent harness）

**不装、留给 agent 自己装的**：
- 冷门/任务特定库（如 `pdfplumber`、特定 SDK）——agent 跑任务时按需 `pip install`。
- 重型依赖（如 playwright/Chromium）——纯文本 195 任务不需要，按需开。

> 现有 `docker/sandbox/Dockerfile` 已基本符合（装了系统工具 + OpenClaw + requirements.txt），只需把 `requirements.txt` 调成"常用基础"清单（去掉冷门、保留 pandas/openpyxl 等常用），不全装。

---

## 4. 落地步骤

1. **`taskspecs/*/files/` → `fs-seeds/<task_id>/`**：写自动化脚本，把每个 task 的 `files/` 原样拷成 seed 目录（含噪声文件）。
2. **更新 `fs-seeds/manifest.json`**：把 14 个 task 注册成 seed 条目（id=task_id、dir=task_id、bucket=该 task 分桶、summary=seed_query 摘要）。
3. **`seed_workspace.sh`**：现有机制已支持（按 `AGENTIC_CL_PERSONA=<task_id>` 铺开），task_id 直接当 persona id 用，**不用改脚本**。
4. **母版 Dockerfile**：`COPY fs-seeds/`（已有一行），`requirements.txt` 调成常用基础清单。
5. **测试**：build 母版 → 起容器 `AGENTIC_CL_PERSONA=<task_id>` → 检查 `/root/workspace` 内容 = 该 task 的 `files/`。

---

## 5. 已确认 / 不动

- workspace ↔ query 1:1（不搞 1:n）。
- 1 seed（files/）→ fork 8 容器跑同一 seed_query（GRPO 8 路，8 路从同一 seed 位级一致起点）。
- `files/` 原样作 seed（含噪声文件，贴合真实场景）。
- 母版只装常用依赖，特定依赖 agent 自己装。
- `seed_workspace.sh` 机制不动（按 task_id 铺开）。
- `rollout.n=8`（GRPO 8 路）不变。

---

## 6. 当前阶段：简化版（1 母版），后续改进方向

**当前**：只需 **1 个母版镜像**——所有 task 共享同一个母版（装 OpenClaw + 常用依赖），差异只在 `fs-seeds/<task_id>/`（启动时 env 选）。足够跑通当前 14 个 task 的训练/采集。

**后续可能改进（模拟环境多样性）**——当前不做，记下方向：
- **多母版**：不同 task_family / 难度 / 桶 用不同母版（如 OfficeQA 母版预装 Office 库、SysOps 母版预装运维工具），减少 agent 运行时装依赖的开销 + 模拟不同用户机器环境。
- **环境扰动**：同 task 多个 seed 变体（文件名/路径/噪声层不同），模拟"同一任务不同机器状态"，增强泛化。
- **依赖版本漂移**：母版里某些库留多版本/不装，让 agent 适配真实环境的不确定性。

> 这些是后续优化，当前阶段先把 1 母版 + taskspec seed 跑通。

> 待 rollout 契约定后，配合 `data_pipeline/`（读 taskspec.yaml）一起落地。
