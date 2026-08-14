# Agent 轨迹 Schema（现状 vs 目标）

> **状态**：本文记录 Agent 轨迹的两种 schema——**现状**（mock 采集产物）与**目标**（buffer/训练消费）。  
> **日期**：2026-06-26  
> **背景**：训练 Agent **仅用 Hermes**（不用 OpenClaw）。冷启动 buffer 轨迹须由 π₀（Qwen3.6-27B）产出。  
> **关联**：`doc/source/CL_Design.md §2.2`、`trainer/trajectory_adapter.py`、`rollout/session_pool.py::Trajectory`、`doc/source/Hermes_Subagent_训练数据方案.md`。

---

## 0. 决策前提（影响 schema 设计）

1. **训练 Agent 仅用 Hermes**：轨迹由 Hermes 产（spawn 同步等待、子轨迹分离），不是 OpenClaw。
2. **冷启动 buffer 轨迹 = π₀ 产出**：`original_logprobs` 须是 π₀ 的（`forgetting_risk` 比的是 π_t vs π₀）、`L_replay` 把 π_new 钉在 π₀ 行为流形。用别的模型（如 Claude）跑的轨迹**不能进 buffer**（破坏抗遗忘），只能用于验证管道。
3. **验证期 actor 可临时换 Claude API**（sufy），跑通后换回 Qwen3.6-27B；**schema 字段不变**，只 actor 来源不同。

---

## 1. 现状 Schema（mock 采集产物）

**来源**：`data/mock/rollouts/test_*/rollouts_*.jsonl`（旧单 slot 冷启动采集，`collect_rollout.py`，OpenClaw 时代）。

**一个 jsonl 文件 = 多个 session，每行一个 session（非单条轨迹）**：

```jsonc
// 一行 = 一个 session
{
  "record_id": "real-001",              // 关联的 task/会话 id
  "seed_query": "Write a Python script that fetches...",  // 首 query
  "persona": "Grace the speechwriter",  // Questioner 人设
  "num_turns": 3,                       // 实际跑了多少轮
  "ended_by": "k_budget",               // 结束原因: k_budget|end_session|patience|scorer_error
  "generated_queries": ["...", "..."],  // 在线生成的后续 query（Questioner 产出）
  "trajectories": [                     // 每轮 winner 的轨迹（per-turn，非 per-slot）
    {
      "trajectory_id": "q0-s0",         // q<turn>-s<slot>
      "messages": [                     // OpenAI chat
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "<toolcall>{...}</toolcall>"},
        {"role": "user", "content": "[Sandbox Output]\n..."},  // tool 结果伪装成 user
        {"role": "assistant", "content": "I created the script..."}
      ],
      "bucket": null,                   // ⚠️ 未分桶
      "response_token_ids": [0,0,...],  // ⚠️ 多为空/占位（mock 无真前向）
      "response_mask": [1,1,...],
      "num_turns": 3
    }
  ],
  "reports": [                          // 每轮 ObservationReport（observer 产出）
    {
      "intermediate": [...],            // 中间结果
      "final": [...],                   // 最终交付物
      "actor_claims": "...",            // ⚠️ 旧字段名（目标 schema 已改名 actor_trajectory）
      "discrepancies": "",              // 红旗
      "file_tree": "..."                // 文件树
    }
  ]
}
```

### 1.1 现状的问题（与目标/buffer 的差距）

| 缺口 | 现状 | 目标 |
|------|------|------|
| **slot 数** | 仅 `s0`（1 slot/单轨迹采集） | 8 slot/query（GRPO 组） |
| **reward** | ❌ 无 | ✅ model judge 分（必填） |
| **original_logprobs** | ❌ 无 | ✅ π₀ 的逐 token logprob（抗遗忘根基） |
| **bucket** | ❌ None | ✅ 9 桶之一（必填，B12 否则跳过） |
| **query_index** | ❌ 隐含在 trajectory_id | ✅ 显式字段 |
| **格式来源** | OpenClaw/旧采集 | Hermes（spawn 同步、子轨迹分离） |
| **actor** | mock/远程对照 | π₀（Qwen3.6-27B） |
| **`actor_claims` 字段名** | 旧名 | 已改名 `actor_trajectory`（pass-through 语义） |

> 现状 mock 是"冷启动单轨迹采集"（1 slot、无 reward、无 logprobs），**不是完整 GRPO 训练轨迹**，不能直接进 buffer。它只用于验证管道/看结构。

---

## 2. 目标 Schema（buffer/训练消费）

**权威来源**：`trainer/trajectory_adapter.py::extract_trajectories_from_batch` + `doc/source/CL_Design.md §2.2` + `rollout/session_pool.py::Trajectory`。

**一个 buffer trajectory = 一个 slot 的一次 query rollout**（GRPO 组内 8 条之一）：

```jsonc
{
  // ---- 标识 ----
  "trajectory_id": "<全局唯一>",        // 必填
  "record_id": "<task_id>",             // 必填，关联 taskspec
  "task_id": "<同 record_id>",          // verl non_tensor 字段（adapter 读 task_id）
  "query_index": 0,                     // 必填，会话内第几个 query（0-based）
  "slot_idx": 0,                        // 必填，0–7（GRPO 组内槽位）

  // ---- 分桶 ----
  "bucket": "workflow",                 // 必填，9 桶之一；unknown/缺失 → 整条跳过（B12）
  "sub_bucket": "workflow",             // 可选，子桶（category，BucketDesign 清单）

  // ---- 对话（Hermes 主轨迹）----
  "messages": [                         // 必填，OpenAI tool-use chat
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<seed_query>"},
    {"role": "assistant", "content": "", "tool_calls": [         // Hermes spawn=同步 toolCall
      {"id":"...", "type":"function", "function":{"name":"sessions_spawn","arguments":"{...}"}}
    ]},
    {"role": "tool", "tool_call_id":"...", "content":"<子 agent 最终输出>"},  // 出栈=含子结果
    {"role": "assistant", "content": "<final answer>"}
  ],                                    // 仅本 query 产生的消息，不含 session 正史前缀
  "tools": [...],                       // 可选，OpenAI tools 定义

  // ---- token 级（抗遗忘 + replay 权重必需）----
  "response_token_ids": [123, 456, ...], // 必填，本 query response 的 token id
  "response_mask": [1, 1, 0, ...],       // 必填，response span 标记（pad=0）
  "original_logprobs": [-0.5, -1.2, ...],// 必填，π₀ 逐 token logprob（采集快照）
  "logprobs": [...],                     // 可选，当前策略 logprob（训练时重算/回填）

  // ---- 奖励 ----
  "reward": 0.72,                        // 必填，model judge 分；null → 整条丢弃（§4）
  "success": true,                       // 建议，judge/checker 二值结果

  // ---- 优先级信号（adapter 回填）----
  "success_rate": 0.625,                 // 组内通过率（within_bucket_difficulty）

  // ---- 元信息 ----
  "meta": {
    "policy": "pi0",                     // 冷启动固定 pi0
    "cold_seed": true,
    "actor": "Qwen3.6-27B",              // 产出 actor（验证期可 Claude，但 buffer 必须 pi0）
    "agent_framework": "Hermes",         // 训练仅用 Hermes
    "sandbox_id": "<task_id>",           // 沙箱 seed 标识（1:1 workspace）
    "persona": "...",                    // Questioner 人设
    "classifier_model": "...",           // 分桶用的模型
    "elapsed_ms": 12345
  }
}
```

### 2.1 子轨迹（subagent）—— 独立成样本，单独训练

Hermes 主 agent 调子 agent 时**同步等待**，子轨迹**独立成一条 buffer 样本**（不入主轨迹，见 `Hermes_Subagent_训练数据方案.md §3.2`）：

```jsonc
{
  "trajectory_id": "<parent>-sub-<child_session_id>",
  "record_id": "<parent_task_id>-sub-<child>",
  "parent_record_id": "<parent_task_id>",       // 父链接
  "spawn_toolcall_id": "<父轨迹 sessions_spawn 的 toolCall id>",
  "bucket": "<按子 task 分桶>",                  // 可二次 classify
  "messages": [                                  // 子 agent 自己的事件流→OpenAI chat
    {"role": "system", "content": "<子角色 prompt>"},
    {"role": "user", "content": "<父 spawn 传入的 task>"},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "<子最终输出>"}
  ],
  "response_token_ids": [...], "response_mask": [...],
  "original_logprobs": [...],                    // π₀ 的
  "reward": ..., "meta": {"is_subagent": true, ...}
}
```

> 主轨迹里 `sessions_spawn` 的 toolResult = 子 agent 最终输出（Hermes 同步，出栈干净）。主轨迹**不内联**子消息。

### 2.2 Session 级封装（采集产出形态）

采集时一个 session 产 1 条 session 记录（含多 query × 8 slot 的轨迹），落盘后由 `route`/`trajectory_adapter` 拆成单条 buffer trajectory：

```jsonc
{
  "session_id": "...", "task_id": "...", "seed_query": "...",
  "persona": "...", "sandbox_id": "...",
  "queries": ["<seed_query>", "<follow-up 1>", ...],  // 首query + Questioner 生成
  "num_turns": N, "ended_by": "end_session|k_budget|...",
  "trajectories": [ /* query_0 × 8 slot + query_1 × 8 slot + ... */ ],
  "reports": [ /* 每轮 ObservationReport（含 actor_trajectory pass-through） */ ]
}
```

---

## 3. 现状 → 目标 的差距与获取路径

| 字段 | 怎么从现状补到目标 |
|------|-------------------|
| 8 slot/query | 现状单 slot → **自跑 GRPO 8 路**（Hermes `SessionSandboxPool`，8 槽同 seed fork） |
| `reward` | **model judge 打分**（taskspec.verifier rubric → reward judge，anthropic/claude-4.8-opus，走 sufy） |
| `original_logprobs` | **π₀ vLLM 前向产出**（必须 Qwen3.6-27B，不能 Claude） |
| `bucket` | **LLM 分桶**（对 seed_query 分 9 桶，复用 classify） |
| `query_index`/`slot_idx` | rollout 时由 Hermes scheduler 填 |
| 格式 | **Hermes 产**（spawn 同步、子分离），非 OpenClaw |

### 获取决策（已定）

- **验证管道/看结构**：自跑 + Claude API（sufy），临时、用完弃，**不进 buffer**。
- **冷启动 buffer（14–20k 轨迹）**：自跑 + **Qwen3.6-27B（π₀）**，集群 GPU vLLM。唯一正确选项。
- 上游拿：仅当上游 actor=Qwen27B 且提供 token 级 logprobs 且能转 Hermes——条件太苛，基本不考虑。

---

## 4. 字段必填性总表（buffer 入库校验）

| 字段 | 必填 | 缺失处理 |
|------|------|---------|
| `trajectory_id` | ✅ | 无则跳过 |
| `record_id`/`task_id` | ✅ | 无则跳过 |
| `query_index`/`slot_idx` | ✅ | 无则跳过 |
| `bucket` | ✅ | unknown/缺失 → **整条跳过**（B12，不静默归默认桶） |
| `messages` | ✅ | 无则跳过 |
| `reward` | ✅ | null → **整条丢弃**（§4，不静默 0 分） |
| `response_token_ids` | ✅ | 无则跳过（replay 权重要） |
| `original_logprobs` | ✅ | 无则跳过（forgetting_risk 要） |
| `response_mask` | ✅ | 无则跳过 |
| `logprobs` | 可选 | 训练时重算 |
| `success`/`success_rate` | 建议 | 缺则 priority 信号降级 |
| `meta` | 建议 | 缺则无溯源 |
| `sub_bucket`/`tools` | 可选 | — |

> 入库前由 `trajectory_adapter.extract_trajectories_from_batch` + buffer.add 校验；不合法轨迹计数 + 跳过（不静默）。
