# 集群采集 Bug 库（持续追加）

> **用途**：商汤 SenseCore（8 机×8 卡 H800）上做 Qwen3.6-27B 多轮 rollout 采集时遇到的 **bug → 解法** 速查库。**持续追加**——以后发现新 bug 继续往文档末尾加，不删改历史条目。
> **配套详情**：更详细的根因分析与排查教训见 `../paper/refs/集群推理采集_经验复盘.md`（条目末尾标注对应章节）。
> **起始日期**：2026-06-13（首批 10 条：已解决 9 + 诊断中 1）。

## 格式约定

每条 bug 一个表格块，字段固定为：

- **编号**：`#NN`，全局唯一、递增、永不复用。
- **现象**：观察到的报错 / 异常行为。
- **根因**：定位到的真实原因。
- **解法**：修复或规避手段。
- **状态**：`已解决` / `已知规避` / `未解决-诊断中`。

追加新条目时：编号接着往下排，按下方「状态汇总」更新计数，正文统一追加到「正文条目」末尾。

---

## 状态汇总

| 状态 | 条目 | 数量 |
|------|------|------|
| 已解决 | #01 #02 #03 #04 #06 #07 #08 #09 | 8 |
| 已知规避 | #05 | 1 |
| 未解决-诊断中 | #10 | 1 |
| **合计** | | **10** |

> 标 ★ 的为重点条目（曾被误判为主因 / 当前最严重）。

---

## 正文条目

### #01 SSH 私钥权限太松，密钥被静默忽略

| 字段 | 内容 |
|------|------|
| **现象** | `Permissions 0644 for id_rsa are too open ... This private key will be ignored` → 私钥被拒、ssh 回退要密码。 |
| **根因** | `~/.ssh/id_rsa` 权限为 0644（过松）。ssh 出于安全会**静默忽略**权限过松的私钥，导致认证回退到密码。 |
| **解法** | `chmod 600 ~/.ssh/id_rsa ~/.ssh/id_ed25519`。私钥必须 600（目录 700 / root:root）。权限正确时改 `authorized_keys` 实时生效，无需重启 sshd。 |
| **状态** | 已解决 |

> 配套：`../paper/refs/集群推理采集_经验复盘.md` §三（权限）、§四（按报错措辞分方向）。

---

### #02 vllm python 路径不固定，worker 起不来

| 字段 | 内容 |
|------|------|
| **现象** | worker 用 `/opt/conda/bin/python` 起 vllm，但镜像里该路径下没有可用 vllm。 |
| **根因** | 镜像内 vllm 不在 `/opt/conda`，python 路径不固定；唯一支持 `qwen3_5` 架构的解释器在别处。 |
| **解法** | 统一用 `/mnt/afs_code/ds32_env/bin/python`（唯一支持 qwen3_5 的环境）起 vllm。 |
| **状态** | 已解决 |

> 配套：`../paper/refs/集群推理采集_经验复盘.md` §五（环境凑齐）。

---

### #03 transformers / vllm 不认 qwen3_5 架构

| 字段 | 内容 |
|------|------|
| **现象** | 加载 Qwen3.6-27B 报架构不识别；vllm0.11 / 0.13 无法 resolve 该架构。 |
| **根因** | Qwen3.6 架构 `model_type=qwen3_5`（attention+mamba 混合）；旧 transformers 不识别，vllm0.11/0.13 的 `model_executor/models/` 里**没有 `qwen3_5.py` 建模文件**，根本不支持。vllm0.13 + transformers5.2 是死结（vllm0.13 要 transformers<5，qwen3_5 要 transformers≥5.2）。 |
| **解法** | 用 `ds32_env` 的 vllm0.16rc（自带 `qwen3_5.py`），可正确 `Resolved architecture: Qwen3_5ForConditionalGeneration`。新架构先查建模文件是否存在再选版本，不在旧 vllm 上硬凑 transformers。 |
| **状态** | 已解决 |

> 配套：`../paper/refs/集群推理采集_经验复盘.md` §五。

---

### #04 vllm / torch ABI 不匹配

| 字段 | 内容 |
|------|------|
| **现象** | `/opt/conda` 的 vllm0.11 + torch2.9.1 → `vllm._C undefined symbol`。 |
| **根因** | vllm 预编译扩展与 torch 版本 ABI 不匹配。 |
| **解法** | 用配套的 `ds32_env`（vllm0.16rc，版本组合自洽）。 |
| **状态** | 已解决 |

> 配套：`../paper/refs/集群推理采集_经验复盘.md` §五。

---

### #05 pip --target 在 quarkfs 上 rmtree 失败（FUSE）

| 字段 | 内容 |
|------|------|
| **现象** | `pip --no-deps --target` 装到 quarkfs 共享盘时 `rmtree` 非空目录失败，`OSError Errno 39`（Directory not empty）；且会缺运行时依赖（如 `model_hosting_container_standards`）。 |
| **根因** | quarkfs 是 FUSE 文件系统，删非空目录不可靠——与后面的 flock 崩溃同属一类 FUSE 行为问题。 |
| **解法** | 不在 quarkfs 上凑包；装到全新空目录 / 直接用配套环境（`ds32_env`）。 |
| **状态** | 已知规避 |

> 配套：`../paper/refs/集群推理采集_经验复盘.md` §五（要点 3）。

---

### #06 ninja 不在 PATH，编译子进程报错

| 字段 | 内容 |
|------|------|
| **现象** | vllm 编译 kernel 时 `FileNotFoundError: 'ninja'`。 |
| **根因** | 用绝对路径 `ds32_env/bin/python` 起 vllm，但编译子进程调命令行 `ninja` 走 PATH，PATH 里没有 `ds32_env/bin`（其实 `ninja` 自带在 `ds32_env/bin/ninja`，只是没进 PATH）。 |
| **解法** | `export PATH=$(dirname $PY):$PATH`（`$PY` = 所用 python 绝对路径）。 |
| **状态** | 已解决 |

> 配套：`../paper/refs/集群推理采集_经验复盘.md` §七点五（次要坑 ninja）。

---

### #07 ★ flashinfer JIT 缓存在 quarkfs → flock 崩溃（8 机崩溃 + GPU 0% 真根因）

| 字段 | 内容 |
|------|------|
| **现象** | 8 机 vllm 起来能服务一阵，推理若干请求后**集体 `Shutting down`**，GPU 掉回 0%。日志：`flashinfer/jit/core.py:314 FileLock → fcntl.flock FileNotFoundError: [Errno 2]`。曾一度误判为 8 机崩溃的笼统主因。 |
| **根因** | Qwen3.6(`qwen3_5`，含 mamba / gated-delta-rule)首次推理时 flashinfer 编译 `gdn_prefill` kernel，其 `FileLock(fcntl.flock)` 缓存默认在 `HOME/.cache/flashinfer`，而 `HOME` 在 quarkfs(FUSE)。FUSE 上锁文件操作不可靠（锁文件会“消失”）→ `FileNotFoundError` → worker 崩 → vllm 连锁 shutdown → GPU 0% → 采集空跑。traceback 链：`qwen3_next.gdn_attention_core → flashinfer/gdn_prefill.py → flashinfer/jit/core.py:314 FileLock → fcntl.flock`。 |
| **解法** | `export FLASHINFER_WORKSPACE_BASE=/tmp/...`（控制变量是 `_BASE`，**不是** `FLASHINFER_WORKSPACE_DIR`）。连带把其它缓存也指向 `/tmp` 本地盘并按 rank 隔离：`VLLM_CACHE_ROOT` / `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` / `XDG_CACHE_HOME`（用 `/tmp/vcache_r${RANK}/...`，注意是 `/tmp` overlay 本地盘，不是 `/root`）。已验证 flock 错误归零、压测不崩、推理非空。 |
| **状态** | 已解决 |

> 排查教训：看 traceback **最内层文件路径**定位模块（是 flashinfer 不是 vllm/triton），别凭猜设缓存 env。smoke(limit=2) 没触发 gdn kernel 首次 JIT，8 机大量请求才触发——小批过≠大规模稳，长稳必压测。配套：`../paper/refs/集群推理采集_经验复盘.md` §七点五。

---

### #08 采集对 actor 不可用不 fail-fast → 空跑产垃圾

| 字段 | 内容 |
|------|------|
| **现象** | vllm 已崩、GPU 0%，但采集脚本 `done=` 计数照常上涨，产出几百条 `{"role":"assistant","content":""}` 的空轨迹，完全无感知，一度误判“8 机正常在跑”。 |
| **根因** | actor endpoint 挂掉后，`HTTPGenerateFn` 的异常在上层 ReAct/session 循环被吞 → 返回空 `GenStep` → 轨迹空但 session 仍计为 done。没有“actor 不可用就停”的 fail-fast。 |
| **解法** | `collect_rollout.py` 启动时先 ping actor，打一个真实请求，空/不通直接 `sys.exit(6)`：`probe = generate_fn([{"role":"user","content":"ping"}]); if not (probe.text or "").strip(): sys.exit(6)`。 |
| **状态** | 已解决 |

> 教训：`done` 计数会骗人；判断采集真在产数据靠**抽查轨迹 assistant 是否非空 + GPU 利用率**。GPU 0% + 采集“在跑” = 必定空跑。配套：`../paper/refs/集群推理采集_经验复盘.md` §七点六、§七点七（体检清单）。

---

### #09 buffer 预热 trajectory_id 重复导致覆盖

| 字段 | 内容 |
|------|------|
| **现象** | `warmup_buffer.py` 灌 1200 条，最终 buffer 里只剩 4 条。 |
| **根因** | 用 jsonl 里原始 `trajectory_id` 作 key，冷采集数据大量重复（如 `q0-s0`），`store.put` 相同 tid 互相覆盖。 |
| **解法** | 入库时强制唯一 tid：`f"warm-{idx}-{原id}"`。 |
| **状态** | 已解决 |

---

### #10 ★★ actor 输出乱码/空（当前最严重，未解决）

| 字段 | 内容 |
|------|------|
| **现象** | Qwen3.6-27B 经 vllm0.16rc 推理输出纯乱码：completion 返回 `",//,årғ────────,//npos₫..."`；chat 模式第一个 token 即 EOS、`content` 为空。 |
| **根因** | 尚未定位。模型权重完整、dtype bf16 与 config 一致、tokenizer 能加载。**已排除**：不是 flashinfer GDN kernel（强制 monkey-patch `ChunkGatedDeltaRule` 走 `forward_native`，patch 确认生效 11 次，仍乱码）；不是 enforce-eager（试过仍乱码）。当前判断根因比 flashinfer 更广，可能是 vllm0.16rc(dev 版) 对 `qwen3_5` 架构整体实现不可靠。 |
| **解法** | 诊断中——后台 agent 正在建干净 vllm 环境 + transformers 原生二分诊断（隔离是 vllm 实现问题还是更底层）。 |
| **状态** | 未解决-诊断中 |

> 与 #07 关系：#07（flashinfer flock 崩溃）是“进程崩 / GPU 0%”，#10 是“进程活着但输出乱码”，两者现象不同、根因不同，已分别独立确认。
