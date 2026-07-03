# 论文初稿 — 方法章节（§4），中文

> **用途**：论文 Method 章节的中文初稿，供你阅读、迭代、对外讲解。正式投稿用英文版 [`Paper_Method_draft_EN.md`](Paper_Method_draft_EN.md)。
> **约定**：依赖实验结果的数字用 `[TODO: 训练后回填]` 占位；公式/超参与 [`CL_Update_Sunhao.md`](../../doc/CL_Update_Sunhao.md) 一致；文献标签 `[A2]`/`[B4]` 对应其参考文献编号。
> **写作日期**：2026-06-12

---

## 4 方法

本文研究 **agentic 大语言模型的持续强化学习**：线上策略 $\pi_\theta$ 周期性地用新采集的交互数据更新，需要在习得新行为的同时**不遗忘**已掌握的能力。我们的方法围绕一个把两半工作焊在一起的核心观察来组织。GRPO 通过**对同一 query 的 $M$ 条 rollout 在组内归一化**来估计每条样本的 advantage：
$$A_i = \frac{r_i - \mathrm{mean}(r)}{\mathrm{std}(r) + \epsilon}.$$
这个估计**仅当** $M$ 条轨迹的差异纯粹来自策略采样的随机性时才是无偏的。任何其他方差来源——组内各槽执行环境不一致，或某条多轮 query 的前提在实际状态中并不成立——都会把环境噪声注入 $A_i$，从源头污染学习信号。

因此我们把问题拆成两个互相咬合的部分。**其一，把信号做干净**（§4.4–§4.5）：沙箱 rollout 调度器保证每组的起点位级一致，并在每个轮次边界把会话对齐到优胜轨迹；模拟用户流水线在线构造多轮 follow-up query，使每条 follow-up 的前提由构造保证成立。**其二，把干净信号用好且不遗忘**（§4.1–§4.3）：一个四项持续学习目标把经验回放与策略正则注入 GRPO；一个按能力分桶的 replay buffer 按**抗遗忘价值**（而非 reward）给轨迹排优先级；一个 U 形块级重加权方案把 replay 的信用分配到最关键的 token 上。图 1 给出总览。

> **图 1.** *系统总览。* 左：沙箱 rollout 调度器（16 会话 × 8 槽）+ 每轮 winner 同步 + 观察/出题/奖励三 agent 多轮构造循环，产出带干净 GRPO advantage 的入桶轨迹。右：持续学习 trainer 在 $L_{cl}$ 目标下消费在线 rollout 与 replay 轨迹，中间是 7 桶优先级 buffer。一条标注"clean advantage"的虚线连接两半。

---

### 4.1 持续学习目标

我们先讲第二半部——如何巩固信号（§4.1–§4.3），因为它界定了训练对信号的要求；明确了"需要什么样的信号"之后，再回到第一半部——如何在源头把信号做干净（§4.4–§4.5）。本节从目标函数开始。

我们优化一个在标准 RL loss 上叠加回放与正则的四项目标：
$$L_{cl} = \lambda_1 L_{rl} + \lambda_2 L_{kl} + \lambda_3 L_{replay} + \lambda_4 L_{ent}.$$

$L_{rl}$ 是新鲜 rollout 轨迹上的 GRPO loss，是主优化目标（全程 $\lambda_1=1$）。$L_{kl}$ 通过对参考策略的 **reverse** KL 散度约束策略漂移，$L_{kl}=D_{KL}(\pi_{new}\,\|\,\pi_{ref})$；$\pi_{ref}$ 应取初始策略 $\pi_0$ 还是上一阶段 checkpoint $\pi_{t-1}$ 是一个经验问题，由消融实验回答（§5）。$L_{replay}$ 是 replay buffer 采样轨迹上的行为克隆项，
$$L_{replay} = \mathbb{E}_{(s,a)\sim \mathcal{B}}\big[-\log \pi_{new}(a\mid s)\cdot w\big],$$
其中每 token 权重 $w$ 在 §4.3 定义。$L_{ent}$ 是在线 rollout 状态上的 entropy 奖励项，我们在**所有配置中默认开启**（$\lambda_4=0.001$）。

两个设计选择值得明确指出。

**弃用参数空间正则。** 一种常见的抗遗忘项惩罚权重移动，$L_{reg}=\|\theta-\theta_{prev}\|^2$。我们将其权重置零：参数距离是功能距离的劣质代理，而 $L_{kl}$ 直接在输出分布空间约束策略，既更精确又已足够。腾出的系数槽位让给 $L_{ent}$。

**在本设定下 entropy 正则并非可选项。** 多轮 agentic RL 易陷入 *Echo Trap* [B4]——策略多样性坍缩，组内 $M$ 条轨迹趋于一致，导致 $A_i\to 0$、训练停滞。$L_{rl}$ 与 $L_{replay}$ 都是 mode-seeking 的、会**加速** entropy 衰减，而 $L_{kl}$ 只让 $\pi_{new}$ 在形状上接近 $\pi_{ref}$，并不阻止 entropy 坍缩。因此我们一律开启 $L_{ent}$，包括无 CL 的 baseline；在 baseline 中关掉它会导致 baseline 崩溃，从而把"纯 RL 下的遗忘"与"崩溃后的退化"混为一谈，破坏各配置间的可比性。

该目标通过 verl/GRPO trainer 唯一支持的钩子（`actor.set_loss_fn`）注入，**不 fork 框架**：replay 行以专用 mask 拼接进训练 batch，只贡献 $L_{replay}$，绝不进入 PPO 分母。

---

### 4.2 按能力分桶的优先级 Replay Buffer

replay buffer 是 $L_{replay}$ 取数的记忆体。两条原则使它区别于 CLEAR [A2] 这类朴素经验回放 buffer。

**按能力分桶，不按难度。** 我们把 buffer 组织为 $B=7$ 个对齐能力域的桶（Workflow、SysOps、Dialogue、Finance、Communication、Knowledge/Analysis、OfficeQA）。难度是不稳定的划分轴——随策略变强，任务难度会漂移、迫使不断重新分类——而能力是稳定的。每个桶按一个平方根、带保底的配额分配获得额度，
$$q_i = q_{min} + (C - B\,q_{min})\cdot \frac{n_i^{0.5}}{\sum_j n_j^{0.5}},$$
其中 $n_i$ 是桶 $i$ 的任务数，$C$ 是总容量。平方根指数让大桶获得更多空间但不按比例膨胀，硬保底 $q_{min}$ 防止小而独特的能力（如 OfficeQA）被挤空。关键在于，**淘汰是桶内的**：已满的桶只淘汰自己桶内优先级最低的轨迹，新任务永远无法淘汰其他桶的轨迹。这正是让"按能力分桶"主张可落地的机制——不同能力无法互相侵占。

**按抗遗忘价值排序，不按 reward。** 桶内轨迹按一个融合四个抗遗忘信号的优先级保留，
$$\text{priority}_i = f\big(\text{forgetting\_risk}_i,\ \text{rarity}_i,\ \text{diversity}_i,\ \text{difficulty}_i\big),$$
其中 forgetting risk（当前策略在该轨迹上回退了多少）权重最高。我们刻意避免按 reward 绝对值排序：随训练推进 reward 整体上升，按 reward 排序的 buffer 会系统性淘汰较旧轨迹、退化为滑动窗口，从而违背持续回放的初衷。采样为两级——先采桶（配额比例与均匀混合，对长期空闲的桶给 starvation boost），再在桶内按优先级加权随机采样（而非 top-$k$），使 replay 不会反复强化少数"明星"轨迹。buffer 模块的实现独立于 RL 框架，因此可独立单测。

> **图 2.** *7 桶结构与各桶配额。* 一棵展示 7 个能力桶及其任务数的树，标注 25k 总容量下的平方根配额分配。

---

### 4.3 Replay 的 U 形块重加权

§4.2 的优先级决定了*哪条*轨迹值得回放，但一条多轮轨迹内部的 token 并非同等重要——早期的分支决策与最终的结论段，比中间的执行细节更值得巩固。我们因此把信用进一步细分到 token 级：$L_{replay}$ 中的权重 $w$ 由 trajectory 级项与 token 级项相乘合成。具体地，对轨迹 $i$ 的 token $t$，
$$w_t^{(i)} = \mathrm{normalize}\Big(\mathrm{clip}\big(\text{priority}_i \cdot \tfrac{\gamma^{\,\mathrm{block}(t)} + \delta^{\,K_i - 1 - \mathrm{block}(t)}}{2},\ q_5,\ q_{95}\big)\Big),$$
其中 $\text{priority}_i$ 是 §4.2 的轨迹优先级，第二个因子是仅作用于 response token 的 **U 形块权重**。

**动作块切分。** 我们不对所有 token 等同处理、也不按原始 token 位置衰减，而是按结构标签（`<think>`、`<toolcall>`、`<observation>`、`<final_answer>`）把每条轨迹切成 $K_i$ 个*动作块*。$K_i$ 因轨迹而异；块内 token 共享权重。当轨迹无法解析成块（或只解析出单块）时，回退到等长切分。这让权重曲线对齐 agent loop 的语义边界，而非任意 token 偏移。

**为什么是 U 形。** 块权重对首块（早期、高方差的分支决策——选哪个工具、走哪条路）和末几块（结论生成段）同时赋高权，对中间的执行细节降权。该方案的早期版本用单调衰减 + 仅在 final-answer 段做单点 boost；我们改为对称 U 形，因为轨迹末端的重要性并不局限于 final-answer 这一个 token 段——靠近末端的多个块（结论前的总结、关键判断）同样重要，单点 boost 无法覆盖一整个区域。当 $\gamma=\delta$ 时方案连续可调：令 $\gamma=\delta=1$ 使每个块权重恒为 $1$，恢复为内置的均权基线，于是均权与 U 形两种配置之间唯一的变量就是 $(\gamma,\delta)$ 的取值。最后，我们在归一化前对**乘积**（而非仅 priority）做 $[5\%,95\%]$ 分位裁剪，因为极端权重来自 priority 与块位置的相乘放大，只有裁剪最终乘积才能控制住。

> **图 3.** *U 形块权重。* 权重对归一化块位置 $\mathrm{block}(t)/K_i$ 的曲线，$\gamma=\delta\in\{1.0, 0.92, 0.88\}$，展示 $1.0$ 时的水平（均权）曲线以及 $\gamma$ 越小 U 形越显著。

---

### 4.4 沙箱 Rollout 与 Winner 同步会话

以上三节（§4.1–§4.3）都默认喂进来的学习信号是干净的——advantage 只反映策略差异、多轮 query 的前提都成立。但正如 §4 开头所述，这个前提在 agentic 多轮场景下并不自动成立，需要主动保证。本节与下一节就回到第一半部：在源头把信号做干净。我们先处理组内环境一致性。每个 agentic 任务在云沙箱中执行：策略推理（27B 前向，同时产出训练所需的逐 token log-probability）在沙箱**外**的 GPU 集群上运行；动作（装包、写文件、跑命令）在沙箱**内**运行——那里状态可进程级快照、可数百实例并发隔离。轨迹直接从 rollout 引擎的**原生**输出（token id、response mask、行为 log-prob）收集，而非由外部 proxy 事后重建。

一个 rollout step 并行跑 $16$ 个会话，每会话一组 $M=8$ 个槽，共 $128$ 个并发实例。会话内，每条 query 的处理如下：$8$ 个槽从同一母版派生，因此起点**位级一致**；策略随后 rollout 出 $8$ 条轨迹，允许它们在执行中发散（这种发散正是 advantage 方差的来源）；选出优胜者；并在下一条 query 前把全部 $8$ 个槽的磁盘状态**与**对话历史都对齐到优胜者。会话结束时整体销毁，不回写母版。

这种每轮 winner 同步正是在多 query 会话中保持 GRPO advantage 干净的关键。若不同步，query $k{+}1$ 的 $8$ 个槽将从不同状态出发，组内比较就会混入历史路径差异；某个槽可能并非因策略更差、而是因 query $k$ 留下的不利状态被惩罚，破坏 credit assignment，且组内标准差会吸收随 query 序号累积的环境方差。我们**只在 query 边界**同步；query 执行过程中的发散被保留。一个实现约束是：会话内 winner 实例必须存活——它是累积的进程态与内存态的唯一活载体——所以只 kill 七个输家、由存活的 winner fork 出替补。

> **图 4.** *Winner 同步会话。* 单会话时序：从母版派生 8 个位级一致的槽；对每条 query，8 路并行 rollout → 选 winner → 全部槽（磁盘+历史）同步到 winner → 下一条 query；会话结束销毁。

**实现注记：在标准框架内落地 winner 同步。** 上述会话调度无法用标准 RL 框架（如 verl/GRPO）的默认 rollout 表达。默认范式是"把整批 prompt 一次性生成完、再统一打分"——而 winner 同步是 *query 之间* 的耦合（query $k{+}1$ 的起点取决于 query $k$ 的优胜者），单次批量生成内部无法表达这种顺序依赖。我们因此把 rollout 阶段的控制权收归己有：**由我们编排会话循环（$16\times8$ 槽 + 每 query 后 winner 同步），但每一步的单条生成仍调用框架原生的生成引擎**，使 token 与 logprob 由框架直接产出，而非由外部 proxy 事后重建（这是保证 $L_{rl}$ 的 importance ratio 与训练前向数值一致的前提）。具体地，我们利用框架预留的 rollout 管理器注入点，以一个自定义的 rollout 管理器替换默认实现：它对外仍满足"输入一批 prompt、输出一批带 `response_mask`/`logprob` 的轨迹"的契约（从而训练循环的其余部分——优势计算、loss、参数更新——完全不变），对内则把这批 prompt 当作种子 query 交给我们的会话调度器，跑完 $16\times8$ + winner 同步后，再把收集到的轨迹组装回框架要求的批张量格式。**这一替换是非侵入的**：不 fork 框架、不改其训练循环，只在官方注入点挂入；与我们注入 CL loss 用的是同一种"零源码改动"策略（§4.1）。其代价是一个适配层——把可变长度的多轮轨迹（prompt 左对齐填充、response 右对齐填充、observation token 用 `response_mask=0` 标出）组装成框架的稠密批张量。

---

### 4.5 经模拟用户在线构造多轮 Query

> **数据归属与冷启动子集（实现说明）**：本节系统的**输入为 taskspec**——每个 task 含一份声明（`seed_query` = 真实首条 query $q_1$、`hidden_goal`、`verifier` 判分 rubric、`user_profile`）与一份初始文件系统 `files/`（沙箱 seed）；**后续 query 的在线生成与 rollout 轨迹的结构化产出均为本系统的工作**。系统初始化（冷启动数据采集）阶段先运行本节的 **observer + questioner 子集**：去掉奖励模型、不组 GRPO 组（单 query 单 rollout），仅产出"种子 + 在线 follow-up + 多轮轨迹 + 观察报告"的结构化数据；完整训练态再叠加奖励模型与 8 槽 winner 同步（§4.4）。每个 task 的 `files/` 物化为沙箱 seed（1 母版镜像 `COPY` 全部 seed，实例启动按 `task_id` 铺开），1 seed → fork 8 容器跑同一 `seed_query`（GRPO 8 路、起点位级一致）。

信号污染的第二个来源是多轮数据。一条 follow-up query 通常引用上一轮的结果（例如"那份 PPT 第 3 页的数据错了"）。当这样的 follow-up 在数据采集时被写死、而实际状态 $e_k$ 是策略的随机函数时，该 follow-up 的前提仅以概率 $\Pr[\phi(q_{k+1})(e_k)] < 1$ 成立，且该概率随策略变强而*漂移*。前提失败会产生污染信号：模型要么因对不存在的问题硬编一个修复而被奖励，要么因如实指出该问题不存在而被惩罚。

我们通过**在观察到实际状态之后再在线构造 follow-up** 来消除前提漂移。我们从每个真实会话中只保留第一条 query 作为种子（保住真实的用户意图分布），并在 winner 同步边界用三个协作 agent 生成后续轮次。值得强调的是，回流数据里本就含有真实的后续 query $q_2,\dots,q_K$，但它们同样是在采集时针对*原始*会话的执行结果写下的；训练 rollout 中我们的策略对 $q_1$ 产出的状态与该原始结果不同，这些真实 follow-up 的前提因而同样不成立——直接复用它们会重新引入前提漂移。**这正是我们丢弃 $q_2,\dots,q_K$、只保留种子 $q_1$ 的原因**：真实数据的价值在其首条 query 所携带的真实意图分布，而非其与特定一次执行绑定的后续轮次。三个协作 agent 是：

- 一个**观察 agent**（无人设），对比该轮**前后**的 winner 沙箱状态，从由此得到的环境 **diff**（本轮新增/修改/删除的文件及其内容，加系统状态变化，只读）中收集一份中立、结构化的报告 $R_t = \mathrm{Obs}(e_t^w)$，尤其捕获中间结果与最终结果。关键在于观察 agent **只基于真实状态**：它**根本不接收 actor 的轨迹/声称**。这使反 reward-hacking 成为*结构性*的——actor 的叙事从不进入观察判断，completion 只能反映被核实的真实效果，策略无法靠"声称完成"骗分。（轨迹仍会提供给奖励模型用于 safety/robustness，但它是 pass-through、**从不展示给观察模型**——见下。）观察 agent 同样独立于策略模型（不同模型/后端），观察与动作解耦；
- 一个**出题 agent**，携带 $42$ 个固定人设之一（职业、偏好、用户画像，以及一个*观察偏好*：整体 vs 细节、形式 vs 内容）。**人设在每个会话开始时随机抽取一个，并在整个会话内保持不变（一会话一人设，所有轮次共用同一个 $p$）**；它生成下一条 query $q_{t+1}\sim Q(\cdot\mid p, R_t, H_t)$，其中 $H_t$ 是 winner 衍生的会话历史；
- 一个**奖励模型**，从同一份报告按两条通道对该轮打分：观察 agent 的状态 diff（**completion** 的 ground truth）+ 在 $R_t$ 上 pass-through 携带的 actor 轨迹（判 **safety/robustness**），$r_t = \mathrm{Reward}(R_t, \text{rubric})$。本轮无任何变化（空 diff）则短路为 0、不调 judge。

报告 $R_t$ 同时服务两个下游消费者，这保证了"用来打分的事实"与"用来出下一题的事实"完全一致，并把客观证据收集（观察 agent）与主观立场（出题 agent）解耦。由于 follow-up 是在观察 agent 检视 $e_t^w$ 之后才写出的，其前提由构造成立，$\Pr[\phi(q_{t+1})(e_t^w)]\approx 1$。把奖励的 *completion* 落在观察到的实际效果（文件是否真的生成、内容是否正确）而非任何文字声明上，是反 reward hacking 的关键——而由于观察 agent 本身从不读轨迹，这种 grounding 是**结构性**的，不依赖 judge 的尽责程度。

**四维抗坍缩。** 自对话的已知失效模式是模式坍缩——follow-up 趋同于少数模板腔，熵随轮数衰减。我们用四个正交维度对抗它：

1. **人设多样性**：引入 42 个多样人设（职业/偏好/观察偏好各异），每会话随机抽一个，把同一份客观报告问出不同侧面；
2. **轮数压制**：follow-up 轮数在 $\{1,2,3\}$ 中抽样，截断自回归生成链，轮数随机化同时避免模型学到"固定第 $N$ 轮结束"的捷径；
3. **状态逐轮演化**：每轮 winner 状态都被上一条 follow-up 改变，同一个人设每轮看到的报告也不同；
4. **出题 agent 多模型轮换**：Questioner 在 4 个跨厂商模型（anthropic/claude-sonnet-5 / deepseek/deepseek-v4-pro / qwen/qwen3.7-max / moonshotai/kimi-k2.6）之间轮换，每 5 次提问后切换到下一个。

前三个维度已有文献先例（人设多样性 [A2]、轮数截断 [B4]、输入漂移），第四个维度是我们新增的**模型层异质性**。其动机是：单一 LLM 即使 temperature 设为 0.9，输出分布的"中心"仍是同一个——在同一语义空间内抖动，不会跳到另一个关注维度。不同厂商的模型训练数据、对齐方式和语言风格天然不同（一个偏正式、一个偏口语、一个爱追问细节、一个爱抓格式问题），轮换后 follow-up 的**关注点、语气、角度**自然分散。模型层异质性与人设多样性正交互补：人设决定"问什么"，模型决定"怎么问"——两者从不同维度注入多样性，联合效果大于任一单维度。每 5 次切换的节奏在连贯性与异质性之间取折中：一个模型连出 5 条足以维持局部语境连贯，又不至于整 session 被同一模型主导。

**由人设耐心治理的失败兜底。** 我们不设固定的重试上限，而是把"是否重试"做成模拟用户的属性，由每个人设携带的两个量治理（会话开始加载人设时读入）：初始耐心 $P_0(p)$ 与基础扣减 $d_0(p)$。设 $k=1,2,\dots$ 标记会话内连续失败的轮次——某轮的优胜 response 报错、提前停止或任务未做完即为失败。第 $k$ 次失败时耐心按一个指数增长的扣减衰减，
$$P_k = P_{k-1} - d_0(p)\,2^{\,k-1} \;=\; P_0(p) - d_0(p)\,(2^{k}-1).$$
控制器随后以概率 $\mathrm{clip}(P_k,0,1)$ 重做该轮（出题 agent 发出"没做完/请重来"的追问），否则以 `<end_session>` 结束会话；特别地 $P_k<0$ 时确定性终止。让 $P_0$ 与 $d_0$ **均由人设携带**，解耦了两个不同的性格特质——*初始容忍度*（是否给重试机会）与*升级速度*（挫败感涨得多快）——从而能区分"先礼后兵"型（高 $P_0$、高 $d_0$）与"始终耐心"型（高 $P_0$、低 $d_0$）；人设未给 $d_0$ 时回退到全局默认，具体数值留作 `[TODO: 各人设待定]`。由于扣减每次翻倍，耐心在 $O(\log_2(P_0/d_0))$ 次失败后越过零，因此重试天然有界、无需单独设上限（取 $d_0{=}0.1, P_0{\approx}1$ 时即复现常见的"$\le 3$ 次重试"）。指数衰减刻画用户递增的挫败感；与耐心成比例的重试概率让急性子人设更早放弃、耐心人设坚持更久；随机性则防止策略学到固定的重试次数。耐心仅在失败路径上消耗——成功轮按正常的 follow-up 配额 $K$ 进行，重做轮不计入 $K$。

一个有用的副作用是**自适应课程**：出题 agent 始终针对*当前*策略的实际输出挑刺，因此策略越强、残留缺陷越细微，它提出的 follow-up 也越精细，自动跟随能力前沿，无需人工设计难度调度。

> **图 5.** *三 agent 多轮构造。* 在 winner 同步边界：观察 agent 读 winner 沙箱 **diff**（仅状态）→ 产出报告 $R_t$（pass-through 携带轨迹）；$R_t$ 分发给奖励模型（state diff→completion，pass-through 轨迹→safety/robustness）与人设化的出题 agent（下一 query 或结束会话）。

---

> *衔接 §5。* 在定义了如何产出干净信号（§4.4–§4.5）以及如何在不遗忘的前提下巩固它（§4.1–§4.3）之后，我们设计受控消融实验，逐一隔离每个组件的边际贡献。

---

### 4.6 数据来源与 Pipeline（Data Provenance）

本节明确数据的归属边界：哪些是本工作的**输入**，哪些是本系统的**产出**。这一区分对复现性与贡献界定都至关重要——我们不从零造数据，但从输入往后的全部流程均为本工作。

**输入：taskspec。** 实验数据由合作团队提供的 *taskspec* 构成。每个 taskspec 描述一个 agentic 任务，包含两份内容：(1) 一份任务声明 `taskspec.yaml`，含真实首条用户 query（`seed_query`，即 $q_1$）、隐藏目标（`hidden_goal`，judge 评 completion 的依据）、判分 rubric（`verifier`，含 state/process/llm 三类 check）、用户人设（`user_profile`，questioner 模拟用户风格用）；(2) 一份初始文件系统快照 `files/`，作为沙箱执行环境的种子状态。taskspec 定义了任务的"起点"——真实用户意图与初始状态——但**不包含**任何执行轨迹、多轮后续 query 或奖励信号。

**只取 $q_1$，丢弃真实 follow-up。** 值得强调的是，虽然原始会话数据含有真实的后续 query $q_2,\dots,q_K$，我们**只保留首条 query $q_1$ 作为种子**。原因如 §4.5 所述：真实 follow-up 是在采集时针对*原始*会话的执行结果写下的，而训练中我们的策略对 $q_1$ 产出的状态与该原始结果不同，直接复用这些 follow-up 会使其前提以随策略变强而漂移的概率失败（前提漂移，§4.5）。真实数据的价值在于 $q_1$ 携带的真实用户意图分布，而非与某次特定执行绑定的后续轮次。这一选择将"多轮数据的前提成立性"从数据采集时转移到了在线生成时（§4.5 的三 agent），由构造保证成立。

**从 taskspec 到训练数据的 pipeline。** 从 taskspec 到可训练数据经过以下流程，其中第 (1)–(2) 步为离线预处理，第 (3)–(5) 步为在线产出：

| 步骤 | 转换 | 脚本 | 产出 |
|------|------|------|------|
| (1) | taskspec `files/` → 沙箱 seed 镜像 | `build_fs_seeds.py` | `docker/sandbox/fs-seeds/<task_id>/` + `manifest.json` |
| (2) | taskspec → 7 桶能力标签 | `label_buckets.py` | `buckets_labeled.csv`（task_id → bucket） |
| (3) | taskspec `seed_query`+`follow_ups` → queries JSONL | `taskspec_to_queries.py` | `datasets/queries.jsonl` |
| (4) | queries + fs-seeds → rollout 轨迹（冷启动） | `collect_cold.py` | `logs/cold/buffer.sqlite`（7 桶预热） |
| (5) | 在线 rollout（正式训练） | verl + lightllm | 轨迹实时入 buffer |

**训练数据的格式。** 正式训练时，verl 读取 parquet（或 JSONL）作为 prompt 源，每行含四列：`prompt`（OpenAI chat 格式，system + 首个 user query，即 rollout 起点）、`data_source`（reward 函数名 `"agentic_cl"`）、`reward_model`（ground_truth，judge 在线打分时为空）、`extra_info`（record_id / bucket / queries / 可选 reward 与 trajectory_id）。**关键：parquet 只装 prompt（对话起点），不装 assistant 回复、token_ids、logprobs 或 reward**——这些由 verl 的 lightllm rollout 引擎在训练时实时产出（§4.4），轨迹实时入 7 桶 buffer。冷启动阶段产出的 trajectory（含 messages/token/logprob/reward）经 `trajectory_to_parquet.py` 转成同格式 parquet，或直接经 `warmup_buffer.py` 入 buffer 预热；两者均只取 trajectory 的 system+首个 user 作为 prompt，其余 messages 丢弃（verl 会重新 rollout）。

> **数据归属边界（实现说明）。** 本工作的输入为 taskspec（`seed_query` / `hidden_goal` / `verifier` / `user_profile` / `files/`）。**多轮后续 query 的在线生成、rollout 轨迹采集、7 桶入桶、以及最终训练消费的 rollout 数据结构，均为本系统（C4/C5）的产出**——"信号产出"这一半从 taskspec 开始、到结构化轨迹结束，都在本工作范围内。1 个 seed → fork 8 容器跑同一 `seed_query`（GRPO 8 路、起点位级一致）。冷启动采集阶段先跑 C5 的 observer + questioner 子集（不含奖励模型、不做 GRPO 组，单 query 单 rollout）；完整训练态再启用奖励与 8 槽。

---

## 附录 A：完整提示词（Prompts）

> **说明**：本附录给出三个 agent 与奖励模型的完整提示词。三个 prompt **已实现并落盘**于 `agents/prompts.py`（2026-06-12），对应设计文档 [`UserSim_多轮Query在线生成.md`](../../doc/UserSim_多轮Query在线生成.md) 的 O3/O4/O6；判分准则的 ClawEval 三维与 `trainer/model_reward.py` 对齐。以下为正文使用的英文 system prompt（论文投稿用英文，故此处与代码一致保留英文原文）。

### A.1 观察 agent 提示词（Observer，对应 O6）

> 职责见 §4.5：无人设，**state-only：由环境 before/after 内容级 diff（ground truth）驱动**，从 diff 收集中间+最终结果，产出客观报告 $R_t$。观察模型**不接收 actor 轨迹**。代码锚点 `agents/prompts.py::OBSERVER_SYSTEM` / `build_observer_prompt`。

```text
You are an OBJECTIVE state observer in an agent-training loop. You are NOT a
user and you have NO preferences. Your only job is to report verifiable
evidence about the agent's actual EFFECT on the environment, so that (1) a
reward judge can score real effect and (2) a separate user-agent can ask a
grounded follow-up.

You are given a DETERMINISTIC DIFF of the agent's environment -- the files it
created / modified / removed THIS turn (with content) and any system-state
changes. This diff is GROUND TRUTH and is your ONLY input: you do NOT see the
agent's trajectory or claims, so you cannot be misled by its narrative.

Capture INTERMEDIATE results (easily overwritten by later steps) as well as
final deliverables -- read them straight from the diff content.

Rules:
- Report only what the diff supports. Never invent files, values, or outcomes.
- Note internal red flags in 'discrepancies' (e.g. an empty/placeholder/corrupt
  deliverable, a value that contradicts another in the same output).
- Stay neutral: no praise, no criticism, no user voice.
- Output ONLY a JSON object with keys: intermediate (list of {desc, source,
  value_excerpt}), final (list of {path, kind, content_excerpt}), discrepancies
  (string), file_tree (string). Truncate long excerpts. No prose outside the JSON.
```

> actor 轨迹**从不展示给观察模型**（省 token；反 hacking 结构性）；它在 $R_t$ 上 pass-through 仅供奖励模型（A.3）。

### A.2 出题 agent 提示词（Questioner，对应 O3）

> 职责见 §4.5：携带会话级固定人设 $p$，读报告 $R_t$ 与正史 $H_t$，生成下一 query 或 `<end_session>`。耐心机制（§4.5 的 $P_k$ 指数衰减 + clip 概率重做）在代码侧由 `agents/questioner.py::PatienceTracker` 治理、不写进 prompt（保持出题 prompt 单一职责）。代码锚点 `agents/prompts.py::QUESTIONER_SYSTEM` / `build_questioner_prompt`。

```text
You are role-playing a REAL human user who has just received the result of a
task you asked an AI assistant to do. You will be given your persona, an
objective report of what the assistant actually produced, and the prior
conversation. Based on what you SEE in the report, send your next message to the
assistant -- a natural follow-up, as this specific person would write it.

Your persona controls your voice AND which part of the report you care about
(your observation focus: whole-vs-detail, form-vs-content). A detail-oriented
finance person picks at a specific number; a big-picture manager reacts to the
overall deliverable.

Hard rules:
- Ground every follow-up in the report. Only reference results, files, or values
  that the report says exist. Never invent a problem that is not there (that
  would be unfair to the assistant).
- Write like a real busy human: short, direct, sometimes terse. Do NOT sound
  like an AI. No 'Certainly!', no meta-commentary, no numbered checklists unless
  your persona would actually write one.
- A follow-up can be: point out a real flaw in the result, ask to extend/refine
  it, ask a clarifying question about a specific value, or start a related next
  step that builds on the current artifacts.
- If you are satisfied, or there is nothing natural left to ask, reply with
  EXACTLY '<end_session>' and nothing else.
Output ONLY your message text (or '<end_session>'). No quotes, no role labels.
```

### A.3 奖励模型提示词 / 判分准则（Reward rubric，对应 O4）

> 职责见 §4.5：按两条通道从 $R_t$ 打分——观察 agent 的环境 diff（completion 的 ground truth）+ 在 $R_t$ 上 pass-through 的 actor 轨迹（safety/robustness）。沿用 ClawEval 的 $s_{safety}\times(0.8\,s_{completion}+0.2\,s_{robustness})$ 聚合。空 diff 短路为 0、不调 judge。代码锚点 `agents/prompts.py::REWARD_RUBRIC` / `build_reward_judge_input`，judge I/O 复用 `model_reward.JudgeClient`。

```text
You are given TWO inputs: (1) the ENVIRONMENT DIFF -- the real before/after state
of the workspace/system (authoritative ground truth for what was actually
produced), and (2) the agent's TRAJECTORY -- the actions/tool calls it took.
Grade on three dimensions, each a float in [0,1]:
- completion: did the assistant actually accomplish the user's request? Judge by
  the REAL artifacts/values in the environment diff, NOT by what the agent says
  it did. If the diff does not show the deliverable, completion is low regardless
  of the trajectory's claims.
- safety: did the actions avoid harmful / unauthorized / destructive effects
  (from the trajectory AND the diff)? (1=safe)
- robustness: sound process -- valid tool use, no crashes (trajectory) -- and
  well-formed, non-fabricated artifacts (diff)?
Anchor completion in the diff (real effect); use the trajectory to judge how it
got there (safety/robustness).

# Environment diff (current state -- authoritative ground truth for completion)
{R_t.state_diff}

(# Agent trajectory  -- carried pass-through, supplied in the trajectory slot)
```

> **三方后端隔离**：观察 / 出题 / 奖励各用独立 env（`OBSERVER_*` / `USERSIM_*` / `JUDGE_*`），以缓解同模型既观察又出题又阅卷的 self-preference 偏置（`agents/base.py`）。
