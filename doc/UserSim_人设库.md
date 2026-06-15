# UserSim 人设库（Questioner Personas）

> **定位**：16 个固定 Questioner 人设的权威参考表 + 设计说明。
> **数据信源**：[`agents/personas.json`](../agents/personas.json)（纯数据，改这里即可增删/调参，无需改代码）。
> **加载代码**：[`agents/personas.py`](../agents/personas.py)（读 JSON → 构造 `Persona`，导出 `PERSONAS` / `sample_persona` / `DEFAULT_PATIENCE_DECAY`）。
> **设计上下文**：[`UserSim_多轮Query在线生成.md`](UserSim_多轮Query在线生成.md) §3.5 / §3.6.5 / §7.5；三 agent 架构见 [`UserSim_三Agent架构与技术设计.md`](UserSim_三Agent架构与技术设计.md)。

---

## 1. 这是什么、为什么存在

三 agent 模拟用户里，**出题 agent（Questioner）带人设**（观察 agent 无人设、奖励模型独立）。每个会话开始时**随机抽 1 个人设并固定整场**——这是三条防模式坍缩机制之一（会话级画像多样性），避免所有会话的 follow-up query 趋同。

人设决定两件事：
1. **说话风格 + 在意什么**（profession / preference / profile / observation_focus）——影响生成的下一轮 query 的语气和侧重。
2. **失败路径下的耐心**（patience / patience_decay）——影响 agent 多轮做不好时用户何时放弃/抱怨/结束会话。

## 2. 配置与代码的解耦

| | |
|---|---|
| **改人设数据** | 编辑 `agents/personas.json`，不碰代码 |
| **字段** | `name` / `profession` / `preference` / `profile` / `observation_focus` / `patience` / `patience_decay`(可省，回退到文件级 `default_patience_decay`) |
| **加载与校验** | `agents/personas.py::_load_personas`：缺字段会报错；断言恰好 16 条 |
| **数量约束** | 必须 16 条（`doc §3.5`，单测 `tests/test_agents.py` 守护） |

> 加 / 删人设：改 JSON + 同步本表；若要突破 16 这个数，需同时改 `personas.py` 的 assert 和 `test_agents.py`。

## 3. 设计轴

### 3.1 观察偏好 observation_focus = {detail | whole} × {content | form}

人设关注观察报告的哪一面：粒度（细节 vs 整体）× 维度（内容对错 vs 形式呈现）。当前配比：

| 组合 | 数量 | 含义 | 谁 |
|------|------|------|----|
| detail × content | 8 | 抠细节、看内容对错 | Mei, Raj, Dr. Lena, Ken, Bella, Omar, Priya, Nina |
| whole × content | 5 | 看整体结果是否成事 | Tom, Carlos, Marcus, Greg, Leo |
| whole × form | 2 | 看整体呈现/可读性 | Anna, Yuki |
| detail × form | 1 | 抠措辞/排版细节 | Sophie |

> ⚠️ **配比偏斜**：content 类 13 / form 类 3。若希望"关注形式/排版/可读性"的反馈信号更足（对 Communication / OfficeQA 桶尤其相关），可在 JSON 里上调 form 类占比。

### 3.2 耐心两轴解耦（§3.6.5）

刻意拆成两个独立维度，避免"急脾气"和"低容忍"混为一谈：

- **P0 = patience**：初始容忍度（愿意给几次机会）。
- **d0 = patience_decay**：脾气 / 升级速度（失败时耐心掉得多快）。
- **放弃前重试次数 ≈ log₂(P0 / d0)**。

这让"一开始就没耐心" vs "起初宽容但崩得快"成为两种不同人格。

## 4. 16 人设总表

按"放弃前重试次数"从急到耐排序：

| # | 名字 | 职业 | 在意什么 | 观察偏好 | P0 | d0 | 重试≈ |
|---|------|------|---------|---------|----|----|------|
| 1 | Tom | 早期创业者 | 速度、大局；烦过度工程 | whole×content | 0.6 | 0.30 | 1.0 |
| 2 | Greg | 高管（没耐心） | 只要结论，不关心过程 | whole×content | 1.0 | 0.40 | 1.3 |
| 3 | Ken | 合规/风险官 | 规则遵守、审计留痕；标风险 | detail×content | 0.7 | 0.25 | 1.5 |
| 4 | Raj | SRE/运维 | 配置正确、可安全应用 | detail×content | 0.8 | 0.20 | 2.0 |
| 5 | Marcus | 运营主管 | 端到端流程跑通、不留半拉子 | whole×content | 0.9 | 0.15 | 2.6 |
| 6 | Carlos | 项目经理 | 交付物对原始需求的完整度 | whole×content | 1.0 | 0.12 | 3.1 |
| 7 | Omar | 采购专员 | 预算/供应商条款逐条准确 | detail×content | 1.0 | 0.12 | 3.1 |
| 8 | Priya | 客服支持工程师 | 修复是否真解决问题 | detail×content | 1.1 | 0.13 | 3.1 |
| 9 | Mei | 财务分析师 | 精确数字、对账；不信整数 | detail×content | 1.0 | 0.10 | 3.3 |
| 10 | Sophie | 内容/传播编辑 | 语气、清晰度、排版 | detail×form | 1.1 | 0.10 | 3.5 |
| 11 | Bella | 数据分析师 | 图表背后的数、分析是否成立 | detail×content | 1.3 | 0.10 | 3.7 |
| 12 | Anna | 办公助理 | 文档条理、能见人 | whole×form | 1.2 | 0.08 | 3.9 |
| 13 | Dr. Lena | 学术研究员 | 方法正确、中间结果可追溯 | detail×content | 1.5 | 0.10 | 3.9 |
| 14 | Yuki | UX 写作 | 终端用户读起来如何；整体感 | whole×form | 1.4 | 0.08 | 4.1 |
| 15 | Leo | 通才实习生 | 期望低、来啥学啥、很少抱怨 | whole×content | 1.2 | 0.06 | 4.3 |
| 16 | Nina | QA 测试 | 边界情况、声称与现实是否一致 | detail×content | 1.6 | 0.07 | 4.5 |

> 表中“#”是耐心排序序号，非 JSON 内顺序。JSON 内顺序见 `agents/personas.json`（前 3 个 Mei/Raj/Anna 对齐 `docker/sandbox/fs-seeds` 的 finance/sysops/office 种子工作区）。

## 5. 覆盖与已知偏斜

- **桶覆盖**：人设职业横跨 7 能力桶（Finance: Mei/Omar；SysOps: Raj/Marcus；Workflow: Carlos/Marcus；Dialogue: Priya；Communication: Sophie/Yuki；Knowledge/Analysis: Dr. Lena/Bella/Nina；OfficeQA: Anna）。
- **耐心分布**：急性子（重试 1–2 次）4 个，中位（3 次档）5 个，耐磨（≥3.5）7 个——失败路径样本不至于全是"秒放弃"或"无限宽容"。
- **已知偏斜**：observation_focus 的 form 维度只占 3/16（见 §3.1）；如需强化形式类反馈可调 JSON。
