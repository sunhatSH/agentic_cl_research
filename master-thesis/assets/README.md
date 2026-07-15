# 论文配图（assets/）

本目录存放论文正文配图。当前 5 张图均为**占位**，编译时用 `\figplaceholder` 命令渲染占位框（不报错）；图绘制完成后导出为 PDF 放入本目录，并把对应 `\figplaceholder` 替换为 `\includegraphics`。

| 文件名 | 图号 | 内容 |
|--------|------|------|
| `fig1_overview.pdf` | 图 1 | 系统总览：左半为干净信号产出（16×8 沙箱 rollout + winner 同步 + 三 agent 多轮构造循环），右半为持续学习 trainer（$L_{cl}$ 目标 + 7 桶 buffer），中间虚线箭头标注「干净 advantage」连接两半。 |
| `fig2_buckets.pdf` | 图 2 | 7 个能力桶结构 + 各桶任务数 + 25k 总容量下的平方根配额分配。 |
| `fig3_ushape.pdf` | 图 3 | U 形块权重曲线：权重对归一化块位置 $\mathrm{block}(t)/K_i$，$\gamma=\delta\in\{1.0, 0.92, 0.88\}$，展示 1.0 时的水平线（均权）与 $\gamma$ 减小时 U 形渐显。 |
| `fig4_winner_sync.pdf` | 图 4 | Winner 同步会话时序：单会话从母版派生 8 个位级一致槽位 → 每条 query 8 路并行 rollout → 选 winner → 全部槽位（磁盘+历史）同步到 winner → 下一条 query → 会话结束销毁。 |
| `fig5_three_agent.pdf` | 图 5 | 三 agent 多轮构造：winner-sync 边界处，观察 agent 读 winner 沙箱 diff（仅状态）→ 产出报告 $R_t$（pass-through 携带轨迹）→ $R_t$ 分发给奖励模型（状态 diff→completion，pass-through 轨迹→safety/robustness）与人设化出题 agent（下一 query 或结束会话）。 |

绘制工具建议：draw.io / TikZ / matplotlib（fig3）。源文件（`.drawio`/`.svg`/`.py`）与导出 PDF 一并放本目录。

> 图注的权威来源是 `草稿/04_方法.md` 与 `latex/chap/chapter4_方法.tex` 中对应图的 caption。
