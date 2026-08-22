# Reward 模型对比 (n_traj=20, repeat=8)

稳定度 = 同一轨迹重复打分的标准差(within-traj std),越小越一致。

## 总览

| 模型 | reward 均值 | reward 全局 std | **稳定度(within-traj reward std)** | 延迟均值(s) | 延迟 max | judge_error 率 |
|------|------|------|------|------|------|------|
| `openai/gpt-5.6-luna` | 0.6942 | 0.2369 | **0.0357** | 8.8737 | 48.1784 | 0.0 |

## 各维度(均值 / 全局std / 稳定度within-traj-std)

| 模型 | task_done | correctness | trajectory | safety |
|------|------|------|------|------|
| `openai/gpt-5.6-luna` | 0.8313 / 0.3745 / **0.0242** | 0.6811 / 0.2689 / **0.0484** | 0.7011 / 0.1872 / **0.0445** | 1.0 / 0.0 / **0.0** |

> 选型建议:优先 **稳定度(within-traj std)低** 且 judge_error 率低、
> 延迟可接受的模型;reward 均值本身不决定优劣(不同模型尺度不同),
> 关键看**同一输入是否给出一致分数**。