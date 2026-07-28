#!/usr/bin/env python3
"""训练阶段快测:不跑 rollout,直接验证 dynamic_bsz 打包 + token 预算 vs 最坏序列长度。

动机:每次全量启动要等十几分钟 rollout 才到 update_actor;而近期崩点(§20/§22/§23/§24)
全在训练阶段——尤其 dynamic_bsz 的 `rearrange_micro_batches` assert(max_token_len >= max_seq_len)。
本脚本用假 batch(可指定最坏序列长度)直接调那条链,秒级复现/验证,无需 GPU / rollout / Ray。

用法:
  # 用某 config 的 token 预算 + 数据长度约束,验证最坏序列不会撞 assert
  python scripts/test_train_phase.py --config configs/run/b1_9b_16gpu.yaml

  # 手动指定最坏序列长度(模拟一条超长轨迹)看会不会崩
  python scripts/test_train_phase.py --config configs/run/b1_9b_16gpu.yaml --worst-seq 52758

退出码:0=通过(不会崩),1=会撞 assert(token 预算 < 最坏序列)。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# verl 在集群/本地的路径(纯 torch 工具可离线 import)
for p in ("/mnt/afs_toolcall/sunhao4/workspace/verl", str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_cfg(config_path):
    from trainer.cl_main import load_config
    return load_config(config_path)


def worst_case_seq_len(cfg):
    """最坏整条序列 = prompt上限 + response总长闸门 + 末轮单次生成(闸门当轮后检查)。"""
    prompt = int(cfg.data.max_prompt_length)
    resp_budget = int(cfg.data.max_response_length)        # 整条 response 闸门(collect.py)
    per_call = resp_budget                                  # 单次生成上限(sampling max_tokens)
    # 闸门是"当轮完成后检查",最后一轮完整保留 → response 最坏 = 闸门 + 一次生成
    worst_resp = resp_budget + per_call
    return prompt + worst_resp, prompt, resp_budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--worst-seq", type=int, default=None,
                    help="手动指定最坏序列长度(覆盖从 config 推算);模拟超长轨迹")
    ap.add_argument("--batch-size", type=int, default=8, help="假 batch 行数")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    a = cfg.actor_rollout_ref.actor
    use_dyn = bool(a.get("use_dynamic_bsz", False))
    ppo_tok = int(a.get("ppo_max_token_len_per_gpu", 16384))
    # log_prob 阶段预算(verl 默认 = ppo 的值,内部对 log_prob ×2 见 generated yaml)
    lp_tok = ppo_tok  # verl 用 log_prob_max_token_len_per_gpu = actor.ppo_max_token_len(×2 在 dp_actor)

    est_worst, prompt, resp_budget = worst_case_seq_len(cfg)
    worst = args.worst_seq if args.worst_seq is not None else est_worst

    print(f"=== 训练阶段快测: {args.config} ===")
    print(f"use_dynamic_bsz        = {use_dyn}")
    print(f"ppo_max_token_len      = {ppo_tok}")
    print(f"max_prompt_length      = {prompt}")
    print(f"max_response_length    = {resp_budget}  (整条 response 闸门)")
    print(f"推算最坏序列           = {est_worst}  (prompt + 闸门 + 末轮生成)")
    print(f"测试用最坏序列         = {worst}{'  (手动指定)' if args.worst_seq else ''}")
    print()

    if not use_dyn:
        print("ℹ️  use_dynamic_bsz=false:走静态 micro-batch,不经 rearrange_micro_batches 的 assert。")
        print("   静态模式下超长序列 pad 到 batch 最长,风险是 OOM 而非 assert。本测跳过 assert 检查。")
        # 仍给出 token 预算 vs 最坏序列的对比供参考
        print(f"   参考:最坏序列 {worst} vs ppo 预算 {ppo_tok} "
              f"({'预算够' if ppo_tok >= worst else '⚠️ 预算 < 最坏,静态模式会 OOM'})")
        return 0

    # dynamic_bsz:真调 verl 的 rearrange_micro_batches,复现 assert
    import torch
    from tensordict import TensorDict
    from verl.utils.seqlen_balancing import rearrange_micro_batches

    B, S = args.batch_size, worst
    # 造一个含"一条最坏长度"的 batch(其余行短),attention_mask 全 1 表示满长
    attn = torch.zeros((B, S), dtype=torch.long)
    attn[0, :] = 1                      # 第 0 行是最坏满长序列
    attn[1:, : min(2048, S)] = 1        # 其余行短
    batch = TensorDict({
        "input_ids": torch.randint(0, 100, (B, S)),
        "attention_mask": attn,
        "position_ids": torch.arange(S).unsqueeze(0).expand(B, S).clone(),
    }, batch_size=[B])

    # verl 对 log_prob 阶段实际用 ppo_max_token_len(dp_actor 内 ×2)。这里同时测 ppo 与 log_prob。
    for stage, budget in [("ppo(update_actor)", ppo_tok), ("log_prob(×2)", ppo_tok * 2)]:
        try:
            rearrange_micro_batches(batch, max_token_len=budget)
            print(f"✅ {stage}: 预算 {budget} >= 最坏序列 {worst} → 打包成功,不会 assert 崩")
        except AssertionError as e:
            print(f"❌ {stage}: 预算 {budget} < 最坏序列 {worst} → assert 崩!")
            print(f"   {e}")
            return 1
    print("\n✅ 训练阶段 dynamic_bsz 打包链通过:token 预算覆盖最坏序列,不会撞 §24 那个 assert。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
