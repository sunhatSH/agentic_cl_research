"""CL 评测入口：复用训练的 RemoteAgentLoopManager 跑 ClawEval + judge 打分。

评测 = 训练 rollout 的一部分（同 lightllm + gateway + hermes harness + 沙箱 + reward
worker），只是不训练、只跑 ClawEval 任务 + 读 judge 四维打分。

完整流程（评测版 TaskRunner 在 trainer/verl_runner.py::CLTaskRunnerEval）：
  ① merge checkpoint(FSDP→HF)：bash scripts/exp_common/merge_ckpt.sh
  ② 本入口起 trainer + RemoteAgentLoopManager（lightllm 加载 merge 后的 HF）
  ③ generate_sequences(ClawEval, validate=True) → trajectory 写 TQ "val" partition
  ④ trainer.replay_buffer.sample(partition_id="val") 读回 batch（KVBatchMeta）
  ⑤ extract_trajectories_from_kvbatch(batch) 读 trajectory + reward 四维
  ⑥ 输出 JSON

用法：
  bash scripts/exp_common/merge_ckpt.sh <ckpt_actor> /tmp/merged_hf
  python -m trainer.cl_eval --config configs/exp1/cl2r_eval.yaml \
    --tasks-file eval/claweval_manifest.json --output eval/results/xxx/per_task.json

注意：只能在集群跑（需 GPU + verl + lightllm + e2b 凭证），本机无 GPU 无法验证。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="CL eval: run ClawEval over a checkpoint via RemoteAgentLoopManager"
    )
    ap.add_argument("--config", required=True, help="评测 config（建议复用训练 config + override model.path）")
    ap.add_argument("--tasks-file", default="eval/claweval_manifest.json")
    ap.add_argument("--output", default=None, help="输出 JSON 路径；默认 eval/results/<config stem>/per_task.json")
    ap.add_argument("--num-runs", type=int, default=3, help="Pass^N 逐次 rollout 数")
    ap.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides，如 actor_rollout_ref.model.path=/tmp/merged_hf cl.buffer.enabled=false")
    return ap.parse_args()


def main():
    args = parse_args()
    from omegaconf import OmegaConf

    from trainer.cl_main import load_config, _apply_overrides

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.overrides)

    # 读 ClawEval 任务（text-only 191 子集）
    from eval.run_eval import load_tasks

    tasks = load_tasks(args.tasks_file, text_only=True)
    print(f"[cl-eval] 加载 {len(tasks)} 个 ClawEval 文本任务", flush=True)

    # 评测版 TaskRunner + run_cl_eval 在 verl_runner.py（复用训练的启动链路）
    from trainer.verl_runner import run_cl_eval

    results = run_cl_eval(cfg, tasks, args.num_runs)

    out = args.output or f"eval/results/{Path(args.config).stem}/per_task.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[cl-eval] ✅ {len(results)} 条结果 → {out}", flush=True)


if __name__ == "__main__":
    main()
