#!/usr/bin/env python3
"""Quick smoke test: vllm 0.19 + Qwen3.6-27B offline inference.

Uses --gdn-prefill-backend triton to avoid FlashInfer GDN deadlock on sm90
(see doc/vllm_upgrade_0.19.md for details).
"""

from vllm import LLM, SamplingParams


def main():
    print("Loading model...", flush=True)
    llm = LLM(
        model="/mnt/afs_toolcall/sunhao4/models/Qwen3.6-27B",
        tensor_parallel_size=4,
        max_model_len=2048,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        additional_config={"gdn_prefill_backend": "triton"},
    )

    print("Model loaded! Generating...", flush=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)
    outputs = llm.generate(["Say hello in one word."], sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Response: {output.outputs[0].text!r}")

    print("✅ Inference successful!")


if __name__ == "__main__":
    main()
