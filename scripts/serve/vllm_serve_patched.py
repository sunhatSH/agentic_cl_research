"""vllm OpenAI server with a runtime patch for Qwen3.6 (qwen3_5) on sm90.

Why: on H800 (compute capability 9.0) vllm 0.16rc forces the gated-delta-rule
linear-attention layer through FlashInfer's `gdn_prefill` kernel, which produces
NUMERICALLY WRONG output -> model emits pure garbage / first-token EOS. The
native Triton (fla) path is correct. We can't edit the shared ds32_env source
(owned by another user), so we monkey-patch in-process: force ChunkGatedDeltaRule
to use forward_native instead of forward_cuda.

Usage: same args as `python -m vllm.entrypoints.openai.api_server ...`
    python scripts/serve/vllm_serve_patched.py --model ... --tensor-parallel-size 8 ...
"""
import runpy
import sys


def _apply_patch():
    from vllm.model_executor.models.qwen3_next import ChunkGatedDeltaRule

    ChunkGatedDeltaRule.forward_cuda = ChunkGatedDeltaRule.forward_native
    _orig_init = ChunkGatedDeltaRule.__init__

    def _patched_init(self):
        _orig_init(self)
        self._forward_method = self.forward_native

    ChunkGatedDeltaRule.__init__ = _patched_init
    print("[patch] ChunkGatedDeltaRule -> forward_native (bypass broken FlashInfer GDN on sm90)", flush=True)


if __name__ == "__main__":
    _apply_patch()
    sys.argv = ["vllm.entrypoints.openai.api_server"] + sys.argv[1:]
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
