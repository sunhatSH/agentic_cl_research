"""Auto-imported by Python at startup (when this dir is on PYTHONPATH).

Applies the Qwen3.6 (qwen3_5) GDN fix in EVERY process, including vllm's spawned
worker subprocesses (where the gated-delta-rule forward actually runs). On H800
(sm90) vllm0.16rc forces FlashInfer's gdn_prefill kernel, which is numerically
broken -> garbled output. We force the correct native Triton path instead.
"""
try:
    from vllm.model_executor.models.qwen3_next import ChunkGatedDeltaRule

    ChunkGatedDeltaRule.forward_cuda = ChunkGatedDeltaRule.forward_native
    _orig_init = ChunkGatedDeltaRule.__init__

    def _patched_init(self):
        _orig_init(self)
        self._forward_method = self.forward_native

    ChunkGatedDeltaRule.__init__ = _patched_init
    import os

    print(f"[gdn-patch pid={os.getpid()}] ChunkGatedDeltaRule -> forward_native", flush=True)
except Exception:
    pass
