# Shim: torch sdpa-based replacements for flash_attn.bert_padding
# PyTorch 2.13 provides built-in sdpa; flash_attn is only used for
# unpad_input / pad_input / rearrange / index_first_axis in verl.
import torch

def index_first_axis(residual, indices):
    return residual[indices]

def unpad_input(hidden_states, attention_mask):
    seqlens = attention_mask.sum(dim=1).to(torch.int32)
    indices = attention_mask.bool().nonzero(as_tuple=True)
    hidden_states_flat = hidden_states[indices]
    max_seqlen = seqlens.max().item()
    cu_seqlens = torch.zeros(seqlens.shape[0] + 1, dtype=torch.int32, device=seqlens.device)
    torch.cumsum(seqlens, dim=0, out=cu_seqlens[1:], dtype=torch.int32)
    return hidden_states_flat, indices, cu_seqlens, max_seqlen

def pad_input(hidden_states, indices, batch, max_seqlen):
    output = torch.zeros(batch, max_seqlen, hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)
    output[indices] = hidden_states
    return output

def rearrange(tensor, indices, cu_seqlens, max_seqlen):
    return tensor

print('flash_attn shim loaded')
