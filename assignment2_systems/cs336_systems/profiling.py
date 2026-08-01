try:
    import torch.cuda.nvtx as nvtx
except Exception:
    nvtx = None
import math
import torch

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q,K,V,mask):

    with nvtx.range("computing attention scores"):
        scores = Q @ K.transpose(-2,-1)
        scores /= math.sqrt(Q.shape[-1])
        if mask is not None:
                # causal (or other) mask: False positions -> -inf so softmax -> 0
                scores = scores.masked_fill(~mask, float("-inf"))
    with nvtx.range("compute softmax"):
         attention = torch.softmax(scores, dim=-1)
    with nvtx.range("compute matmul"):
         return attention @ V
         