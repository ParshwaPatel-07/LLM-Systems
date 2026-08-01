import torch
import torch.nn as nn
from math import sqrt, cos, pi
from einops import rearrange
from collections.abc import Callable, Iterable
from typing import Optional
import numpy as np 
import os
import typing
import argparse
from dataclasses import dataclass
import wandb
import time
from tqdm.auto import tqdm

# =============================================================================
# Notation used throughout (Transformer LM shapes):
#   B  = batch_size
#   S  = seq_len  (context / sequence length)
#   V  = vocab_size
#   D  = d_model  (model / embedding dimension)
#   H  = num_heads
#   d_k = d_v = D / H   (per-head key/query/value dim)
#   d_ff = feed-forward hidden width (SwiGLU intermediate)
#
# High-level forward flow (TransformerLM):
#   tokens [B, S]
#     -> Embedding           -> x [B, S, D]
#     -> N x TransformerBlock (pre-norm + residual + RoPE attention + SwiGLU)
#     -> final RMSNorm       -> x [B, S, D]
#     -> lm_head Linear      -> logits [B, S, V]
# =============================================================================


class Linear(nn.Module):
    """
    Affine map without bias: y = x @ W^T
    Matches the assignment's Linear (no bias term).
    """

    def __init__(self, in_features:int, out_features:int, device:torch.device | None=None, dtype:torch.dtype | None=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # Truncated normal init with std = sqrt(2 / (fan_in + fan_out))
        std = sqrt(2.0 / (in_features + out_features))

        # W: [out_features, in_features]  (stored as out x in, applied as x @ W.T)
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3, b=3)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        # W.T: [in_features, out_features]
        # out: [..., out_features]
        return x @ self.W.T
    
class Embedding(nn.Module):
    """
    Token embedding lookup table.
    Maps discrete token ids -> continuous d_model vectors.
    """

    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device | None=None, dtype:torch.dtype | None=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # embedding table: [vocab_size, d_model]
        self.embedding = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding, mean=0, std=1, a=-3, b=3)

    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        # token_ids: [B, S]  (integer token indices)
        # out:       [B, S, d_model]  via advanced indexing into the table
        return self.embedding[token_ids]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (no mean centering, no bias).
    RMSNorm(x) = (x / RMS(x)) * gain
    where RMS(x) = sqrt(mean(x^2) + eps) over the last dim (d_model).
    """

    def __init__(self, d_model:int, eps:float=1e-5, device:torch.device | None=None, dtype:torch.dtype | None=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps 
        self.device = device
        self.dtype = dtype

        # learnable per-channel scale: [d_model]
        self.gain = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.gain, mean=0, std=1, a=-3, b=3)
    
    def forward(self, x:torch.Tensor) -> torch.tensor:
        # x: [..., d_model]  (typically [B, S, D])
        in_dtype = x.dtype
        # upcast for numerical stability of the mean-of-squares
        x = x.to(torch.float32)
        # RMS_x: [..., 1]  (keepdim so it broadcasts over d_model)
        RMS_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # result: [..., d_model]
        result = (x / RMS_x) * self.gain

        return result.to(in_dtype)

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network (gated activation):
      FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )
    where SiLU(z) = z * sigmoid(z)  ("swish").

    Shape flow:
      x    [B, S, D]
      W1x  [B, S, d_ff]
      W3x  [B, S, d_ff]
      gate [B, S, d_ff]  = SiLU(W1x) * W3x
      out  [B, S, D]     = W2(gate)
    """

    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff)   # up-projection for SiLU branch
        self.w2 = Linear(d_ff, d_model)   # down-projection back to d_model
        self.w3 = Linear(d_model, d_ff)   # up-projection for gate branch

    def forward(self, x):
        # x: [B, S, D]
        w1x = self.w1(x)  # [B, S, d_ff]
        w3x = self.w3(x)  # [B, S, d_ff]

        # SiLU(w1x) * w3x -> [B, S, d_ff]; then W2 -> [B, S, D]
        return self.w2(((w1x * torch.sigmoid(w1x)) * w3x))

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE: rotates pairs of features in Q/K by an angle that depends on
    absolute token position. Relative position info then appears in QK^T.

    Precompute for each position i and pair index j:
      theta_ij = i / (base^(2j/d_k))
      cos[i, j], sin[i, j]

    For each feature pair (x_even, x_odd):
      [x'_even] = [cos  -sin] [x_even]
      [x'_odd ]   [sin   cos] [x_odd ]
    """

    def __init__(self,theta:float,d_k:int,max_seq_len:int,device:torch.device | None=None):
        
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # inv_freq: [d_k/2]  — frequencies for each 2D rotation plane
        inv_freq = 1.0 / (theta ** (torch.arange(0,d_k,2,device=device) / d_k))
        # positions: [max_seq_len]
        positions  = torch.arange(0,max_seq_len, device=device)
        # angle: [max_seq_len, d_k/2]  = positions outer inv_freq
        angle = torch.outer(positions,inv_freq)

        # buffers (not trained): cos/sin tables [max_seq_len, d_k/2]
        self.register_buffer("cos", torch.cos(angle), persistent=False)
        self.register_buffer("sin", torch.sin(angle), persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        # x:               [..., S, d_k]   (e.g. [B, H, S, d_k] for multi-head Q/K)
        # token_positions: [S] or broadcastable index into the cos/sin tables
        # cos, sin:        [..., S, d_k/2] after indexing
        cos = self.cos[token_positions] 
        sin = self.sin[token_positions]

        # split last dim into even/odd pairs: each [..., S, d_k/2]
        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        # 2D rotation in each plane
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        # interleave even/odd back to [..., S, d_k]
        out = torch.empty_like(x)
        out[...,0::2] = rot_even
        out[...,1::2] = rot_odd
        
        return out

def softmax(x:torch.Tensor, i:int):
    """
    Numerically stable softmax along dim i:
      softmax(x)_j = exp(x_j - max(x)) / sum_k exp(x_k - max(x))
    """

    # subtract max for stability (does not change softmax result)
    stabelized_x = x - torch.max(x, dim=i, keepdim=True).values
    return torch.softmax(stabelized_x, dim=i) 

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor | None=None):
    """
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + mask ) V

    Shape flow (with optional leading batch/head dims):
      Q:      [..., S_q, d_k]
      K:      [..., S_k, d_k]
      V:      [..., S_k, d_v]
      scores: [..., S_q, S_k]  = Q @ K^T / sqrt(d_k)
      attn:   [..., S_q, S_k]  after softmax over keys
      out:    [..., S_q, d_v]  = attn @ V
    """

    # weight / scores: [..., S_q, S_k]
    weight = Q @ K.transpose(-2,-1)
    d_k = Q.shape[-1]
    # scale by 1/sqrt(d_k) to keep variance ~O(1) before softmax
    weight /= sqrt(d_k)
    if mask is not None:
        # causal (or other) mask: False positions -> -inf so softmax -> 0
        weight = weight.masked_fill(~mask, float("-inf"))
    Attention = softmax(weight, -1)  # [..., S_q, S_k]

    return Attention @ V # dimension: (batch_size, seq_len, d_V)

class Multihead_Self_Attention(nn.Module):
    """
    Multi-head self-attention (optional RoPE on Q and K) + causal mask.

    Shape flow:
      x:     [B, S, D]
      QKV:   [B, S, D] each (after W_q / W_k / W_v)
      split: [B, H, S, d_k]  with D = H * d_k
      (+RoPE on Q, K if enabled)
      attn:  [B, H, S, d_k]
      concat:[B, S, D]
      out:   [B, S, D] via W_o
    """

    def __init__(self, d_model:int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
       
        # per-head dims; require d_model % num_heads == 0
        self.d_k = self.d_v = self.d_model // self.num_heads

        # projections: D -> D (packed heads), then D -> D output
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)


    def forward(self, x:torch.Tensor, token_positions:torch.Tensor | None=None, theta:float = 10000, max_seq_len:int | None=None, use_rope: bool = False):
        # x: [B, S, D]
        seq_len = x.shape[-2]
        if token_positions is None:
            # default absolute positions 0..S-1
            token_positions = torch.arange(seq_len, device=x.device)

        if max_seq_len is None:
            max_seq_len = seq_len

        # project to Q, K, V: each [B, S, D]
        W_qx = self.W_q(x)
        W_kx = self.W_k(x)
        W_vx = self.W_v(x)
       

        # d_model = (heads d_k)
        # reshape packed heads: [B, S, D] -> [B, H, S, d_k]
        Q = rearrange(W_qx, "batch seq_len (heads d_k) ->  batch heads seq_len d_k", heads=self.num_heads, d_k=self.d_k)
        K = rearrange(W_kx, "batch seq_len (heads d_k) ->  batch heads seq_len d_k", heads=self.num_heads, d_k=self.d_k)
        V = rearrange(W_vx, "batch seq_len (heads d_k) ->  batch heads seq_len d_k", heads=self.num_heads, d_k=self.d_k)

        if use_rope:
            # apply same RoPE rotation to Q and K (not V)
            rope = RotaryPositionalEmbedding(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len,device=x.device)

            Q = rope(Q, token_positions)  # [B, H, S, d_k]
            K = rope(K, token_positions)  # [B, H, S, d_k]

        # causal lower-triangular mask: [S, S], True = allowed
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        # head: [B, H, S, d_k]
        head = scaled_dot_product_attention(Q, K, V, mask=mask)
        # merge heads: [B, H, S, d_k] -> [B, S, D]
        MultiHead = rearrange(head, "batch heads seq_len d_k ->  batch seq_len (heads d_k) ")
        # output projection: [B, S, D]
        W_o = self.W_o(MultiHead)
        
        return W_o

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with residual connections:

      y = x + MHA(RMSNorm(x))     # attention sublayer
      z = y + FFN(RMSNorm(y))     # feed-forward sublayer (SwiGLU)

    All tensors stay [B, S, D] through the block.
    """

    def __init__(self, d_model:int, num_heads:int, d_ff:int):
        
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = RMSNorm(d_model, eps=1e-5)          # pre-norm before attention
        self.ln2 = RMSNorm(d_model, eps=1e-5)          # pre-norm before FFN
        self.Attention = Multihead_Self_Attention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor | None=None, theta:float =10000, max_seq_len:int = None, use_rope:bool=True):
        # x: [B, S, D]
        # Attention sublayer + residual
        MHA = self.Attention(self.ln1(x), token_positions=token_positions, theta=theta, max_seq_len=max_seq_len, use_rope=use_rope)  # [B, S, D]
        y = x + MHA  # [B, S, D]
        # FFN sublayer + residual
        FFN = self.ffn(self.ln2(y))  # [B, S, D]
        z = y + FFN  # [B, S, D]

        return z

class TransformerLM(nn.Module):
    """
    Decoder-only Transformer language model.

    Flow:
      tokens [B, S]
        -> token Embedding          [B, S, D]
        -> num_layers x Block       [B, S, D]  (RoPE causal attention + SwiGLU)
        -> final RMSNorm            [B, S, D]
        -> lm_head Linear           [B, S, V]  (next-token logits)
    """

    def __init__(self, vocab_size:int, context_length:int, num_layers:int, d_model:int, num_heads:int, d_ff:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
            )
            for _ in range(num_layers)
        ])
        self.final_ln = RMSNorm(d_model)
        # untied output projection: D -> V (no weight tying with embedding here)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, tokens:torch.Tensor,rope_theta:float=10000.0):
        # tokens: [B, S] integer ids
        x = self.embedding(tokens)  # [B, S, D]

        for layer in self.layers:
            # each layer: [B, S, D] -> [B, S, D]
            x = layer(x, theta=rope_theta, max_seq_len=self.context_length, use_rope=True)

        x = self.final_ln(x)  # [B, S, D]
        logits = self.lm_head(x) # shape: (batch_size, seq_len, vocab_size)

        return logits 

def cross_entropy(logits:torch.Tensor, targets:torch.Tensor):
    """
    Mean cross-entropy for next-token prediction (numerically stable log-softmax).

    Shape flow:
      logits:  [B, S, V]  (or any [..., V])
      targets: [B, S]     integer class indices
      loss:    scalar     = -mean( log_softmax(logits)[targets] )
    """

    # max subtract for stable exp
    max_logits = torch.max(logits, dim=-1, keepdim=True).values  # [..., 1]
    exp_logits = torch.exp(logits - max_logits)                  # [..., V]
    sum_exp_logits = torch.sum(exp_logits, dim=-1, keepdim=True) # [..., 1]
    # log_softmax = log(exp(x - max) / sum) = (x - max) - log(sum)
    log_softmax = (logits - max_logits) - torch.log(sum_exp_logits)  # [..., V]

    # pick log-prob of the true token at each position -> [B, S]
    targeted_log_prob = torch.gather(
        log_softmax,
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = -torch.mean(targeted_log_prob)

    return loss

class SGD(torch.optim.Optimizer):
    """
    SGD with decaying step size: lr_t = lr / sqrt(t+1)
    """
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
        state = self.state[p] # Get state associated with p.
        t = state.get("t", 0) # Get iteration number from the state, or 0.
        grad = p.grad.data # Get the gradient of loss with respect to p.
        p.data -= lr / sqrt(t + 1) * grad # Update weight tensor in-place.
        state["t"] = t + 1 # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    """
    AdamW: decoupled weight decay + Adam moments.

    Per-parameter state:
      m: 1st moment (EMA of grad)     same shape as p
      v: 2nd moment (EMA of grad^2)   same shape as p
      t: step count

    Update:
      p <- p - lr * wd * p                         # decoupled decay
      m <- beta1 m + (1-beta1) g
      v <- beta2 v + (1-beta2) g^2
      p <- p - (lr * sqrt(1-beta2^t)/(1-beta1^t)) * m / (sqrt(v)+eps)
    """

    def __init__(self, params:torch.Tensor,lr, betas, eps, weight_decay):
        defaults = {
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "weight_decay": weight_decay,
    }
        super().__init__(params, defaults)
       
    #     optimizer = AdamW([
    # {"params": embedding.parameters(), "lr": 1e-5},
    # {"params": transformer.parameters(), "lr": 3e-4},
    # {"params": lm_head.parameters(), "lr": 1e-3},
    # ])
    def step(self, closure=None):

        for group in self.param_groups:

            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue
            
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]

                # decoupled weight decay (applied to params, not to the gradient)
                p.data -= lr * weight_decay * p.data
                m = state["m"]
                m.mul_(beta1)
                m.add_((1-beta1) * grad) 
                v = state["v"]
                v.mul_(beta2)
                v.add_((1-beta2) * grad * grad)


                # bias-corrected step size
                alpha_t = lr * sqrt(1 - beta2**t) / (1- beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)

def learning_rate_schedule(t, lr_max, lr_min, warmup_iters, cosine_cycle_iters):
    """
    Linear warmup -> cosine decay -> constant min lr.

      t < warmup:           lr = (t / warmup) * lr_max
      warmup <= t <= cosine: cosine anneal from lr_max down to lr_min
      t > cosine:           lr = lr_min
    """

    if t < warmup_iters:
        return (t / warmup_iters) * lr_max
    elif warmup_iters <= t and t <= cosine_cycle_iters:
        lr_t = lr_min + 0.5 * (1 + cos(((t - warmup_iters)/(cosine_cycle_iters - warmup_iters))*pi)) * (lr_max - lr_min)
        return lr_t
    else:
        return lr_min


def gradient_clipping(params, max_l2_norm):
    """
    Global L2 grad clipping: if ||g||_2 > max_norm, scale all grads by max_norm / (||g||+eps).
    """

    l2_norms = [torch.linalg.vector_norm(p.grad) for p in params if p.grad is not None]
    global_norm = sqrt(sum(p**2 for p in l2_norms))

    if global_norm <= max_l2_norm:
        return None
    else:
        eps = 1e-6
        scale = max_l2_norm / (global_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)
        return None
        
def data_loading(x, batch_size, context_len, device:torch.device):
    """
    Sample random contiguous windows from a 1D token array for LM training.

    For each start index s:
      input  = x[s : s+context_len]       next-token inputs
      target = x[s+1 : s+context_len+1]   labels shifted by 1

    Returns:
      inputs_tensor:  [B, context_len]
      targets_tensor: [B, context_len]
    """

    starts = torch.randint(high=len(x) - context_len, size=(batch_size,), device=device)

    inputs = [x[s: s + context_len] for s in starts.tolist()]
    targets = [x[s+1: s + context_len + 1] for s in starts.tolist()]

    inputs = np.stack(inputs)
    targets = np.stack(targets)

    inputs_tensor = torch.from_numpy(inputs).to(device).long()
    targets_tensor = torch.from_numpy(targets).to(device).long()

    return inputs_tensor, targets_tensor

# def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):

#     checkpoint = {
#     "model": model.state_dict(),
#     "optimizer": optimizer.state_dict(),
#     "step": iteration,
# }
#     torch.save(checkpoint, out)

# def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model:torch.nn.Module, optimizer:torch.optim.Optimizer):

#     checkpoint = torch.load(src)
#     model.load_state_dict(checkpoint["model"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
    
#     return checkpoint["iteration"]

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler,  # Added scheduler
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),  # Save scheduler state
        "step": iteration,  # Standardized key name
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler  # Added scheduler
):
    # Use weights_only=True for security unless custom objects are inside
    checkpoint = torch.load(src, map_location="cpu") 
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint and scheduler is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    return checkpoint["step"]  # Fixed key match

@dataclass
class TrainingConfig:
    # Model
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int

    # Training
    batch_size: int
    learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int
    max_iters: int
    checkpoint_iter: int
    checkpoint_path : str
    train_path: str
    val_path: str
    resume_checkpoint: str | None = None
    # Optimizer
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1

    # Runtime
    eval_interval: int = 100
    device: str = "cuda"

    max_learning_rate: float = 3e-4
    max_l2_norm: float = 1.0
    rope_theta: float = 10000.0
    



def training_together(config: TrainingConfig):
    """
    Training loop:
      sample (x,y) -> forward logits -> cross_entropy -> backward
      -> clip grads -> set lr from schedule -> AdamW step
      periodic val eval + wandb log + checkpointing
    """

    train = np.memmap(
            config.train_path,
            dtype=np.uint16,
            mode="r"
    )
    val = np.memmap(
            config.val_path,
            dtype=np.uint16,
            mode="r"
    )

    model = TransformerLM(vocab_size=config.vocab_size, 
                  context_length=config.context_length, 
                  num_layers=config.num_layers, 
                  d_model=config.d_model,
                  num_heads=config.num_heads,
                  d_ff=config.d_ff,)
    model.to(config.device)
    
    optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1,config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
    )
    start_step = 0

    if config.resume_checkpoint is not None:
        checkpoint = torch.load(
            config.resume_checkpoint,
            map_location=config.device,
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        if "step" in checkpoint:
            start_step = checkpoint["step"] + 1
        elif "iteration" in checkpoint:
            start_step = checkpoint["iteration"] + 1
        else:
            raise KeyError("Invalid checkpoint format")
        
        current_lr = learning_rate_schedule(
            t=start_step,
            lr_max=config.max_learning_rate,
            lr_min=config.min_learning_rate,
            warmup_iters=config.warmup_iters,
            cosine_cycle_iters=config.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    print(f"✅ Resuming training from step {start_step}")

    wandb.init(
        project="cs336-assignment",
        name="tiny-debug",
        config=config.__dict__,
    )
    start_time = time.time()

    for step in tqdm(
        range(start_step, config.max_iters),
        desc="Training",
        initial=start_step,
        total=config.max_iters,
    ):

        # x: [B, S] inputs, y: [B, S] next-token targets
        x, y = data_loading(
                train,
                batch_size=config.batch_size,
                context_len=config.context_length,
                device=config.device
        )

        logits = model(x, config.rope_theta)  # [B, S, V]
        loss = cross_entropy(logits=logits, targets=y)
        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), config.max_l2_norm)
        current_lr = learning_rate_schedule(
                t=step,
                lr_max=config.max_learning_rate,
                lr_min=config.min_learning_rate,
                warmup_iters=config.warmup_iters,
                cosine_cycle_iters=config.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        optimizer.step()

        if step % config.eval_interval == 0:

            model.eval()
            with torch.no_grad():

                x_val, y_val = data_loading(
                    val,
                    batch_size=config.batch_size,
                    context_len=config.context_length,
                    device=config.device
                )
                logits_val = model(x_val, config.rope_theta)
                loss_val = cross_entropy(logits=logits_val, targets=y_val)
                elapsed = time.time() - start_time
                wandb.log({
                        "step": step,
                        "train_loss": loss.item(),
                        "val_loss": loss_val.item(),
                        "learning_rate": current_lr,
                        "elapsed_time": elapsed,
                    })
            model.train()

            # print("=" * 70)
            # print(f"Step          : {step}")
            # print(f"Train Loss    : {loss.item():.4f}")
            # print(f"Validation    : {loss_val.item():.4f}")
            # print(f"Learning Rate : {current_lr:.2e}")
            # print(f"Elapsed Time  : {elapsed:.1f}s")
            # print("=" * 70)
            print(
            f"Step {step:6d} | "
            f"Train {loss.item():.4f} | "
            f"Val {loss_val:.4f} | "
            f"LR {current_lr:.2e} | "
            f"{elapsed/60:.1f} min"
        )
        
        if step > 0 and step % config.checkpoint_iter == 0:
            path = os.path.join(
                config.checkpoint_path,
                f"checkpoint_{step}.pt"
            )
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=path,scheduler=None)  # Pass scheduler if used
            print("=" * 60)
            print(f"Loaded checkpoint : {config.resume_checkpoint}")
            print(f"Resuming at step  : {start_step}")
            print("=" * 60)

    wandb.finish()

def main():

    parser = argparse.ArgumentParser(
    description="Train a Transformer Language Model"
)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)

    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)

    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=100000)

    parser.add_argument("--max_iters", type=int, default=100000)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--beta1",type=float,default=0.9,)

    parser.add_argument("--beta2",type=float,default=0.95,)
    parser.add_argument("--eps",type=float,default=1e-8,)
    parser.add_argument("--weight_decay",type=float,default=0.1,)

    parser.add_argument("--eval_interval",type=int,default=100,)
    parser.add_argument("--max_l2_norm",type=float,default=1.0,)
    parser.add_argument("--checkpoint_iter",type=int,default=5000,)
    parser.add_argument("--max_learning_rate",type=float,default=3e-4,)
    parser.add_argument("--rope_theta", type=float, default=10000.0,)
    args = parser.parse_args()

    config = TrainingConfig(
    train_path=args.train_path,
    val_path=args.val_path,
    checkpoint_path=args.checkpoint_path,

    vocab_size=args.vocab_size,
    context_length=args.context_length,

    num_layers=args.num_layers,
    d_model=args.d_model,
    num_heads=args.num_heads,
    d_ff=args.d_ff,

    batch_size=args.batch_size,

    learning_rate=args.learning_rate,
    min_learning_rate=args.min_learning_rate,

    warmup_iters=args.warmup_iters,
    cosine_cycle_iters=args.cosine_cycle_iters,
    max_iters=args.max_iters,

    device=args.device,
    max_learning_rate=args.max_learning_rate,
    checkpoint_iter=args.checkpoint_iter,

    beta1=args.beta1,
    beta2=args.beta2,
    eps=args.eps,
    weight_decay=args.weight_decay,

    eval_interval=args.eval_interval,
    rope_theta=args.rope_theta,
    max_l2_norm=args.max_l2_norm,
)
    
    training_together(config=config)

if __name__ == "__main__":
    main()

def decoding(model: TransformerLM, tokenizer, prompt, max_newtokens=100, tempreture=1, top_p = 0.9):
    """
    Autoregressive decode with temperature + nucleus (top-p) sampling.

    Shape flow each step:
      tokens:     [1, T]
      logits:     [1, T, V]
      next_logit: [1, V]          last position only
      probs:      [1, V]          softmax(logit / temperature)
      sample:     [1, 1]          multinomial over truncated nucleus
      tokens:     [1, T+1]        append sampled id
    """

    tokens = tokenizer(prompt)
    for _ in range(max_newtokens):

        logits = model(tokens)
        next_logit = logits[:,-1,:]
        probs = torch.softmax(next_logit / tempreture, dim=-1)
        # nucleus sampling: keep smallest set of tokens with cumprob >= top_p
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_sum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_sum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs.masked_fill_(mask, 0)
        new_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sample = torch.multinomial(new_probs, num_samples=1)
        token_id = torch.gather(sorted_indices,dim=-1,index=sample)
        token_id = token_id.view((1,1))
        tokens = torch.cat((tokens, token_id), dim=-1)
        if token_id.item() == tokenizer.eot_token:
            break
    
    return tokenizer.decode(tokens[0].tolist())
        




    




