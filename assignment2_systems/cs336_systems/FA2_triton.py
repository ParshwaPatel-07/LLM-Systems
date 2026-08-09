import torch
import triton
import triton.language as tl
import math

@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_pointer = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES,D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0),
    )

    L_ptr_batch = L_ptr + batch_index * stride_lb + query_tile_index * Q_TILE_SIZE * stride_lq

    Q = tl.load(Q_block_ptr)
    O = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        K = tl.load(K_block_ptr)
        V = tl.load(V_block_ptr)

        S = tl.dot(Q, tl.trans(K))
        S *= scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            S = tl.where(causal_mask, S, -1e6)
        m_old = m
        m = tl.maximum(m_old, tl.max(S, axis=1))
        P = tl.exp(S - m[:,None])
        l_old = l
        m_diff = tl.exp(m_old - m)
        l = m_diff*l_old + tl.sum(P, axis=1)
        O_old = O
        P = P.to(V.dtype)
        O = m_diff[:,None]*O_old 
        O = tl.dot(P, V, acc=O)

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE,0))

    O = O/l[:,None]
    L = m + tl.log(l)
    tl.store(L_ptr_batch + tl.arange(0, Q_TILE_SIZE) * stride_lq, L)
    tl.store(O_block_pointer, O.to(O_block_pointer.type.element_ty))

class FlashAttention2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, S, D = Q.shape
        O = torch.empty_like(Q)
        L = torch.empty((B,S), device=Q.device, dtype=torch.float32,)
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        grid = (triton.cdiv(S, Q_TILE_SIZE), B)
        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            S,
            S,
            1 / math.sqrt(D),
            D=D,
            Q_TILE_SIZE=16,
            K_TILE_SIZE=16,
            is_causal=is_causal,
        )
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError
