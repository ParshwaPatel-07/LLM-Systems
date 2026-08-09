import torch
import torch.autograd
import math
from einops import rearrange

def flash_backward(Q, K, V, O, dO, L, is_causal):
    d = Q.shape[2]
    scale = 1 / math.sqrt(d)
    S = Q @ torch.transpose(K, 1, 2) * scale
    if is_causal:
        n_queries = Q.shape[1]
        n_keys = K.shape[1]
        causal_mask = torch.triu(torch.ones(n_queries, n_keys, device=Q.device, dtype=torch.bool),diagonal=1,)
        S = S.masked_fill(causal_mask, float("-inf"))
    # Recompute the backward pass in FP32. This avoids mixed-dtype matmul
    # errors when the benchmark uses FP16 inputs, while keeping the
    # gradients returned to autograd in the original input dtype.
    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()
    O_f = O.float()
    dO_f = dO.float()
    L_f = L.float()

    P = torch.exp(S - L_f.unsqueeze(-1))
    dV = torch.transpose(P, 1, 2) @ dO_f
    dP = dO_f @ torch.transpose(V_f, 1, 2)
    D_vec = (O_f * dO_f).sum(dim=-1)
    dS = P * (dP - D_vec.unsqueeze(-1))
    dQ = (dS @ K_f) * scale
    dK = (torch.transpose(dS, 2, 1) @ Q_f) * scale

    return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype)
compiled_flash_backward = torch.compile(flash_backward)

class FlashAttention2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        BLOCK_M = 16
        BLOCK_N = 16

        B, S, D = Q.shape

        O = torch.zeros_like(Q)
        L = torch.zeros(B, S, device=Q.device)

        for q_start in range(0, S, BLOCK_M):

            q_end = q_start + BLOCK_M
            Q_i = Q[:, q_start:q_end, :]

            O_i = O[:, q_start:q_end, :]
            l_i = torch.zeros((B, BLOCK_M, 1), device=Q.device)
            m_i = torch.full((B, BLOCK_M,1), float('-inf'), device=Q.device)

            for k_start in range(0, S, BLOCK_N):

                k_end = k_start + BLOCK_N
                K_j = K[:, k_start:k_end, :]
                V_j = V[:, k_start:k_end, :]
                # Q_i has shape (B, BLOCK_M, D)
                #  K_j has shape (B, BLOCK_N, D)
                s_ij = (Q_i @ torch.transpose(K_j, 1, 2)) / math.sqrt(D)
                rowmax = torch.amax(s_ij, dim=-1, keepdim=True)
                m_old = m_i
                m_ij = torch.maximum(m_old,rowmax)
                P_ij = torch.exp(s_ij - m_ij)
                # (B, BLOCK_M, 1) - (B, BLOCK_M, 1) -> perfectly aligned
                l_ij = (torch.exp(m_old-m_ij))*l_i + torch.sum(P_ij, dim=-1, keepdim=True)
                O_ij = torch.exp(m_old - m_ij)*O_i + P_ij.float() @ V_j.float()
                m_i = m_ij
                l_i = l_ij
                O_i = O_ij

            O_i = O_ij / l_ij
            L_i = m_ij + torch.log(l_ij)
            #Write 𝑶𝑖 to global memory as the 𝑖-th tile of 𝑶.
            O[:, q_start:q_end, :] = O_i
            #Write 𝐿𝑖 to global memory as the 𝑖-th tile of 𝐿.
            L[:, q_start:q_end] = L_i.squeeze(dim=-1)
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        # print("s_ij.shape:", s_ij.shape)
        # print("m_ij.shape:", m_ij.shape)
        # print("P_ij.shape:", P_ij.shape)
        # print("l_ij.shape:", l_ij.shape)
        # print("O_ij.shape:", O_ij.shape)
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = compiled_flash_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None




# if __name__ == "__main__":
#     # Example usage
#     B, S, D = 1, 16, 8  # Batch size, sequence length, embedding dimension
#     Q = torch.randn(B, S, D)
#     K = torch.randn(B, S, D)
#     V = torch.randn(B, S, D)

#     O_flash = FlashAttention2.apply(Q, K, V, False)
#     scores = Q @ K.transpose(-1, -2) / math.sqrt(D)
#     P = torch.softmax(scores, dim=-1)
#     O_ref = P @ V
#     print(torch.max(torch.abs(O_flash - O_ref)))
#     # print("Output O shape:", O.shape)
#     # print("Output L shape:", L.shape)