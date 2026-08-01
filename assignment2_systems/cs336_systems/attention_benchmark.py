import torch
from cs336_basics.model import scaled_dot_product_attention
import timeit

batch_size = 8
# d_model = [16, 32, 64, 128]
# seq_len = [256, 1024, 4096, 8192, 16384]
# num_fwd_pass, num_bwd_pass = 100, 100
seq_len = [256, 1024]
d_model = [16, 32]
num_fwd_pass = 10
num_bwd_pass = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
num_warmup = 5

results = []

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

def benchmark_fwd(num_fwd_pass, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor | None=None):
    step_timings = []
    for _ in range(num_fwd_pass):
        synchronize()
        start_tick = timeit.default_timer()
        out = scaled_dot_product_attention(Q, K, V, mask)
        synchronize()
        end_tick = timeit.default_timer()
        step_timings.append(end_tick - start_tick)
    return step_timings

def benchmark_bwd(num_bwd_pass, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor | None=None):
    step_timings = []
    for _ in range(num_bwd_pass):
        synchronize()
        start_tick = timeit.default_timer()
        out = scaled_dot_product_attention(Q, K, V, mask)
        # memory_before_backward = (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        if torch.backends.mps.is_available():
            memory_before_backward = torch.mps.current_allocated_memory()
        loss = out.sum()
        loss.backward()
        synchronize()
        if Q.grad is not None:
            Q.grad.zero_()
        if K.grad is not None:
            K.grad.zero_()
        if V.grad is not None:
            V.grad.zero_()
        end_tick = timeit.default_timer()
        step_timings.append(end_tick - start_tick)
    return step_timings, memory_before_backward

def pytorch_attention(batch_size, num_warmup, num_fwd_pass, num_bwd_pass):
        for d in d_model:
            for s in seq_len:
                try:
                    q = torch.randn(batch_size, s, d,device=device,requires_grad=True)
                    k = torch.randn(batch_size, s, d,device=device,requires_grad=True)
                    v = torch.randn(batch_size, s, d,device=device,requires_grad=True)

                    for _ in range(num_warmup):
                        scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)
                        synchronize()

                    #call forward benchmark
                    step_timings_fwd = benchmark_fwd(num_fwd_pass, Q=q, K=k, V=v, mask=None)
                    #call backward benchmark
                    step_timings_bwd, memory_usage = benchmark_bwd(num_bwd_pass, Q=q, K=k, V=v, mask=None)

                    forward_avg = sum(step_timings_fwd) / len(step_timings_fwd)
                    backward_avg = sum(step_timings_bwd) / len(step_timings_bwd)
                    results.append({
                        "d_model": d,
                        "seq_len": s,
                        "forward_time": forward_avg,
                        "backward_time": backward_avg,
                        "memory": memory_usage
                    })
                    print(
                        f"d={d:3d} "
                        f"seq={s:5d} "
                        f"fwd={forward_avg:.5f}s "
                        f"bwd={backward_avg:.5f}s "
                        f"mem={memory_usage/1024**2:.10f} MB"
                    )
                except torch.OutOfMemoryError:
                        print(f"Out of memory for d_model={d} and seq_len={s}. Skipping this configuration.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.mps.is_available():
                            torch.mps.empty_cache()
    
if __name__ == "__main__":
    pytorch_attention(
        batch_size=batch_size,
        num_warmup=num_warmup,
        num_fwd_pass=num_fwd_pass,
        num_bwd_pass=num_bwd_pass,
    )

    