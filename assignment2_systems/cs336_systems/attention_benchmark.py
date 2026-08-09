import torch
from cs336_basics.model import scaled_dot_product_attention
import timeit
import pandas as pd

batch_size = 8
# d_model = [16, 32, 64, 128]
# seq_len = [256, 1024, 4096, 8192, 16384]
# num_fwd_pass, num_bwd_pass = 100, 100
seq_len = [256, 1024]
d_model = [16, 32]
num_fwd_pass = 10
num_bwd_pass = 10
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
num_warmup = 5

results = []

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

compiled_attention = torch.compile(scaled_dot_product_attention)

def benchmark_fwd(num_fwd_pass, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor | None=None, attention_fn:callable=None):
    step_timings = []
    for _ in range(num_fwd_pass):
        synchronize()
        start_tick = timeit.default_timer()
        out = attention_fn(Q, K, V, mask)
        synchronize()
        end_tick = timeit.default_timer()
        step_timings.append(end_tick - start_tick)
    return step_timings

def benchmark_bwd(num_bwd_pass, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, mask:torch.Tensor | None=None, attention_fn:callable=None):
    step_timings = []
    memory = []
    for _ in range(num_bwd_pass):
        synchronize()
        start_tick = timeit.default_timer()
        out = attention_fn(Q, K, V, mask)
        # memory_before_backward = (torch.cuda.memory_allocated() if torch.cuda.is_available() else 0)
        loss = out.sum()
        if torch.backends.mps.is_available():
            memory.append(torch.mps.current_allocated_memory())
        loss.backward()
        synchronize()
        Q.grad = None
        K.grad = None
        V.grad = None
        end_tick = timeit.default_timer()
        step_timings.append(end_tick - start_tick)
    return step_timings, sum(memory)/len(memory)

def pytorch_attention(batch_size, num_warmup, num_fwd_pass, num_bwd_pass):
        for d in d_model:
            for s in seq_len:
                try:
                    q = torch.randn(batch_size, s, d,device=device,requires_grad=True)
                    k = torch.randn(batch_size, s, d,device=device,requires_grad=True)
                    v = torch.randn(batch_size, s, d,device=device,requires_grad=True)

                    for _ in range(num_warmup):
                        scaled_dot_product_attention(q, k, v, None)
                        synchronize()
                    #call forward benchmark
                    step_timings_fwd = benchmark_fwd(num_fwd_pass, Q=q, K=k, V=v, mask=None, attention_fn=scaled_dot_product_attention)
                    #call backward benchmark
                    step_timings_bwd, memory_usage = benchmark_bwd(num_bwd_pass, Q=q, K=k, V=v, mask=None, attention_fn=scaled_dot_product_attention)

                    forward_avg = sum(step_timings_fwd) / len(step_timings_fwd)
                    backward_avg = sum(step_timings_bwd) / len(step_timings_bwd)

                    for _ in range(num_warmup):
                        compiled_attention(Q=q, K=k, V=v, mask=None)
                        synchronize()
                    # call compiled forward benchmark
                    step_timings_fwd_compiled = benchmark_fwd(num_fwd_pass, Q=q, K=k, V=v, mask=None, attention_fn=compiled_attention)
                    # call compiled backward benchmark
                    step_timings_bwd_compiled, memory_usage_compiled = benchmark_bwd(num_bwd_pass, Q=q, K=k, V=v, mask=None, attention_fn=compiled_attention)

                    forward_avg_compiled = sum(step_timings_fwd_compiled) / len(step_timings_fwd_compiled)
                    backward_avg_compiled = sum(step_timings_bwd_compiled) / len(step_timings_bwd_compiled)

                    results.append({
                            "d_model": d,
                            "seq_len": s,
                            "fwd": forward_avg,
                            "fwd_compiled": forward_avg_compiled,
                            "bwd": backward_avg,
                            "bwd_compiled": backward_avg_compiled,
                            "memory": memory_usage,
                            "memory_compiled": memory_usage_compiled,
                        })
                    print(f"Completed benchmark for d_model={d} and seq_len={s}.")
                    print(f"Forward Avg: {forward_avg:.6f} seconds, Forward Compiled Avg: {forward_avg_compiled:.6f} seconds")
                    print(f"Backward Avg: {backward_avg:.6f} seconds, Backward Compiled Avg: {backward_avg_compiled:.6f} seconds")
                    print(f"Memory Usage: {memory_usage/1e6:.2f} MB, Memory Usage Compiled: {memory_usage_compiled/1e6:.2f} MB")
                    print("-"*50)
                except torch.OutOfMemoryError:
                        print(f"Out of memory for d_model={d} and seq_len={s}. Skipping this configuration.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.mps.is_available():
                            torch.mps.empty_cache()
        df = pd.DataFrame(results)
        df.to_csv("attention_compile_results.csv", index=False)
        print(df) 
if __name__ == "__main__":
    pytorch_attention(
        batch_size=batch_size,
        num_warmup=num_warmup,
        num_fwd_pass=num_fwd_pass,
        num_bwd_pass=num_bwd_pass,
    )

    