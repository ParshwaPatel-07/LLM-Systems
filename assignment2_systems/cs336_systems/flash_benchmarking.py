import torch
import timeit

from cs336_systems.FA2_pytorch import FlashAttention2 as FlashAttentionPytorch
from cs336_systems.FA2_triton import FlashAttention2 as FlashAttentionTriton

BATCH_SIZE = 1

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DIMS = [16, 32, 64, 128]

DTYPES = [torch.float16, torch.float32]

NUM_WARMUP = 5
NUM_ITERATIONS = 20

def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def make_inputs(seq_len, d, dtype):
    Q = torch.randn(
        BATCH_SIZE, seq_len, d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    return Q, K, V

def benchmark(fn, num_warmup=NUM_WARMUP, num_iterations=NUM_ITERATIONS):
    for _ in range(num_warmup):
        fn()

    synchronize()

    timings = []

    for _ in range(num_iterations):
        synchronize()
        start = timeit.default_timer()

        fn()

        synchronize()
        end = timeit.default_timer()

        timings.append(end - start)

    return sum(timings) / len(timings)

def benchmark_backward(attention_fn, Q, K, V):
    def step():
        Q.grad = None
        K.grad = None
        V.grad = None

        O = attention_fn(Q, K, V, True)
        O.sum().backward()

    return benchmark(step)

def print_results(seq_len, results):
    print(f"\n{'=' * 20} seq_len={seq_len} {'=' * 20}")
    print()
    print(
        f"{'dtype':<8}"
        f"{'d':<6}"
        f"{'PyTorch Fwd':<16}"
        f"{'Triton Fwd':<16}"
        f"{'PyTorch Bwd':<16}"
    )

    for result in results:
        print(
            f"{result['dtype']:<8}"
            f"{result['d']:<6}"
            f"{result['pytorch_fwd'] * 1000:>10.2f} ms    "
            f"{result['triton_fwd'] * 1000:>10.2f} ms    "
            f"{result['pytorch_bwd'] * 1000:>10.2f} ms"
        )

def main():
    results = []

    for seq_len in SEQ_LENS:

        seq_results = []

        for dtype in DTYPES:
            for d in DIMS:

                try:
                    # create Q, K, V
                    Q, K, V = make_inputs(seq_len, d, dtype)
                    # run benchmarks
                    pytorch_fwd_time = benchmark(lambda: FlashAttentionPytorch.apply(Q, K, V, True))
                    triton_fwd_time = benchmark(lambda: FlashAttentionTriton.apply(Q, K, V, True))
                    pytorch_bwd_time = benchmark_backward(FlashAttentionPytorch.apply, Q, K, V)
                    #e2e_time = benchmark(lambda: FlashAttentionPytorch.apply(Q, K, V, True))

                    result = {
                        "dtype": "fp16" if dtype == torch.float16 else "fp32",
                        "d": d,
                        "pytorch_fwd": pytorch_fwd_time,
                        "triton_fwd": triton_fwd_time,
                        "pytorch_bwd": pytorch_bwd_time,
                        #"e2e": e2e_time,
                    }

                    seq_results.append(result)
                    results.append({
                        "seq_len": seq_len,
                        **result,
                    })

                except torch.cuda.OutOfMemoryError:
                    print(f"OOM: dtype={dtype}, d={d}")
                    torch.cuda.empty_cache()

        print_results(seq_len, seq_results)

    return results


if __name__ == "__main__":
    main()
