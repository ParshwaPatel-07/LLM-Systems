import os
import time
import argparse

import torch
import torch.distributed as dist

from cs336_basics.model import (
    TransformerLM,
    AdamW,
    cross_entropy,
    gradient_clipping,
)

from cs336_systems.ddp_overlap_individual_parameters import DDP


# ============================================================
# Benchmark configuration
# ============================================================

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
BATCH_SIZE = 2

NUM_LAYERS = 32
D_MODEL = 2560
NUM_HEADS = 32
D_FF = 10240 

ROPE_THETA = 10000.0

WARMUP_STEPS = 5
BENCHMARK_STEPS = 10

LEARNING_RATE = 3e-4
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
WEIGHT_DECAY = 0.1
MAX_L2_NORM = 1.0


# ============================================================
# Distributed setup
# ============================================================

def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    return rank, local_rank, world_size


# ============================================================
# Main benchmark
# ============================================================

def main():

    rank, local_rank, world_size = setup_distributed()

    assert world_size == 2, (
        f"This benchmark expects 2 GPUs, got {world_size}"
    )

    device = torch.device(f"cuda:{local_rank}")

    # --------------------------------------------------------
    # Measure gradient communication without modifying DDP
    # --------------------------------------------------------

    # original_all_reduce = dist.all_reduce

    # communication_time = 0.0

    # def timed_all_reduce(*args, **kwargs):
    #     nonlocal communication_time

    #     # Make sure previous GPU work has completed.
    #     torch.cuda.synchronize()

    #     start = time.perf_counter()

    #     result = original_all_reduce(*args, **kwargs)

    #     # Your naive DDP uses synchronous all_reduce.
    #     torch.cuda.synchronize()

    #     communication_time += time.perf_counter() - start

    #     return result

    # dist.all_reduce = timed_all_reduce

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    torch.manual_seed(42)

    model = TransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
    ).to(device)

    # Explicitly FP32.
    model = model.float()

    ddp_model = DDP(model)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        ddp_model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Synthetic data
    # --------------------------------------------------------

    torch.manual_seed(1234 + rank)

    x = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, CONTEXT_LENGTH),
        device=device,
        dtype=torch.long,
    )

    y = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, CONTEXT_LENGTH),
        device=device,
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Training step
    # --------------------------------------------------------

    def step():

        optimizer.zero_grad()

        logits = ddp_model(
            x,
            ROPE_THETA,
        )

        loss = cross_entropy(
            logits=logits,
            targets=y,
        )

        loss.backward()

        # Your DDP implementation.
        ddp_model.finish_gradient_synchronization()

        gradient_clipping(
            ddp_model.parameters(),
            MAX_L2_NORM,
        )

        optimizer.step()

        return loss

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    if rank == 0:
        print("Running warmup...")

    for _ in range(WARMUP_STEPS):
        step()

    torch.cuda.synchronize()
    dist.barrier()

    # Reset communication measurement after warmup.
    communication_time = 0.0

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    step_times = []

    if rank == 0:
        print("\nStarting benchmark...\n")

    for i in range(BENCHMARK_STEPS):

        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()

        step()

        torch.cuda.synchronize()

        end = time.perf_counter()

        local_step_time = end - start

        # Use the slowest GPU's time
        step_tensor = torch.tensor(
            local_step_time,
            device=device,
            dtype=torch.float64,
        )

        dist.all_reduce(
            step_tensor,
            op=dist.ReduceOp.MAX,
        )

        step_time = step_tensor.item()
        step_times.append(step_time)

        if rank == 0:
            print(
                f"Step {i + 1:02d} | "
                f"total: {step_time:.4f}s"
            )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    if rank == 0:

        avg_step_time = sum(step_times) / len(step_times)

        print("\n" + "=" * 65)
        print("OVERLAP INDIVIDUAL PARAMETERS DDP BENCHMARK RESULTS")
        print("=" * 65)

        print(f"Model:                 XL")
        print(f"GPUs:                  2 × NVIDIA T4")
        print(f"Precision:             FP32")
        print(f"d_model:               {D_MODEL}")
        print(f"d_ff:                  {D_FF}")
        print(f"num_layers:            {NUM_LAYERS}")
        print(f"num_heads:             {NUM_HEADS}")
        print(f"batch size / GPU:      {BATCH_SIZE}")
        print(f"context length:        {CONTEXT_LENGTH}")
        print(f"vocab size:            {VOCAB_SIZE}")

        print("-" * 65)

        print(f"Average step time:     {avg_step_time:.4f} s")

        print("=" * 65)


if __name__ == "__main__":
    main()


#     Starting benchmark...

# Step 01 | total: 1.4648s | communication: 0.4319s | 29.49%
# Step 02 | total: 1.4663s | communication: 0.4324s | 29.49%
# Step 03 | total: 1.4717s | communication: 0.4326s | 29.40%
# Step 04 | total: 1.4696s | communication: 0.4298s | 29.24%
# Step 05 | total: 1.4764s | communication: 0.4346s | 29.43%
# Step 06 | total: 1.4786s | communication: 0.4325s | 29.25%
# Step 07 | total: 1.4785s | communication: 0.4320s | 29.22%
# Step 08 | total: 1.4815s | communication: 0.4401s | 29.71%
# Step 09 | total: 1.4805s | communication: 0.4309s | 29.10%
# Step 10 | total: 1.4872s | communication: 0.4364s | 29.35%
# Step 11 | total: 1.4872s | communication: 0.4352s | 29.26%
# Step 12 | total: 1.4920s | communication: 0.4361s | 29.23%
# Step 13 | total: 1.4942s | communication: 0.4331s | 28.98%
# Step 14 | total: 1.4869s | communication: 0.4326s | 29.09%
# Step 15 | total: 1.4912s | communication: 0.4306s | 28.87%
# Step 16 | total: 1.4960s | communication: 0.4318s | 28.86%
# Step 17 | total: 1.4974s | communication: 0.4366s | 29.16%
# Step 18 | total: 1.4992s | communication: 0.4319s | 28.81%
# Step 19 | total: 1.5010s | communication: 0.4321s | 28.79%
# Step 20 | total: 1.5064s | communication: 0.4312s | 28.62%

# =================================================================
# NAIVE DDP BENCHMARK RESULTS
# =================================================================
# Model:                 Medium
# Precision:             FP32
# GPUs:                  2 × NVIDIA T4
# Batch / GPU:           2
# Context length:        512
# d_model:               1024
# d_ff:                  4096
# num_layers:            24
# num_heads:             16
# -----------------------------------------------------------------
# Average step time:     1.4853 s
# Average communication: 0.4332 s
# Communication:         29.17%
# =================================================================

# Starting benchmark...

# Step 01 | total: 1.6804s | communication: 0.4448s | communication: 26.47%
# Step 02 | total: 1.6738s | communication: 0.4533s | communication: 27.08%
# Step 03 | total: 1.6842s | communication: 0.4532s | communication: 26.91%
# Step 04 | total: 1.6858s | communication: 0.4545s | communication: 26.96%
# Step 05 | total: 1.6937s | communication: 0.4556s | communication: 26.90%
# Step 06 | total: 1.6977s | communication: 0.4606s | communication: 27.13%
# Step 07 | total: 1.7024s | communication: 0.4635s | communication: 27.22%
# Step 08 | total: 1.7073s | communication: 0.4699s | communication: 27.52%
# Step 09 | total: 1.7177s | communication: 0.4758s | communication: 27.70%
# Step 10 | total: 1.7233s | communication: 0.4752s | communication: 27.57%
# Step 11 | total: 1.7287s | communication: 0.4781s | communication: 27.66%
# Step 12 | total: 1.7476s | communication: 0.4900s | communication: 28.04%
# Step 13 | total: 1.7385s | communication: 0.4873s | communication: 28.03%
# Step 14 | total: 1.7483s | communication: 0.4940s | communication: 28.26%
# Step 15 | total: 1.7545s | communication: 0.4977s | communication: 28.36%
# Step 16 | total: 1.7273s | communication: 0.4742s | communication: 27.45%
# Step 17 | total: 1.7612s | communication: 0.5030s | communication: 28.56%
# Step 18 | total: 1.7669s | communication: 0.5108s | communication: 28.91%
# Step 19 | total: 1.7238s | communication: 0.4657s | communication: 27.02%
# Step 20 | total: 1.7681s | communication: 0.5008s | communication: 28.32%

# =================================================================
# FLAT DDP BENCHMARK RESULTS
# =================================================================
# Model:                 Medium
# GPUs:                  2 × NVIDIA T4
# Precision:             FP32
# d_model:               1024
# d_ff:                  4096
# num_layers:            24
# num_heads:             16
# batch size / GPU:      2
# context length:        512
# vocab size:            10000
# -----------------------------------------------------------------
# Average step time:     1.7216 s
# Average communication: 0.4754 s
# Communication fraction:27.61%
# =================================================================