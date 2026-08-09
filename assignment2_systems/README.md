# Assignment 2 — Systems

Benchmarking, profiling, mixed precision, and **FlashAttention2** on top of the Assignment 1 Transformer.

**Status:** FlashAttention2 (PyTorch + Triton) implemented and benchmarked · other A2 units continuing

## Layout

- `cs336_systems/` — systems code
- `cs336-basics/` — local dependency: copy of A1 `cs336_basics`

## FlashAttention2

| File | Purpose |
|------|---------|
| `FA2_pytorch.py` | Tiled FlashAttention2 in PyTorch (forward + backward) |
| `FA2_triton.py` | Triton forward kernel |
| `flash_benchmarking.py` | Causal fwd timing sweep (PyTorch vs Triton) |

Headline (T4, causal, fp16, `d=128`, `seq_len=1024`): Triton **2.52 ms** vs PyTorch **1226.90 ms** (~**487×**).

## Other modules

| File | Purpose |
|------|---------|
| `benchmarking.py` | Model step timing |
| `benchmarking_mixed_pricision.py` | Mixed-precision benches |
| `mixed_precision_accumulation.py` | Accumulation experiments |
| `profiling.py` | Profiling helpers |
| `attention_benchmark.py` | Baseline attention timing |
| `autograd_experiment.py` | Autograd experiments |

## Setup

```bash
uv sync
# CUDA required for Triton / flash_benchmarking:
PYTHONPATH=. python cs336_systems/flash_benchmarking.py
```
