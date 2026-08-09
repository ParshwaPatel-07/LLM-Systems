# Results

Figures live under `docs/assets/`. Numbers below are from logged runs (not estimates).

## Assignment 1 — Basics

### TinyStories

<!-- ![TinyStories train/val loss](assets/tinystories_loss.png) -->

| Run | Model | Data | Notes |
|-----|-------|------|-------|
| tiny-debug / ts-baseline | D=384, L=6, H=6 | TinyStories | Separate debug / baseline run — not part of the OWT progression |

### OpenWebText Training

Training was run across four resumed sessions (`owt_run1` → `owt_run4`) due to Colab/Kaggle
session time limits, using checkpointing to resume from the last saved state each time.
Total planned schedule: 100,000 steps (cosine LR cycle).

| Run | Step (cumulative) | Train Loss | Val Loss | Runtime |
|-----|-------------------|------------|----------|---------|
| owt_run1 | 10,100 | 3.616 | 3.524 | 3h 11m |
| owt_run2 | 17,000 | 3.539 | 3.438 | 2h 05m |
| owt_run3 | 26,600 | 3.318 | 3.344 | 2h 46m |
| owt_run4 | 37,100 | 3.361 | 3.349 | 3h 00m |

Currently at **37,100 / 100,000** planned steps (**~37%**). Latest checkpoint: train_loss **3.361**, val_loss **3.349**.

<!-- Drop a clean export at docs/assets/owt_loss_curve.png, then uncomment:
![OWT loss curve](assets/owt_loss_curve.png)
-->

### Ablations (optional)

| Change | Effect on val loss | Notes |
|--------|--------------------|-------|
| Pre-norm vs post-norm | | |
| RoPE vs no positions | | |
| SwiGLU vs SiLU FFN | | |

## Assignment 2 — Systems

### FlashAttention2 forward (causal, T4 GPU)

Benchmarked with `cs336_systems/flash_benchmarking.py` (PyTorch tiled FA2 vs Triton kernel).

| seq_len | dtype | d | PyTorch Fwd | Triton Fwd | Speedup |
|---------|-------|---|-------------|------------|---------|
| 128 | fp16 | 128 | 27.71 ms | 0.20 ms | ~139× |
| 512 | fp16 | 128 | 327.67 ms | 0.72 ms | ~455× |
| 1024 | fp16 | 128 | 1226.90 ms | 2.52 ms | ~487× |

Headline: at `seq_len=1024`, fp16, `d=128` — **~487×** Triton vs PyTorch forward.

### Other experiments

| Experiment | Metric | Result | Notes |
|------------|--------|--------|-------|
| Mixed precision | throughput / memory | — | |
| Model step benchmark | ms/step | — | `benchmarking.py` |
| Profiling | top kernels | — | |

### Plots

<!-- ![FlashAttention latency vs sequence length](assets/flash_attn_bench.png) -->
