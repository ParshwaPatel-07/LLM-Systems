# Results

Add figures under `docs/assets/` and link them here. Numbers below are placeholders — replace with your runs.

## Assignment 1 — Basics

### Training curves

<!-- ![TinyStories train/val loss](assets/tinystories_loss.png) -->

| Run | Model | Data | Steps | Peak LR | Val loss | Val PPL | Notes |
|-----|-------|------|-------|---------|----------|---------|-------|
| ts-baseline | D=384, L=6, H=6 | TinyStories | — | 3e-4 | — | — | |
| owt | — | OpenWebText sample | ~35% | — | 3.18 | — | In progress |

### Ablations (optional)

| Change | Effect on val loss | Notes |
|--------|--------------------|-------|
| Pre-norm vs post-norm | | |
| RoPE vs no positions | | |
| SwiGLU vs SiLU FFN | | |

## Assignment 2 — Systems

| Experiment | Metric | Result | Notes |
|------------|--------|--------|-------|
| Forward/backward benchmark | ms/step | — | `cs336_systems/benchmarking.py` |
| Mixed precision | throughput / memory | — | |
| Attention benchmark | ms vs seq len | — | |
| Profiling | top kernels | — | |

### Plots

<!-- ![Attention latency vs sequence length](assets/attention_bench.png) -->
