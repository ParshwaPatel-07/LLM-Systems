# Assignment 2 — Systems

Benchmarking, profiling, mixed precision, and attention / systems experiments on top of the Assignment 1 Transformer.

Official handout: [`cs336_assignment2_systems.pdf`](./cs336_assignment2_systems.pdf)

**Status:** in progress — more pieces will land here as I finish them.

## Layout

- `cs336_systems/` — systems code (benchmarks, profiling, mixed precision, …)
- `cs336-basics/` — local dependency: copy of my A1 `cs336_basics` package
- `tests/` — course unit tests

## Setup

```bash
uv sync
uv run pytest
```

## Current modules

| File | Purpose |
|------|---------|
| `benchmarking.py` | Step / forward timing |
| `benchmarking_mixed_pricision.py` | Mixed-precision benches |
| `mixed_precision_accumulation.py` | Accumulation experiments |
| `profiling.py` | Profiling helpers |
| `attention_benchmark.py` | Attention timing |
| `autograd_experiment.py` | Autograd experiments |
