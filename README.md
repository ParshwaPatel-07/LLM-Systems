# CS336: Language Models from Scratch

Personal solutions and experiments for [Stanford CS336](https://cs336.stanford.edu/) — building language models end-to-end: tokenization, Transformer architecture, training, and systems optimizations.

> **Status:** Assignment 1 core implementation complete · OpenWebText training in progress (~35%) · Assignment 2 in progress

<!-- Optional: add LinkedIn post link after publishing
**LinkedIn:** [post title](https://www.linkedin.com/...)
-->

---

## What’s in this repo

| Folder | Focus | Status |
|--------|--------|--------|
| [`assignment1_basics/`](./assignment1_basics/) | BPE tokenizer, modern Transformer LM, AdamW training loop, decoding | Core done · OWT ~35% |
| [`assignment2_systems/`](./assignment2_systems/) | Benchmarking, profiling, mixed precision, attention systems work | In progress |
| [`common/`](./common/) | Notes on shared code between assignments | — |
| [`docs/results.md`](./docs/results.md) | Loss curves, ablations, benchmark numbers | Fill as you run |

Course scaffolding / tests retain Stanford’s MIT license; implementation code in `cs336_basics/` and `cs336_systems/` is my work for the assignments.

---

## Architecture (Assignment 1)

Decoder-only Transformer LM, aligned with current open-model defaults:

| Component | Choice |
|-----------|--------|
| Block layout | Pre-norm residual stream |
| Normalization | RMSNorm |
| Position | RoPE on Q/K |
| FFN | SwiGLU |
| Attention | Causal multi-head self-attention |
| Optimizer | AdamW + decoupled weight decay |
| LR schedule | Linear warmup → cosine decay |
| Tokenizer | Byte-level BPE (GPT-2-style pretokenization) |

**Key references:** Vaswani et al. 2017 (Transformer) · Sennrich et al. 2016 (BPE) · Xiong et al. 2020 (Pre-LN Transformer placement) · Su et al. 2021 (RoPE) · Shazeer 2020 (SwiGLU) · Zhang & Sennrich 2019 (RMSNorm) · Loshchilov & Hutter 2017 (AdamW)

```
tokens [B, S]
  → Embedding [B, S, D]
  → N × (RMSNorm → Causal MHA+RoPE → + → RMSNorm → SwiGLU → +)
  → Final RMSNorm
  → LM head → logits [B, S, V]
```

---

## Results

> Drop a loss-curve image into `docs/assets/` and uncomment the line below.

<!-- ![TinyStories training curve](docs/assets/tinystories_loss.png) -->

| Run | Dataset | Notes | Val loss / PPL | Status |
|-----|---------|-------|----------------|--------|
| A1 baseline | TinyStories | Transformer LM + BPE | _add number_ | Done |
| A1 OWT | OpenWebText sample | Leaderboard-style run | loss 3.18 @ ~35% steps | In progress |
| A2 systems | — | Profiling / mixed precision / attention benches | _add_ | In progress |

Full tables and figures: [`docs/results.md`](./docs/results.md)

---

## Repository layout

```text
cs336-llm-from-scratch/
├── README.md
├── assignment1_basics/      # tokenizer, model, train, decode + tests
├── assignment2_systems/     # systems experiments (uses A1 code as cs336-basics)
├── common/                  # shared-code notes
└── docs/
    ├── results.md
    └── assets/              # loss curves, benchmark plots
```

Large artifacts (`*.pt`, `*.bin`, datasets, `wandb/`) are **not** committed — see `.gitignore`.

---

## Quick start

Each assignment is a self-contained `uv` project:

```bash
# Assignment 1
cd assignment1_basics
uv sync
uv run pytest

# Assignment 2
cd ../assignment2_systems
uv sync
uv run pytest
```

Datasets: follow the download steps in each assignment `README.md` / handout PDF.

---

## Progress log

| When | Update |
|------|--------|
| — | A1: BPE + Transformer LM + training/decoding (core complete) |
| — | A1: OpenWebText training in progress (~35%, loss 3.18) |
| — | A2: started — benchmarking, profiling, mixed precision |
| _next_ | Finish OWT run · continue A2 systems |

---

## Disclaimer

This repository is for portfolio / learning. If you are a current CS336 student, do your own work — the value of the course is in implementing these pieces yourself.
