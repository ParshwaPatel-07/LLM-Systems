# CS336: Language Models from Scratch

Personal solutions and experiments for [Stanford CS336](https://cs336.stanford.edu/) — building language models end-to-end: tokenization, Transformer architecture, training, and systems optimizations.

> **Status:** Assignment 1 core implementation complete · OpenWebText training in progress (~37%) · Assignment 2: FlashAttention2 + systems benchmarking done (more A2 units ahead)

<!-- Optional: add LinkedIn post link after publishing
**LinkedIn:** [post title](https://www.linkedin.com/...)
-->

---

## What’s in this repo

| Folder | Focus | Status |
|--------|--------|--------|
| [`assignment1_basics/`](./assignment1_basics/) | BPE tokenizer, modern Transformer LM, AdamW training loop, decoding | Core done · OWT ~37% |
| [`assignment2_systems/`](./assignment2_systems/) | Profiling, mixed precision, **FlashAttention2** (PyTorch + Triton), benches | FlashAttn done · continuing |
| [`common/`](./common/) | Notes on shared code between assignments | — |
| [`docs/results.md`](./docs/results.md) | Loss curves, ablations, benchmark numbers | Updated |

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

**Key references:** Vaswani et al. 2017 (Transformer) · Sennrich et al. 2016 (BPE) · Xiong et al. 2020 (Pre-LN Transformer placement) · Su et al. 2021 (RoPE) · Shazeer 2020 (SwiGLU) · Zhang & Sennrich 2019 (RMSNorm) · Loshchilov & Hutter 2017 (AdamW) · Dao et al. 2022/2023 (FlashAttention / FlashAttention-2)

```
tokens [B, S]
  → Embedding [B, S, D]
  → N × (RMSNorm → Causal MHA+RoPE → + → RMSNorm → SwiGLU → +)
  → Final RMSNorm
  → LM head → logits [B, S, V]
```

---

## Results

### Assignment 1 — OpenWebText

| Run | Dataset | Notes | Val loss / PPL | Status |
|-----|---------|-------|----------------|--------|
| A1 baseline | TinyStories | Transformer LM + BPE | _add number_ | Done |
| A1 OWT | OpenWebText | 4 resumed runs (checkpointed across sessions) | val_loss 3.349 @ 37.1% steps (37.1k/100k) | In progress |

### Assignment 2 — FlashAttention2 forward (T4 GPU, causal)

PyTorch tiled FlashAttention2 vs custom **Triton** kernel. Headline at `seq_len=1024`, **fp16**, `d=128`:

| Impl | Time | Speedup vs PyTorch |
|------|------|--------------------|
| Triton | **2.52 ms** | **~487×** |
| PyTorch | 1226.90 ms | 1× |

Scaling (fp16, `d=128`):

| seq_len | PyTorch | Triton | Speedup |
|---------|---------|--------|---------|
| 128 | 27.71 ms | 0.20 ms | ~139× |
| 512 | 327.67 ms | 0.72 ms | ~455× |
| 1024 | 1226.90 ms | 2.52 ms | ~487× |

Code: [`assignment2_systems/cs336_systems/FA2_pytorch.py`](./assignment2_systems/cs336_systems/FA2_pytorch.py), [`FA2_triton.py`](./assignment2_systems/cs336_systems/FA2_triton.py), bench script [`flash_benchmarking.py`](./assignment2_systems/cs336_systems/flash_benchmarking.py).

Full tables: [`docs/results.md`](./docs/results.md)

---

## Repository layout

```text
cs336-llm-from-scratch/
├── README.md
├── assignment1_basics/      # tokenizer, model, train, decode
├── assignment2_systems/     # systems + FlashAttention2
├── common/
└── docs/
    ├── results.md
    └── assets/
```

Large artifacts (`*.pt`, `*.bin`, datasets, `wandb/`) are **not** committed — see `.gitignore`.

---

## Quick start

```bash
# Assignment 1
cd assignment1_basics
uv sync

# Assignment 2
cd ../assignment2_systems
uv sync
# FlashAttention bench (needs CUDA):
# PYTHONPATH=. python cs336_systems/flash_benchmarking.py
```

---

## Progress log

| When | Update |
|------|--------|
| — | A1: BPE + Transformer LM + training/decoding (core complete) |
| — | A1: OpenWebText training in progress (37.1k/100k, val_loss 3.349) |
| — | A2: benchmarking, profiling, mixed precision |
| — | A2: FlashAttention2 (PyTorch + Triton) implemented & benchmarked — up to ~487× fwd speedup @ 1024/fp16/d=128 |
| _next_ | Finish OWT run · continue remaining A2 systems units |

---

## Disclaimer

This repository is for portfolio / learning. If you are a current CS336 student, do your own work — the value of the course is in implementing these pieces yourself.
