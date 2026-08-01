# Assignment 1 — Basics

Byte-Pair Encoding tokenizer, decoder-only Transformer LM (pre-norm, RMSNorm, RoPE, SwiGLU), AdamW training loop, checkpointing, and temperature / top-p decoding.

Official handout: [`cs336_assignment1_basics.pdf`](./cs336_assignment1_basics.pdf)

## Layout

- `cs336_basics/` — implementation (tokenizer, model, training, decoding)
- `tests/` — course unit tests via `tests/adapters.py`
- `pyproject.toml` — `uv` project

## Setup

```bash
# Install uv if needed: https://docs.astral.sh/uv/
uv sync
uv run pytest
```

### Data (not in git)

```bash
mkdir -p data && cd data
# TinyStories + OWT sample — see course handout / original assignment README for URLs
```

## Main entry points

| Piece | Module |
|-------|--------|
| BPE train / encode | `cs336_basics/tokenizer.py` |
| Transformer LM | `cs336_basics/model.py` |
| Training loop | `cs336_basics.model.training_together` |
| Decoding | `cs336_basics.model.decoding` |
