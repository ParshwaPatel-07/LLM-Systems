
"""
Colab-ready training script for CS336 Transformer.
Adjust the paths in the PATHS section before running.
"""

import os
import time
import numpy as np
import torch
import wandb
from tqdm.auto import tqdm
from torchinfo import summary

from model import (
    TransformerLM,
    AdamW,
    cross_entropy,
    data_loading,
    gradient_clipping,
    learning_rate_schedule,
)

# ==========================
# PATHS (EDIT FOR YOUR SETUP)
# ==========================
ROOT = "/content/drive/MyDrive/Transformer AS-1"

TRAIN_BIN = f"{ROOT}/data/train.bin"
VAL_BIN = f"{ROOT}/data/val.bin"
CHECKPOINT_DIR = f"{ROOT}/checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================
# DEVICE
# ==========================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

torch.set_float32_matmul_precision("high")

# ==========================
# HYPERPARAMETERS
# ==========================
VOCAB_SIZE = 16000
CONTEXT_LENGTH = 256

NUM_LAYERS = 6
D_MODEL = 384
NUM_HEADS = 6
D_FF = 1536

BATCH_SIZE = 64

MAX_ITERS = 100000

LR_MAX = 3e-4
LR_MIN = 3e-5

WARMUP_ITERS = 1000
COSINE_ITERS = 100000

EVAL_INTERVAL = 500
EVAL_BATCHES = 20

CHECKPOINT_INTERVAL = 5000

# ==========================
# DATA
# ==========================
train = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
val = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")

# ==========================
# MODEL
# ==========================
model = TransformerLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
).to(DEVICE)

dummy = torch.randint(
    0,
    VOCAB_SIZE,
    (1, CONTEXT_LENGTH),
    device=DEVICE,
)

print(summary(model, input_data=(dummy,), verbose=1))

optimizer = AdamW(
    model.parameters(),
    lr=LR_MAX,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)

wandb.login()
wandb.init(
    project="cs336-assignment",
    name="6L-384D-full-run",
    config={
        "batch_size": BATCH_SIZE,
        "lr_max": LR_MAX,
        "layers": NUM_LAYERS,
        "d_model": D_MODEL,
    },
)

best_val = float("inf")
start = time.time()

def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            x, y = data_loading(
                val,
                batch_size=BATCH_SIZE,
                context_len=CONTEXT_LENGTH,
                device=DEVICE,
            )
            logits = model(x)
            losses.append(cross_entropy(logits, y).item())
    model.train()
    return sum(losses) / len(losses)

for step in tqdm(range(MAX_ITERS), desc="Training"):

    lr = learning_rate_schedule(
        t=step,
        lr_max=LR_MAX,
        lr_min=LR_MIN,
        warmup_iters=WARMUP_ITERS,
        cosine_cycle_iters=COSINE_ITERS,
    )

    for g in optimizer.param_groups:
        g["lr"] = lr

    x, y = data_loading(
        train,
        batch_size=BATCH_SIZE,
        context_len=CONTEXT_LENGTH,
        device=DEVICE,
    )

    optimizer.zero_grad()

    logits = model(x)
    loss = cross_entropy(logits, y)

    loss.backward()
    gradient_clipping(model.parameters(), max_l2_norm=1.0)
    optimizer.step()

    if step % EVAL_INTERVAL == 0:
        val_loss = evaluate()

        elapsed = time.time() - start

        print(
            f"Step {step:6d} | "
            f"Train {loss.item():.4f} | "
            f"Val {val_loss:.4f} | "
            f"LR {lr:.2e} | "
            f"{elapsed/60:.1f} min"
        )

        wandb.log({
            "step": step,
            "train_loss": loss.item(),
            "val_loss": val_loss,
            "learning_rate": lr,
            "elapsed_minutes": elapsed / 60,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                },
                os.path.join(CHECKPOINT_DIR, "best.pt"),
            )

    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            },
            os.path.join(CHECKPOINT_DIR, f"checkpoint_{step}.pt"),
        )

torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": MAX_ITERS,
    },
    os.path.join(CHECKPOINT_DIR, "final.pt"),
)

wandb.finish()

print("Training complete.")
