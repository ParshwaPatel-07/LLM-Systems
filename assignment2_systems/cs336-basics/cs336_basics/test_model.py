import time
import numpy as np
import torch
import wandb
import os
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
# Hyperparameters
# ==========================

DEVICE = "mps"

VOCAB_SIZE = 16000
CONTEXT_LENGTH = 256

NUM_LAYERS = 6
D_MODEL = 384
NUM_HEADS = 6
D_FF = 1536

BATCH_SIZE = 16

MAX_ITERS = 500

LR_MAX = 3e-4
LR_MIN = 3e-5

WARMUP_ITERS = 50
COSINE_ITERS = 500

# ==========================
# Load Dataset
# ==========================

train = np.memmap(
    "train.bin",
    dtype=np.uint16,
    mode="r",
)

val = np.memmap(
    "val.bin",
    dtype=np.uint16,
    mode="r",
)

# ==========================
# Model
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

summary(
    model,
    input_data=(dummy,),
)

optimizer = AdamW(
    model.parameters(),
    lr=LR_MAX,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)
os.environ["WANDB_API_KEY"] = "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"
wandb.login()
wandb.init(
    project="cs336-assignment",
    name="local-smoke-test",
)

start = time.time()

# ==========================
# Training
# ==========================

for step in range(MAX_ITERS):

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

    gradient_clipping(
        model.parameters(),
        max_l2_norm=1.0,
    )

    optimizer.step()

    # Validation every 25 steps
    if step % 25 == 0:

        model.eval()

        with torch.no_grad():

            x_val, y_val = data_loading(
                val,
                batch_size=BATCH_SIZE,
                context_len=CONTEXT_LENGTH,
                device=DEVICE,
            )

            logits_val = model(x_val)

            val_loss = cross_entropy(
                logits_val,
                y_val,
            )

        model.train()

        elapsed = time.time() - start

        print(
            f"Step {step:4d} | "
            f"Train {loss.item():.4f} | "
            f"Val {val_loss.item():.4f} | "
            f"LR {lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        wandb.log({
            "step": step,
            "train_loss": loss.item(),
            "val_loss": val_loss.item(),
            "learning_rate": lr,
            "elapsed_time": elapsed,
        })

wandb.finish()

print("\nTraining completed successfully!")