import torch

from model import TransformerLM
from tokenizer import BPETokenizer

# ==========================================================
# Configuration
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE = 16000
CONTEXT_LENGTH = 256
NUM_LAYERS = 6
D_MODEL = 384
NUM_HEADS = 6
D_FF = 1536

CHECKPOINT = "checkpoint_37000.pt"

VOCAB = "vocab.pkl"
MERGES = "merges.pkl"

TEMPERATURE = 0.5
TOP_K = 40
MAX_NEW_TOKENS = 300

PROMPT = "that day"

# ==========================================================
# Load tokenizer
# ==========================================================

tokenizer = BPETokenizer.from_files(
    vocab_path=VOCAB,
    merge_path=MERGES,
    special_tokens=["<|endoftext|>"],
)

# ==========================================================
# Load model
# ==========================================================

model = TransformerLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF,
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)

model.load_state_dict(checkpoint["model"])
model.eval()

print("✓ Model loaded")

# ==========================================================
# Encode prompt
# ==========================================================

tokens = tokenizer.encode(PROMPT)

tokens = torch.tensor(
    tokens,
    dtype=torch.long,
    device=DEVICE,
).unsqueeze(0)

# ==========================================================
# Generate
# ==========================================================

with torch.no_grad():

    for _ in range(MAX_NEW_TOKENS):

        # Keep only last context_length tokens
        tokens = tokens[:, -CONTEXT_LENGTH:]

        logits = model(tokens)

        logits = logits[:, -1, :]

        # Temperature
        logits = logits / TEMPERATURE

        # Top-k sampling
        values, indices = torch.topk(logits, TOP_K)

        probs = torch.softmax(values, dim=-1)

        sample = torch.multinomial(probs, 1)

        next_token = indices.gather(-1, sample)

        tokens = torch.cat([tokens, next_token], dim=1)
        if tokenizer.decode(next_token[0].tolist()) == "<|endoftext|>":
            break

# ==========================================================
# Decode
# ==========================================================

generated = tokenizer.decode(tokens[0].tolist())

print("\n" + "=" * 80)
print(generated)
print("=" * 80)