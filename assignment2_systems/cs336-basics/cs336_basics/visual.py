import torch
from cs336_basics.model import TransformerLM

VOCAB_SIZE = 16000
CONTEXT_LENGTH = 256
NUM_LAYERS = 6
D_MODEL = 384
NUM_HEADS = 6
D_FF = 1536

# 1. Initialize your model architecture
model = TransformerLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    d_ff=D_FF
)

# 2. Load your saved state dict weights
# Using weights_only=True is recommended for safety when loading files
checkpoint = torch.load("checkpoint_11000.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()

# 3. Create mock token input (Batch Size = 1, Sequence Length = 50 tokens)
dummy_input = torch.randint(0, VOCAB_SIZE, (1, 50), dtype=torch.long)

# 4. Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx", 
    export_params=True,
    opset_version=14, # Modern opset version ensures better transformer support
    input_names=['input_ids'],
    output_names=['logits']
)
print("Export complete! Upload 'model.onnx' to Netron.app now.")
