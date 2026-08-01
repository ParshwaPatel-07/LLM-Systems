import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    # def forward(self, x):
    #     x = self.relu(self.fc1(x))
    #     x = self.ln(x)
    #     x = self.fc2(x)
    #     return x
    def forward(self, x):
        x = self.fc1(x)
        print("fc1:", x.dtype)
        x = self.relu(x)
        x = self.ln(x)
        print("ln:", x.dtype)
        x = self.fc2(x)
        print("fc2:", x.dtype)

        return x

"""Suppose we are training the model on a GPU and that the model parameters are originally in
FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
• the model parameters within the autocast context?
• the output of the first feed-forward layer (ToyModel.fc1)?
• the output of layer norm (ToyModel.ln)?
• the model’s predicted logits?
• the loss?
• the model’s gradients?"""

# answer:
model = ToyModel(5, 2).to(device='mps')
x = torch.randn(1, 5).to(device='mps')
with torch.autocast(device_type="mps", dtype=torch.float16):
    y = model(x)

"""| Component                   | Data type       |
| --------------------------- | --------------- |
| Model parameters            | `torch.float32` |
| Output of `fc1`             | `torch.float16` |
| Output of `LayerNorm`       | `torch.float32` |
| Model logits (`fc2` output) | `torch.float16` |
| Loss                        | `torch.float32` |
| Gradients                   | `torch.float32` |
"""
