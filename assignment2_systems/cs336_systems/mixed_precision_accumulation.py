import torch
import time
# all this in device=mps
s = torch.tensor(0,dtype=torch.float32).to(device='mps')
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32).to(device='mps')
print(s)
start = time.time()
s = torch.tensor(0,dtype=torch.float16).to(device='mps')
for i in range(100):
    s += torch.tensor(0.1,dtype=torch.float16).to(device='mps')
print(s)
end = time.time()
print("Time taken for 100 iterations of float16 addition:", end - start)
s = torch.tensor(0,dtype=torch.float32).to(device='mps')
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16).to(device='mps')
print(s)
s = torch.tensor(0,dtype=torch.float32).to(device='mps')
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16).to(device='mps')
    s += x.type(torch.float32)
print(s)
"""Traceback (most recent call last):
  File "/Users/parshwapatel/CS336/Assignments/AS-2/assignment2-systems-main/cs336_systems/mixed_precision_accumulation.py", line 6, in <module>
    s += torch.tensor(0.01,dtype=torch.float32).to(device='mps')
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!"""