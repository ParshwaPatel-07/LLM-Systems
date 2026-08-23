import torch
import torch.nn as nn
import torch.distributed as dist
import time

class DDP(nn.Module):

    def __init__(self, module:nn.Module):
        super().__init__()
        self.module = module
        self.communication_time = 0.0
        self.hook_handles = []
        self.work_handles = []
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(self.make_hook())
                self.hook_handles.append(handle)

    def make_hook(self):
        def hook(param):
            start_time = time.perf_counter()
            grad = param.grad
            work_hoook = dist.all_reduce(grad, async_op=True)
            self.work_handles.append(work_hoook)
            self.communication_time += time.perf_counter() - start_time
        return hook       

    def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for handle in self.work_handles:
                handle.wait()
        self.work_handles.clear()

        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.div_(world_size)

