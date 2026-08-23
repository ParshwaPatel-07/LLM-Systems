import torch
import torch.nn as nn
import torch.distributed as dist
from cs336_basics.model import Linear, Embedding


class _AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shard, world_size):
        ctx.world_size = world_size
        ctx.rank = dist.get_rank()

        gathered = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(gathered, shard)
        gathered[dist.get_rank()] = shard.clone()
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        grad_output /= ctx.world_size
        chunks = torch.chunk(grad_output, ctx.world_size, dim=0)
        grad_shard = chunks[dist.get_rank()].clone()
        return grad_shard, None
    
class ShardedLinear(nn.Module):

    def __init__(self, module:nn.Module, fsdp):
        super().__init__()
        self.module = module
        object.__setattr__(self, "fsdp", fsdp)

    def forward(self, x):
        weight = self.fsdp._all_gather_weights(self.module)
        return x @ weight.T


class ShardedEmbedding(nn.Module):
    def __init__(self, module:nn.Module, fsdp):
        super().__init__()
        self.module = module
        object.__setattr__(self, "fsdp", fsdp)

    def forward(self, x):
        weight = self.fsdp._all_gather_weights(self.module)
        return weight[x]


class FSDP(nn.Module):

    def __init__(self, module:nn.Module, compute_dtype:torch.dtype | None = None):
        super().__init__()

        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.shard_module = []
        self.work_handles = []

        for module in self.module.modules():
            if isinstance(module, (Linear, Embedding)):
                self.shard_module.append(module)

        for module in self.shard_module:
            self._shard_parameter(module)

        self._replace_modules(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _shard_parameter(self, module):
        if isinstance(module, Linear):
            param = module.W
        else:
            param = module.embedding

        full_weight = param.data

        chunks = torch.chunk(
            full_weight,
            chunks=self.world_size,
            dim=0,
        )

        sharded_weight = chunks[self.rank].clone()

        if isinstance(module, Linear):
            module.W = nn.Parameter(
                sharded_weight,
                requires_grad=param.requires_grad,
            )
        else:
            module.embedding = nn.Parameter(
                sharded_weight,
                requires_grad=param.requires_grad,
            )

    def _all_gather_weights(self, module, cast_to_compute_dtype=True):

        if isinstance(module, Linear):
            shard = module.W
        else:
            shard = module.embedding

        full_weight = _AllGather.apply(shard, self.world_size)

        if cast_to_compute_dtype and self.compute_dtype is not None:
            full_weight = full_weight.to(self.compute_dtype)


        return full_weight

    def _forward_linear(self, module:nn.Module, inputs):

        full_weight = self._all_gather_weights(module)
        output = inputs @ full_weight.T
        return output

    def _replace_modules(self, parent=None):
        if parent is None:
            parent = self.module

        for name, child in parent.named_children():
            if isinstance(child, Linear):
                parent._modules[name] = ShardedLinear(child, self)

            elif isinstance(child, Embedding):
                parent._modules[name] = ShardedEmbedding(child, self)

            else:
                self._replace_modules(child)

    def _reduce_scatter_gradient(self, module:nn.Module):

        if isinstance(module, Linear):
            grad = module.W.grad
        else:
            grad = module.embedding.grad

        chunks = list(torch.chunk(grad, chunks=self.world_size, dim=0))
        output = torch.empty_like(chunks[self.rank])
        dist.reduce_scatter(output, chunks, op=dist.ReduceOp.SUM)
        return output

    def finish_gradient_synchronization(self):

        sharded_originals = set(id(m) for m in self.shard_module)
        for module in self.module.modules():
            if isinstance(module, (ShardedLinear, ShardedEmbedding, Linear, Embedding)) or id(module) in sharded_originals:
                continue

            for param in module.parameters(recurse=False):
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= self.world_size

        for handle in self.work_handles:
            handle.wait()

        self.work_handles.clear()

    def gather_full_params(self):
        full_params = {}
        sharded_originals = set(id(m) for m in self.shard_module)

        for name, module in self.module.named_modules():
            if isinstance(module, ShardedLinear):
                full_params[f"{name}.W"] = self._all_gather_weights(module.module, cast_to_compute_dtype=False)
            elif isinstance(module, ShardedEmbedding):
                full_params[f"{name}.embedding"] = self._all_gather_weights(module.module, cast_to_compute_dtype=False)
            elif id(module) in sharded_originals:
                continue  # the wrapped original Linear/Embedding — already handled above
            else:
                for param_name, param in module.named_parameters(recurse=False):
                    full_params[f"{name}.{param_name}" if name else param_name] = param.data

        return full_params
        