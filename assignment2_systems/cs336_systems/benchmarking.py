import argparse
import timeit
from dataclasses import dataclass
import torch
from cs336_basics.model import TransformerLM, AdamW, cross_entropy
import numpy as np
from einops import rearrange
from contextlib import nullcontext

"""Modify your benchmarking script to optionally run the model using mixed precision with BF16."""

@dataclass
class ModelConfiguration:

    d_model:int
    d_ff:int
    num_layers:int
    num_heads:int
    context_len:int
    vocab_size:int 
    beta1:float = 0.9
    beta2:float = 0.95
    eps:float = 1e-8
    weight_decay:float = 0.1
    learning_rate:float = 1e-4


    TIERS = {
        "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
        # "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
        # "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
        # "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
        # "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
    }
    @classmethod
    def from_tier(cls, tier_name: str, vocab_size: int, context_len: int) -> "ModelConfiguration":
        if tier_name not in cls.TIERS:
            raise ValueError(f"Unknown model tier: {tier_name}. Choose from {list(cls.TIERS.keys())}")
        tier_data = cls.TIERS[tier_name].copy()
        tier_data["vocab_size"] = vocab_size
        tier_data["context_len"] = context_len
        return cls(**tier_data)



def create_model(config : ModelConfiguration):

    model = TransformerLM(
        vocab_size=config.vocab_size, 
        context_length=config.context_len, 
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_heads=config.num_heads,
        num_layers=config.num_layers
    )
    return model

def create_random_batch(config:ModelConfiguration, batch_size, device:str):

    random_tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.context_len),
        device=device,
        dtype=torch.long
    )
    batch = random_tokens

    inputs  = batch[:, :-1]
    targets = batch[:, 1:]

    return inputs, targets

def benchmark(model, inputs,targets,mode, num_warmup: int, num_steps: int, optimizer=None, use_mixed_precision: bool = False, memory_profile=False):
    device_type = inputs.device.type
    autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if use_mixed_precision else nullcontext()
    for _ in range(num_warmup):
        # Inside main():
        with autocast:
            outputs = model(inputs)
            flat_logits = rearrange(outputs, 'b s v -> (b s) v')
            flat_targets = rearrange(targets, 'b s -> (b s)')
            loss = cross_entropy(flat_logits, targets=flat_targets)

        if mode == "fwd":
            if torch.cuda.is_available(): 
                torch.cuda.synchronize()
            continue
            
        loss.backward()
        if mode == "fwd_bwd":
            if optimizer: optimizer.zero_grad()
            if torch.cuda.is_available(): 
                torch.cuda.synchronize()
            continue
            
        if mode == "fwd_bwd_opt" and optimizer:
            optimizer.step()
            optimizer.zero_grad()
            if torch.cuda.is_available(): 
                torch.cuda.synchronize()

    step_timings = []

    if memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(num_steps):

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_tick = timeit.default_timer()

        with autocast:
            outputs = model(inputs)
            flat_logits = rearrange(outputs, 'b s v -> (b s) v')
            flat_targets = rearrange(targets, 'b s -> (b s)')
            loss = cross_entropy(flat_logits, targets=flat_targets)

        if mode != "fwd":
            loss.backward()
            if mode == "fwd_bwd":
                if optimizer: optimizer.zero_grad()
            elif mode == "fwd_bwd_opt" and optimizer:
                optimizer.step()
                optimizer.zero_grad()

        if torch.cuda.is_available(): 
            torch.cuda.synchronize()
        end_tick = timeit.default_timer()

        step_timings.append(end_tick - start_tick)

    if memory_profile:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enable=None)

    return step_timings

def main():

        parser = argparse.ArgumentParser(description="Benchmark Transformer models.")

        parser.add_argument("--model", type=str, choices=list(ModelConfiguration.TIERS.keys()), required=False, default="small")
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--vocab_size", type=int, default=10000)
        parser.add_argument("--context_len", type=int, default=512)
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
        parser.add_argument("-w", "--warmup", type=int, default=5)
        parser.add_argument("-n", "--steps", type=int, default=10)
        parser.add_argument("--mode", type=str, default="fwd", choices=["fwd", "fwd_bwd", "fwd_bwd_opt"])
        parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision (BF16) for benchmarking.")
        parser.add_argument("--memory_profile", action="store_true", help="Enable CUDA memory profiling")
        args = parser.parse_args()

        config = ModelConfiguration.from_tier(
                tier_name=args.model,
                vocab_size=args.vocab_size,
                context_len=args.context_len
            )
        inputs, targets = create_random_batch(config, args.batch_size, args.device)

        model = create_model(config).to(args.device)
        model.to(device=args.device)

        optimizer = AdamW(
            model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1,config.beta2),
                eps=config.eps,
                weight_decay=config.weight_decay
        )
        print(f"🚀 Benchmarking {args.model} ({args.mode}) on {args.device}...")
        raw_times = benchmark(
            model=model, 
            inputs=inputs,
            targets=targets, 
            mode=args.mode, 
            num_warmup=args.warmup, 
            num_steps=args.steps, 
            optimizer=optimizer,
            use_mixed_precision=args.mixed_precision,
            memory_profile=args.memory_profile
        )

        mean_latency = np.mean(raw_times)
        std_latency = np.std(raw_times)
        throughput = args.batch_size / mean_latency 

        print("\n📊 Benchmark Results:")
        print(f"  • Model Size:        {args.model.upper()}")
        print(f"  • Mean Step Time:    {mean_latency:.5f} seconds")
        print(f"  • Std Deviation:     {std_latency:.5f} seconds")
        print(f"  • Throughput:        {throughput:.2f} sequences/sec")
        # model_sizes = list(ModelConfiguration.TIERS.keys())
        # precisions = [False, True]   # FP32, BF16

        # results = []

        # for model_name in model_sizes:
        #     config = ModelConfiguration.from_tier(
        #         tier_name=model_name,
        #         vocab_size=args.vocab_size,
        #         context_len=args.context_len,
        #     )

        #     model = create_model(config).to(args.device)

        #     optimizer = AdamW(
        #         model.parameters(),
        #         lr=config.learning_rate,
        #         betas=(config.beta1, config.beta2),
        #         eps=config.eps,
        #         weight_decay=config.weight_decay,
        #     )

        #     inputs, targets = create_random_batch(
        #         config,
        #         args.batch_size,
        #         args.device,
        #     )

        #     for mixed in precisions:
        #         times = benchmark(
        #             model=model,
        #             inputs=inputs,
        #             targets=targets,
        #             mode=args.mode,
        #             num_warmup=args.warmup,
        #             num_steps=args.steps,
        #             optimizer=optimizer,
        #             use_mixed_precision=mixed,
        #         )

        #         results.append({
        #             "Model": model_name,
        #             "Precision": "BF16" if mixed else "FP32",
        #             "Mean (ms)": np.mean(times) * 1000,
        #             "Std (ms)": np.std(times) * 1000,
        #             "Throughput": args.batch_size / np.mean(times),
        #         })
        #         print(f"✓ {model_name} {'BF16' if mixed else 'FP32'} completed")
        # print(f"Running {model_name} - {'BF16' if mixed else 'FP32'}...", flush=True)
        # print("\nBenchmark Results")
        # print("-" * 70)
        # print(f"{'Model':<8}{'Precision':<10}{'Mean(ms)':<12}{'Std(ms)':<12}{'Seq/s':<12}")

        # for r in results:
        #     print(
        #         f"{r['Model']:<8}"
        #         f"{r['Precision']:<10}"
        #         f"{r['Mean (ms)']:<12.2f}"
        #         f"{r['Std (ms)']:<12.2f}"
        #         f"{r['Throughput']:<12.2f}"
        #   )
if __name__ == "__main__":
    main()


