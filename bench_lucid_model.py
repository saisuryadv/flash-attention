"""Full model benchmark: LUCID attention vs standard attention in Llama models.

Measures training throughput (tokens/sec), inference throughput (tokens/sec),
and peak GPU memory across sequence lengths and model sizes.

Usage:
    python bench_lucid_model.py
"""

import sys
import types
import os

# Mock C extension modules before any flash_attn imports
for mod_name in ('flash_attn_2_cuda', 'flash_attn_3_cuda'):
    sys.modules[mod_name] = types.ModuleType(mod_name)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import math

from flash_attn.cute.forward_sub_interface import _forward_sub_fwd, _forward_sub_bwd


# ─── LUCID Autograd Function ───

class LUCIDForwardSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v):
        """k, v: (batch, seqlen, heads, head_dim) BSHD layout."""
        v_prime = _forward_sub_fwd(k, v, use_diag_solve=True)
        ctx.save_for_backward(k, v_prime)
        ctx.head_dim = k.shape[-1]
        return v_prime

    @staticmethod
    def backward(ctx, dv_prime):
        k, v_prime = ctx.saved_tensors
        dv, dk = _forward_sub_bwd(k, v_prime, dv_prime,
                                   block_size=ctx.head_dim,
                                   use_diag_solve=True,
                                   use_combined_kernel=True)
        return dk, dv


# ─── Minimal Llama-style Model (no transformers dependency) ───

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=131072):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen, device):
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)  # (1, S, D)


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(2)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, use_lucid=False, kv_heads=None, is_sliding=False, sliding_window=1024):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads or heads
        self.head_dim = head_dim
        self.use_lucid = use_lucid
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.rope = RotaryEmbedding(head_dim)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.kv_heads, self.head_dim)

        cos, sin = self.rope(S, x.device)
        q, k = apply_rope(q, k, cos, sin)

        q = q.transpose(1, 2)  # (B, H_q, S, D)
        k = k.transpose(1, 2)  # (B, H_kv, S, D)
        v = v.transpose(1, 2)

        if self.use_lucid and not self.is_sliding:
            # Global layer with LUCID: triangular solve on KV heads
            k_bshd = k.transpose(1, 2).contiguous()  # back to (B, S, H_kv, D)
            v_bshd = v.transpose(1, 2).contiguous()
            k_bshd_scaled = k_bshd * (self.head_dim ** -0.25)
            v_prime_bshd = LUCIDForwardSub.apply(k_bshd_scaled, v_bshd)
            k = k_bshd_scaled.transpose(1, 2)
            v = v_prime_bshd.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        elif self.is_sliding:
            # Sliding window attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            # Standard global causal attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)

        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, -1))


class FeedForward(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, intermediate, use_lucid=False, kv_heads=None,
                 is_sliding=False, sliding_window=1024):
        super().__init__()
        self.attn_norm = RMSNorm(hidden)
        self.attn = Attention(hidden, heads, head_dim, use_lucid, kv_heads, is_sliding, sliding_window)
        self.ff_norm = RMSNorm(hidden)
        self.ff = FeedForward(hidden, intermediate)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(self.ff_norm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, vocab, hidden, heads, head_dim, layers, intermediate, use_lucid=False,
                 kv_heads=None, layer_types=None, sliding_window=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        if layer_types is None:
            layer_types = ['full_attention'] * layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, head_dim, intermediate,
                           use_lucid=use_lucid, kv_heads=kv_heads,
                           is_sliding=(lt == 'sliding_attention'),
                           sliding_window=sliding_window)
            for lt in layer_types
        ])
        self.norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, use_checkpoint=False):
        x = self.embed(input_ids)
        if use_checkpoint and self.training:
            # Checkpoint every block — saves only block boundary activations
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False,
                )
        else:
            for block in self.blocks:
                x = block(x)
        return self.lm_head(self.norm(x))


# ─── Benchmarking ───

def bench_step(fn, warmup=3, iters=5):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# Gemma 3 layer pattern: every 6th layer is global (full_attention), rest sliding window
def gemma3_layer_types(num_layers):
    return ['full_attention' if (i + 1) % 6 == 0 else 'sliding_attention'
            for i in range(num_layers)]

# (name, hidden, q_heads, layers, kv_heads, intermediate, sliding_window)
MODEL_CONFIGS = [
    # Gemma 3-4B style adapted to head_dim=128
    # Original: hidden=2560, heads=8, kv_heads=4, head_dim=256, 34 layers, inter=10240
    # Adapted:  hidden=1024, heads=8, kv_heads=4, head_dim=128, 34 layers, inter=10240
    # 5 global layers (LUCID), 29 sliding window layers (standard SDPA)
    ("Gemma3-4B-style", 1024, 8, 34, 4, 10240, 1024),
]

def create_model(name, hidden, heads, layers, kv_heads, intermediate, sliding_window,
                 use_lucid=False, vocab=32000):
    head_dim = hidden // heads
    layer_types = gemma3_layer_types(layers)
    model = LlamaModel(vocab, hidden, heads, head_dim, layers, intermediate,
                       use_lucid=use_lucid, kv_heads=kv_heads,
                       layer_types=layer_types, sliding_window=sliding_window)
    return model.cuda().bfloat16()


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def main():
    device = "cuda"
    batch = 1

    gpu_name = torch.cuda.get_device_name()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"{'='*110}")
    print(f"LUCID Full Model Benchmark — Gemma 3-style architecture (global + sliding window)")
    print(f"{'='*110}")
    print(f"Hardware: {gpu_name}, {gpu_mem_gb:.0f} GB HBM")
    print(f"Software: PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Config:   batch={batch}, dtype=bf16, head_dim=128, use_diag_solve=True")
    print(f"Metrics:  Inference = forward-only throughput (tokens/sec)")
    print(f"          Training  = forward+backward throughput (tokens/sec)")
    print(f"          Peak Mem  = max GPU memory during training step (GB)")
    print()

    seqlens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    for model_name, hidden, heads, layers, kv_heads, intermediate, sliding_window in MODEL_CONFIGS:
        head_dim = hidden // heads
        gqa_ratio = heads // kv_heads
        lt = gemma3_layer_types(layers)
        n_global = sum(1 for t in lt if t == 'full_attention')
        n_sliding = layers - n_global
        params_est = count_params(create_model(model_name, hidden, heads, layers, kv_heads, intermediate, sliding_window))
        print(f"{'='*110}")
        print(f"Model: {model_name} ({params_est:.0f}M params)")
        print(f"  hidden={hidden}, q_heads={heads}, kv_heads={kv_heads}, GQA={gqa_ratio}:1, head_dim={head_dim}")
        print(f"  {layers} layers: {n_global} global (LUCID) + {n_sliding} sliding window (sw={sliding_window})")
        print(f"  FFN intermediate={intermediate}")
        print(f"{'='*110}")

        hdr = (f"{'Seqlen':>8} |"
               f" {'LUCID Infer':>14} {'LUCID Train':>14} {'Mem':>6} |"
               f" {'Softmax Infer':>14} {'Softmax Train':>14} {'Mem':>6} |"
               f" {'Infer x':>8} {'Train x':>8}")
        print(hdr)
        print("-" * len(hdr))

        for seqlen in seqlens:
            # Ensure seqlen divisible by block_size=128
            if seqlen % 128 != 0:
                continue

            results = {}
            for mode in ["standard", "lucid"]:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()

                try:
                    use_lucid = (mode == "lucid")
                    model = create_model(model_name, hidden, heads, layers, kv_heads,
                                        intermediate, sliding_window, use_lucid=use_lucid)
                    params_m = count_params(model)

                    tokens = torch.randint(0, 32000, (batch, seqlen), device=device)

                    # Inference (forward only)
                    model.eval()
                    with torch.no_grad():
                        infer_sec = bench_step(lambda: model(tokens))

                    # Training (forward + backward)
                    model.train()
                    torch.cuda.reset_peak_memory_stats()

                    labels = tokens[:, 1:].contiguous()

                    def train_step():
                        model.zero_grad(set_to_none=True)
                        logits = model(tokens, use_checkpoint=True)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.size(-1)),
                            labels.reshape(-1),
                        )
                        loss.backward()

                    train_sec = bench_step(train_step)
                    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

                    toks = batch * seqlen
                    results[mode] = {
                        'infer_tps': toks / infer_sec,
                        'train_tps': toks / train_sec,
                        'peak_gb': peak_mem_gb,
                        'params_m': params_m,
                    }

                    del model, tokens
                    torch.cuda.empty_cache()
                    gc.collect()

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "out of memory" in str(e).lower():
                        results[mode] = None
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        raise

            def fmt(tps):
                if tps >= 1e6: return f"{tps/1e6:.1f}M tok/s"
                if tps >= 1e3: return f"{tps/1e3:.0f}K tok/s"
                return f"{tps:.0f} tok/s"

            std = results.get("standard")
            luc = results.get("lucid")

            if luc and std:
                ir = (std['infer_tps'] / luc['infer_tps'])
                tr = (std['train_tps'] / luc['train_tps'])
                print(f"{seqlen:>8} |"
                      f" {fmt(luc['infer_tps']):>14} {fmt(luc['train_tps']):>14} {luc['peak_gb']:>5.1f}G |"
                      f" {fmt(std['infer_tps']):>14} {fmt(std['train_tps']):>14} {std['peak_gb']:>5.1f}G |"
                      f" {ir:>7.2f}x {tr:>7.2f}x")
            elif std:
                print(f"{seqlen:>8} |"
                      f" {'OOM':>14} {'OOM':>14} {'OOM':>6} |"
                      f" {fmt(std['infer_tps']):>14} {fmt(std['train_tps']):>14} {std['peak_gb']:>5.1f}G |"
                      f" {'---':>8} {'---':>8}")
            elif luc:
                print(f"{seqlen:>8} |"
                      f" {fmt(luc['infer_tps']):>14} {fmt(luc['train_tps']):>14} {luc['peak_gb']:>5.1f}G |"
                      f" {'OOM':>14} {'OOM':>14} {'OOM':>6} |"
                      f" {'---':>8} {'---':>8}")
            else:
                print(f"{seqlen:>8} | {'OOM':>14} {'OOM':>14} {'OOM':>6} | {'OOM':>14} {'OOM':>14} {'OOM':>6} | {'---':>8} {'---':>8}")

            sys.stdout.flush()

        print()


if __name__ == "__main__":
    main()
