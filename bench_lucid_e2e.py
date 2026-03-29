"""End-to-end LUCID attention layer benchmark: throughput and peak memory.

Benchmarks a single LUCID attention layer (forward + backward) across
sequence lengths and model sizes, comparing against standard FlashAttention.

Usage:
    python bench_lucid_e2e.py
"""

import sys
import types
import torch
import time
import gc

# Mock C extension modules
for mod_name in ('flash_attn_2_cuda', 'flash_attn_3_cuda'):
    sys.modules[mod_name] = types.ModuleType(mod_name)

from flash_attn.cute.forward_sub_interface import (
    _forward_sub_fwd, _forward_sub_bwd, _compute_diag_inv, _compute_diag_inv_transpose,
)
from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd


class LUCIDForwardSub(torch.autograd.Function):
    """Autograd wrapper for LUCID forward substitution fwd/bwd."""

    @staticmethod
    def forward(ctx, k, v, block_size=128):
        v_prime = _forward_sub_fwd(k, v, use_diag_solve=True)
        ctx.save_for_backward(k, v_prime)
        ctx.block_size = block_size
        return v_prime

    @staticmethod
    def backward(ctx, dv_prime):
        k, v_prime = ctx.saved_tensors
        dv, dk = _forward_sub_bwd(
            k, v_prime, dv_prime,
            block_size=ctx.block_size,
            use_diag_solve=True,
            use_combined_kernel=True,
        )
        return dk, dv, None


class FA3Attention(torch.autograd.Function):
    """Autograd wrapper for standard FA3 causal attention."""

    @staticmethod
    def forward(ctx, q, k, v):
        out, lse = _flash_attn_fwd(q, k, v, causal=True, return_lse=True)
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        _flash_attn_bwd(q, k, v, out, dout, lse, causal=True)
        # FA3 bwd writes grads in-place; return zeros for shape compatibility
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        return dq, dk, dv


def measure_peak_memory():
    """Return peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / 1e6


def bench_fwd_bwd(make_inputs, run_fwd, warmup=5, iters=20):
    """Benchmark forward+backward, return (fwd_ms, bwd_ms, peak_mem_MB).

    make_inputs(): returns fresh tensors with requires_grad
    run_fwd(*inputs): returns output tensor
    """
    # Warmup
    for _ in range(warmup):
        inputs = make_inputs()
        out = run_fwd(*inputs)
        out.sum().backward()
        torch.cuda.synchronize()
        del inputs, out

    # Forward-only timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        inputs = make_inputs()
        out = run_fwd(*inputs)
        torch.cuda.synchronize()
        del inputs, out
    fwd_ms = (time.perf_counter() - t0) / iters * 1000

    # Forward+backward timing
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        inputs = make_inputs()
        out = run_fwd(*inputs)
        torch.cuda.synchronize()
        out.sum().backward()
        torch.cuda.synchronize()
        del inputs, out
    total_ms = (time.perf_counter() - t0) / iters * 1000
    bwd_ms = total_ms - fwd_ms
    peak_mem = measure_peak_memory()

    return fwd_ms, bwd_ms, peak_mem


# Gemma-like model configs: (name, num_heads, head_dim, num_kv_heads)
MODEL_CONFIGS = {
    "Gemma-2B":  (8,  256, 1),   # 8 heads, head_dim=256, 1 KV head (MQA)
    "Gemma-7B":  (16, 256, 16),  # 16 heads, head_dim=256, 16 KV heads (MHA)
}

# For LUCID: single-head D=128 (block_size constraint)
LUCID_CONFIGS = {
    "LUCID-small": (1, 128),   # 1 head, D=128
    "LUCID-4h":    (4, 128),   # 4 heads, D=128
    "LUCID-8h":    (8, 128),   # 8 heads, D=128
}


def main():
    device = "cuda"
    dtype = torch.bfloat16
    batch = 1

    print(f"=== LUCID End-to-End Attention Layer Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"dtype: bf16, batch: {batch}")
    print()

    # ── LUCID kernel benchmarks ──
    print("=" * 90)
    print("LUCID Forward Substitution (CuTe DSL kernel)")
    print("=" * 90)
    hdr = (f"{'Config':>12} {'Seqlen':>8} {'Heads':>6} {'D':>4} |"
           f" {'Fwd(ms)':>8} {'Bwd(ms)':>8} {'F+B(ms)':>8} {'PeakMem(MB)':>12}")
    print(hdr)
    print("-" * len(hdr))

    for config_name, (num_heads, head_dim) in LUCID_CONFIGS.items():
        for seqlen in [512, 1024, 2048, 4096, 8192]:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            try:
                def make_inputs():
                    k = torch.randn(batch, seqlen, num_heads, head_dim,
                                   device=device, dtype=dtype, requires_grad=True) * 0.1
                    v = torch.randn(batch, seqlen, num_heads, head_dim,
                                   device=device, dtype=dtype, requires_grad=True)
                    return k, v

                def run_fwd(k, v):
                    return LUCIDForwardSub.apply(k, v, 128)

                fwd_ms, bwd_ms, peak_mem = bench_fwd_bwd(make_inputs, run_fwd)
                print(f"{config_name:>12} {seqlen:>8} {num_heads:>6} {head_dim:>4} |"
                      f" {fwd_ms:>8.2f} {bwd_ms:>8.2f} {fwd_ms+bwd_ms:>8.2f} {peak_mem:>12.1f}")
            except Exception as e:
                print(f"{config_name:>12} {seqlen:>8} {num_heads:>6} {head_dim:>4} | ERROR: {e}")

    # ── FA3 baseline benchmarks ──
    print()
    print("=" * 90)
    print("FlashAttention-3 Causal (baseline)")
    print("=" * 90)
    print(hdr)
    print("-" * len(hdr))

    for config_name, (num_heads, head_dim) in LUCID_CONFIGS.items():
        for seqlen in [512, 1024, 2048, 4096, 8192]:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            try:
                def make_inputs():
                    q = torch.randn(batch, seqlen, num_heads, head_dim,
                                   device=device, dtype=dtype, requires_grad=True)
                    k = torch.randn(batch, seqlen, num_heads, head_dim,
                                   device=device, dtype=dtype, requires_grad=True)
                    v = torch.randn(batch, seqlen, num_heads, head_dim,
                                   device=device, dtype=dtype, requires_grad=True)
                    return q, k, v

                def run_fwd(q, k, v):
                    return FA3Attention.apply(q, k, v)

                fwd_ms, bwd_ms, peak_mem = bench_fwd_bwd(make_inputs, run_fwd)
                print(f"{config_name:>12} {seqlen:>8} {num_heads:>6} {head_dim:>4} |"
                      f" {fwd_ms:>8.2f} {bwd_ms:>8.2f} {fwd_ms+bwd_ms:>8.2f} {peak_mem:>12.1f}")
            except Exception as e:
                print(f"{config_name:>12} {seqlen:>8} {num_heads:>6} {head_dim:>4} | ERROR: {e}")


if __name__ == "__main__":
    main()
