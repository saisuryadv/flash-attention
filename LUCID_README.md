# LUCID Forward Substitution Kernels

Custom SM90 (Hopper) CuTe DSL kernels for LUCID's block-triangular forward substitution:

```
V'_i = inv(tril(exp(K_i @ K_i^T))) @ (V_i - sum_{j<i} exp(K_i @ K_j^T) @ V'_j)
```

## Files

| File | Description |
|------|-------------|
| `flash_attn/cute/forward_sub_fwd.py` | Forward kernel (`ForwardSubSm90`) |
| `flash_attn/cute/forward_sub_bwd.py` | Backward kernel (`BackwardSubSm90`) — dV + dK |
| `flash_attn/cute/forward_sub_interface.py` | Python interface for fwd/bwd |
| `bench_lucid.py` | Kernel benchmark vs FA3 |

## Usage

```python
from flash_attn.cute.forward_sub_interface import _forward_sub_fwd, _forward_sub_bwd

# Forward: V' = L^{-1} V
v_prime = _forward_sub_fwd(k, v, use_diag_solve=True)

# Backward: dV, dK from upstream gradient dV'
dv, dk = _forward_sub_bwd(k, v_prime, dv_prime,
                           use_diag_solve=True,
                           use_combined_kernel=True)
```

## Kernel Benchmark

**Hardware:** NVIDIA GH200 120GB (SM90 Hopper), CUDA 12.8
**Software:** PyTorch 2.7, CUTLASS 4.4, CuTe DSL
**Config:** B=1, H=1, D=128, BS=128, dtype=bf16, warmup=10, iters=50

```
python bench_lucid.py
```

### LUCID vs FA3 Kernel Timing (microseconds)

| T | Seqlen | FA3 fwd | FA3 bwd | LUCID fwd | LUCID dV+dK | fwd/FA3 | bwd/FA3 |
|---|--------|---------|---------|-----------|-------------|---------|---------|
| 2 | 256 | 55 | 113 | 109 | 118 | 2.0x | 1.0x |
| 4 | 512 | 58 | 73 | 115 | 85 | 2.0x | 1.2x |
| 8 | 1024 | 64 | 88 | 80 | 164 | 1.3x | 1.9x |
| 16 | 2048 | 76 | 166 | 156 | 324 | 2.1x | 1.9x |
| 32 | 4096 | 148 | 324 | 260 | 595 | 1.8x | 1.8x |
| 64 | 8192 | 244 | 591 | 513 | 1144 | 2.1x | 1.9x |

**Notes:**
- FA3 = standard causal softmax attention (different operation, shown as hardware roofline)
- LUCID fwd = block-triangular forward substitution with inter-CTA flag sync
- LUCID dV+dK = combined backward kernel (Phase 1 upper triangle dV + Phase 2 lower triangle dK)
- `use_diag_solve=False` for raw kernel timing (excludes host-side diagonal inverse)
- LUCID backward is ~1.0-1.9x FA3 backward despite sequential inter-block dependencies

### End-to-End Attention Layer (forward + backward with autograd)

**Config:** B=1, dtype=bf16, `use_diag_solve=True` (includes diagonal inverse overhead)

```
python bench_lucid_e2e.py
```

#### LUCID Forward Substitution

| Config | Seqlen | Heads | D | Fwd (ms) | Bwd (ms) | F+B (ms) | Peak Mem (MB) |
|--------|--------|-------|---|----------|----------|----------|---------------|
| 1 head | 512 | 1 | 128 | 0.46 | 0.85 | 1.31 | 70 |
| 1 head | 1024 | 1 | 128 | 0.78 | 0.87 | 1.65 | 73 |
| 1 head | 2048 | 1 | 128 | 0.52 | 0.92 | 1.45 | 79 |
| 1 head | 4096 | 1 | 128 | 0.82 | 0.86 | 1.68 | 91 |
| 1 head | 8192 | 1 | 128 | 1.02 | 1.08 | 2.10 | 115 |
| 4 heads | 512 | 4 | 128 | 0.64 | 1.03 | 1.67 | 80 |
| 4 heads | 2048 | 4 | 128 | 0.78 | 0.90 | 1.68 | 120 |
| 4 heads | 8192 | 4 | 128 | 1.08 | 1.86 | 2.94 | 277 |
| 8 heads | 512 | 8 | 128 | 0.67 | 0.90 | 1.58 | 93 |
| 8 heads | 2048 | 8 | 128 | 0.78 | 0.89 | 1.68 | 172 |
| 8 heads | 8192 | 8 | 128 | 1.46 | 3.01 | 4.46 | 487 |

#### FlashAttention-3 Causal (baseline, different operation)

| Config | Seqlen | Heads | D | Fwd (ms) | Bwd (ms) | F+B (ms) | Peak Mem (MB) |
|--------|--------|-------|---|----------|----------|----------|---------------|
| 1 head | 512 | 1 | 128 | 0.16 | 0.38 | 0.54 | 68 |
| 1 head | 1024 | 1 | 128 | 0.25 | 0.40 | 0.65 | 70 |
| 1 head | 2048 | 1 | 128 | 0.24 | 0.41 | 0.65 | 72 |
| 1 head | 4096 | 1 | 128 | 0.26 | 0.41 | 0.67 | 78 |
| 1 head | 8192 | 1 | 128 | 0.31 | 0.54 | 0.85 | 88 |
| 4 heads | 512 | 4 | 128 | 0.24 | 0.41 | 0.65 | 72 |
| 4 heads | 2048 | 4 | 128 | 0.24 | 0.40 | 0.64 | 88 |
| 4 heads | 8192 | 4 | 128 | 0.32 | 0.68 | 1.00 | 151 |
| 8 heads | 512 | 8 | 128 | 0.24 | 0.39 | 0.63 | 78 |
| 8 heads | 2048 | 8 | 128 | 0.24 | 0.40 | 0.64 | 109 |
| 8 heads | 8192 | 8 | 128 | 0.43 | 0.90 | 1.33 | 236 |

**Hardware/Software:**
- GPU: NVIDIA GH200 120GB (SM90 Hopper), 102 GB HBM3
- CUDA 12.8, PyTorch 2.7, CUTLASS 4.4, CuTe DSL
- `nvidia-cutlass-dsl>=4.4.1`, `quack-kernels>=0.2.10`, `apache-tvm-ffi`

## Architecture

### Forward Kernel
- Producer/consumer warp specialization (32 producer + 256 consumer threads)
- 2-stage TMA async pipeline for K_j and V'_j streaming
- Flag synchronization for inter-CTA data dependencies (dV_j must be ready before downstream CTAs)
- Optional diagonal block solve via WGMMA

### Backward Kernel
- **Phase 1 (upper triangle, sequential):** dV_i via reverse-order triangular solve + upper-triangle dK accumulation. 5 GEMMs per block: S, G, P@dV, dS@K.
- **Phase 2 (lower triangle, parallel):** Lower-triangle dK accumulation. 4 GEMMs per block: S, G, dS@K.
- K_i reload between phases (sO/sQ shared memory alias fix)
- sVfixed R2S reload (solved dV_i from registers to smem for Phase 2 GEMM2)
- Diagonal dK correction computed on host

## Correctness

Tested against float64 autograd reference (PyTorch `solve_triangular`):
- dV relative error: <1% (bf16)
- dK relative error: <2% (bf16)
- Verified at T=2, 3, 4, 8
