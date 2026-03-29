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
