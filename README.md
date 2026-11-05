# FlagGems-lightx2v

LightX2V-specific numerical contracts, separated from model loading, scheduling,
communication and framework patching. The layout follows FlagGems-vllm and
FlagGems-sglang: public APIs, ordinary Triton operators, vendor/architecture
specializations, independent tests and benchmarks.

Generic SageAttention is contributed to **FlagGems** and is not duplicated here.

## Public APIs

| API | Contract |
|---|---|
| `h3_rms_norm` | BF16 D128/D5376; FP32 accumulation and weight multiplication before final cast |
| `h3_swiglu` | Packed `[value | gate]`; SiLU materialized in FP16/BF16 before multiplication |
| `h3_vae_qk_norm_rope` | VAE D64 Q/K normalization and48-channel split RoPE |
| `h3_vae_residual` | FP32 state/scale with FP16 update |
| `h3_vae_residual_norm` | D2048 FP32 residual/RMS with FP16 normalized output |
| `h3_reference_blend` | FP32 H3 schedule update with separate rounded operations |
| `h3_rotary_tables` | SM90 native-order position tables, FP64 frequency power and FP32 trigonometry |

Inputs are inference tensors. Operations do not own model state, persistent
weight caches, global ATen registration or request lifecycle. Public APIs
validate unsupported inputs before launch; execution errors propagate.

## Feature tiers

| Tier | Features | Validation boundary |
|---|---|---|
| `sm90` | Architecture-specific RMS with Gluon/PTX | H100; requires compatible FlagTree/Gluon |
| `tle` | Ordinary Triton plus TLE RMS | MTT S5000; other domestic devices unverified |
| `baseline` | Ordinary Triton RMS; no TLE/Gluon/vendor assembly | Upstream Triton H100 and S5000 prototype validation; A100 compile validation only |

VAE fusions and reference blend currently reuse the basic Triton implementation
under all feature budgets; they are not advertised as separate vendor speedups.
Their model-level validation covers H100. S5000 RMS validation does not establish
full H3 video-generation support, nor validation of every VAE fusion on MUSA.

```python
import flaggems_lightx2v as ops

y = ops.h3_rms_norm(x, weight, eps=1e-5, tier="baseline")
hidden = ops.h3_swiglu(packed_value_gate)
```

Install the device-matched PyTorch/Triton or FlagTree stack first; optional vendor
compilers are never installed implicitly. Then use `python -m pip install .`.

The maintained documents describe [optimization design](docs/optimization.md)
and [environment, installation and reproduction](docs/reproduction.md). Runtime
validation must distinguish baseline, TLE and SM90 arithmetic and actual tested
hardware. Per-run performance evidence belongs in the associated PRs.
