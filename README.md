# FlagGems-lightx2v

## Introduction

FlagGems-lightx2v provides LightX2V-specific operators implemented with Triton
and FlagTree, with generic implementations and vendor/architecture backends.

## Relationship with FlagGems and lightx2v-plugin-FL

- [FlagGems](https://github.com/flagos-ai/FlagGems) maintains general-purpose
  operators and shared backend utilities.
- **FlagGems-lightx2v** exposes LightX2V-specific numerical operators through the
  `flaggems_lightx2v` Python package.
- [lightx2v-plugin-FL](https://github.com/ModelTC/lightx2v-plugin-FL) owns model
  configuration, loading, scheduling, communication and operator integration.

## Operators

| API | Contract |
|---|---|
| `h3_rms_norm` | BF16 D128/D5376; FP32 weight multiplication before the final cast |
| `h3_swiglu` | Packed `[value, gate]`; input-precision SiLU before multiplication |
| `h3_vae_qk_norm_rope` | D64 Q/K normalization and 48-channel split rotary embedding |
| `h3_vae_residual` | FP32 state/scale with FP16 update |
| `h3_vae_residual_norm` | D2048 FP32 residual/RMS with FP16 normalized output |
| `h3_reference_blend` | FP32 reference-blend update |
| `h3_rotary_tables` | SM90 rotary position tables |

The RMS `tier` selects ordinary Triton (`baseline`), MTT S5000 TLE (`tle`),
or NVIDIA SM90 Gluon/PTX (`sm90`). Advanced tiers require a compatible compiler.
Other operators use their ordinary Triton implementations; rotary tables require
SM90. See the public API docstrings for supported dtypes, layouts and arguments.

## Quick Installation

Install the device-matched PyTorch, Triton/FlagTree and FlagGems builds first.

```bash
python -m pip install --no-deps -e .
python -m pip install pytest numpy
```

## Usage Example

```python
import torch
import flaggems_lightx2v

x = torch.randn((44, 5376), device="cuda", dtype=torch.bfloat16)
weight = torch.ones(5376, device=x.device, dtype=x.dtype)
y = flaggems_lightx2v.h3_rms_norm(x, weight, eps=1e-5, tier="baseline")
```

## Tests and Benchmarks

Tests and benchmarks use per-operator pytest files and markers.

```bash
# Check test discovery without executing device workloads.
python -m pytest tests benchmark --collect-only -q

# CPU API contracts.
python -m pytest tests/test_contracts.py -q

# Device numerical tests; fail if the requested accelerator is unavailable.
python -m pytest tests --accelerator-device cuda:0 --require-accelerator -q

# Operator benchmarks, including allocation and five explicit warmup calls.
python -m pytest benchmark --accelerator-device cuda:0 --require-accelerator \
  --tier sm90 --warmup 25 --iter 100 -s
```

Use `--accelerator-device musa:0 --tier tle` for the S5000 RMS benchmark.
Timing warmup and sample durations are in milliseconds. The RMS benchmark
reference is an unfused Torch expression with weight multiplication before the
final cast. The large-address SwiGLU test allocates the 1080p packed shape.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
