# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public inference APIs; no model patching, persistent cache or hidden fallback."""

import logging
import math

import torch

from ..runtime.op_registrar import resolve_kernel

logger = logging.getLogger(__name__)


def _inference(*tensors):
    if any(x.requires_grad for x in tensors) and torch.is_grad_enabled():
        raise NotImplementedError("LightX2V fusion APIs are inference-only")
    if not tensors or tensors[0].device.type not in ("cuda", "musa"):
        raise NotImplementedError("A CUDA or MUSA tensor is required")
    if any(x.device != tensors[0].device for x in tensors[1:]):
        raise ValueError("All inputs must reside on the same device")


def h3_rms_norm(x, weight, eps=1e-6, *, tier="auto"):
    """BF16 D128/D5376 RMS; FP32 weight multiplication before the final cast.

    The caller owns persistent weight staging. Baseline/TLE require a same-device
    BF16 weight; SM90 additionally supports the verified mapped pinned-host ABI.
    """
    logger.debug("L2V H3_RMS_NORM tier=%s", tier)
    return resolve_kernel("h3_rms_norm", device=x.device, tier=tier)(x, weight, eps)


def h3_swiglu(projection, *, tier="baseline"):
    """Packed [value|gate] SiLU product, materializing SiLU in input precision."""
    logger.debug("L2V H3_SWIGLU tier=%s", tier)
    _inference(projection)
    if projection.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Packed SwiGLU requires FP16 or BF16")
    if (
        projection.ndim < 2
        or projection.shape[-1] % 2
        or not projection.is_contiguous()
    ):
        raise ValueError("Packed SwiGLU requires contiguous [...,2D] input")
    if projection.shape[-1] == 0:
        raise ValueError("Packed SwiGLU requires a nonzero final dimension")
    fn = resolve_kernel("h3_swiglu", device=projection.device, tier=tier)
    with getattr(torch, projection.device.type).device(projection.device):
        return fn(projection, block=512, warps=4, exp=0)


def h3_vae_qk_norm_rope(q, k, cos, sin, eps=1e-5, *, tier="baseline"):
    """[B,N,H,64] RMS and48-channel split RoPE; FP16/BF16 boundaries retained."""
    logger.debug("L2V H3_VAE_QK_NORM_ROPE tier=%s", tier)
    _inference(q, k, cos, sin)
    fn = resolve_kernel("h3_vae_qk_norm_rope", device=q.device, tier=tier)
    with getattr(torch, q.device.type).device(q.device):
        return fn(q, k, cos, sin, eps)


def _residual_contract(x, update, scale):
    _inference(x, update, scale)
    if (
        x.dtype != torch.float32
        or update.dtype != torch.float16
        or scale.dtype != torch.float32
    ):
        raise TypeError("H3 VAE residual requires FP32 state/scale and FP16 update")
    if x.ndim < 1 or x.shape != update.shape or scale.shape != (x.shape[-1],):
        raise ValueError("Residual requires matching state/update and last-axis scale")
    if not all(t.is_contiguous() for t in (x, update, scale)):
        raise ValueError("Residual requires contiguous tensors")


def h3_vae_residual(x, update, scale, *, tier="baseline"):
    """FP32 x + float32(FP16 update)*FP32 scale; returns independent storage."""
    logger.debug("L2V H3_VAE_RESIDUAL tier=%s", tier)
    _residual_contract(x, update, scale)
    fn = resolve_kernel("h3_vae_residual", device=x.device, tier=tier)
    if x.numel() == 0:
        return torch.empty_like(x)
    with getattr(torch, x.device.type).device(x.device):
        return fn(x, update, scale, block=1024, warps=4)


def h3_vae_residual_norm(
    x, weight, eps=1e-6, *, update=None, scale=None, tier="baseline"
):
    """D2048 FP32 residual/RMS; returns FP16 norm, or (FP32 residual, FP16 norm)."""
    logger.debug("L2V H3_VAE_RESIDUAL_NORM tier=%s", tier)
    _inference(x, weight)
    if x.ndim < 1 or x.shape[-1] != 2048 or x.dtype != torch.float32:
        raise ValueError("H3 VAE residual RMS requires FP32 D2048 rows")
    if weight.shape != (2048,) or weight.dtype != torch.float32:
        raise ValueError("H3 VAE residual RMS requires FP32 D2048 weight")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("H3 VAE residual RMS requires contiguous tensors")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (update is None) != (scale is None):
        raise ValueError("update and scale must be supplied together")
    if update is not None:
        _residual_contract(x, update, scale)
    fn = resolve_kernel("h3_vae_residual_norm", device=x.device, tier=tier)
    if not x.numel():
        norm = torch.empty_like(x, dtype=torch.float16)
        return (torch.empty_like(x), norm) if update is not None else norm
    with getattr(torch, x.device.type).device(x.device):
        return fn(x, weight, eps, a=update, scale=scale, warps=4)


def h3_rotary_tables(positions, freq_dim=16, theta=10000.0, *, tier="sm90"):
    """H100 native-order FP32 position tables, with FP64 frequency power."""
    logger.debug("L2V H3_ROTARY_TABLES")
    if positions.requires_grad and torch.is_grad_enabled():
        raise NotImplementedError("H3 rotary tables are inference-only")
    fn = resolve_kernel("h3_rotary_tables", device=positions.device, tier=tier)
    return fn(positions, freq_dim, theta, double=True)
