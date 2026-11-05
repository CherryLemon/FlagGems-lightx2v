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

"""Tier 3: ordinary Triton only, BF16 weighted RMS with explicit FP32 order."""

import math

import torch
import triton as tr
import triton.language as tl


@tr.jit
def ordered_inverse(
    x,
    EPS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
):
    values = tl.arange(0, V)
    lanes = tl.arange(0, L)
    acc = tl.full((R, L), 0, tl.float32)
    for i in tl.static_range(8 if D == 128 else 88):
        value = tl.sum(tl.where(values[None, None, :] == i, x, 0), 2)
        acc = tl.fma(value, value, acc)
    for i in tl.static_range(4 if L == 16 else 5):
        indices = tl.broadcast_to((lanes ^ (1 << i))[None, :], (R, L))
        acc = acc + tl.gather(acc, indices, 1)
    if L == 64:
        partial = tl.max(tl.reshape(acc, (R, 2, 32)), 2)
        total = tl.sum(partial, 1)
    else:
        total = tl.max(acc, 1)
    return tl.rsqrt(tl.div_rn(total, D) + EPS)


@tr.jit
def baseline_kernel(
    X,
    W,
    Y,
    M,
    EPS: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    V: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    lanes = tl.arange(0, L)
    values = tl.arange(0, V)
    cols = lanes[:, None] * 8 + (values[None, :] // 8) * (L * 8) + values[None, :] % 8
    mask = (rows[:, None, None] < M) & (cols[None, :, :] < D)
    x = tl.load(X + rows[:, None, None] * D + cols[None, :, :], mask, other=0).to(
        tl.float32
    )
    inv = ordered_inverse(x, EPS, D, R, L, V)
    weight = tl.load(W + cols, cols < D, other=0).to(tl.float32)
    tl.store(
        Y + rows[:, None, None] * D + cols[None, :, :],
        x * inv[:, None, None] * weight[None, :, :],
        mask,
    )


def validate(x, weight, eps):
    if x.ndim < 1 or x.shape[-1] not in (128, 5376):
        raise ValueError("D128/D5376 required")
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype:
        raise TypeError("BF16 required")
    if weight.shape != (x.shape[-1],) or weight.device != x.device:
        raise ValueError("same-device last-axis weight required")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("contiguous rows required")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("finite positive epsilon required")
    if x.requires_grad or weight.requires_grad:
        raise NotImplementedError("inference only")


def rms_norm_baseline(x, weight, eps=1e-6):
    validate(x, weight, eps)
    d = x.shape[-1]
    m = x.numel() // d
    r = 8 if d == 128 else 1
    lanes = 16 if d == 128 else 64
    v = 8 if d == 128 else 128
    y = torch.empty_like(x)
    if m:
        with getattr(torch, x.device.type).device(x.device):
            baseline_kernel[(tr.cdiv(m, r),)](
                x,
                weight,
                y,
                m,
                eps,
                d,
                r,
                lanes,
                v,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return y
