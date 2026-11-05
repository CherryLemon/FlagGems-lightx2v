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

"""Basic Triton VAE fusions. Native FP32/FP16 boundaries retained."""

import torch
import triton
import triton.language as tl
from flag_gems.utils import tl_extra_shim as libdevice


@triton.jit
def _swiglu(
    P, Y, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, EXP: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    col = i % D
    x = tl.load(P + row * (2 * D) + col, i < N, 0).to(tl.float32)
    g = tl.load(P + row * (2 * D) + D + col, i < N, 0).to(tl.float32)
    if EXP == 0:
        z = libdevice.exp(-g)
    else:
        z = tl.exp(-g)
    # SiLU is a separate native FP16/BF16 tensor before the product.
    act = tl.div_rn(g, 1.0 + z).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, x * act, i < N)


def swiglu(p, *, block=1024, warps=4, exp=0, artifacts=None):
    assert p.device.type in ("cuda", "musa") and p.is_contiguous() and p.ndim >= 2
    assert p.dtype in (torch.float16, torch.bfloat16) and p.shape[-1] % 2 == 0
    d = p.shape[-1] // 2
    out = torch.empty((*p.shape[:-1], d), dtype=p.dtype, device=p.device)
    if out.numel():
        k = _swiglu[(triton.cdiv(out.numel(), block),)](
            p, out, out.numel(), d, block, exp, num_warps=warps, enable_fp_fusion=False
        )
        if artifacts is not None:
            artifacts.append(k)
    return out


@triton.jit
def _residual_norm(
    X,
    A,
    S,
    W,
    R,
    Y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    RESIDUAL: tl.constexpr,
    DEBUG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, 128)
    v = tl.arange(0, 4)
    acc = tl.full((128,), 0.0, tl.float32)
    # Native: four contiguous values, four visits per CUDA thread, FMA.
    # Keep the 128-thread native reduction grouping before normalization.
    for it in tl.static_range(4):
        d = (it * 128 + t[:, None]) * 4 + v[None, :]
        x = tl.load(X + row * 2048 + d)
        if RESIDUAL:
            a = tl.load(A + row * 2048 + d).to(tl.float32)
            s = tl.load(S + d)
            x = x + a * s
        for j in tl.static_range(4):
            q = tl.sum(tl.where(v[None, :] == j, x, 0.0), 1)
            acc = tl.fma(q, q, acc)
    total = tl.sum(acc, 0)
    inv = tl.rsqrt(total * (1.0 / 2048.0) + EPS)
    d = tl.arange(0, 2048)
    x = tl.load(X + row * 2048 + d)
    if RESIDUAL:
        a = tl.load(A + row * 2048 + d).to(tl.float32)
        s = tl.load(S + d)
        x = x + a * s
        tl.store(R + row * 2048 + d, x)
    w = tl.load(W + d)
    y = w * (x * inv)
    tl.store(Y + row * 2048 + d, y)


def residual_norm(
    x,
    weight,
    eps=1e-6,
    *,
    a=None,
    scale=None,
    warps=4,
    output_dtype=torch.float16,
    artifacts=None,
):
    assert (
        x.device.type in ("cuda", "musa")
        and x.is_contiguous()
        and x.dtype == torch.float32
        and x.shape[-1] == 2048
    )
    assert (
        weight.dtype == torch.float32
        and weight.shape == (2048,)
        and weight.is_contiguous()
    )
    if a is not None:
        assert a.shape == x.shape and a.is_contiguous() and a.dtype == torch.float16
        assert (
            scale.shape == (2048,)
            and scale.dtype == torch.float32
            and scale.is_contiguous()
        )
    y = torch.empty_like(x, dtype=output_dtype)
    r = torch.empty_like(x) if a is not None else x
    k = _residual_norm[(x.numel() // 2048,)](
        x,
        a if a is not None else x,
        scale if scale is not None else weight,
        weight,
        r,
        y,
        x.numel() // 2048,
        eps,
        a is not None,
        False,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    if artifacts is not None:
        artifacts.append(k)
    return (r, y) if a is not None else y


@triton.jit
def _residual(X, A, S, Y, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    s = tl.load(S + i % D, i < N, 0)
    tl.store(Y + i, x + a * s, i < N)


def residual(x, a, scale, *, block=1024, warps=4, artifacts=None):
    assert (
        x.device.type in ("cuda", "musa")
        and x.dtype == torch.float32
        and a.dtype == torch.float16
        and x.shape == a.shape
    )
    assert (
        x.is_contiguous()
        and a.is_contiguous()
        and scale.is_contiguous()
        and scale.dtype == torch.float32
    )
    assert scale.shape == (x.shape[-1],)
    y = torch.empty_like(x)
    k = _residual[(triton.cdiv(x.numel(), block),)](
        x,
        a,
        scale,
        y,
        x.numel(),
        x.shape[-1],
        block,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    if artifacts is not None:
        artifacts.append(k)
    return y
