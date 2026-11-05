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

"""Basic Triton VAE Q/K RMSNorm + RoPE, preserving native cast boundaries.

No Gluon, TLE, TMA, inline assembly, or vendor-specific scheduling is used.
The FP32 reduction follows PyTorch 2.11's four-value serial accumulation.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    NQ,
    NK,
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    QS2: tl.constexpr,
    KS0: tl.constexpr,
    KS1: tl.constexpr,
    KS2: tl.constexpr,
    CS0: tl.constexpr,
    CS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    DEBUG: tl.constexpr,
):
    r = tl.program_id(0) * R + tl.arange(0, R)
    g = tl.arange(0, 16)
    v = tl.arange(0, 4)
    d = g[:, None] * 4 + v[None, :]
    b = r // (N * H)
    n = (r // H) % N
    h = r % H
    if tl.program_id(1) == 0:
        ptr = Q + (b * QS0 + n * QS1 + h * QS2)[:, None, None] + d[None, :, :]
    else:
        ptr = K + (b * KS0 + n * KS1 + h * KS2)[:, None, None] + d[None, :, :]
    x = tl.load(ptr, r[:, None, None] < B * N * H, 0).to(tl.float32)
    acc = tl.full((R, 16), 0.0, tl.float32)
    for i in tl.static_range(4):
        t = tl.sum(tl.where(v[None, None, :] == i, x, 0.0), 2)
        acc = tl.fma(t, t, acc)
    total = tl.sum(acc, 1)
    inv = tl.rsqrt(total * (1.0 / 64.0) + EPS)
    normalized = (x * inv[:, None, None]).reshape((R, 64))
    # Native emits FP32 RMS output, then converts to the original inference dtype.
    normalized = normalized.to(OQ.dtype.element_ty).to(tl.float32)
    col = tl.arange(0, 64)
    rotcol = tl.where(col < 24, col + 24, tl.where(col < 48, col - 24, col))
    rotated = tl.gather(normalized, tl.broadcast_to(rotcol[None, :], (R, 64)), 1)
    rotated = tl.where(col[None, :] < 24, -rotated, rotated)
    co = (
        tl.load(
            C + b[:, None] * CS0 + n[:, None] * CS1 + col[None, :],
            (r[:, None] < B * N * H) & (col[None, :] < 48),
            0,
        )
        .to(OQ.dtype.element_ty)
        .to(tl.float32)
    )
    si = (
        tl.load(
            S + b[:, None] * SS0 + n[:, None] * SS1 + col[None, :],
            (r[:, None] < B * N * H) & (col[None, :] < 48),
            0,
        )
        .to(OQ.dtype.element_ty)
        .to(tl.float32)
    )
    lhs = (normalized * co).to(OQ.dtype.element_ty).to(tl.float32)
    rhs = (rotated * si).to(OQ.dtype.element_ty).to(tl.float32)
    y = tl.where(col[None, :] < 48, lhs + rhs, normalized)
    if tl.program_id(1) == 0:
        tl.store(OQ + r[:, None] * 64 + col[None, :], y, r[:, None] < B * N * H)
        if DEBUG:
            tl.store(
                NQ + r[:, None] * 64 + col[None, :], normalized, r[:, None] < B * N * H
            )
    else:
        tl.store(OK + r[:, None] * 64 + col[None, :], y, r[:, None] < B * N * H)
        if DEBUG:
            tl.store(
                NK + r[:, None] * 64 + col[None, :], normalized, r[:, None] < B * N * H
            )


def qk_norm_rope(
    q, k, cos, sin, eps=1e-5, *, rows=4, warps=4, debug=False, artifacts=None
):
    if q.ndim != 4 or q.shape != k.shape or q.shape[-1] != 64:
        raise ValueError("Q/K must share [B,N,H,64] shape")
    if (
        q.device.type not in ("cuda", "musa")
        or q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or k.device != q.device
    ):
        raise TypeError("Q/K require matching CUDA FP16 or BF16")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("Q/K last dimension must be contiguous")
    b, n, h, _ = q.shape
    for x in (cos, sin):
        if (
            x.ndim != 4
            or x.shape[0] not in (1, b)
            or x.shape[1:] != (n, 1, 48)
            or x.stride(-1) != 1
            or x.device != q.device
            or x.dtype != torch.float32
        ):
            raise ValueError(
                "RoPE tables require FP32 [1|B,N,1,48] on the input device"
            )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (rows, warps) not in ((4, 4), (8, 8), (32, 8)):
        raise ValueError("configuration does not preserve the verified reduction order")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, cos, sin)):
        raise NotImplementedError("inference only")
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty_like(oq)
    nq = torch.empty_like(oq) if debug else oq
    nk = torch.empty_like(ok) if debug else ok
    if b * n * h:
        with getattr(torch, q.device.type).device(q.device):
            compiled = _qk_norm_rope[(triton.cdiv(b * n * h, rows), 2)](
                q,
                k,
                cos,
                sin,
                oq,
                ok,
                nq,
                nk,
                b,
                n,
                h,
                *q.stride()[:3],
                *k.stride()[:3],
                cos.stride(0) if cos.shape[0] > 1 else 0,
                cos.stride(1),
                sin.stride(0) if sin.shape[0] > 1 else 0,
                sin.stride(1),
                eps,
                rows,
                debug,
                num_warps=warps,
                enable_fp_fusion=False,
            )
        if artifacts is not None:
            artifacts.append(compiled)
    return (oq, ok, nq, nk) if debug else (oq, ok)
