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
):
    row = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, 128)
    v = tl.arange(0, 4)
    acc = tl.full((128,), 0.0, tl.float32)
    d0 = (0 * 128 + t[:, None]) * 4 + v[None, :]
    x0 = tl.load(X + row * 2048 + d0)
    if RESIDUAL:
        a0 = tl.load(A + row * 2048 + d0).to(tl.float32)
        s0 = tl.load(S + d0)
        x0 = x0 + a0 * s0
    for j in tl.static_range(4):
        q = tl.sum(tl.where(v[None, :] == j, x0, 0.0), 1)
        acc = tl.fma(q, q, acc)
    d1 = (1 * 128 + t[:, None]) * 4 + v[None, :]
    x1 = tl.load(X + row * 2048 + d1)
    if RESIDUAL:
        a1 = tl.load(A + row * 2048 + d1).to(tl.float32)
        s1 = tl.load(S + d1)
        x1 = x1 + a1 * s1
    for j in tl.static_range(4):
        q = tl.sum(tl.where(v[None, :] == j, x1, 0.0), 1)
        acc = tl.fma(q, q, acc)
    d2 = (2 * 128 + t[:, None]) * 4 + v[None, :]
    x2 = tl.load(X + row * 2048 + d2)
    if RESIDUAL:
        a2 = tl.load(A + row * 2048 + d2).to(tl.float32)
        s2 = tl.load(S + d2)
        x2 = x2 + a2 * s2
    for j in tl.static_range(4):
        q = tl.sum(tl.where(v[None, :] == j, x2, 0.0), 1)
        acc = tl.fma(q, q, acc)
    d3 = (3 * 128 + t[:, None]) * 4 + v[None, :]
    x3 = tl.load(X + row * 2048 + d3)
    if RESIDUAL:
        a3 = tl.load(A + row * 2048 + d3).to(tl.float32)
        s3 = tl.load(S + d3)
        x3 = x3 + a3 * s3
    for j in tl.static_range(4):
        q = tl.sum(tl.where(v[None, :] == j, x3, 0.0), 1)
        acc = tl.fma(q, q, acc)
    total = tl.sum(acc, 0)
    inv = tl.rsqrt(total * (1.0 / 2048.0) + EPS)
    if RESIDUAL:
        tl.store(R + row * 2048 + d0, x0)
    w0 = tl.load(W + d0)
    y0 = w0 * (x0 * inv)
    tl.store(Y + row * 2048 + d0, y0)
    if RESIDUAL:
        tl.store(R + row * 2048 + d1, x1)
    w1 = tl.load(W + d1)
    y1 = w1 * (x1 * inv)
    tl.store(Y + row * 2048 + d1, y1)
    if RESIDUAL:
        tl.store(R + row * 2048 + d2, x2)
    w2 = tl.load(W + d2)
    y2 = w2 * (x2 * inv)
    tl.store(Y + row * 2048 + d2, y2)
    if RESIDUAL:
        tl.store(R + row * 2048 + d3, x3)
    w3 = tl.load(W + d3)
    y3 = w3 * (x3 * inv)
    tl.store(Y + row * 2048 + d3, y3)


def residual_norm(
    x,
    weight,
    eps=1e-6,
    *,
    a=None,
    scale=None,
    warps=4,
    output_dtype=torch.float16,
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
    _residual_norm[(x.numel() // 2048,)](
        x,
        a if a is not None else x,
        scale if scale is not None else weight,
        weight,
        r,
        y,
        x.numel() // 2048,
        eps,
        a is not None,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return (r, y) if a is not None else y
