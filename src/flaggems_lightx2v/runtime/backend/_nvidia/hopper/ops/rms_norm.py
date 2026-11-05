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

# Copyright 2026 The FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SM90 inference RMSNorm with the pinned FlashInfer/SGL arithmetic order.

This explicit operation multiplies the weight in FP32 before the final cast.
It must not replace the early-cast public rms_norm semantics globally.
Validated against the installed H3 native path on BF16 D128 and D5376.
"""

import math

import torch
import triton as tr
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@g.jit
def native128(
    X,
    W,
    Y,
    M: gl.constexpr,
    EPS: gl.constexpr,
    R: gl.constexpr,
    SERIAL: gl.constexpr,
    ASC: gl.constexpr,
):
    # Match installed FlashInfer's 16 threads/row and eight values/thread.
    layout: gl.constexpr = BlockedLayout([1, 1, 8], [2, 16, 1], [4, 1, 1], [2, 1, 0])
    rl: gl.constexpr = SliceLayout(1, SliceLayout(2, layout))
    ll: gl.constexpr = SliceLayout(0, SliceLayout(2, layout))
    vl: gl.constexpr = SliceLayout(0, SliceLayout(1, layout))
    rows = gl.program_id(0) * R + gl.arange(0, R, layout=rl)
    lanes = gl.arange(0, 16, layout=ll)
    v = gl.arange(0, 8, layout=vl)
    x = gl.load(
        X + rows[:, None, None] * 128 + lanes[None, :, None] * 8 + v[None, None, :],
        rows[:, None, None] < M,
        other=0,
    ).to(gl.float32)
    sq = x * x
    if SERIAL:
        ss = gl.sum(gl.where(v[None, None, :] == 0, sq, 0), 2)
        for i in gl.static_range(1, 8):
            ss = ss + gl.sum(gl.where(v[None, None, :] == i, sq, 0), 2)
    else:
        ss = gl.sum(sq, 2)
    for i in gl.static_range(0, 4):
        if ASC:
            offset = 1 << i
        else:
            offset = 8 >> i
        other = gl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
            constraints="=f,f,r",
            args=[ss, offset],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        ss = ss + other
    inv = gl.rsqrt(ss * (1.0 / 128) + EPS)
    w = gl.load(
        W + rows[:, None, None] * 0 + lanes[None, :, None] * 8 + v[None, None, :]
    ).to(gl.float32)
    gl.store(
        Y + rows[:, None, None] * 128 + lanes[None, :, None] * 8 + v[None, None, :],
        x * inv[:, :, None] * w,
        rows[:, None, None] < M,
    )


@g.jit
def native5376(X, W, Y, M, EPS: gl.constexpr, PRELOAD: gl.constexpr = False):
    # Native PTX: 88 serial FMAs/thread, then xor 1/2/4/8/16,
    # two-warp sum and div.rn.f32 (a reciprocal multiply is not equivalent).
    layout: gl.constexpr = BlockedLayout([1, 1, 8], [1, 32, 1], [2, 2, 1], [2, 1, 0])
    rr = gl.program_id(0) * 2 + gl.arange(
        0, 2, layout=SliceLayout(1, SliceLayout(2, layout))
    )
    ll = gl.arange(0, 64, layout=SliceLayout(0, SliceLayout(2, layout)))
    vv = gl.arange(
        0, 128 if PRELOAD else 8, layout=SliceLayout(0, SliceLayout(1, layout))
    )
    acc = gl.full((2, 64), 0, gl.float32, SliceLayout(2, layout))
    if PRELOAD:
        cols = (
            ll[None, :, None] * 8
            + (vv[None, None, :] // 8) * 512
            + vv[None, None, :] % 8
        )
        x = gl.load(
            X + rr[:, None, None] * 5376 + cols,
            (rr[:, None, None] < M) & (cols < 5376),
            other=0,
        ).to(gl.float32)
        for i in gl.static_range(88):
            value = gl.sum(gl.where(vv[None, None, :] == i, x, 0), 2)
            acc = gl.fma(value, value, acc)
    else:
        for chunk in gl.static_range(11):
            cols = ll[None, :, None] * 8 + vv[None, None, :] + chunk * 512
            x = gl.load(
                X + rr[:, None, None] * 5376 + cols,
                (rr[:, None, None] < M) & (cols < 5376),
                other=0,
            ).to(gl.float32)
            for i in gl.static_range(8):
                value = gl.sum(gl.where(vv[None, None, :] == i, x, 0), 2)
                acc = gl.fma(value, value, acc)
    for i in gl.static_range(5):
        other = gl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
            constraints="=f,f,r",
            args=[acc, 1 << i],
            dtype=gl.float32,
            is_pure=True,
            pack=1,
        )
        acc = acc + other
    partial = gl.max(acc.reshape((2, 2, 32)), 2)
    total = gl.sum(partial, 1)
    mean = gl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, 0f45A80000;",
        constraints="=f,f",
        args=[total],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    inv = gl.rsqrt(mean + EPS)
    inv = gl.convert_layout(inv, SliceLayout(1, SliceLayout(2, layout)))
    if PRELOAD:
        w = gl.load(W + rr[:, None, None] * 0 + cols, cols < 5376, other=0).to(
            gl.float32
        )
        gl.store(
            Y + rr[:, None, None] * 5376 + cols,
            x * inv[:, None, None] * w,
            (rr[:, None, None] < M) & (cols < 5376),
        )
    else:
        for chunk in gl.static_range(11):
            cols = ll[None, :, None] * 8 + vv[None, None, :] + chunk * 512
            x = gl.load(
                X + rr[:, None, None] * 5376 + cols,
                (rr[:, None, None] < M) & (cols < 5376),
                other=0,
            ).to(gl.float32)
            w = gl.load(W + rr[:, None, None] * 0 + cols, cols < 5376, other=0).to(
                gl.float32
            )
            gl.store(
                Y + rr[:, None, None] * 5376 + cols,
                x * inv[:, None, None] * w,
                (rr[:, None, None] < M) & (cols < 5376),
            )


def rms_norm_sgl(x, weight, eps=1e-6):
    """Normalize contiguous BF16 H3 rows, matching native weight-before-cast.

    Supports H100 (SM90), D128 or D5376, finite positive epsilon, inference.
    Weight may reside on the input device or in CUDA-mapped pinned host memory.
    Unsupported inputs raise; this operator has no silent fallback.
    """
    if not x.is_cuda or torch.cuda.get_device_capability(x.device) != (9, 0):
        raise NotImplementedError("rms_norm_sgl requires SM90")
    weight_accessible = weight.device == x.device or (
        weight.device.type == "cpu" and weight.is_pinned()
    )
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or not weight_accessible:
        raise TypeError(
            "rms_norm_sgl requires BF16 input and device or mapped pinned-host weight"
        )
    if x.ndim < 1 or x.shape[-1] not in (128, 5376):
        raise ValueError("rms_norm_sgl supports D128 and D5376")
    if (
        weight.shape != (x.shape[-1],)
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("rms_norm_sgl requires contiguous rows and last-axis weight")
    if x.requires_grad or weight.requires_grad:
        raise NotImplementedError("rms_norm_sgl supports inference only")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be finite and positive")
    y = torch.empty_like(x)
    d = x.shape[-1]
    m = x.numel() // d
    if m == 0:
        return y
    with torch.cuda.device(x.device):
        if d == 128:
            native128[(tr.cdiv(m, 8),)](
                x, weight, y, m, eps, 8, True, True, num_warps=4, enable_fp_fusion=False
            )
        else:
            native5376[(tr.cdiv(m, 2),)](
                x, weight, y, m, eps, True, num_warps=4, enable_fp_fusion=False
            )
    return y
