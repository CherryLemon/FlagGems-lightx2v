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

"""Tier 2 candidate: retain ordered register reduction; share only row inverses.

The previous full input-tile staging expands into thousands of MP31 shared
operations. This keeps the arithmetic contract and bounds TLE storage by R.
"""

import torch
import triton as tr
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_lightx2v.ops._rms._ordered import ordered_inverse, validate


@tr.jit
def small_shared_kernel(
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
    rr = tl.arange(0, R)
    rows = tl.program_id(0) * R + rr
    lanes = tl.arange(0, L)
    values = tl.arange(0, V)
    cols = lanes[:, None] * 8 + (values[None, :] // 8) * (L * 8) + values[None, :] % 8
    mask = (rows[:, None, None] < M) & (cols[None, :, :] < D)
    x = tl.load(X + rows[:, None, None] * D + cols[None, :, :], mask, other=0).to(
        tl.float32
    )
    inv = ordered_inverse(x, EPS, D, R, L, V)
    inverse_buffer = tle.gpu.alloc(
        [R],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    inverse_ptr = tle.gpu.local_ptr(inverse_buffer, (rr,))
    tl.store(inverse_ptr, inv)
    inv = tl.load(inverse_ptr)
    weight = tl.load(W + cols, cols < D, other=0).to(tl.float32)
    tl.store(
        Y + rows[:, None, None] * D + cols[None, :, :],
        x * inv[:, None, None] * weight[None, :, :],
        mask,
    )


def rms_norm_tle(x, weight, eps=1e-6):
    validate(x, weight, eps)
    d = x.shape[-1]
    m = x.numel() // d
    r, l, v = (8, 16, 8) if d == 128 else (1, 64, 128)
    y = torch.empty_like(x)
    if m:
        with getattr(torch, x.device.type).device(x.device):
            small_shared_kernel[(tr.cdiv(m, r),)](
                x, weight, y, m, eps, d, r, l, v, num_warps=4, enable_fp_fusion=False
            )
    return y
