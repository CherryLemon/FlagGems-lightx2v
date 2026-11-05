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

"""Native-order SM90 rotary position tables; two kernels, no global overrides."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=["theta"])
def inverse_frequency(
    Inv, theta, F: tl.constexpr, B: tl.constexpr, DOUBLE: tl.constexpr = False
):
    i = tl.arange(0, B)
    exponent = i.to(tl.float32) / F
    if DOUBLE:
        power = libdevice.pow(theta.to(tl.float64), exponent.to(tl.float64)).to(
            tl.float32
        )
    else:
        power = libdevice.pow(theta, exponent)
    inv = tl.div_rn(1.0, power)
    tl.store(Inv + i, inv, i < F)


@triton.jit
def position_tables(P, Inv, C, S, N, F: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    cols = F * 6
    row = i // cols
    axis = (i % (F * 3)) // F
    freq = i % F
    position = tl.load(P + row * 3 + axis, row < N, other=0)
    inv = tl.load(Inv + freq)
    phase = position * inv
    tl.store(C + i, libdevice.cos(phase), row < N)
    tl.store(S + i, libdevice.sin(phase), row < N)


def rotary_position_tables(positions, freq_dim=16, theta=10000.0, *, double=False):
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or positions.dtype != torch.float32
        or not positions.is_contiguous()
    ):
        raise ValueError("contiguous float32 [N,3] positions required")
    if positions.device.type != "cuda" or torch.cuda.get_device_capability(
        positions.device
    ) != (9, 0):
        raise NotImplementedError("SM90 only")
    if (
        not isinstance(freq_dim, int)
        or not 1 <= freq_dim <= 128
        or not math.isfinite(theta)
        or theta <= 0
    ):
        raise ValueError("positive dimension and finite theta required")
    inv = torch.empty((freq_dim,), device=positions.device, dtype=torch.float32)
    cos = torch.empty(
        (positions.shape[0], freq_dim * 6), device=positions.device, dtype=torch.float32
    )
    sin = torch.empty_like(cos)
    with torch.cuda.device(positions.device):
        inverse_frequency[(1,)](
            inv,
            float(theta),
            freq_dim,
            triton.next_power_of_2(freq_dim),
            double,
            num_warps=1,
            enable_fp_fusion=False,
        )
        if positions.shape[0]:
            position_tables[(triton.cdiv(cos.numel(), 256),)](
                positions,
                inv,
                cos,
                sin,
                positions.shape[0],
                freq_dim,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return cos, sin
