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

"""Triton+TLE D5376 candidate, sharing only the final inverse scalar."""

import torch
import triton as tr
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_lightx2v.ops._rms._ordered import validate
from flaggems_lightx2v.ops._rms._stream import stream_inverse


@tr.jit
def stream_tle_kernel(X, W, Y, EPS: tl.constexpr):
    row = tl.program_id(0)
    inv = stream_inverse(X, row, EPS)
    buf = tle.gpu.alloc(
        [1],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    ptr = tle.gpu.local_ptr(buf, (tl.arange(0, 1),))
    tl.store(ptr, inv)
    inv = tl.sum(tl.load(ptr), 0)
    columns = tl.arange(0, 8192)
    mask = columns < 5376
    x = tl.load(X + row * 5376 + columns, mask, other=0).to(tl.float32)
    w = tl.load(W + columns, mask, other=0).to(tl.float32)
    tl.store(Y + row * 5376 + columns, x * inv * w, mask)


def run(x, w, eps=1e-5, *, num_warps=4):
    validate(x, w, eps)
    if x.shape[-1] != 5376:
        raise ValueError("D5376 only")
    y = torch.empty_like(x)
    n = x.numel() // 5376
    if n:
        with getattr(torch, x.device.type).device(x.device):
            stream_tle_kernel[(n,)](
                x, w, y, eps, num_warps=num_warps, enable_fp_fusion=False
            )
    return y
