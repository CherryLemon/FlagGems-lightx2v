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

"""Ordinary Triton D5376 candidate: stream the native ordered accumulator."""

import torch
import triton as tr
import triton.language as tl

from ._ordered import validate


@tr.jit
def stream_inverse(X, row, EPS: tl.constexpr):
    lanes = tl.arange(0, 64)
    acc = tl.full((64,), 0, tl.float32)
    for j in tl.static_range(88):
        columns = lanes * 8 + (j // 8) * 512 + j % 8
        x = tl.load(X + row * 5376 + columns, columns < 5376, other=0).to(tl.float32)
        acc = tl.fma(x, x, acc)
    for j in tl.static_range(5):
        acc = acc + tl.gather(acc, lanes ^ (1 << j), 0)
    partial = tl.max(tl.reshape(acc, (2, 32)), 1)
    total = tl.sum(partial, 0)
    return tl.rsqrt(tl.div_rn(total, 5376) + EPS)


@tr.jit
def stream_kernel(X, W, Y, EPS: tl.constexpr):
    row = tl.program_id(0)
    inv = stream_inverse(X, row, EPS)
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
            stream_kernel[(n,)](
                x, w, y, eps, num_warps=num_warps, enable_fp_fusion=False
            )
    return y
