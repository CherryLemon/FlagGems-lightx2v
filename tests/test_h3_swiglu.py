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

import pytest
import torch
import torch.nn.functional as F

import flaggems_lightx2v as ops

pytestmark = pytest.mark.h3_swiglu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 31, 1108])
def test_swiglu_native_boundary(accelerator, dtype, rows):
    generator = torch.Generator(device=accelerator).manual_seed(124)
    packed = torch.randn(
        (rows, 28672), generator=generator, device=accelerator, dtype=dtype
    )
    value, gate = packed.chunk(2, -1)
    expected = value * F.silu(gate)
    actual = ops.h3_swiglu(packed)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


def test_swiglu_above_int32_address_boundary(accelerator):
    packed = torch.randn((75938, 28672), device=accelerator, dtype=torch.bfloat16)
    assert packed.numel() > 2**31
    value, gate = packed.chunk(2, -1)
    expected = value * F.silu(gate)
    actual = ops.h3_swiglu(packed)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[-1], expected[-1])
