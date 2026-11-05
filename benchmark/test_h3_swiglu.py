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


@pytest.mark.h3_swiglu
def test_h3_swiglu(accelerator, benchmark):
    packed = torch.randn((1108, 28672), device=accelerator, dtype=torch.bfloat16)

    def native_swiglu():
        value, gate = packed.chunk(2, -1)
        return value * F.silu(gate)

    benchmark(
        native_swiglu,
        lambda: ops.h3_swiglu(packed),
        shape=packed.shape,
        dtype=packed.dtype,
    )
