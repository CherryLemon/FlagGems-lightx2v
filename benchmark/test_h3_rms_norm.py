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

import flaggems_lightx2v as ops


@pytest.mark.h3_rms_norm
def test_h3_rms_norm(accelerator, benchmark, request):
    x = torch.randn((1108, 5376), device=accelerator, dtype=torch.bfloat16)
    weight = torch.randn(5376, device=accelerator, dtype=x.dtype)
    tier = request.config.getoption("--tier")

    def native_rms():
        xf = x.float()
        return (
            xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-5) * weight.float()
        ).to(x.dtype)

    benchmark(
        native_rms,
        lambda: ops.h3_rms_norm(x, weight, 1e-5, tier=tier),
        shape=x.shape,
        dtype=x.dtype,
    )
