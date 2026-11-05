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

pytestmark = [pytest.mark.h3_vae_residual, pytest.mark.h3_vae_residual_norm]


def test_vae_residual_and_norm(accelerator):
    x = torch.randn((3, 2048), device=accelerator, dtype=torch.float32)
    weight = torch.randn(2048, device=accelerator, dtype=torch.float32)
    update = torch.randn_like(x, dtype=torch.float16)
    scale = torch.randn_like(weight)
    expected_residual = x + update.float() * scale
    residual = ops.h3_vae_residual(x, update, scale)
    assert torch.equal(residual, expected_residual)
    norm = ops.h3_vae_residual_norm(x, weight, 1e-5)
    assert torch.equal(norm, F.rms_norm(x, (2048,), weight, 1e-5).half())
    fused_residual, fused_norm = ops.h3_vae_residual_norm(
        x, weight, 1e-5, update=update, scale=scale
    )
    assert torch.equal(fused_residual, expected_residual)
    assert torch.equal(
        fused_norm, F.rms_norm(expected_residual, (2048,), weight, 1e-5).half()
    )
