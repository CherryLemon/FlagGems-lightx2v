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

pytestmark = pytest.mark.h3_reference_blend


def test_reference_blend_order(accelerator):
    sample, prediction = [
        torch.randn((127, 96), device=accelerator, dtype=torch.float32)
        for _ in range(2)
    ]
    timestep = torch.tensor([0.73], device=accelerator, dtype=torch.float32)
    sigmas = torch.linspace(1.0, 0.0, 30, device=accelerator, dtype=torch.float32)
    for step in [0, 13, 28]:
        ratio = sigmas[step + 1] / sigmas[step]
        denoised = sample + (1.0 - timestep) * prediction
        expected = ratio * sample + (1.0 - ratio) * denoised
        assert torch.equal(
            ops.h3_reference_blend(sample, prediction, timestep, sigmas, step), expected
        )
