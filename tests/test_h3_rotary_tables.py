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

pytestmark = pytest.mark.h3_rotary_tables


def test_rotary_native_tables(accelerator):
    if accelerator.type != "cuda":
        pytest.skip("The native-order frequency implementation targets SM90")
    positions = torch.randint(0, 100, (137, 3), device=accelerator).float()
    freq_dim, theta = 16, 10000.0
    inv = 1.0 / (
        theta
        ** (
            torch.arange(0, 2 * freq_dim, 2, device=accelerator, dtype=torch.float32)
            / (2 * freq_dim)
        )
    )
    phase = positions.unsqueeze(-1) * inv.view(1, 1, -1)
    phase = torch.cat(tuple(phase.unbind(1)), -1)
    phase = torch.cat((phase, phase), -1)
    actual = ops.h3_rotary_tables(positions, freq_dim, theta)
    for candidate, reference in zip(actual, (phase.cos(), phase.sin())):
        assert torch.equal(candidate, reference)
