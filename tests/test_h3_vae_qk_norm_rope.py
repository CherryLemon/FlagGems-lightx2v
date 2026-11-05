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

pytestmark = pytest.mark.h3_vae_qk_norm_rope


def test_vae_qk_native_chain(accelerator):
    q, k = [
        torch.randn((2, 17, 2, 64), device=accelerator, dtype=torch.float16)
        for _ in range(2)
    ]
    phase = torch.randn((1, 17, 1, 48), device=accelerator, dtype=torch.float32)
    cos, sin = phase.cos(), phase.sin()
    actual = ops.h3_vae_qk_norm_rope(q, k, cos, sin, 1e-5)
    for original, output in zip((q, k), actual):
        normalized = F.rms_norm(original.float(), (64,), None, 1e-5).to(original.dtype)
        first, second = normalized[..., :48].chunk(2, -1)
        rotate = torch.cat((-second, first), -1)
        expected = torch.cat(
            (
                normalized[..., :48] * cos.to(q.dtype) + rotate * sin.to(q.dtype),
                normalized[..., 48:],
            ),
            -1,
        )
        assert torch.equal(output, expected)
