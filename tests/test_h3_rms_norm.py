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

pytestmark = pytest.mark.h3_rms_norm


@pytest.mark.parametrize("dimension", [128, 5376])
def test_rms_tiers(accelerator, dimension):
    generator = torch.Generator(device=accelerator).manual_seed(123)
    x = torch.randn(
        (3, dimension), generator=generator, device=accelerator, dtype=torch.bfloat16
    )
    weight = torch.randn(
        dimension, generator=generator, device=accelerator, dtype=x.dtype
    )
    xd, wd = x.cpu().double(), weight.cpu().double()
    expected = (xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + 1e-5) * wd).to(
        x.dtype
    )
    tiers = ["baseline", "sm90"] if accelerator.type == "cuda" else ["baseline", "tle"]
    before = x.clone(), weight.clone()
    for tier in tiers:
        actual = ops.h3_rms_norm(x, weight, 1e-5, tier=tier)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-3, atol=1e-3)
    assert torch.equal(x, before[0]) and torch.equal(weight, before[1])


def test_rms_graph_refresh(accelerator):
    if accelerator.type != "cuda":
        pytest.skip(
            "This graph API test is CUDA-specific; MUSA has separate acceptance"
        )
    x = torch.randn((44, 5376), device=accelerator, dtype=torch.bfloat16)
    weight = torch.ones(5376, device=accelerator, dtype=x.dtype)
    ops.h3_rms_norm(x, weight, 1e-5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ops.h3_rms_norm(x, weight, 1e-5)
    for _ in range(3):
        x.normal_()
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected = ops.h3_rms_norm(x, weight, 1e-5)
        assert torch.equal(output, expected)
