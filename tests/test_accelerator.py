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

"""Independent native references and refreshed graphs through public APIs."""

import pytest
import torch
import torch.nn.functional as F

import flaggems_lightx2v as ops


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
