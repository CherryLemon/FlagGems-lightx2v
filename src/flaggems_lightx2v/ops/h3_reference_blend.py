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

"""The H3 reference-blend numerical update, independent of scheduler state."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _blend(X, PRED, TIME, SIGMAS, Y, N, STEP, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, other=0)
    pred = tl.load(PRED + offsets, offsets < N, other=0)
    timestep = tl.load(TIME)
    sigma = tl.load(SIGMAS + STEP)
    sigma_next = tl.load(SIGMAS + STEP + 1)
    denoised = x + (1.0 - timestep) * pred
    ratio = tl.div_rn(sigma_next, sigma)
    result = ratio * x + (1.0 - ratio) * denoised
    tl.store(Y + offsets, result, offsets < N)


def h3_reference_blend(sample, prediction, timestep, sigmas, step_index):
    """FP32 update; the caller owns the schedule, index and state replacement.

    No input is mutated. Eager/graph callers must provide a valid nonzero current
    sigma; checking schedule values would add a forbidden device-to-host read.
    """
    logger.debug("L2V H3_REFERENCE_BLEND")
    tensors = (sample, prediction, timestep, sigmas)
    if sample.device.type not in ("cuda", "musa"):
        raise NotImplementedError("Reference blend requires CUDA or MUSA")
    if any(t.dtype != torch.float32 or t.device != sample.device for t in tensors):
        raise ValueError("Reference blend requires same-device FP32 tensors")
    if sample.shape != prediction.shape or not all(t.is_contiguous() for t in tensors):
        raise ValueError(
            "Reference blend requires contiguous matching sample/prediction"
        )
    if timestep.numel() != 1 or sigmas.ndim != 1 or not isinstance(step_index, int):
        raise ValueError("Invalid scalar timestep, schedule or index")
    if not 0 <= step_index < sigmas.numel() - 1:
        raise ValueError("step_index is outside the schedule")
    if any(t.requires_grad for t in tensors) and torch.is_grad_enabled():
        raise NotImplementedError("Reference blend is inference-only")
    output = torch.empty_like(sample)
    if sample.numel():
        with getattr(torch, sample.device.type).device(sample.device):
            _blend[(triton.cdiv(sample.numel(), 1024),)](
                sample,
                prediction,
                timestep,
                sigmas,
                output,
                sample.numel(),
                step_index,
                BLOCK=1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return output
