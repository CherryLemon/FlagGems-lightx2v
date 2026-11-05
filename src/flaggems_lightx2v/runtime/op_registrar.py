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

"""Generic < vendor < architecture resolution, without eager optional imports."""

from functools import lru_cache
from importlib import import_module

_OPERATORS = {
    "h3_rotary_tables": (
        "runtime.backend._nvidia.hopper.ops.rotary_tables",
        "rotary_position_tables",
    ),
    "h3_reference_blend": ("ops.h3_reference_blend", "h3_reference_blend"),
    "h3_rms_norm": ("ops._rms.baseline", "rms_norm_baseline"),
    "h3_swiglu": ("ops._vae.basic_other_fusions", "swiglu"),
    "h3_vae_qk_norm_rope": ("ops._vae.basic_qk_norm_rope", "qk_norm_rope"),
    "h3_vae_residual": ("ops._vae.basic_other_fusions", "residual"),
    "h3_vae_residual_norm": ("ops._vae.basic_other_fusions_v3", "residual_norm"),
}


def all_registered_ops():
    return sorted(_OPERATORS)


@lru_cache(maxsize=32)
def _load(module, symbol):
    # Exceptions are not cached and no fallback is attempted after execution.
    return getattr(import_module("flaggems_lightx2v." + module), symbol)


def resolve_kernel(name, *, device=None, tier="baseline"):
    """Resolve before allocation/launch. Only RMS has distinct advanced kernels.

    VAE fusions currently share their ordinary-Triton implementation across all
    feature budgets. This is reuse, not an unmeasured vendor optimization.
    """
    if name not in _OPERATORS:
        raise ValueError(f"Unknown LightX2V operator: {name}")
    if tier not in ("auto", "baseline", "tle", "sm90"):
        raise ValueError("tier must be auto, baseline, tle, or sm90")
    if device is None:
        if tier != "baseline":
            raise ValueError("A device is required for capability-based selection")
    elif device.type not in ("cuda", "musa"):
        raise NotImplementedError("LightX2V kernels require a CUDA or MUSA device")
    if name == "h3_rotary_tables":
        import torch

        if (
            device is None
            or device.type != "cuda"
            or torch.cuda.get_device_capability(device) != (9, 0)
        ):
            raise NotImplementedError(
                "Native-order H3 FP64 frequency tables are validated on SM90 only"
            )
    if name == "h3_rms_norm":
        if tier == "auto":
            if device is not None and device.type == "cuda":
                import torch

                tier = (
                    "sm90"
                    if torch.cuda.get_device_capability(device) == (9, 0)
                    else "baseline"
                )
            else:
                tier = "baseline"
        if tier == "sm90":
            import torch

            if (
                device is None
                or device.type != "cuda"
                or torch.cuda.get_device_capability(device) != (9, 0)
            ):
                raise NotImplementedError(
                    "The SM90 RMS implementation requires CUDA SM90"
                )
            return _load("runtime.backend._nvidia.hopper.ops.rms_norm", "rms_norm_sgl")
        if tier == "tle":
            if device is None or device.type != "musa":
                raise NotImplementedError(
                    "The packaged TLE RMS backend targets MUSA S5000"
                )
            return _load("runtime.backend._mthreads.ops.tle", "rms_norm_tle")
    return _load(*_OPERATORS[name])


def get_op(name, *, device=None, tier="baseline"):
    """Return a validated public API, never an unchecked private kernel wrapper."""
    from functools import partial

    if name not in _OPERATORS:
        raise ValueError(f"Unknown LightX2V operator: {name}")
    if tier not in ("auto", "baseline", "tle", "sm90"):
        raise ValueError("tier must be auto, baseline, tle, or sm90")
    if device is not None and device.type not in ("cuda", "musa"):
        raise NotImplementedError("LightX2V kernels require CUDA or MUSA")
    if name == "h3_reference_blend":
        return _load("ops.h3_reference_blend", name)
    from ..ops import h3

    return partial(getattr(h3, name), tier=tier)
