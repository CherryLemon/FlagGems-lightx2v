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

"""LightX2V operators; import does not install hooks or load advanced kernels."""

from .ops.h3 import (
    h3_rms_norm,
    h3_rotary_tables,
    h3_swiglu,
    h3_vae_qk_norm_rope,
    h3_vae_residual,
    h3_vae_residual_norm,
)
from .ops.h3_reference_blend import h3_reference_blend
from .runtime.op_registrar import all_registered_ops, get_op

__version__ = "0.1.0"
__all__ = [
    "all_registered_ops",
    "get_op",
    "h3_reference_blend",
    "h3_rms_norm",
    "h3_rotary_tables",
    "h3_swiglu",
    "h3_vae_qk_norm_rope",
    "h3_vae_residual",
    "h3_vae_residual_norm",
]
