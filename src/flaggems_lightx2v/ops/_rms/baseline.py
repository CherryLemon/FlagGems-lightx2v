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

"""Tier 3: ordinary Triton only; no TLE, Gluon, or vendor instructions."""

from ._ordered import rms_norm_baseline as ordered
from ._ordered import validate
from ._stream import run as stream


def rms_norm_baseline(x, weight, eps=1e-6):
    validate(x, weight, eps)
    if x.shape[-1] == 5376:
        return stream(x, weight, eps, num_warps=8)
    return ordered(x, weight, eps)
