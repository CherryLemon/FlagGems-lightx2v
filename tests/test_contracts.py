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

"""CPU contract tests; these deliberately make no GPU numerical claim."""

import subprocess
import sys

import pytest
import torch

import flaggems_lightx2v as ops


def test_import_does_not_load_advanced_compiler_or_install_hooks():
    script = """import sys, torch
before=torch.nn.functional.silu
import flaggems_lightx2v
assert torch.nn.functional.silu is before
assert not any('experimental.gluon' in n or 'experimental.tle' in n for n in sys.modules)
assert 'flag_gems' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_operator_inventory_and_unknown_lookup():
    names = ops.all_registered_ops()
    assert names == sorted(ops.__all__[2:])
    for name in names:
        assert callable(ops.get_op(name))
    with pytest.raises(ValueError, match="Unknown"):
        ops.get_op("not_an_operator")
    with pytest.raises(ValueError, match="tier"):
        ops.get_op("h3_swiglu", tier="not_a_tier")


@pytest.mark.parametrize(
    "name,args",
    [
        ("h3_swiglu", lambda: (torch.empty(1, 8),)),
        ("h3_rms_norm", lambda: (torch.empty(1, 128), torch.empty(128))),
        (
            "h3_vae_residual",
            lambda: (torch.empty(1, 2048), torch.empty(1, 2048), torch.empty(2048)),
        ),
        ("h3_vae_residual_norm", lambda: (torch.empty(1, 2048), torch.empty(2048))),
        (
            "h3_vae_qk_norm_rope",
            lambda: (
                torch.empty(1, 1, 1, 64),
                torch.empty(1, 1, 1, 64),
                torch.empty(1, 1, 1, 48),
                torch.empty(1, 1, 1, 48),
            ),
        ),
        (
            "h3_reference_blend",
            lambda: (torch.empty(1), torch.empty(1), torch.empty(1), torch.ones(2), 0),
        ),
    ],
)
def test_cpu_is_rejected_without_loading_advanced_implementations(name, args):
    loaded_before = set(sys.modules)
    with pytest.raises(NotImplementedError):
        getattr(ops, name)(*args())
    assert not any(
        "runtime.backend._nvidia" in n or "runtime.backend._mthreads" in n
        for n in set(sys.modules) - loaded_before
        if n.startswith("flaggems_lightx2v")
    )


def test_lookup_cannot_bypass_public_input_guards():
    fn = ops.get_op("h3_swiglu")
    with pytest.raises(NotImplementedError):
        fn(torch.empty((1, 8)))
