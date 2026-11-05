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

import gc

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--accelerator-device", default=None)
    parser.addoption("--require-accelerator", action="store_true")


@pytest.fixture
def accelerator(request):
    selected = request.config.getoption("--accelerator-device")
    if selected and selected.startswith("musa"):
        import torch_musa  # noqa: F401
    if selected is None and torch.cuda.is_available():
        selected = "cuda:0"
    if selected is None:
        if request.config.getoption("--require-accelerator"):
            pytest.fail("An accelerator was required but none was selected/available")
        pytest.skip("Accelerator numerical tests were not requested")
    device = torch.device(selected)
    backend = getattr(torch, device.type)
    if not backend.is_available():
        pytest.fail(f"Requested accelerator is unavailable: {selected}")
    backend.set_device(device)
    with torch.no_grad():
        yield device
    backend.synchronize()
    gc.collect()
    backend.empty_cache()
