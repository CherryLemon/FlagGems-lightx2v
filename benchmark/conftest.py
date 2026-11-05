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


def pytest_addoption(parser):
    parser.addoption("--tier", choices=["baseline", "tle", "sm90"], default="baseline")
    parser.addoption(
        "--warmup", type=float, default=25, help="Timing warmup in milliseconds"
    )
    parser.addoption(
        "--iter", type=float, default=100, help="Timing samples in milliseconds"
    )


@pytest.fixture
def benchmark(request, accelerator, record_property):
    import triton.testing

    warmup = request.config.getoption("--warmup")
    repetitions = request.config.getoption("--iter")
    if warmup <= 0 or repetitions <= 0:
        pytest.fail("Benchmark warmup and sample durations must be positive")
    backend = getattr(torch, accelerator.type)

    def measure(reference, candidate, *, shape, dtype):
        with torch.inference_mode():
            for _ in range(5):
                reference()
                candidate()
            backend.synchronize()
            reference_ms = triton.testing.do_bench(
                reference, warmup=warmup, rep=repetitions
            )
            candidate_ms = triton.testing.do_bench(
                candidate, warmup=warmup, rep=repetitions
            )
        record_property("reference_ms", reference_ms)
        record_property("candidate_ms", candidate_ms)
        record_property("speedup", reference_ms / candidate_ms)
        record_property("shape", str(tuple(shape)))
        record_property("dtype", str(dtype))
        record_property("device", str(accelerator))
        record_property("tier", request.config.getoption("--tier"))
        print(
            f"{request.node.name}: reference={reference_ms:.6f}ms candidate={candidate_ms:.6f}ms "
            f"speedup={reference_ms / candidate_ms:.3f}x"
        )

    return measure
