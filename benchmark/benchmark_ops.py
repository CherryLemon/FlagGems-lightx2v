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

"""Controlled public-call timings; run correctness tests before this script."""

import argparse
import json

import torch
import torch.nn.functional as F
import triton.testing

import flaggems_lightx2v as ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tier", choices=["baseline", "tle", "sm90"], default="baseline"
    )
    parser.add_argument("--rows", type=int, default=1108)
    args = parser.parse_args()
    if args.device.startswith("musa"):
        import torch_musa  # noqa: F401
    device = torch.device(args.device)
    backend = getattr(torch, device.type)
    backend.set_device(device)
    with torch.inference_mode():
        x = torch.randn((args.rows, 5376), device=device, dtype=torch.bfloat16)
        weight = torch.randn(5376, device=device, dtype=x.dtype)
        packed = torch.randn((args.rows, 28672), device=device, dtype=x.dtype)

        def native_rms():
            xf = x.float()
            return (
                xf
                * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + 1e-5)
                * weight.float()
            ).to(x.dtype)

        def native_swiglu():
            value, gate = packed.chunk(2, -1)
            return value * F.silu(gate)

        rows = []
        for name, reference, candidate in [
            (
                "h3_rms_norm",
                native_rms,
                lambda: ops.h3_rms_norm(x, weight, 1e-5, tier=args.tier),
            ),
            ("h3_swiglu", native_swiglu, lambda: ops.h3_swiglu(packed)),
        ]:
            for _ in range(5):
                reference()
                candidate()
            backend.synchronize()
            rows.append(
                {
                    "op": name,
                    "reference_ms": triton.testing.do_bench(
                        reference, warmup=25, rep=100
                    ),
                    "candidate_ms": triton.testing.do_bench(
                        candidate, warmup=25, rep=100
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "device": str(device),
                    "rows": args.rows,
                    "tier": args.tier,
                    "scope": "complete eager API call including allocation; five explicit warmups",
                    "results": rows,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
