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

"""A minimal device/public-API smoke test, not full numerical acceptance."""

import argparse
import logging


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", required=True)
    p.add_argument("--tier", choices=["baseline", "tle", "sm90"], default="baseline")
    args = p.parse_args()
    import torch

    if args.device.startswith("musa"):
        import torch_musa  # noqa: F401
    import flaggems_lightx2v as ops

    device = torch.device(args.device)
    backend = getattr(torch, device.type)
    if not backend.is_available():
        raise RuntimeError(f"{device.type} is unavailable")
    backend.set_device(device)
    with torch.no_grad():
        weight = torch.ones(5376, device=device, dtype=torch.bfloat16)
        x = torch.ones((44, 5376), device=device, dtype=torch.bfloat16)
        y = ops.h3_rms_norm(x, weight, 1e-6, tier=args.tier)
        if not torch.equal(y, x):
            raise RuntimeError("Constant-one RMS smoke check failed")
        x.zero_()
        y = ops.h3_rms_norm(x, weight, 1e-6, tier=args.tier)
        if not torch.equal(y, x):
            raise RuntimeError("Constant-zero RMS smoke check failed")
    backend.synchronize()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "Public RMS smoke passed: device=%s tier=%s", device, args.tier
    )


if __name__ == "__main__":
    main()
