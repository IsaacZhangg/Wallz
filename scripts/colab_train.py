"""Non-interactive Colab entry point with explicit accelerator reporting."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from wallzero.pipeline import preset, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("smoke", "bootstrap", "colab", "expert"),
        default="colab",
    )
    parser.add_argument("--output", type=Path, default=Path("/content/wallzero-output"))
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()

    hardware = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    }
    print(json.dumps({"hardware": hardware}, sort_keys=True), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("the Colab training entry point requires a GPU runtime")
    run_training(
        args.output,
        preset(args.preset),
        device_name="cuda",
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
