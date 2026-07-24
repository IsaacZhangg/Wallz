"""A100 correctness and throughput probe before a metered training iteration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from wallzero.pipeline import preset, run_training

OUTPUT = Path("/content/wallzero-smoke")

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "bf16": torch.cuda.is_bf16_supported()
            if torch.cuda.is_available()
            else False,
        },
        sort_keys=True,
    ),
    flush=True,
)
if not torch.cuda.is_available():
    raise RuntimeError("validation requires the allocated GPU")
run_training(OUTPUT, preset("smoke"), device_name="cuda", iterations=1)
