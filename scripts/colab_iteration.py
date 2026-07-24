"""Run exactly one additional resumable A100 training iteration."""

from __future__ import annotations

import json
from pathlib import Path

from wallzero.pipeline import preset, run_training

OUTPUT = Path("/content/wallzero-output")
STATE = OUTPUT / "run-state.json"
completed = 0
if STATE.exists():
    completed = int(
        json.loads(STATE.read_text(encoding="utf-8"))["completed_iterations"]
    )

run_training(
    OUTPUT,
    preset("colab"),
    device_name="cuda",
    iterations=completed + 1,
)
