"""One gateless kata training round on a worker node.

Same round as scripts/colab_round_kata.py (60K window, 3,000 steps, always
adopt), but env-driven like the other node scripts so it runs on any machine
that holds the replay shards and best.pt. The distance head already exists on
the incumbent, so no candidate_network/warm_start is needed here.

    WALLZERO_OUTPUT=~/wallzero/output/wallzero-output \
    python scripts/node_round_kata.py
"""

import os
from dataclasses import replace
from pathlib import Path

from wallzero.campaign import run_training_round
from wallzero.pipeline import preset

if __name__ == "__main__":
    output = Path(
        os.environ.get("WALLZERO_OUTPUT", "artifacts/runs/node/wallzero-output")
    ).expanduser()
    run_training_round(
        output,
        replace(preset("colab"), replay_window=60_000),
        # 3,000 steps measured as the convergence point for a ~60K window
        # (round33-lowreuse-metrics.jsonl falsified the low-step hypothesis).
        training_steps=int(os.environ.get("WALLZERO_STEPS", "3000")),
        arena_games=0,
        device_name=os.environ.get("WALLZERO_DEVICE", "cuda"),
    )
