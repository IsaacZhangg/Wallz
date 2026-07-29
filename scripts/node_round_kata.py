"""One gateless kata training round on a worker node (era-3 growing window).

Era-2 post-mortem (2026-07-28): the fixed newest-60K window with 3,000 steps
meant ~26 epochs per round over the last ~1,100 games — each round overwrote
the last, knowledge never accumulated, and the lineage measured flat across
45 rounds (r59 vs r31 0.47, vs r14 0.48; classical KPI 0-40) while drifting
into a wall-heavy style. AlphaZero/KataGo train on windows orders of
magnitude larger (500K games / growing multi-million-position windows).

Era 3 trains on ALL accumulated shards, with steps scaled to ~3 epochs per
round and capped at 3,000 (the ~50-minute GPU budget on the 1080). Past
~512K positions the cap means each round samples under 3 epochs — the
KataGo-like regime where per-round reuse keeps falling as data grows.

    WALLZERO_OUTPUT=~/wallzero/output/wallzero-output \
    python scripts/node_round_kata.py
"""

import os
from dataclasses import replace
from pathlib import Path

from wallzero.campaign import run_training_round
from wallzero.pipeline import preset

MEAN_SHARD_POSITIONS = 5_100  # measured era-2 mean; only steers the step count
BATCH = 512
TARGET_EPOCHS = 3

if __name__ == "__main__":
    output = Path(
        os.environ.get("WALLZERO_OUTPUT", "artifacts/runs/node/wallzero-output")
    ).expanduser()
    shards = len(list((output / "replay").glob("chunk-*.npz")))
    est_positions = shards * MEAN_SHARD_POSITIONS
    steps_env = os.environ.get("WALLZERO_STEPS")
    steps = (
        int(steps_env)
        if steps_env
        else min(3_000, max(500, est_positions * TARGET_EPOCHS // BATCH))
    )
    run_training_round(
        output,
        replace(preset("colab"), replay_window=10_000_000),
        training_steps=steps,
        arena_games=0,
        device_name=os.environ.get("WALLZERO_DEVICE", "cuda"),
    )
