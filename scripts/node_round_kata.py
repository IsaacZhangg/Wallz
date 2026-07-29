"""One gateless kata training round on a worker node (era-3 growing window).

Era-2 post-mortem (2026-07-28): the fixed newest-60K window with 3,000 steps
meant ~26 epochs per round over the last ~1,100 games — each round overwrote
the last, knowledge never accumulated, and the lineage measured flat across
45 rounds (r59 vs r31 0.47, vs r14 0.48; classical KPI 0-40) while drifting
into a wall-heavy style. AlphaZero/KataGo train on windows orders of
magnitude larger (500K games / growing multi-million-position windows).

Era 3 trains on ALL accumulated shards, with the step count governed by
KataGo's sample-reuse rule, not by epochs over the window: at most
MAX_TRAIN_PER_DATA training samples per fresh data row
(synchronous_loop.sh uses 8 and warns larger risks overfitting; the async
large-scale rule is 4). The flywheel passes the fresh-row count via
WALLZERO_FRESH_ROWS; steps = 8 x fresh / batch, so a ~21K-position cycle
trains ~330 steps (~6 min) and the GPU spends ~90% of wall time
generating. Era 2 ran at ~77 samples per data row — nearly 10x KataGo's
warning threshold — which is the overfit-to-recent mechanism behind the
flat lineage.

    WALLZERO_OUTPUT=~/wallzero/output/wallzero-output \
    python scripts/node_round_kata.py
"""

import os
from dataclasses import replace
from pathlib import Path

from wallzero.campaign import run_training_round
from wallzero.pipeline import preset

BATCH = 512
MAX_TRAIN_PER_DATA = 8  # KataGo synchronous_loop.sh MAX_TRAIN_PER_DATA
# KataGo trains with a continuous piecewise-CONSTANT learning rate across
# the whole run — it never anneals per cycle. Our per-round warmup+cosine
# (2e-3 -> 2e-5 inside every round) was a WallZero-only invention; with
# era-3's short rounds it meant a 100x LR swing every ~7 minutes. Setting
# lr == min_lr collapses the schedule to warmup-then-flat (audit round 2,
# 2026-07-29). 3e-4 approximates the old cosine's time-weighted average.
CONSTANT_LR = float(os.environ.get("WALLZERO_LR", "3e-4"))

if __name__ == "__main__":
    output = Path(
        os.environ.get("WALLZERO_OUTPUT", "artifacts/runs/node/wallzero-output")
    ).expanduser()
    fresh = int(os.environ.get("WALLZERO_FRESH_ROWS", "21000"))
    steps_env = os.environ.get("WALLZERO_STEPS")
    steps = (
        int(steps_env)
        if steps_env
        else min(3_000, max(150, fresh * MAX_TRAIN_PER_DATA // BATCH))
    )
    base = preset("colab")
    run_training_round(
        output,
        replace(
            base,
            replay_window=10_000_000,
            train=replace(
                base.train,
                learning_rate=CONSTANT_LR,
                minimum_learning_rate=CONSTANT_LR,
            ),
        ),
        training_steps=steps,
        arena_games=0,
        device_name=os.environ.get("WALLZERO_DEVICE", "cuda"),
    )
