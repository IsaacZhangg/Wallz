"""Train one candidate on the clean-schedule window and gate it, leaf-batched.

Identical training recipe to colab_round_big.py (4000 steps, 14K window,
60-game arena at 256 simulations, strict-win threshold 0.5); the arena search
uses leaf batching to cut wall time. The independent multi-seed control suite
remains the arbiter for strength claims.
"""

from dataclasses import replace

from wallzero.campaign import run_training_round
from wallzero.pipeline import preset

_base = preset("colab")

run_training_round(
    "/content/wallzero-output",
    replace(
        _base,
        replay_window=14_000,
        arena=replace(_base.arena, promotion_threshold=0.5),
        arena_mcts=replace(_base.arena_mcts, simulations=256, leaf_batch=8),
    ),
    training_steps=4_000,
    arena_games=60,
    arena_workers=10,
    device_name="cuda",
)
