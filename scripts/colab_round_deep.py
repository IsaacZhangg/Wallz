"""Train and gate one candidate with a fresher window and stricter arena.

Promotion uses threshold 0.5 plus the strict wins>losses rule: any candidate
that wins the 60-game match replaces the incumbent. This prevents a stylistic
dead-end incumbent from surviving on gate variance alone.
"""

from dataclasses import replace

from wallzero.campaign import run_training_round
from wallzero.pipeline import preset

_base = preset("colab")

run_training_round(
    "/content/wallzero-output",
    replace(
        _base,
        replay_window=8_000,
        arena=replace(_base.arena, promotion_threshold=0.5),
    ),
    training_steps=1_200,
    arena_games=60,
    arena_simulations=256,
    device_name="cuda",
)
