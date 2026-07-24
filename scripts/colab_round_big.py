"""Train one candidate hard on the clean-schedule window, then gate it.

Four thousand steps over a 14K window of preset-temperature 512-simulation
shards; the 60-game arena records lineage while the independent multi-seed
suite remains the arbiter for strength claims.
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
    ),
    training_steps=4_000,
    arena_games=60,
    arena_simulations=256,
    device_name="cuda",
)
