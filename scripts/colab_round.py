"""Train and arena-gate one candidate from durable Colab replay shards."""

from wallzero.campaign import run_training_round
from wallzero.pipeline import preset

run_training_round(
    "/content/wallzero-output",
    preset("colab"),
    training_steps=600,
    arena_games=20,
    arena_simulations=192,
    device_name="cuda",
)
