"""Generate one durable shard: 512-sim search, preset temperature schedule."""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

run_self_play_chunk(
    "/content/wallzero-output",
    preset("colab"),
    games=24,
    simulations=512,
    temperature_moves=24,
    endgame_temperature=0.05,
    device_name="cuda",
)
