"""Generate one durable full-board self-play shard on the active A100."""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

run_self_play_chunk(
    "/content/wallzero-output",
    preset("colab"),
    games=16,
    simulations=160,
    device_name="cuda",
)
