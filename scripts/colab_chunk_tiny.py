"""Prove one durable A100 round trip with a deliberately tiny self-play chunk."""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

run_self_play_chunk(
    "/content/wallzero-output",
    preset("colab"),
    games=4,
    simulations=64,
    device_name="cuda",
)
