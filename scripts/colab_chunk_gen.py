"""Generate one full generation of deep self-play data in a single chunk.

96 games at 512 simulations with the preset temperature schedule — the same
data distribution as four colab_chunk_deep.py chunks, packaged as one shard so
worker stragglers and spawn overhead amortize across the whole generation.
"""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

run_self_play_chunk(
    "/content/wallzero-output",
    preset("colab"),
    games=96,
    simulations=512,
    temperature_moves=24,
    endgame_temperature=0.05,
    workers=10,
    leaf_batch=8,
    device_name="cuda",
)
