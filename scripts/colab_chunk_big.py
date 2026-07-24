"""Generate one 30M-network self-play generation on the A100 at full feed.

Same data recipe (96 games, 512 simulations, preset temperature schedule);
leaf_batch 16 doubles the states per inference call (~160 per worker
request) because the 29.65M-parameter network finally rewards large batches.
Ten actors on the VM's 12 vCPUs, one batched server on the A100.
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
    leaf_batch=16,
    device_name="cuda",
)
