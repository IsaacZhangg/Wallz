"""Generate one deep self-play chunk with multiprocess actors and leaf batching.

Same data recipe as colab_chunk_deep.py (24 games, 512 simulations, preset
temperature schedule); only the execution engine changed. A100 VMs expose
12 vCPUs: ten actors, one inference server thread, one spare.
"""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

run_self_play_chunk(
    "/content/wallzero-output",
    preset("colab"),
    games=24,
    simulations=512,
    temperature_moves=24,
    endgame_temperature=0.05,
    workers=10,
    leaf_batch=8,
    device_name="cuda",
)
