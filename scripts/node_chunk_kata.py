"""Generate one kata-recipe self-play chunk on a worker node.

Used by the always-on generation loop on secondary machines. The output
directory, worker count, and leaf batch come from the environment so the same
script serves every node:

    WALLZERO_OUTPUT=~/wallzero/output/wallzero-output \
    WALLZERO_WORKERS=12 WALLZERO_LEAF_BATCH=16 \
    python scripts/node_chunk_kata.py
"""

import os
from pathlib import Path

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

if __name__ == "__main__":
    output = Path(
        os.environ.get("WALLZERO_OUTPUT", "artifacts/runs/node/wallzero-output")
    ).expanduser()
    run_self_play_chunk(
        output,
        preset("colab"),
        games=int(os.environ.get("WALLZERO_GAMES", "192")),
        simulations=int(os.environ.get("WALLZERO_SIMULATIONS", "800")),
        temperature_moves=24,
        endgame_temperature=0.05,
        workers=int(os.environ.get("WALLZERO_WORKERS", "12")),
        leaf_batch=int(os.environ.get("WALLZERO_LEAF_BATCH", "16")),
        full_search_probability=0.25,
        fast_simulations=200,
        surprise_weighting=True,
        forced_playout_scale=2.0,
        root_policy_temperature=1.2,
        dirichlet_concentration=10.83,
        device_name=os.environ.get("WALLZERO_DEVICE", "cuda"),
    )
