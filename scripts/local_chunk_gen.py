"""Generate one full generation of deep self-play data on the local Mac.

Identical data recipe to colab_chunk_gen.py (96 games, 512 simulations,
preset temperature schedule, leaf_batch=8); the M4 Pro's cores make local
generation as fast as the Colab A100 for this CPU-bound phase, at zero
compute-unit cost. Chunks land in the local campaign mirror, whose replay
directory is uploaded as the resume bundle for the next A100 session.

Eight workers stay on the performance cores and leave headroom for
interactive use.
"""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

if __name__ == "__main__":
    # The guard is required: spawn-based workers re-import __main__, and an
    # unguarded module would re-enter the campaign entry point in every actor.
    run_self_play_chunk(
        "artifacts/runs/a100-campaign-local/wallzero-output",
        preset("colab"),
        games=96,
        simulations=512,
        temperature_moves=24,
        endgame_temperature=0.05,
        workers=8,
        leaf_batch=8,
        device_name="mps",
    )
