"""Generate three kata-recipe self-play generations in one call.

Bundling several chunks into a single exec removes CLI round-trip dead time
between them; each chunk still fsyncs its own shard, so a session death costs
at most the chunk in flight.
"""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

_PRESET = preset("colab")

for _ in range(3):
    run_self_play_chunk(
        "/content/wallzero-output",
        _PRESET,
        games=192,
        simulations=800,
        temperature_moves=24,
        endgame_temperature=0.05,
        workers=10,
        leaf_batch=16,
        full_search_probability=0.25,
        fast_simulations=200,
        surprise_weighting=True,
        forced_playout_scale=2.0,
        root_policy_temperature=1.2,
        dirichlet_concentration=10.83,
        device_name="cuda",
    )
