"""Generate one KataGo-recipe self-play generation on the A100.

Playout cap randomization (25% full searches at 800 simulations with forced
playouts, pruned policy targets, shaped noise, and root softmax temperature;
75% fast value-only searches at 200), policy surprise weighting, and exact
distance targets on every position. 192 games per chunk.
"""

from wallzero.campaign import run_self_play_chunk
from wallzero.pipeline import preset

run_self_play_chunk(
    "/content/wallzero-output",
    preset("colab"),
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
