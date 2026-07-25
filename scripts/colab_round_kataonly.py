"""Diagnosis round: retrain from round-24 on the kata-only window, gated.

The replay directory holds only chunks 69-75 (pure kata-recipe data:
cap-randomized searches, pruned policy targets, surprise weights, exact
distance targets). Candidate = round-24 trunk warm-started with the distance
head, 4000 steps, judged by the restored 60-game arena against round 24.
Isolates whether legacy-data dilution explains the flat overnight lineage.
"""

from dataclasses import replace

from wallzero.campaign import run_training_round
from wallzero.network import NetworkConfig
from wallzero.pipeline import preset

_base = preset("colab")

run_training_round(
    "/content/wallzero-output",
    replace(
        _base,
        replay_window=65_000,
        arena=replace(_base.arena, promotion_threshold=0.5),
        arena_mcts=replace(_base.arena_mcts, simulations=256, leaf_batch=16),
    ),
    # KataGo reuse ratio: ~4 epochs over the window (window*4/batch), not
    # the fixed 2M samples that silently overtrained every earlier round.
    training_steps=600,
    arena_games=60,
    arena_workers=10,
    candidate_network=NetworkConfig(
        channels=256, blocks=24, value_hidden=512, distance_head=True
    ),
    warm_start=True,
    device_name="cuda",
)
