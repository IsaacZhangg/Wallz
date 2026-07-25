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
    # Measured on the 65K kata-only window (round32-kataonly-metrics.jsonl vs
    # round33-lowreuse-metrics.jsonl): 600 steps left the net underfit
    # (policy loss 2.01 vs 1.54) and scored 0.475 vs 0.500 in the same gate.
    # KataGo's 1-4 epoch norm reflects its millions-of-positions firehose, not
    # a universal law; with a small window, steps-to-convergence governs.
    training_steps=4_000,
    arena_games=60,
    arena_workers=10,
    candidate_network=NetworkConfig(
        channels=256, blocks=24, value_hidden=512, distance_head=True
    ),
    warm_start=True,
    device_name="cuda",
)
