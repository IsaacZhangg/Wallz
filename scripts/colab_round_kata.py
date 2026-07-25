"""One gateless KataGo-recipe training round: train the latest, always adopt.

The 60K window spans the recent large generations; strength judgments belong
exclusively to the independent suites (control gate, classical-engine KPI).
The first kata round passes candidate_network with distance_head=True and
warm_start so the incumbent trunk is kept while the auxiliary head is added;
afterwards the architecture simply carries forward.
"""

from dataclasses import replace

from wallzero.campaign import run_training_round
from wallzero.network import NetworkConfig
from wallzero.pipeline import preset

_base = preset("colab")

run_training_round(
    "/content/wallzero-output",
    replace(_base, replay_window=60_000),
    # Measured on the 65K kata-only window (round32-kataonly-metrics.jsonl vs
    # round33-lowreuse-metrics.jsonl): 600 steps left the net underfit
    # (policy loss 2.01 vs 1.54) and scored 0.475 vs 0.500 in the same gate.
    # KataGo's 1-4 epoch norm reflects its millions-of-positions firehose, not
    # a universal law; with a small window, steps-to-convergence governs.
    training_steps=3_000,
    arena_games=0,
    candidate_network=NetworkConfig(
        channels=256, blocks=24, value_hidden=512, distance_head=True
    ),
    warm_start=True,
    device_name="cuda",
)
