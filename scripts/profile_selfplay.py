"""Profile rule/search overhead independently of neural inference."""

from __future__ import annotations

import cProfile
import pstats

from wallzero.mcts import MCTSConfig, UniformEvaluator
from wallzero.selfplay import SelfPlayConfig, generate_self_play


def workload() -> None:
    _, stats = generate_self_play(
        UniformEvaluator(),
        MCTSConfig(simulations=32, max_plies=96),
        SelfPlayConfig(
            games=8,
            parallel_games=8,
            temperature_moves=64,
            repetition_limit=8,
            seed=20260722,
        ),
    )
    print(stats)


profile = cProfile.Profile()
profile.enable()
workload()
profile.disable()
pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(30)
