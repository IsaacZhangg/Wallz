from __future__ import annotations

import numpy as np

from wallzero.constants import ACTION_SIZE
from wallzero.encoding import absolute_action, canonical_action
from wallzero.game import State, cell, pawn_action
from wallzero.mcts import MCTSConfig, UniformEvaluator
from wallzero.selfplay import SelfPlayConfig, generate_self_play


class OscillatingEvaluator:
    """Prefer shuffling between the start cell and the cell ahead of it."""

    def __call__(self, states: list[State]) -> tuple[np.ndarray, np.ndarray]:
        logits = np.zeros((len(states), ACTION_SIZE), dtype=np.float32)
        for index, state in enumerate(states):
            player = state.to_play
            home = 4 if player == 0 else 76
            forward = 13 if player == 0 else 67
            target = forward if state.pawns[player] == home else home
            if target != state.pawns[1 - player]:
                logits[index, canonical_action(state, pawn_action(target))] = 50.0
        return logits, np.zeros(len(states), dtype=np.float32)


def test_repetition_draws_are_reported_separately() -> None:
    _, stats = generate_self_play(
        OscillatingEvaluator(),
        MCTSConfig(simulations=8, dirichlet_fraction=0.0),
        SelfPlayConfig(
            games=2,
            parallel_games=2,
            temperature_moves=0,
            endgame_temperature=0.0,
            repetition_limit=3,
            seed=5,
        ),
    )

    assert stats.games == 2
    assert stats.draws == 2
    assert stats.repetition_draws == 2
    assert stats.max_ply_draws == 0


def test_max_ply_draws_are_reported_separately() -> None:
    _, stats = generate_self_play(
        UniformEvaluator(),
        MCTSConfig(simulations=2, max_plies=6),
        SelfPlayConfig(
            games=2,
            parallel_games=2,
            temperature_moves=2,
            repetition_limit=8,
            seed=6,
        ),
    )

    assert stats.games == 2
    assert stats.draws == 2
    assert stats.max_ply_draws == 2
    assert stats.repetition_draws == 0


def test_solver_finishes_wall_free_games_decisively() -> None:
    start = State(
        pawns=(cell(4, 2), cell(4, 6)),
        walls_remaining=(0, 0),
        to_play=0,
    )
    examples, stats = generate_self_play(
        UniformEvaluator(),
        MCTSConfig(simulations=2),
        SelfPlayConfig(
            games=4,
            parallel_games=4,
            temperature_moves=0,
            repetition_limit=3,
            seed=7,
        ),
        initial_state=start,
    )

    assert stats.games == 4
    assert stats.draws == 0
    assert stats.player_one_wins + stats.player_two_wins == 4
    assert stats.solver_moves == stats.positions
    for example in examples:
        support = np.flatnonzero(example.policy)
        assert support.size >= 1
        for canonical in support:
            action = absolute_action(example.state, int(canonical))
            assert action in example.state.legal_actions()
