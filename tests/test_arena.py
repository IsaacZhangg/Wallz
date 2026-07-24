from __future__ import annotations

from wallzero.arena import ArenaConfig, _paired_opening, evaluate_candidate
from wallzero.game import State, cell
from wallzero.mcts import MCTSConfig, UniformEvaluator


def test_wall_free_positions_are_adjudicated_exactly() -> None:
    start = State(
        pawns=(cell(4, 1), cell(4, 7)),
        walls_remaining=(0, 0),
        to_play=0,
    )
    result = evaluate_candidate(
        UniformEvaluator(),
        UniformEvaluator(),
        MCTSConfig(simulations=2, dirichlet_fraction=0.0),
        ArenaConfig(games=2, parallel_games=2, seed=3),
        initial_state=start,
    )

    assert result.games == 2
    assert result.adjudications == 2
    assert result.draws == 0
    assert result.candidate_wins + result.incumbent_wins == 2


def test_arena_draw_causes_default_to_zero() -> None:
    result = evaluate_candidate(
        UniformEvaluator(),
        UniformEvaluator(),
        MCTSConfig(simulations=2, max_plies=4),
        ArenaConfig(games=2, parallel_games=2, seed=4, opening_random_plies=0),
    )

    assert result.games == 2
    assert result.draws == result.repetition_draws + result.max_ply_draws


def test_paired_openings_are_shared_within_a_color_pair() -> None:
    config = ArenaConfig(games=8, parallel_games=4, seed=12)
    base = State.initial()

    for pair in range(4):
        first = _paired_opening(base, pair, config)
        second = _paired_opening(base, pair, config)
        assert first == second
        assert first.ply == config.opening_random_plies

    openings = {_paired_opening(base, pair, config).position_key for pair in range(4)}
    assert len(openings) > 1


def test_zero_opening_plies_preserves_the_initial_position() -> None:
    config = ArenaConfig(games=2, parallel_games=2, seed=13, opening_random_plies=0)
    assert _paired_opening(State.initial(), 0, config) == State.initial()
