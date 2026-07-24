from __future__ import annotations

import numpy as np
import pytest

from wallzero.game import (
    State,
    cell,
    exact_pawn_race,
    exact_pawn_race_moves,
    horizontal_action,
    vertical_action,
)


def test_winning_side_receives_only_fastest_winning_actions() -> None:
    state = State(
        pawns=(cell(4, 7), cell(4, 2)),
        walls_remaining=(0, 0),
        to_play=0,
    )
    outcome, distance, actions = exact_pawn_race_moves(state)

    assert (outcome, distance) == (1, 1)
    assert actions == (cell(4, 8),)


def test_losing_side_receives_maximum_resistance_actions() -> None:
    state = State(
        pawns=(cell(4, 7), cell(4, 2)),
        walls_remaining=(0, 0),
        to_play=1,
    )
    outcome, distance, actions = exact_pawn_race_moves(state)

    assert outcome == -1
    assert actions
    for action in actions:
        successor = state.play(action)
        next_outcome, next_distance, _ = exact_pawn_race_moves(successor)
        assert next_outcome == 1
        assert next_distance == distance - 1


def test_solver_rejects_states_with_wall_reserves() -> None:
    with pytest.raises(ValueError, match="wall reserves"):
        exact_pawn_race_moves(State.initial())


def _random_wall_free_state(rng: np.random.Generator) -> State | None:
    walls = State.initial()
    for _ in range(int(rng.integers(0, 12))):
        wall_actions = [action for action in walls.legal_actions() if action >= 81]
        if not wall_actions:
            break
        walls = walls.play(int(rng.choice(wall_actions)))
    pawn_one = int(rng.integers(0, 72))
    pawn_two = int(rng.integers(9, 81))
    if pawn_one == pawn_two:
        return None
    try:
        state = State(
            pawns=(pawn_one, pawn_two),
            walls_remaining=(0, 0),
            horizontal=walls.horizontal,
            vertical=walls.vertical,
            to_play=int(rng.integers(0, 2)),
        )
    except ValueError:
        return None
    if state.shortest_distance(0) < 0 or state.shortest_distance(1) < 0:
        return None
    return state


def test_optimal_playout_matches_predicted_outcome_and_distance() -> None:
    rng = np.random.default_rng(20260723)
    verified = 0
    while verified < 12:
        state = _random_wall_free_state(rng)
        if state is None:
            continue
        outcome, distance, actions = exact_pawn_race_moves(state)
        if outcome == 0:
            assert actions == ()
            continue
        expected_winner = state.to_play if outcome == 1 else 1 - state.to_play
        current = state
        plies = 0
        while current.terminal_value() is None:
            _, _, optimal = exact_pawn_race_moves(current)
            assert optimal, f"solved position lacks optimal actions: {current}"
            current = current.play(optimal[0])
            plies += 1
            assert plies <= distance
        assert plies == distance
        assert current.winner == expected_winner
        verified += 1


def test_solver_outcome_is_antisymmetric_with_the_leaf_oracle() -> None:
    state = State(
        pawns=(cell(0, 3), cell(8, 5)),
        walls_remaining=(0, 0),
        horizontal=1 << 27,
        to_play=0,
    )
    outcome, distance = exact_pawn_race(state)
    moves_outcome, moves_distance, actions = exact_pawn_race_moves(state)

    assert (moves_outcome, moves_distance) == (outcome, distance)
    if outcome == 1:
        for action in actions:
            next_outcome, _, _ = exact_pawn_race_moves(state.play(action))
            assert next_outcome == -1


def test_walled_geometry_changes_the_race_result() -> None:
    open_race = State(
        pawns=(cell(4, 4), cell(3, 4)),
        walls_remaining=(0, 0),
        to_play=0,
    )
    fenced = State(
        pawns=(cell(4, 4), cell(3, 4)),
        walls_remaining=(0, 0),
        horizontal=(1 << 35) | (1 << 37),
        to_play=0,
    )
    open_outcome, open_distance, _ = exact_pawn_race_moves(open_race)
    fenced_outcome, fenced_distance, _ = exact_pawn_race_moves(fenced)

    assert open_outcome == 1
    assert (fenced_outcome, fenced_distance) != (open_outcome, open_distance)


def test_wall_actions_are_never_reported_when_reserves_are_empty() -> None:
    state = State(
        pawns=(cell(2, 2), cell(6, 6)),
        walls_remaining=(0, 0),
        to_play=0,
    )
    _, _, actions = exact_pawn_race_moves(state)
    assert all(action < 81 for action in actions)
    assert horizontal_action(0, 0) not in actions
    assert vertical_action(0, 0) not in actions
