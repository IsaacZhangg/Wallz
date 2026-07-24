from __future__ import annotations

import numpy as np
import pytest

from wallzero.constants import (
    CELL_COUNT,
    HORIZONTAL_ACTION_OFFSET,
    VERTICAL_ACTION_OFFSET,
    WALL_COUNT,
    WALL_GRID_SIZE,
)
from wallzero.game import (
    State,
    cell,
    exact_pawn_race,
    horizontal_action,
    vertical_action,
    wall_coordinates,
    wall_index,
)


def test_initial_position_has_the_expected_full_board_action_space() -> None:
    state = State.initial()

    assert state.legal_pawn_actions() == (cell(3, 0), cell(5, 0), cell(4, 1))
    assert len(state.legal_actions()) == 131
    assert state.shortest_distance(0) == 8
    assert state.shortest_distance(1) == 8


def test_straight_and_diagonal_jump_rules() -> None:
    straight = State(pawns=(cell(4, 4), cell(4, 5)), to_play=0)
    assert cell(4, 6) in straight.legal_pawn_actions()
    assert cell(4, 5) not in straight.legal_pawn_actions()

    blocked = State(
        pawns=(cell(4, 4), cell(4, 5)),
        horizontal=1 << wall_index(4, 5),
        walls_remaining=(9, 10),
        to_play=0,
    )
    assert cell(3, 5) in blocked.legal_pawn_actions()
    assert cell(5, 5) in blocked.legal_pawn_actions()
    assert cell(4, 6) not in blocked.legal_pawn_actions()


def test_walls_cannot_cross_or_overlap() -> None:
    state = State.initial().play(horizontal_action(3, 3))
    legal = set(state.legal_actions())

    assert vertical_action(3, 3) not in legal
    assert horizontal_action(2, 3) not in legal
    assert horizontal_action(4, 3) not in legal
    assert horizontal_action(3, 4) in legal


def test_path_blocking_wall_is_illegal() -> None:
    horizontal = 0
    for x in (0, 2, 4, 6):
        horizontal |= 1 << wall_index(x, 0)
    state = State(
        horizontal=horizontal,
        walls_remaining=(6, 10),
        to_play=0,
    )

    assert horizontal_action(7, 0) not in state.legal_actions()
    assert state.shortest_distance(0) > 0


def test_play_updates_turn_counts_and_terminal_value() -> None:
    state = State(
        pawns=(cell(4, 7), cell(4, 1)),
        walls_remaining=(2, 3),
        to_play=0,
        ply=42,
    )
    terminal = state.play(cell(4, 8))

    assert terminal.winner == 0
    assert terminal.to_play == 1
    assert terminal.terminal_value() == -1.0
    assert terminal.legal_actions() == ()


def test_illegal_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="illegal action"):
        State.initial().play(CELL_COUNT - 1)


def test_action_layout_matches_userscript_contract() -> None:
    assert horizontal_action(0, 0) == HORIZONTAL_ACTION_OFFSET
    assert horizontal_action(7, 7) == HORIZONTAL_ACTION_OFFSET + 63
    assert vertical_action(0, 0) == HORIZONTAL_ACTION_OFFSET + 64


def test_edge_of_board_jump_falls_back_to_diagonals() -> None:
    state = State(pawns=(cell(4, 7), cell(4, 8)), to_play=0)
    moves = state.legal_pawn_actions()

    assert cell(3, 8) in moves
    assert cell(5, 8) in moves
    assert cell(4, 8) not in moves


def test_sideways_jump_over_an_adjacent_opponent() -> None:
    state = State(pawns=(cell(3, 4), cell(4, 4)), to_play=0)
    assert cell(5, 4) in state.legal_pawn_actions()


def test_diagonal_jump_respects_side_walls() -> None:
    state = State(
        pawns=(cell(4, 4), cell(4, 5)),
        horizontal=1 << wall_index(4, 5),
        vertical=1 << wall_index(3, 4),
        walls_remaining=(9, 9),
        to_play=0,
    )
    moves = state.legal_pawn_actions()

    assert cell(4, 6) not in moves
    assert cell(3, 5) not in moves
    assert cell(5, 5) in moves


def _brute_force_wall_actions(state: State) -> set[int]:
    """Re-derive legal wall placements from first principles for comparison."""
    if state.walls_remaining[state.to_play] == 0:
        return set()
    legal: set[int] = set()
    for index in range(WALL_COUNT):
        x, y = wall_coordinates(index)
        east_neighbor = (
            state.horizontal & (1 << wall_index(x + 1, y))
            if x + 1 < WALL_GRID_SIZE
            else 0
        )
        horizontal_blocked = (
            state.horizontal & (1 << index)
            or state.vertical & (1 << index)
            or (x > 0 and state.horizontal & (1 << wall_index(x - 1, y)))
            or east_neighbor
        )
        vertical_blocked = (
            state.horizontal & (1 << index)
            or state.vertical & (1 << index)
            or (y > 0 and state.vertical & (1 << wall_index(x, y - 1)))
            or (y + 1 < WALL_GRID_SIZE and state.vertical & (1 << wall_index(x, y + 1)))
        )
        for blocked, bits, offset in (
            (horizontal_blocked, "horizontal", HORIZONTAL_ACTION_OFFSET),
            (vertical_blocked, "vertical", VERTICAL_ACTION_OFFSET),
        ):
            if blocked:
                continue
            remaining = list(state.walls_remaining)
            remaining[state.to_play] -= 1
            new_horizontal = 1 << index if bits == "horizontal" else 0
            new_vertical = 1 << index if bits == "vertical" else 0
            candidate = State(
                pawns=state.pawns,
                walls_remaining=(remaining[0], remaining[1]),
                horizontal=state.horizontal | new_horizontal,
                vertical=state.vertical | new_vertical,
                to_play=1 - state.to_play,
                ply=state.ply + 1,
            )
            if (
                candidate.shortest_distance(0) >= 0
                and candidate.shortest_distance(1) >= 0
            ):
                legal.add(offset + index)
    return legal


def test_wall_generation_matches_brute_force_on_random_positions() -> None:
    rng = np.random.default_rng(20260722)
    state = State.initial()
    checked = 0
    for _ in range(60):
        generated = {action for action in state.legal_actions() if action >= 81}
        assert generated == _brute_force_wall_actions(state)
        checked += 1
        legal = state.legal_actions()
        if not legal:
            break
        state = state.play(int(rng.choice(legal)))
    assert checked >= 40


def test_exact_no_wall_retrograde_solves_an_immediate_race_win() -> None:
    state = State(
        pawns=(cell(4, 7), cell(4, 2)),
        walls_remaining=(0, 0),
        to_play=0,
    )

    assert exact_pawn_race(state) == (1, 1)
