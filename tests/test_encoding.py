from __future__ import annotations

import numpy as np

from wallzero.constants import ACTION_SIZE, BOARD_SIZE, INPUT_PLANES
from wallzero.encoding import (
    absolute_action,
    canonical_action,
    encode_state,
    mirror_action,
    mirror_policy,
    mirror_state,
)
from wallzero.game import State, cell, horizontal_action, vertical_action


def test_canonical_encoding_rotates_and_swaps_the_side_to_move() -> None:
    state = State(
        pawns=(cell(2, 3), cell(6, 7)),
        walls_remaining=(7, 4),
        to_play=1,
        ply=11,
    )
    encoded = encode_state(state)

    assert encoded.shape == (INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert encoded[0, 1, 2] == 1.0
    assert encoded[1, 5, 6] == 1.0
    assert np.all(encoded[6] == 0.4)
    assert np.all(encoded[7] == 0.7)


def test_action_canonicalization_is_an_involution() -> None:
    state = State(to_play=1)
    actions = [cell(2, 3), horizontal_action(1, 6), vertical_action(7, 0)]

    for action in actions:
        assert absolute_action(state, canonical_action(state, action)) == action


def test_horizontal_mirror_is_consistent_for_states_and_policies() -> None:
    state = State.initial().play(horizontal_action(1, 2)).play(vertical_action(6, 4))
    assert mirror_state(mirror_state(state)) == state

    policy = np.arange(ACTION_SIZE, dtype=np.float32)
    assert np.array_equal(mirror_policy(mirror_policy(policy)), policy)
    for action in range(ACTION_SIZE):
        assert mirror_action(mirror_action(action)) == action
