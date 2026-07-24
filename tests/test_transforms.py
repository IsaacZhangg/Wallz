from __future__ import annotations

import numpy as np

from wallzero.constants import ACTION_SIZE
from wallzero.encoding import (
    canonical_action,
    mirror_action,
    mirror_state,
)
from wallzero.game import State


def _random_states(seed: int, count: int, plies: int) -> list[State]:
    rng = np.random.default_rng(seed)
    states: list[State] = []
    for _ in range(count):
        state = State.initial()
        for _ in range(plies):
            legal = state.legal_actions()
            if not legal:
                break
            state = state.play(int(rng.choice(legal)))
        states.append(state)
    return states


def test_canonical_action_is_a_bijection_for_both_players() -> None:
    for to_play in (0, 1):
        state = State(to_play=to_play, pawns=(4, 76))
        mapped = {canonical_action(state, action) for action in range(ACTION_SIZE)}
        assert mapped == set(range(ACTION_SIZE))


def test_mirror_action_is_a_bijection() -> None:
    mapped = {mirror_action(action) for action in range(ACTION_SIZE)}
    assert mapped == set(range(ACTION_SIZE))


def test_mirror_state_preserves_legality_exactly() -> None:
    for state in _random_states(seed=11, count=6, plies=24):
        mirrored = mirror_state(state)
        expected = {mirror_action(action) for action in state.legal_actions()}
        assert set(mirrored.legal_actions()) == expected


def test_mirror_commutes_with_play() -> None:
    rng = np.random.default_rng(13)
    for state in _random_states(seed=17, count=6, plies=16):
        legal = state.legal_actions()
        if not legal:
            continue
        action = int(rng.choice(legal))
        played_then_mirrored = mirror_state(state.play(action))
        mirrored_then_played = mirror_state(state).play(mirror_action(action))
        assert played_then_mirrored == mirrored_then_played


def test_mirror_preserves_shortest_distances() -> None:
    for state in _random_states(seed=23, count=6, plies=30):
        mirrored = mirror_state(state)
        assert mirrored.shortest_distance(0) == state.shortest_distance(0)
        assert mirrored.shortest_distance(1) == state.shortest_distance(1)
