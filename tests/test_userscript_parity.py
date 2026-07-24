from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from wallzero.constants import BOARD_SIZE, WALL_COUNT
from wallzero.game import State, wall_coordinates


def _walls(state: State) -> list[dict[str, int | str]]:
    walls: list[dict[str, int | str]] = []
    for index in range(WALL_COUNT):
        bit = 1 << index
        x, y = wall_coordinates(index)
        if state.horizontal & bit:
            walls.append({"o": "h", "x": x, "y": y})
        if state.vertical & bit:
            walls.append({"o": "v", "x": x, "y": y})
    return walls


def _wallz_state(state: State) -> dict[str, object]:
    p1_x, p1_y = state.pawns[0] % BOARD_SIZE, state.pawns[0] // BOARD_SIZE
    p2_x, p2_y = state.pawns[1] % BOARD_SIZE, state.pawns[1] // BOARD_SIZE
    winner = state.winner
    return {
        "pawns": {
            "p1": {"x": p1_x, "y": p1_y},
            "p2": {"x": p2_x, "y": p2_y},
        },
        "turn": "p1" if state.to_play == 0 else "p2",
        "walls": _walls(state),
        "wallsRemaining": {
            "p1": state.walls_remaining[0],
            "p2": state.walls_remaining[1],
        },
        "winner": None if winner is None else f"p{winner + 1}",
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_python_rules_match_the_existing_userscript_across_random_play() -> None:
    rng = random.Random(20260722)
    states: list[State] = []
    state = State.initial()
    while len(states) < 24:
        states.append(state)
        actions = state.legal_actions()
        if not actions or state.ply >= 80:
            state = State.initial()
            continue
        state = state.play(rng.choice(actions), validate=False)

    oracle = Path(__file__).with_name("js_rule_oracle.cjs")
    completed = subprocess.run(
        ["node", str(oracle)],
        input=json.dumps([_wallz_state(item) for item in states]),
        text=True,
        capture_output=True,
        check=True,
    )
    javascript_actions = json.loads(completed.stdout)

    for position, expected in zip(states, javascript_actions, strict=True):
        assert sorted(position.legal_actions()) == expected
