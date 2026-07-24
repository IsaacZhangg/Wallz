"""Canonical neural representation and exact action symmetries."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from wallzero.constants import (
    ACTION_SIZE,
    BOARD_SIZE,
    CELL_COUNT,
    HORIZONTAL_ACTION_OFFSET,
    INPUT_PLANES,
    MAX_GAME_PLIES,
    MAX_WALLS,
    VERTICAL_ACTION_OFFSET,
    WALL_COUNT,
    WALL_GRID_SIZE,
)
from wallzero.game import State, blocked_edge_masks, decode_action

FloatArray = NDArray[np.float32]


def rotate_cell(index: int) -> int:
    return CELL_COUNT - 1 - index


def rotate_wall_index(index: int) -> int:
    return WALL_COUNT - 1 - index


def mirror_cell(index: int) -> int:
    x = index % BOARD_SIZE
    y = index // BOARD_SIZE
    return y * BOARD_SIZE + (BOARD_SIZE - 1 - x)


def mirror_wall_index(index: int) -> int:
    x = index % WALL_GRID_SIZE
    y = index // WALL_GRID_SIZE
    return y * WALL_GRID_SIZE + (WALL_GRID_SIZE - 1 - x)


def _remap_bits(bits: int, mapper: tuple[int, ...]) -> int:
    mapped = 0
    while bits:
        least = bits & -bits
        index = least.bit_length() - 1
        mapped |= 1 << mapper[index]
        bits ^= least
    return mapped


ROTATED_WALL_INDEX = tuple(rotate_wall_index(i) for i in range(WALL_COUNT))
MIRRORED_WALL_INDEX = tuple(mirror_wall_index(i) for i in range(WALL_COUNT))


@lru_cache(maxsize=262_144)
def rotate_wall_bits(bits: int) -> int:
    return _remap_bits(bits, ROTATED_WALL_INDEX)


@lru_cache(maxsize=262_144)
def mirror_wall_bits(bits: int) -> int:
    return _remap_bits(bits, MIRRORED_WALL_INDEX)


def canonical_action(state: State, action: int) -> int:
    """Map an absolute action into the side-to-move's forward-facing frame."""
    if state.to_play == 0:
        return action
    kind, target = decode_action(action)
    if kind == "pawn":
        return rotate_cell(target)
    mapped = rotate_wall_index(target)
    offset = (
        HORIZONTAL_ACTION_OFFSET if kind == "horizontal" else VERTICAL_ACTION_OFFSET
    )
    return offset + mapped


def absolute_action(state: State, action: int) -> int:
    """Invert :func:`canonical_action`; a 180-degree rotation is self-inverse."""
    return canonical_action(state, action)


def mirror_action(action: int) -> int:
    kind, target = decode_action(action)
    if kind == "pawn":
        return mirror_cell(target)
    mapped = mirror_wall_index(target)
    offset = (
        HORIZONTAL_ACTION_OFFSET if kind == "horizontal" else VERTICAL_ACTION_OFFSET
    )
    return offset + mapped


def mirror_policy(policy: FloatArray) -> FloatArray:
    if policy.shape != (ACTION_SIZE,):
        raise ValueError(f"expected policy shape {(ACTION_SIZE,)}, got {policy.shape}")
    mirrored = np.empty_like(policy)
    for action in range(ACTION_SIZE):
        mirrored[mirror_action(action)] = policy[action]
    return mirrored


def mirror_state(state: State) -> State:
    return State(
        pawns=(mirror_cell(state.pawns[0]), mirror_cell(state.pawns[1])),
        walls_remaining=state.walls_remaining,
        horizontal=mirror_wall_bits(state.horizontal),
        vertical=mirror_wall_bits(state.vertical),
        to_play=state.to_play,
        ply=state.ply,
    )


def _set_bit_plane(plane: FloatArray, bits: int, width: int) -> None:
    while bits:
        least = bits & -bits
        index = least.bit_length() - 1
        plane[index // width, index % width] = 1.0
        bits ^= least


def encode_state(state: State) -> FloatArray:
    """Encode a position with the side to move always racing toward row 8.

    The planes contain only rule-visible state and geometry. No human games,
    opening priors, path heuristics, or handcrafted evaluations enter the net.
    """
    if state.to_play == 0:
        current = state.pawns[0]
        opponent = state.pawns[1]
        current_walls, opponent_walls = state.walls_remaining
        horizontal = state.horizontal
        vertical = state.vertical
    else:
        current = rotate_cell(state.pawns[1])
        opponent = rotate_cell(state.pawns[0])
        current_walls = state.walls_remaining[1]
        opponent_walls = state.walls_remaining[0]
        horizontal = rotate_wall_bits(state.horizontal)
        vertical = rotate_wall_bits(state.vertical)

    planes = np.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    planes[0, current // BOARD_SIZE, current % BOARD_SIZE] = 1.0
    planes[1, opponent // BOARD_SIZE, opponent % BOARD_SIZE] = 1.0
    _set_bit_plane(planes[2, :WALL_GRID_SIZE, :WALL_GRID_SIZE], horizontal, 8)
    _set_bit_plane(planes[3, :WALL_GRID_SIZE, :WALL_GRID_SIZE], vertical, 8)

    south, east = blocked_edge_masks(horizontal, vertical)
    _set_bit_plane(planes[4], south, BOARD_SIZE)
    _set_bit_plane(planes[5], east, BOARD_SIZE)
    planes[6].fill(current_walls / MAX_WALLS)
    planes[7].fill(opponent_walls / MAX_WALLS)
    for action in state.legal_pawn_actions():
        canonical = canonical_action(state, action)
        planes[8, canonical // BOARD_SIZE, canonical % BOARD_SIZE] = 1.0
    planes[9].fill(min(state.ply, MAX_GAME_PLIES) / MAX_GAME_PLIES)
    planes[10] = np.broadcast_to(
        np.linspace(0.0, 1.0, BOARD_SIZE, dtype=np.float32),
        (BOARD_SIZE, BOARD_SIZE),
    )
    planes[11] = planes[10].T
    planes[12].fill(1.0)
    return planes
