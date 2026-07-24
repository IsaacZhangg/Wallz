"""Exact full-size Quoridor rules with a compact immutable bitboard state."""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeAlias

from wallzero.constants import (
    ACTION_SIZE,
    BOARD_SIZE,
    CELL_COUNT,
    HORIZONTAL_ACTION_OFFSET,
    MAX_GAME_PLIES,
    MAX_WALLS,
    VERTICAL_ACTION_OFFSET,
    WALL_COUNT,
    WALL_GRID_SIZE,
)

ActionKind: TypeAlias = Literal["pawn", "horizontal", "vertical"]

BOARD_MASK = (1 << CELL_COUNT) - 1
TOP_ROW_MASK = (1 << BOARD_SIZE) - 1
BOTTOM_ROW_MASK = TOP_ROW_MASK << (CELL_COUNT - BOARD_SIZE)
LEFT_COLUMN_MASK = sum(1 << (row * BOARD_SIZE) for row in range(BOARD_SIZE))
RIGHT_COLUMN_MASK = LEFT_COLUMN_MASK << (BOARD_SIZE - 1)


def cell(x: int, y: int) -> int:
    """Return the row-major cell index for an in-bounds coordinate."""
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"cell out of bounds: ({x}, {y})")
    return y * BOARD_SIZE + x


def coordinates(index: int) -> tuple[int, int]:
    """Return ``(x, y)`` for a cell index."""
    if not 0 <= index < CELL_COUNT:
        raise ValueError(f"cell index out of bounds: {index}")
    return index % BOARD_SIZE, index // BOARD_SIZE


def wall_index(x: int, y: int) -> int:
    """Return the row-major 8x8 wall-anchor index."""
    if not (0 <= x < WALL_GRID_SIZE and 0 <= y < WALL_GRID_SIZE):
        raise ValueError(f"wall anchor out of bounds: ({x}, {y})")
    return y * WALL_GRID_SIZE + x


def wall_coordinates(index: int) -> tuple[int, int]:
    """Return ``(x, y)`` for a wall-anchor index."""
    if not 0 <= index < WALL_COUNT:
        raise ValueError(f"wall index out of bounds: {index}")
    return index % WALL_GRID_SIZE, index // WALL_GRID_SIZE


def pawn_action(destination: int) -> int:
    if not 0 <= destination < CELL_COUNT:
        raise ValueError(f"pawn destination out of bounds: {destination}")
    return destination


def horizontal_action(x: int, y: int) -> int:
    return HORIZONTAL_ACTION_OFFSET + wall_index(x, y)


def vertical_action(x: int, y: int) -> int:
    return VERTICAL_ACTION_OFFSET + wall_index(x, y)


def decode_action(action: int) -> tuple[ActionKind, int]:
    """Decode an action into its kind and cell or wall-anchor index."""
    if not 0 <= action < ACTION_SIZE:
        raise ValueError(f"action out of bounds: {action}")
    if action < HORIZONTAL_ACTION_OFFSET:
        return "pawn", action
    if action < VERTICAL_ACTION_OFFSET:
        return "horizontal", action - HORIZONTAL_ACTION_OFFSET
    return "vertical", action - VERTICAL_ACTION_OFFSET


def _wall_edge_tables() -> tuple[tuple[int, ...], tuple[int, ...]]:
    horizontal: list[int] = []
    vertical: list[int] = []
    for index in range(WALL_COUNT):
        x, y = wall_coordinates(index)
        horizontal.append((1 << cell(x, y)) | (1 << cell(x + 1, y)))
        vertical.append((1 << cell(x, y)) | (1 << cell(x, y + 1)))
    return tuple(horizontal), tuple(vertical)


HORIZONTAL_EDGE_MASKS, VERTICAL_EDGE_MASKS = _wall_edge_tables()


def _iter_set_bits(bits: int) -> Iterable[int]:
    while bits:
        least = bits & -bits
        yield least.bit_length() - 1
        bits ^= least


@lru_cache(maxsize=262_144)
def blocked_edge_masks(horizontal: int, vertical: int) -> tuple[int, int]:
    """Return origin-cell masks for blocked south and east edges."""
    blocked_south = 0
    blocked_east = 0
    for index in _iter_set_bits(horizontal):
        blocked_south |= HORIZONTAL_EDGE_MASKS[index]
    for index in _iter_set_bits(vertical):
        blocked_east |= VERTICAL_EDGE_MASKS[index]
    return blocked_south, blocked_east


def _edge_open(origin: int, destination: int, south: int, east: int) -> bool:
    difference = destination - origin
    if difference == BOARD_SIZE:
        return not (south >> origin) & 1
    if difference == -BOARD_SIZE:
        return not (south >> destination) & 1
    if difference == 1:
        return not (east >> origin) & 1
    if difference == -1:
        return not (east >> destination) & 1
    return False


def _adjacent(index: int, dx: int, dy: int) -> int | None:
    x, y = coordinates(index)
    next_x = x + dx
    next_y = y + dy
    if not (0 <= next_x < BOARD_SIZE and 0 <= next_y < BOARD_SIZE):
        return None
    return next_y * BOARD_SIZE + next_x


def _pawn_destinations(
    pawn: int, opponent: int, south: int, east: int
) -> tuple[int, ...]:
    moves: set[int] = set()
    for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        adjacent = _adjacent(pawn, dx, dy)
        if adjacent is None or not _edge_open(pawn, adjacent, south, east):
            continue
        if adjacent != opponent:
            moves.add(adjacent)
            continue

        beyond = _adjacent(adjacent, dx, dy)
        if beyond is not None and _edge_open(adjacent, beyond, south, east):
            moves.add(beyond)
            continue

        sides = ((1, 0), (-1, 0)) if dx == 0 else ((0, 1), (0, -1))
        for side_x, side_y in sides:
            diagonal = _adjacent(adjacent, side_x, side_y)
            if diagonal is not None and _edge_open(adjacent, diagonal, south, east):
                moves.add(diagonal)
    return tuple(sorted(moves))


def _reachable(start: int, goal_mask: int, south: int, east: int) -> bool:
    """Bit-parallel flood fill over the board's open edges."""
    reached = 1 << start
    while True:
        southward = ((reached & ~south) << BOARD_SIZE) & BOARD_MASK
        northward = (reached >> BOARD_SIZE) & ~south
        eastward = ((reached & ~east & ~RIGHT_COLUMN_MASK) << 1) & BOARD_MASK
        westward = (reached >> 1) & ~east & ~RIGHT_COLUMN_MASK
        expanded = reached | southward | northward | eastward | westward
        if expanded & goal_mask:
            return True
        if expanded == reached:
            return False
        reached = expanded


def _shortest_distance(start: int, goal_mask: int, south: int, east: int) -> int:
    frontier = 1 << start
    reached = frontier
    distance = 0
    while frontier:
        if frontier & goal_mask:
            return distance
        southward = ((frontier & ~south) << BOARD_SIZE) & BOARD_MASK
        northward = (frontier >> BOARD_SIZE) & ~south
        eastward = ((frontier & ~east & ~RIGHT_COLUMN_MASK) << 1) & BOARD_MASK
        westward = (frontier >> 1) & ~east & ~RIGHT_COLUMN_MASK
        frontier = (southward | northward | eastward | westward) & ~reached
        reached |= frontier
        distance += 1
    return -1


@lru_cache(maxsize=524_288)
def _path_witness(start: int, goal_mask: int, south: int, east: int) -> tuple[int, int]:
    """Return south/east edge masks for one valid path to the goal."""
    previous = [-1] * CELL_COUNT
    queue = [0] * CELL_COUNT
    seen = 1 << start
    queue[0] = start
    head = 0
    tail = 1
    finish = -1
    while head < tail:
        origin = queue[head]
        head += 1
        if (1 << origin) & goal_mask:
            finish = origin
            break
        for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
            destination = _adjacent(origin, dx, dy)
            if destination is None or (seen >> destination) & 1:
                continue
            if not _edge_open(origin, destination, south, east):
                continue
            seen |= 1 << destination
            previous[destination] = origin
            queue[tail] = destination
            tail += 1

    if finish < 0:
        return -1, -1
    path_south = 0
    path_east = 0
    cursor = finish
    while previous[cursor] >= 0:
        parent = previous[cursor]
        difference = cursor - parent
        if abs(difference) == BOARD_SIZE:
            path_south |= 1 << min(cursor, parent)
        else:
            path_east |= 1 << min(cursor, parent)
        cursor = parent
    return path_south, path_east


@dataclass(frozen=True, slots=True)
class State:
    """An immutable Quoridor position in absolute board coordinates.

    Player 0 begins at the top and races toward row 8. Player 1 begins at the
    bottom and races toward row 0. Placed walls are intentionally ownerless;
    only the remaining wall counts affect future play.
    """

    pawns: tuple[int, int] = (4, 76)
    walls_remaining: tuple[int, int] = (MAX_WALLS, MAX_WALLS)
    horizontal: int = 0
    vertical: int = 0
    to_play: int = 0
    ply: int = 0

    def __post_init__(self) -> None:
        if self.to_play not in (0, 1):
            raise ValueError(f"invalid player: {self.to_play}")
        if self.pawns[0] == self.pawns[1]:
            raise ValueError("pawns cannot occupy the same cell")
        if any(not 0 <= pawn < CELL_COUNT for pawn in self.pawns):
            raise ValueError(f"invalid pawn positions: {self.pawns}")
        if any(not 0 <= count <= MAX_WALLS for count in self.walls_remaining):
            raise ValueError(f"invalid remaining walls: {self.walls_remaining}")
        wall_mask = (1 << WALL_COUNT) - 1
        if self.horizontal & ~wall_mask or self.vertical & ~wall_mask:
            raise ValueError("wall bitboard contains an out-of-bounds anchor")
        if self.horizontal & self.vertical:
            raise ValueError("horizontal and vertical walls cannot cross")
        if self.ply < 0:
            raise ValueError("ply must be non-negative")

    @classmethod
    def initial(cls) -> State:
        return cls()

    @property
    def position_key(self) -> tuple[object, ...]:
        """A repetition key that deliberately excludes the move counter."""
        return (
            self.pawns,
            self.walls_remaining,
            self.horizontal,
            self.vertical,
            self.to_play,
        )

    @property
    def winner(self) -> int | None:
        if self.pawns[0] // BOARD_SIZE == BOARD_SIZE - 1:
            return 0
        if self.pawns[1] // BOARD_SIZE == 0:
            return 1
        return None

    def terminal_value(self, max_plies: int = MAX_GAME_PLIES) -> float | None:
        """Return the outcome from the side-to-move perspective, or ``None``."""
        winner = self.winner
        if winner is not None:
            return 1.0 if winner == self.to_play else -1.0
        if self.ply >= max_plies:
            return 0.0
        return None

    def legal_pawn_actions(self) -> tuple[int, ...]:
        return legal_pawn_actions(self)

    def legal_actions(self) -> tuple[int, ...]:
        return legal_actions(self)

    def play(self, action: int, *, validate: bool = True) -> State:
        return play(self, action, validate=validate)

    def shortest_distance(self, player: int) -> int:
        if player not in (0, 1):
            raise ValueError(f"invalid player: {player}")
        south, east = blocked_edge_masks(self.horizontal, self.vertical)
        goal = BOTTOM_ROW_MASK if player == 0 else TOP_ROW_MASK
        return _shortest_distance(self.pawns[player], goal, south, east)


@lru_cache(maxsize=524_288)
def legal_pawn_actions(state: State) -> tuple[int, ...]:
    """Generate ordinary, jump, and blocked-jump diagonal pawn moves."""
    player = state.to_play
    pawn = state.pawns[player]
    opponent = state.pawns[1 - player]
    south, east = blocked_edge_masks(state.horizontal, state.vertical)
    return _pawn_destinations(pawn, opponent, south, east)


def _race_code(pawn_one: int, pawn_two: int, turn: int) -> int:
    return (pawn_one * CELL_COUNT + pawn_two) * 2 + turn


@lru_cache(maxsize=64)
def _pawn_race_table(horizontal: int, vertical: int) -> tuple[array, array]:
    """Solve every pawn-only state for one fixed wall geometry by retrograde."""
    state_count = CELL_COUNT * CELL_COUNT * 2
    outcomes = array("b", [0]) * state_count
    distances = array("H", [0]) * state_count
    resolved_successors = array("B", [0]) * state_count
    successors: list[list[int] | None] = [None] * state_count
    predecessors: list[list[int] | None] = [None] * state_count
    buckets: list[list[int]] = [[], []]
    south, east = blocked_edge_masks(horizontal, vertical)

    for pawn_one in range(CELL_COUNT):
        if pawn_one // BOARD_SIZE == BOARD_SIZE - 1:
            continue
        for pawn_two in range(CELL_COUNT):
            if pawn_one == pawn_two or pawn_two // BOARD_SIZE == 0:
                continue
            for turn in (0, 1):
                code = _race_code(pawn_one, pawn_two, turn)
                pawns = (pawn_one, pawn_two)
                next_states: list[int] = []
                immediate_win = False
                target_row = BOARD_SIZE - 1 if turn == 0 else 0
                for destination in _pawn_destinations(
                    pawns[turn], pawns[1 - turn], south, east
                ):
                    if destination // BOARD_SIZE == target_row:
                        immediate_win = True
                        continue
                    next_one = destination if turn == 0 else pawn_one
                    next_two = destination if turn == 1 else pawn_two
                    next_code = _race_code(next_one, next_two, 1 - turn)
                    next_states.append(next_code)
                    previous_states = predecessors[next_code]
                    if previous_states is None:
                        previous_states = []
                        predecessors[next_code] = previous_states
                    previous_states.append(code)
                successors[code] = next_states
                if immediate_win:
                    outcomes[code] = 1
                    distances[code] = 1
                    buckets[1].append(code)

    distance = 1
    while distance < len(buckets):
        for code in buckets[distance]:
            for previous in predecessors[code] or ():
                if outcomes[previous] != 0:
                    continue
                if outcomes[code] == -1:
                    outcomes[previous] = 1
                    distances[previous] = distance + 1
                    while len(buckets) <= distance + 1:
                        buckets.append([])
                    buckets[distance + 1].append(previous)
                    continue
                resolved_successors[previous] += 1
                previous_successors = successors[previous]
                if previous_successors is not None and resolved_successors[
                    previous
                ] == len(previous_successors):
                    outcomes[previous] = -1
                    distances[previous] = distance + 1
                    while len(buckets) <= distance + 1:
                        buckets.append([])
                    buckets[distance + 1].append(previous)
        distance += 1
    return outcomes, distances


def exact_pawn_race(state: State) -> tuple[int, int]:
    """Return exact side-to-move outcome and distance when no walls remain."""
    if state.walls_remaining != (0, 0):
        raise ValueError("exact pawn-race solving requires both wall reserves empty")
    terminal = state.terminal_value()
    if terminal is not None:
        return int(terminal), 0
    outcomes, distances = _pawn_race_table(state.horizontal, state.vertical)
    code = _race_code(state.pawns[0], state.pawns[1], state.to_play)
    return int(outcomes[code]), int(distances[code])


def exact_pawn_race_moves(state: State) -> tuple[int, int, tuple[int, ...]]:
    """Return exact outcome, distance, and optimal root actions with no walls left.

    A winning side to move receives every fastest winning pawn action; a losing
    side receives every maximum-resistance action. Unresolved positions return
    an empty action tuple. All values derive from the retrograde rule table.
    """
    outcome, distance = exact_pawn_race(state)
    if outcome == 0 or distance == 0:
        return outcome, distance, ()
    outcomes, distances = _pawn_race_table(state.horizontal, state.vertical)
    player = state.to_play
    pawns = state.pawns
    south, east = blocked_edge_masks(state.horizontal, state.vertical)
    target_row = BOARD_SIZE - 1 if player == 0 else 0
    optimal: list[int] = []
    for destination in _pawn_destinations(
        pawns[player], pawns[1 - player], south, east
    ):
        if destination // BOARD_SIZE == target_row:
            if outcome == 1 and distance == 1:
                optimal.append(destination)
            continue
        next_one = destination if player == 0 else pawns[0]
        next_two = destination if player == 1 else pawns[1]
        next_code = _race_code(next_one, next_two, 1 - player)
        if (
            int(outcomes[next_code]) == -outcome
            and int(distances[next_code]) == distance - 1
        ):
            optimal.append(destination)
    return outcome, distance, tuple(optimal)


def _wall_conflicts(
    horizontal: int, vertical: int, index: int, kind: ActionKind
) -> bool:
    bit = 1 << index
    x, y = wall_coordinates(index)
    if kind == "horizontal":
        if (horizontal | vertical) & bit:
            return True
        if x > 0 and horizontal & (bit >> 1):
            return True
        return x + 1 < WALL_GRID_SIZE and bool(horizontal & (bit << 1))
    if (horizontal | vertical) & bit:
        return True
    if y > 0 and vertical & (bit >> WALL_GRID_SIZE):
        return True
    return y + 1 < WALL_GRID_SIZE and bool(vertical & (bit << WALL_GRID_SIZE))


def _wall_preserves_paths(
    state: State,
    index: int,
    kind: ActionKind,
    witnesses: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    south, east = blocked_edge_masks(state.horizontal, state.vertical)
    if kind == "horizontal":
        candidate = HORIZONTAL_EDGE_MASKS[index]
        south |= candidate
    else:
        candidate = VERTICAL_EDGE_MASKS[index]
        east |= candidate

    for player, goal in ((0, BOTTOM_ROW_MASK), (1, TOP_ROW_MASK)):
        path_south, path_east = witnesses[player]
        path_edges = path_south if kind == "horizontal" else path_east
        if candidate & path_edges and not _reachable(
            state.pawns[player], goal, south, east
        ):
            return False
    return True


@lru_cache(maxsize=262_144)
def legal_actions(state: State) -> tuple[int, ...]:
    if state.terminal_value() is not None:
        return ()
    actions = list(legal_pawn_actions(state))
    if state.walls_remaining[state.to_play] == 0:
        return tuple(actions)

    south, east = blocked_edge_masks(state.horizontal, state.vertical)
    witnesses = (
        _path_witness(state.pawns[0], BOTTOM_ROW_MASK, south, east),
        _path_witness(state.pawns[1], TOP_ROW_MASK, south, east),
    )
    occupied = state.horizontal | state.vertical
    for index in range(WALL_COUNT):
        bit = 1 << index
        if occupied & bit:
            continue
        if not _wall_conflicts(
            state.horizontal, state.vertical, index, "horizontal"
        ) and _wall_preserves_paths(state, index, "horizontal", witnesses):
            actions.append(HORIZONTAL_ACTION_OFFSET + index)
        if not _wall_conflicts(
            state.horizontal, state.vertical, index, "vertical"
        ) and _wall_preserves_paths(state, index, "vertical", witnesses):
            actions.append(VERTICAL_ACTION_OFFSET + index)
    return tuple(actions)


def play(state: State, action: int, *, validate: bool = True) -> State:
    kind, target = decode_action(action)
    if validate and action not in state.legal_actions():
        raise ValueError(f"illegal action {action} in position {state.position_key}")

    player = state.to_play
    pawns = state.pawns
    walls_remaining = state.walls_remaining
    horizontal = state.horizontal
    vertical = state.vertical

    if kind == "pawn":
        pawns_list = list(pawns)
        pawns_list[player] = target
        pawns = (pawns_list[0], pawns_list[1])
    else:
        remaining = list(walls_remaining)
        remaining[player] -= 1
        walls_remaining = (remaining[0], remaining[1])
        if kind == "horizontal":
            horizontal |= 1 << target
        else:
            vertical |= 1 << target

    return State(
        pawns=pawns,
        walls_remaining=walls_remaining,
        horizontal=horizontal,
        vertical=vertical,
        to_play=1 - player,
        ply=state.ply + 1,
    )


def action_to_dict(action: int) -> dict[str, object]:
    kind, target = decode_action(action)
    if kind == "pawn":
        x, y = coordinates(target)
        return {"type": "pawn", "to": {"x": x, "y": y}}
    x, y = wall_coordinates(target)
    return {
        "type": "wall",
        "wall": {"o": "h" if kind == "horizontal" else "v", "x": x, "y": y},
    }


def _required_int(payload: object, key: str) -> int:
    if not isinstance(payload, dict):
        raise ValueError("coordinate payload must be an object")
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def action_from_dict(payload: dict[str, object]) -> int:
    if payload.get("type") == "pawn":
        target = payload.get("to")
        if not isinstance(target, dict):
            raise ValueError("pawn action requires a 'to' object")
        return pawn_action(cell(_required_int(target, "x"), _required_int(target, "y")))
    if payload.get("type") == "wall":
        wall = payload.get("wall")
        if not isinstance(wall, dict):
            raise ValueError("wall action requires a 'wall' object")
        x = _required_int(wall, "x")
        y = _required_int(wall, "y")
        if wall.get("o") == "h":
            return horizontal_action(x, y)
        if wall.get("o") == "v":
            return vertical_action(x, y)
    raise ValueError(f"unrecognized action: {payload}")
