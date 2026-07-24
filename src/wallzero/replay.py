"""Compact, representation-independent self-play replay shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from wallzero.constants import ACTION_SIZE
from wallzero.game import State

FloatArray = NDArray[np.float32]


@dataclass(slots=True)
class TrainingExample:
    state: State
    policy: FloatArray
    value: float

    def __post_init__(self) -> None:
        if self.policy.shape != (ACTION_SIZE,):
            raise ValueError(f"invalid policy shape: {self.policy.shape}")
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"invalid outcome: {self.value}")


def save_shard(path: str | Path, examples: list[TrainingExample]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = len(examples)
    pawns = np.empty((count, 2), dtype=np.uint8)
    remaining = np.empty((count, 2), dtype=np.uint8)
    horizontal = np.empty(count, dtype=np.uint64)
    vertical = np.empty(count, dtype=np.uint64)
    to_play = np.empty(count, dtype=np.uint8)
    ply = np.empty(count, dtype=np.uint16)
    policies = np.empty((count, ACTION_SIZE), dtype=np.float16)
    values = np.empty(count, dtype=np.int8)
    for index, example in enumerate(examples):
        state = example.state
        pawns[index] = state.pawns
        remaining[index] = state.walls_remaining
        horizontal[index] = state.horizontal
        vertical[index] = state.vertical
        to_play[index] = state.to_play
        ply[index] = state.ply
        policies[index] = example.policy
        values[index] = round(example.value)
    np.savez_compressed(
        destination,
        schema=np.array("wallzero.replay.v1"),
        pawns=pawns,
        remaining=remaining,
        horizontal=horizontal,
        vertical=vertical,
        to_play=to_play,
        ply=ply,
        policies=policies,
        values=values,
    )
    return destination


def load_shard(path: str | Path) -> list[TrainingExample]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if str(data["schema"]) != "wallzero.replay.v1":
            raise ValueError(f"unsupported replay schema in {source}")
        examples = []
        for index in range(len(data["values"])):
            pawn_row = data["pawns"][index]
            remaining_row = data["remaining"][index]
            state = State(
                pawns=(int(pawn_row[0]), int(pawn_row[1])),
                walls_remaining=(int(remaining_row[0]), int(remaining_row[1])),
                horizontal=int(data["horizontal"][index]),
                vertical=int(data["vertical"][index]),
                to_play=int(data["to_play"][index]),
                ply=int(data["ply"][index]),
            )
            examples.append(
                TrainingExample(
                    state=state,
                    policy=data["policies"][index].astype(np.float32),
                    value=float(data["values"][index]),
                )
            )
    return examples


def load_replay_window(
    directory: str | Path,
    *,
    max_samples: int,
) -> list[TrainingExample]:
    shards = sorted(
        (
            shard
            for shard in Path(directory).glob("*.npz")
            if not shard.name.startswith((".", "_"))
        ),
        reverse=True,
    )
    selected: list[TrainingExample] = []
    for shard in shards:
        selected[0:0] = load_shard(shard)
        if len(selected) >= max_samples:
            return selected[-max_samples:]
    return selected
