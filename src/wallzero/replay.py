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
    # KataGo-style targets; defaults reproduce the original schema exactly.
    policy_weight: float = 1.0
    weight: float = 1.0
    own_distance: int = -1
    opp_distance: int = -1

    def __post_init__(self) -> None:
        if self.policy.shape != (ACTION_SIZE,):
            raise ValueError(f"invalid policy shape: {self.policy.shape}")
        if not -1.0 <= self.value <= 1.0:
            raise ValueError(f"invalid outcome: {self.value}")
        if self.weight < 0.0 or not 0.0 <= self.policy_weight <= 1.0:
            raise ValueError("invalid example weights")


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
    policy_weights = np.empty(count, dtype=np.float16)
    weights = np.empty(count, dtype=np.float16)
    own_distances = np.empty(count, dtype=np.int16)
    opp_distances = np.empty(count, dtype=np.int16)
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
        policy_weights[index] = example.policy_weight
        weights[index] = example.weight
        own_distances[index] = example.own_distance
        opp_distances[index] = example.opp_distance
    np.savez_compressed(
        destination,
        schema=np.array("wallzero.replay.v2"),
        policy_weights=policy_weights,
        weights=weights,
        own_distances=own_distances,
        opp_distances=opp_distances,
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
        schema = str(data["schema"])
        if schema not in ("wallzero.replay.v1", "wallzero.replay.v2"):
            raise ValueError(f"unsupported replay schema in {source}")
        second = schema == "wallzero.replay.v2"
        # Every `data[key]` decompresses that entire array out of the zip
        # again, so each column is read exactly once, up front. Indexing the
        # NpzFile inside the loop cost ~18s per 5K-row shard instead of ~0.1s,
        # which pushed a 71-shard window load past the flywheel's watchdog.
        pawns = data["pawns"].tolist()
        remaining = data["remaining"].tolist()
        horizontal = data["horizontal"].tolist()
        vertical = data["vertical"].tolist()
        to_play = data["to_play"].tolist()
        ply = data["ply"].tolist()
        values = data["values"].tolist()
        policies = data["policies"].astype(np.float32)
        if second:
            policy_weights = data["policy_weights"].tolist()
            weights = data["weights"].tolist()
            own_distances = data["own_distances"].tolist()
            opp_distances = data["opp_distances"].tolist()

    examples = []
    for index in range(len(values)):
        pawn_row = pawns[index]
        remaining_row = remaining[index]
        state = State(
            pawns=(pawn_row[0], pawn_row[1]),
            walls_remaining=(remaining_row[0], remaining_row[1]),
            horizontal=horizontal[index],
            vertical=vertical[index],
            to_play=to_play[index],
            ply=ply[index],
        )
        if second:
            policy_weight = policy_weights[index]
            weight = weights[index]
            own_distance = own_distances[index]
            opp_distance = opp_distances[index]
        else:
            policy_weight = 1.0
            weight = 1.0
            own_distance = state.shortest_distance(state.to_play)
            opp_distance = state.shortest_distance(1 - state.to_play)
        examples.append(
            TrainingExample(
                state=state,
                policy=policies[index],
                value=float(values[index]),
                policy_weight=policy_weight,
                weight=weight,
                own_distance=own_distance,
                opp_distance=opp_distance,
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
