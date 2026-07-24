from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wallzero.arena import ArenaConfig
from wallzero.mcts import MCTSConfig, UniformEvaluator
from wallzero.network import NetworkConfig, load_checkpoint
from wallzero.pipeline import RunConfig, run_training
from wallzero.replay import TrainingExample
from wallzero.selfplay import SelfPlayConfig, SelfPlayStats, generate_self_play
from wallzero.training import TrainConfig


def _tiny_config() -> RunConfig:
    return RunConfig(
        name="tiny-determinism",
        iterations=2,
        network=NetworkConfig(channels=16, blocks=1, value_hidden=32),
        self_play_mcts=MCTSConfig(simulations=4, max_plies=48),
        self_play=SelfPlayConfig(
            games=2, parallel_games=2, temperature_moves=4, seed=71
        ),
        train=TrainConfig(steps=2, batch_size=8, seed=72),
        arena_mcts=MCTSConfig(simulations=4, dirichlet_fraction=0.0, max_plies=48),
        arena=ArenaConfig(games=2, parallel_games=2, promotion_threshold=0.5, seed=73),
        replay_window=10_000,
    )


def test_self_play_generation_is_deterministic_for_a_fixed_seed() -> None:
    def run() -> tuple[list[TrainingExample], SelfPlayStats]:
        return generate_self_play(
            UniformEvaluator(),
            MCTSConfig(simulations=8, max_plies=32),
            SelfPlayConfig(games=2, parallel_games=2, temperature_moves=6, seed=99),
        )

    first, first_stats = run()
    second, second_stats = run()

    assert len(first) == len(second)
    assert first_stats.positions == second_stats.positions
    for left, right in zip(first, second, strict=True):
        assert left.state == right.state
        assert left.value == right.value
        assert np.array_equal(left.policy, right.policy)


def _metrics_without_timing(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.pop("elapsed_seconds", None)
        row.pop("device", None)
        row.get("self_play", {}).pop("elapsed_seconds", None)
        row.get("training", {}).pop("elapsed_seconds", None)
        rows.append(row)
    return rows


def test_resumed_training_matches_a_straight_run(tmp_path: Path) -> None:
    config = _tiny_config()
    straight = tmp_path / "straight"
    resumed = tmp_path / "resumed"

    run_training(straight, config, device_name="cpu")
    run_training(resumed, config, device_name="cpu", iterations=1)
    run_training(resumed, config, device_name="cpu")

    straight_state = json.loads((straight / "run-state.json").read_text())
    resumed_state = json.loads((resumed / "run-state.json").read_text())
    assert straight_state["completed_iterations"] == 2
    assert resumed_state["completed_iterations"] == 2
    assert straight_state["last_promoted"] == resumed_state["last_promoted"]

    straight_metrics = _metrics_without_timing(straight / "metrics.jsonl")
    resumed_metrics = _metrics_without_timing(resumed / "metrics.jsonl")
    assert straight_metrics == resumed_metrics

    straight_best, _ = load_checkpoint(straight / "best.pt")
    resumed_best, _ = load_checkpoint(resumed / "best.pt")
    for key, tensor in straight_best.state_dict().items():
        assert torch.equal(tensor, resumed_best.state_dict()[key]), key
