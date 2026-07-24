"""Resumable AlphaZero training iterations and reproducible presets."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from wallzero.arena import ArenaConfig, evaluate_candidate
from wallzero.mcts import MCTSConfig
from wallzero.network import (
    NetworkConfig,
    PolicyValueNet,
    TorchEvaluator,
    load_checkpoint,
    save_checkpoint,
    select_device,
)
from wallzero.replay import load_replay_window, save_shard
from wallzero.selfplay import SelfPlayConfig, generate_self_play
from wallzero.training import TrainConfig, train_candidate


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    iterations: int
    network: NetworkConfig
    self_play_mcts: MCTSConfig
    self_play: SelfPlayConfig
    train: TrainConfig
    arena_mcts: MCTSConfig
    arena: ArenaConfig
    replay_window: int


def preset(name: str) -> RunConfig:
    if name == "smoke":
        return RunConfig(
            name=name,
            iterations=1,
            network=NetworkConfig(channels=32, blocks=2, value_hidden=64),
            self_play_mcts=MCTSConfig(simulations=4),
            self_play=SelfPlayConfig(
                games=2, parallel_games=2, temperature_moves=4, seed=101
            ),
            train=TrainConfig(steps=2, batch_size=16, seed=102),
            arena_mcts=MCTSConfig(simulations=4, dirichlet_fraction=0.0),
            arena=ArenaConfig(
                games=2,
                parallel_games=2,
                promotion_threshold=0.5,
                seed=103,
            ),
            replay_window=2_000,
        )
    if name == "bootstrap":
        return RunConfig(
            name=name,
            iterations=3,
            network=NetworkConfig(channels=64, blocks=6, value_hidden=128),
            self_play_mcts=MCTSConfig(simulations=64),
            self_play=SelfPlayConfig(
                games=32, parallel_games=8, temperature_moves=20, seed=1_001
            ),
            train=TrainConfig(steps=300, batch_size=256, seed=1_002),
            arena_mcts=MCTSConfig(simulations=96, dirichlet_fraction=0.0),
            arena=ArenaConfig(games=16, parallel_games=8, seed=1_003),
            replay_window=100_000,
        )
    if name == "colab":
        return RunConfig(
            name=name,
            iterations=12,
            network=NetworkConfig(channels=128, blocks=10, value_hidden=256),
            self_play_mcts=MCTSConfig(simulations=160),
            self_play=SelfPlayConfig(
                games=128, parallel_games=32, temperature_moves=24, seed=10_001
            ),
            train=TrainConfig(steps=1_000, batch_size=512, seed=10_002),
            arena_mcts=MCTSConfig(simulations=256, dirichlet_fraction=0.0),
            arena=ArenaConfig(games=40, parallel_games=20, seed=10_003),
            replay_window=500_000,
        )
    if name == "expert":
        return RunConfig(
            name=name,
            iterations=100,
            network=NetworkConfig(channels=192, blocks=15, value_hidden=384),
            self_play_mcts=MCTSConfig(simulations=800),
            self_play=SelfPlayConfig(
                games=1_024,
                parallel_games=64,
                temperature_moves=30,
                seed=100_001,
            ),
            train=TrainConfig(
                steps=4_000,
                batch_size=1_024,
                learning_rate=1e-3,
                seed=100_002,
            ),
            arena_mcts=MCTSConfig(simulations=1_200, dirichlet_fraction=0.0),
            arena=ArenaConfig(
                games=200,
                parallel_games=40,
                promotion_threshold=0.55,
                seed=100_003,
            ),
            replay_window=2_000_000,
        )
    raise ValueError(f"unknown training preset: {name}")


def _append_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _resume_iteration(state_path: Path) -> int:
    if not state_path.exists():
        return 0
    data = json.loads(state_path.read_text(encoding="utf-8"))
    return int(data.get("completed_iterations", 0))


def run_training(
    output: str | Path,
    config: RunConfig,
    *,
    device_name: str = "auto",
    iterations: int | None = None,
) -> Path:
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_dir = output_dir / "replay"
    candidate_dir = output_dir / "candidates"
    best_path = output_dir / "best.pt"
    state_path = output_dir / "run-state.json"
    metrics_path = output_dir / "metrics.jsonl"
    device = select_device(device_name)
    iteration_count = config.iterations if iterations is None else iterations

    if not best_path.exists():
        torch.manual_seed(config.self_play.seed)
        initial = PolicyValueNet(config.network)
        save_checkpoint(
            best_path,
            initial,
            metadata={
                "status": "random-initialization",
                "preset": config.name,
                "zero_human_data": True,
            },
        )

    start_iteration = _resume_iteration(state_path)
    for iteration in range(start_iteration, iteration_count):
        iteration_started = time.monotonic()
        best_model, best_payload = load_checkpoint(best_path, device=device)
        self_play_config = replace(
            config.self_play,
            seed=config.self_play.seed + iteration,
        )
        if iteration < 2:
            # Random networks otherwise settle into short deterministic cycles.
            # Extra early temperature is generic exploration, not game strategy.
            self_play_config = replace(
                self_play_config,
                temperature_moves=max(self_play_config.temperature_moves, 64),
                endgame_temperature=max(self_play_config.endgame_temperature, 0.25),
                repetition_limit=max(self_play_config.repetition_limit, 8),
            )
        examples, self_play_stats = generate_self_play(
            TorchEvaluator(best_model, device),
            config.self_play_mcts,
            self_play_config,
        )
        shard_path = replay_dir / f"iteration-{iteration:05d}.npz"
        save_shard(shard_path, examples)
        replay = load_replay_window(
            replay_dir,
            max_samples=config.replay_window,
        )

        candidate = PolicyValueNet(best_model.config)
        candidate.load_state_dict(best_model.state_dict())
        train_config = replace(config.train, seed=config.train.seed + iteration)
        train_metrics, optimizer = train_candidate(
            candidate,
            replay,
            device,
            train_config,
        )
        candidate_path = candidate_dir / f"iteration-{iteration:05d}.pt"
        save_checkpoint(
            candidate_path,
            candidate,
            optimizer=optimizer,
            metadata={
                "iteration": iteration,
                "parent_metadata": best_payload.get("metadata", {}),
                "replay_samples": len(replay),
                "zero_human_data": True,
            },
        )

        arena_config = replace(config.arena, seed=config.arena.seed + iteration)
        arena_result = evaluate_candidate(
            TorchEvaluator(candidate, device),
            TorchEvaluator(best_model, device),
            config.arena_mcts,
            arena_config,
        )
        promoted = arena_result.should_promote(arena_config.promotion_threshold)
        if promoted:
            save_checkpoint(
                best_path,
                candidate,
                optimizer=optimizer,
                metadata={
                    "status": "promoted",
                    "iteration": iteration,
                    "arena": asdict(arena_result),
                    "zero_human_data": True,
                },
            )

        metric = {
            "schema": "wallzero.metrics.v1",
            "preset": config.name,
            "iteration": iteration,
            "device": str(device),
            "self_play": asdict(self_play_stats),
            "training": asdict(train_metrics),
            "arena": {
                **asdict(arena_result),
                "score_rate": arena_result.score_rate,
                "decisive_win_rate": arena_result.decisive_win_rate,
                "threshold": arena_config.promotion_threshold,
            },
            "replay_samples": len(replay),
            "promoted": promoted,
            "elapsed_seconds": time.monotonic() - iteration_started,
        }
        _append_json(metrics_path, metric)
        _atomic_json(
            state_path,
            {
                "schema": "wallzero.run-state.v1",
                "preset": config.name,
                "completed_iterations": iteration + 1,
                "best_checkpoint": str(best_path),
                "last_candidate": str(candidate_path),
                "last_promoted": promoted,
            },
        )
        print(json.dumps(metric, sort_keys=True), flush=True)

        del best_model, candidate
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return best_path
