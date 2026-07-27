"""Durable, disconnect-tolerant phases for metered Colab training."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from wallzero.arena import ArenaResult, evaluate_candidate
from wallzero.network import (
    NetworkConfig,
    PolicyValueNet,
    TorchEvaluator,
    load_checkpoint,
    save_checkpoint,
    select_device,
)
from wallzero.parallel import evaluate_candidate_parallel, generate_self_play_parallel
from wallzero.pipeline import RunConfig
from wallzero.replay import load_replay_window, save_shard
from wallzero.selfplay import SelfPlayStats, generate_self_play
from wallzero.training import train_candidate


def _emit(event: str, payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {"schema": "wallzero.campaign-event.v1", "event": event, **payload},
            sort_keys=True,
        ),
        flush=True,
    )


def _append_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _ensure_best(output: Path, config: RunConfig) -> Path:
    best = output / "best.pt"
    if best.exists():
        return best
    torch.manual_seed(config.self_play.seed)
    save_checkpoint(
        best,
        PolicyValueNet(config.network),
        metadata={
            "status": "random-initialization",
            "preset": config.name,
            "zero_human_data": True,
        },
    )
    return best


class _GpuSampler:
    """Sample GPU utilization in the background so throughput claims are real.

    Silently degrades to no measurement when the driver query is unavailable,
    since this is instrumentation and must never fail a training run.
    """

    def __init__(self, device: torch.device, interval: float = 5.0) -> None:
        self.device = device
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _GpuSampler:
        if self.device.type != "cuda":
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.samples.append(int(torch.cuda.utilization(self.device)))
            except Exception:
                return

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def report(self) -> dict[str, float | int] | None:
        if not self.samples:
            return None
        return {
            "samples": len(self.samples),
            "mean_percent": round(sum(self.samples) / len(self.samples), 1),
            "max_percent": max(self.samples),
            "min_percent": min(self.samples),
        }


def run_self_play_chunk(
    output: str | Path,
    config: RunConfig,
    *,
    games: int = 16,
    simulations: int | None = None,
    temperature_moves: int | None = None,
    endgame_temperature: float | None = None,
    workers: int = 1,
    leaf_batch: int | None = None,
    full_search_probability: float = 1.0,
    fast_simulations: int | None = None,
    surprise_weighting: bool = False,
    forced_playout_scale: float | None = None,
    root_policy_temperature: float | None = None,
    dirichlet_concentration: float | None = None,
    eval_cache_entries: int = 0,
    device_name: str = "auto",
) -> Path:
    """Generate and fsync one replay shard before returning control.

    The cold-start default keeps the aggressive exploration override so a
    random network cannot settle into short cycles. Later generations should
    pass the preset temperature schedule explicitly: with the wall-free solver
    finishing games and paired arena openings, long high-temperature prefixes
    only degrade trajectory quality.
    """
    output_dir = Path(output)
    replay_dir = output_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    existing = list(replay_dir.glob("chunk-*.npz"))
    chunk_index = (
        max(int(shard.stem.split("-")[1]) for shard in existing) + 1 if existing else 0
    )
    best_path = _ensure_best(output_dir, config)
    device = select_device(device_name)
    model, _ = load_checkpoint(best_path, device=device)
    self_play_config = replace(
        config.self_play,
        games=games,
        parallel_games=min(games, config.self_play.parallel_games),
        temperature_moves=(
            temperature_moves
            if temperature_moves is not None
            else max(config.self_play.temperature_moves, 64)
        ),
        endgame_temperature=(
            endgame_temperature
            if endgame_temperature is not None
            else max(config.self_play.endgame_temperature, 0.25)
        ),
        repetition_limit=max(config.self_play.repetition_limit, 8),
        seed=config.self_play.seed + chunk_index,
        full_search_probability=full_search_probability,
        fast_simulations=fast_simulations,
        surprise_weighting=surprise_weighting,
    )
    mcts_config = (
        config.self_play_mcts
        if simulations is None
        else replace(config.self_play_mcts, simulations=simulations)
    )
    if leaf_batch is not None:
        mcts_config = replace(mcts_config, leaf_batch=leaf_batch)
    if forced_playout_scale is not None:
        mcts_config = replace(mcts_config, forced_playout_scale=forced_playout_scale)
    if root_policy_temperature is not None:
        mcts_config = replace(
            mcts_config, root_policy_temperature=root_policy_temperature
        )
    if dirichlet_concentration is not None:
        mcts_config = replace(
            mcts_config, dirichlet_concentration=dirichlet_concentration
        )
    started = time.monotonic()
    _emit(
        "self-play-start",
        {
            "chunk": chunk_index,
            "games": games,
            "simulations": mcts_config.simulations,
            "workers": workers,
            "leaf_batch": mcts_config.leaf_batch,
            "device": str(device),
        },
    )

    def report(stats: SelfPlayStats) -> None:
        _emit("self-play-progress", {"chunk": chunk_index, **asdict(stats)})

    evaluator = TorchEvaluator(model, device)
    with _GpuSampler(device) as sampler:
        if workers > 1:
            examples, stats = generate_self_play_parallel(
                evaluator.evaluate_planes,
                mcts_config,
                self_play_config,
                workers=workers,
                progress=report,
                eval_cache_entries=eval_cache_entries,
            )
        else:
            examples, stats = generate_self_play(
                evaluator,
                mcts_config,
                self_play_config,
                progress=report,
            )
    shard = replay_dir / f"chunk-{chunk_index:05d}.npz"
    save_shard(shard, examples)
    metric = {
        "schema": "wallzero.chunk-metrics.v1",
        "chunk": chunk_index,
        "checkpoint": str(best_path),
        "games": games,
        "simulations": mcts_config.simulations,
        "workers": workers,
        "leaf_batch": mcts_config.leaf_batch,
        "device": str(device),
        "gpu_utilization": sampler.report(),
        "self_play": asdict(stats),
        "examples": len(examples),
        "elapsed_seconds": time.monotonic() - started,
        "zero_human_data": True,
    }
    _append_json(output_dir / "chunk-metrics.jsonl", metric)
    _emit("self-play-complete", metric)
    return shard


def run_training_round(
    output: str | Path,
    config: RunConfig,
    *,
    training_steps: int | None = None,
    arena_games: int | None = None,
    arena_simulations: int | None = None,
    arena_workers: int = 1,
    candidate_network: NetworkConfig | None = None,
    warm_start: bool = False,
    device_name: str = "auto",
) -> Path:
    """Train one candidate from durable shards, then gate it in the arena.

    By default the candidate clones the incumbent's weights. Passing
    candidate_network instead trains a freshly initialized network of a new
    architecture on the same replay window — the scale-up path — and gates it
    against the incumbent under the identical rules; the checkpoint carries
    its own architecture, so later rounds continue from whichever wins.
    """
    output_dir = Path(output)
    state_path = output_dir / "campaign-state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {"completed_rounds": 0}
    )
    round_index = int(state["completed_rounds"])
    best_path = _ensure_best(output_dir, config)
    device = select_device(device_name)
    best_model, best_payload = load_checkpoint(best_path, device=device)
    replay = load_replay_window(output_dir / "replay", max_samples=config.replay_window)
    if not replay:
        raise ValueError("training round requires at least one durable replay shard")

    train_config = replace(
        config.train,
        steps=training_steps or config.train.steps,
        seed=config.train.seed + round_index,
    )
    if candidate_network is None:
        candidate = PolicyValueNet(best_model.config)
        candidate.load_state_dict(best_model.state_dict())
    else:
        torch.manual_seed(train_config.seed)
        candidate = PolicyValueNet(candidate_network)
        if warm_start:
            # Adopt every architecturally compatible weight from the incumbent
            # (e.g. adding auxiliary heads keeps the whole trunk).
            candidate.load_state_dict(best_model.state_dict(), strict=False)
    _emit(
        "training-start",
        {
            "round": round_index,
            "replay_samples": len(replay),
            "steps": train_config.steps,
            "device": str(device),
            "candidate_network": asdict(candidate.config),
            "scale_up": candidate_network is not None,
        },
    )
    train_metrics, optimizer = train_candidate(candidate, replay, device, train_config)
    candidate_path = output_dir / "candidates" / f"round-{round_index:05d}.pt"
    save_checkpoint(
        candidate_path,
        candidate,
        optimizer=optimizer,
        metadata={
            "round": round_index,
            "parent_metadata": best_payload.get("metadata", {}),
            "replay_samples": len(replay),
            "scale_up": candidate_network is not None,
            "zero_human_data": True,
        },
    )
    _emit("training-complete", {"round": round_index, **asdict(train_metrics)})

    if arena_games == 0:
        # Gateless mode (AlphaZero-style): always adopt the latest candidate;
        # strength judgments belong to the independent evaluation suites.
        save_checkpoint(
            best_path,
            candidate,
            optimizer=optimizer,
            metadata={
                "status": "promoted-gateless",
                "round": round_index,
                "zero_human_data": True,
            },
        )
        metric = {
            "schema": "wallzero.round-metrics.v1",
            "round": round_index,
            "candidate_network": asdict(candidate.config),
            "scale_up": candidate_network is not None,
            "training": asdict(train_metrics),
            "arena": None,
            "replay_samples": len(replay),
            "promoted": True,
            "gateless": True,
            "zero_human_data": True,
        }
        _append_json(output_dir / "round-metrics.jsonl", metric)
        _atomic_json(
            state_path,
            {
                "schema": "wallzero.campaign-state.v1",
                "completed_rounds": round_index + 1,
                "last_candidate": str(candidate_path),
                "last_promoted": True,
            },
        )
        _emit("round-complete-gateless", metric)
        return best_path

    arena_config = replace(
        config.arena,
        games=arena_games or config.arena.games,
        parallel_games=min(
            arena_games or config.arena.games, config.arena.parallel_games
        ),
        seed=config.arena.seed + round_index,
    )
    mcts_config = (
        config.arena_mcts
        if arena_simulations is None
        else replace(config.arena_mcts, simulations=arena_simulations)
    )

    def report(result: ArenaResult) -> None:
        _emit("arena-progress", {"round": round_index, **asdict(result)})

    _emit(
        "arena-start",
        {
            "round": round_index,
            "games": arena_config.games,
            "simulations": mcts_config.simulations,
            "workers": arena_workers,
            "leaf_batch": mcts_config.leaf_batch,
        },
    )
    if arena_workers > 1:
        result = evaluate_candidate_parallel(
            TorchEvaluator(candidate, device).evaluate_planes,
            TorchEvaluator(best_model, device).evaluate_planes,
            mcts_config,
            arena_config,
            workers=arena_workers,
            progress=report,
        )
    else:
        result = evaluate_candidate(
            TorchEvaluator(candidate, device),
            TorchEvaluator(best_model, device),
            mcts_config,
            arena_config,
            progress=report,
        )
    promoted = result.should_promote(arena_config.promotion_threshold)
    if promoted:
        save_checkpoint(
            best_path,
            candidate,
            optimizer=optimizer,
            metadata={
                "status": "promoted",
                "round": round_index,
                "arena": asdict(result),
                "zero_human_data": True,
            },
        )
    metric = {
        "schema": "wallzero.round-metrics.v1",
        "round": round_index,
        "candidate_network": asdict(candidate.config),
        "scale_up": candidate_network is not None,
        "training": asdict(train_metrics),
        "arena": {
            **asdict(result),
            "score_rate": result.score_rate,
            "decisive_win_rate": result.decisive_win_rate,
            "threshold": arena_config.promotion_threshold,
        },
        "replay_samples": len(replay),
        "promoted": promoted,
        "zero_human_data": True,
    }
    _append_json(output_dir / "round-metrics.jsonl", metric)
    _atomic_json(
        state_path,
        {
            "schema": "wallzero.campaign-state.v1",
            "completed_rounds": round_index + 1,
            "last_candidate": str(candidate_path),
            "last_promoted": promoted,
        },
    )
    _emit("arena-complete", metric)
    return best_path
