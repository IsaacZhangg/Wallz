"""Concurrent, GPU-batched pure self-play generation."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from wallzero.constants import ACTION_SIZE
from wallzero.encoding import canonical_action
from wallzero.game import State, exact_pawn_race_moves
from wallzero.mcts import Evaluator, MCTSConfig, SearchTree, run_batched_search
from wallzero.replay import TrainingExample


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    games: int = 64
    parallel_games: int = 16
    temperature_moves: int = 20
    opening_temperature: float = 1.0
    endgame_temperature: float = 0.05
    repetition_limit: int = 3
    seed: int = 1

    def __post_init__(self) -> None:
        if self.games < 1 or self.parallel_games < 1:
            raise ValueError("self-play game counts must be positive")
        if self.repetition_limit < 2:
            raise ValueError("repetition_limit must be at least two")


@dataclass(slots=True)
class SelfPlayStats:
    games: int = 0
    positions: int = 0
    player_one_wins: int = 0
    player_two_wins: int = 0
    draws: int = 0
    repetition_draws: int = 0
    max_ply_draws: int = 0
    solver_moves: int = 0
    elapsed_seconds: float = 0.0

    @property
    def positions_per_second(self) -> float:
        return self.positions / self.elapsed_seconds if self.elapsed_seconds else 0.0


@dataclass(slots=True)
class _Game:
    tree: SearchTree
    rng: np.random.Generator
    history: list[tuple[State, np.ndarray, int]] = field(default_factory=list)
    repetitions: Counter[tuple[object, ...]] = field(default_factory=Counter)
    draw: bool = False

    def __post_init__(self) -> None:
        self.repetitions[self.tree.state.position_key] += 1


def _solved_root_actions(state: State) -> tuple[int, ...] | None:
    """Return the exact optimal root actions once both wall reserves are empty."""
    if state.walls_remaining != (0, 0):
        return None
    outcome, _, actions = exact_pawn_race_moves(state)
    if outcome == 0 or not actions:
        return None
    return actions


def _exact_policy(state: State, actions: tuple[int, ...]) -> np.ndarray:
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    weight = 1.0 / len(actions)
    for action in actions:
        policy[canonical_action(state, action)] = weight
    return policy


def generate_self_play(
    evaluator: Evaluator,
    mcts_config: MCTSConfig,
    config: SelfPlayConfig,
    *,
    progress: Callable[[SelfPlayStats], None] | None = None,
    initial_state: State | None = None,
) -> tuple[list[TrainingExample], SelfPlayStats]:
    """Generate self-play without human data or handcrafted evaluations."""
    started = time.monotonic()
    all_examples: list[TrainingExample] = []
    stats = SelfPlayStats()
    seed_sequence = np.random.SeedSequence(config.seed)
    game_seeds = iter(seed_sequence.spawn(config.games))

    while stats.games < config.games:
        wave_size = min(config.parallel_games, config.games - stats.games)
        games = [
            _Game(
                tree=SearchTree.from_state(initial_state or State.initial()),
                rng=np.random.default_rng(next(game_seeds)),
            )
            for _ in range(wave_size)
        ]

        while games:
            solved: dict[int, tuple[int, ...]] = {
                id(game): actions
                for game in games
                if (actions := _solved_root_actions(game.tree.state)) is not None
            }
            searching = [game for game in games if id(game) not in solved]
            if searching:
                run_batched_search(
                    [game.tree for game in searching],
                    evaluator,
                    mcts_config,
                    add_noise=True,
                    rngs=[game.rng for game in searching],
                )
            finished: list[_Game] = []
            for game in games:
                state = game.tree.state
                actor = state.to_play
                exact_actions = solved.get(id(game))
                if exact_actions is not None:
                    policy = _exact_policy(state, exact_actions)
                    choice = int(game.rng.integers(len(exact_actions)))
                    action = exact_actions[choice]
                    stats.solver_moves += 1
                else:
                    policy = game.tree.policy()
                    temperature = (
                        config.opening_temperature
                        if state.ply < config.temperature_moves
                        else config.endgame_temperature
                    )
                    action = game.tree.select_action(temperature, game.rng)
                game.history.append((state, policy, actor))
                game.tree.advance(action)
                next_state = game.tree.state
                game.repetitions[next_state.position_key] += 1
                game.draw = (
                    game.repetitions[next_state.position_key] >= config.repetition_limit
                )
                if (
                    game.draw
                    or next_state.terminal_value(mcts_config.max_plies) is not None
                ):
                    finished.append(game)

            for game in finished:
                games.remove(game)
                state = game.tree.state
                winner = None if game.draw else state.winner
                if winner == 0:
                    stats.player_one_wins += 1
                elif winner == 1:
                    stats.player_two_wins += 1
                else:
                    stats.draws += 1
                    if game.draw:
                        stats.repetition_draws += 1
                    else:
                        stats.max_ply_draws += 1
                for history_state, policy, actor in game.history:
                    value = (
                        0.0 if winner is None else (1.0 if actor == winner else -1.0)
                    )
                    all_examples.append(
                        TrainingExample(history_state, policy.astype(np.float32), value)
                    )
                stats.games += 1
                stats.positions += len(game.history)
            if finished and progress is not None:
                stats.elapsed_seconds = time.monotonic() - started
                progress(stats)

    stats.elapsed_seconds = time.monotonic() - started
    return all_examples, stats
