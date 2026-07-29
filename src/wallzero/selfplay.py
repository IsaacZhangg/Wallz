"""Concurrent, GPU-batched pure self-play generation."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

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
    # KataGo playout cap randomization: most moves get a cheap search that
    # feeds only the value target; a random fraction get the full search and
    # produce (pruned) policy targets. Defaults keep the original behavior.
    full_search_probability: float = 1.0
    fast_simulations: int | None = None
    surprise_weighting: bool = False

    def __post_init__(self) -> None:
        if self.games < 1 or self.parallel_games < 1:
            raise ValueError("self-play game counts must be positive")
        if self.repetition_limit < 2:
            raise ValueError("repetition_limit must be at least two")
        if not 0.0 < self.full_search_probability <= 1.0:
            raise ValueError("full_search_probability must be in (0, 1]")


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
    # Aggregate policy surprise over full-search moves (policy_weight >= 1),
    # exposed per chunk so the flywheel can eventually trigger training on
    # "the model is out of date" instead of a fixed shard count. Stored as
    # sum + count (not mean) so per-worker stats merge by addition.
    surprise_sum: float = 0.0
    surprise_moves: int = 0
    elapsed_seconds: float = 0.0

    @property
    def positions_per_second(self) -> float:
        return self.positions / self.elapsed_seconds if self.elapsed_seconds else 0.0


@dataclass(slots=True)
class _Game:
    tree: SearchTree
    rng: np.random.Generator
    history: list[tuple[State, np.ndarray, int, float, float]] = field(
        default_factory=list
    )
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


def _policy_surprise(tree: SearchTree, target: np.ndarray) -> float:
    """KL divergence from the (noised) root prior to the recorded target."""
    root = tree.root
    if not root.children or root.state is None:
        return 0.0
    divergence = 0.0
    for action, child in root.children.items():
        probability = float(target[canonical_action(root.state, action)])
        if probability <= 0.0:
            continue
        prior = max(float(child.prior), 1e-9)
        divergence += probability * float(np.log(probability / prior))
    return max(0.0, divergence)


def _frequency_weights(
    history: list[tuple[State, np.ndarray, int, float, float]],
    config: SelfPlayConfig,
) -> list[float]:
    """Policy-surprise weighting: half uniform, half proportional to KL."""
    if not config.surprise_weighting:
        return [1.0] * len(history)
    full_indices = [index for index, entry in enumerate(history) if entry[3] >= 1.0]
    weights = [1.0] * len(history)
    total_surprise = sum(history[index][4] for index in full_indices)
    if not full_indices or total_surprise <= 0.0:
        return weights
    count = len(full_indices)
    for index in full_indices:
        weights[index] = 0.5 + 0.5 * count * history[index][4] / total_surprise
    return weights


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
            full_searches: set[int] = set()
            full_group: list[_Game] = []
            fast_group: list[_Game] = []
            for game in searching:
                is_full = (
                    config.full_search_probability >= 1.0
                    or float(game.rng.random()) < config.full_search_probability
                )
                if is_full:
                    full_searches.add(id(game))
                    full_group.append(game)
                else:
                    fast_group.append(game)
            if full_group:
                run_batched_search(
                    [game.tree for game in full_group],
                    evaluator,
                    mcts_config,
                    add_noise=True,
                    rngs=[game.rng for game in full_group],
                )
            if fast_group:
                fast_config = replace(
                    mcts_config,
                    simulations=config.fast_simulations
                    or max(1, mcts_config.simulations // 4),
                    forced_playout_scale=0.0,
                )
                run_batched_search(
                    [game.tree for game in fast_group],
                    evaluator,
                    fast_config,
                    add_noise=False,
                    rngs=[game.rng for game in fast_group],
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
                    policy_weight = 1.0
                    surprise = 0.0
                else:
                    full = id(game) in full_searches
                    policy = (
                        game.tree.pruned_policy(mcts_config)
                        if full
                        else game.tree.policy()
                    )
                    policy_weight = 1.0 if full else 0.0
                    surprise = _policy_surprise(game.tree, policy) if full else 0.0
                    temperature = (
                        config.opening_temperature
                        if state.ply < config.temperature_moves
                        else config.endgame_temperature
                    )
                    action = game.tree.select_action(temperature, game.rng)
                game.history.append((state, policy, actor, policy_weight, surprise))
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
                frequency = _frequency_weights(game.history, config)
                for entry, weight in zip(game.history, frequency, strict=True):
                    history_state, policy, actor, policy_weight, _ = entry
                    value = (
                        0.0 if winner is None else (1.0 if actor == winner else -1.0)
                    )
                    all_examples.append(
                        TrainingExample(
                            history_state,
                            policy.astype(np.float32),
                            value,
                            policy_weight=policy_weight,
                            weight=weight,
                            own_distance=history_state.shortest_distance(
                                history_state.to_play
                            ),
                            opp_distance=history_state.shortest_distance(
                                1 - history_state.to_play
                            ),
                        )
                    )
                stats.games += 1
                stats.positions += len(game.history)
                for entry in game.history:
                    if entry[3] >= 1.0:
                        stats.surprise_sum += entry[4]
                        stats.surprise_moves += 1
            if finished and progress is not None:
                stats.elapsed_seconds = time.monotonic() - started
                progress(stats)

    stats.elapsed_seconds = time.monotonic() - started
    return all_examples, stats
