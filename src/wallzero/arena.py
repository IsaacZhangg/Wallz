"""Color-balanced neural MCTS arena used for checkpoint promotion."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from wallzero.game import State, exact_pawn_race
from wallzero.mcts import Evaluator, MCTSConfig, SearchTree, run_batched_search


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    games: int = 40
    parallel_games: int = 16
    repetition_limit: int = 3
    promotion_threshold: float = 0.55
    opening_random_plies: int = 6
    seed: int = 10_000

    def __post_init__(self) -> None:
        if self.games < 2 or self.games % 2:
            raise ValueError("arena games must be a positive even number")
        if self.parallel_games < 1:
            raise ValueError("parallel_games must be positive")
        if not 0.5 <= self.promotion_threshold <= 1.0:
            raise ValueError("promotion_threshold must be between 0.5 and 1.0")
        if not 0 <= self.opening_random_plies <= 20:
            raise ValueError("opening_random_plies must be between 0 and 20")


@dataclass(slots=True)
class ArenaResult:
    candidate_wins: int = 0
    incumbent_wins: int = 0
    draws: int = 0
    repetition_draws: int = 0
    max_ply_draws: int = 0
    adjudications: int = 0

    @property
    def games(self) -> int:
        return self.candidate_wins + self.incumbent_wins + self.draws

    @property
    def score_rate(self) -> float:
        if not self.games:
            return 0.0
        return (self.candidate_wins + 0.5 * self.draws) / self.games

    @property
    def decisive_win_rate(self) -> float:
        decisive = self.candidate_wins + self.incumbent_wins
        return self.candidate_wins / decisive if decisive else 0.0

    def should_promote(self, threshold: float) -> bool:
        return (
            self.score_rate >= threshold and self.candidate_wins > self.incumbent_wins
        )


@dataclass(slots=True)
class _ArenaGame:
    tree: SearchTree
    candidate_player: int
    rng: np.random.Generator
    repetitions: Counter[tuple[object, ...]] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.repetitions[self.tree.state.position_key] += 1


def _adjudicated_winner(state: State) -> int | None:
    """Return the exact winner once both wall reserves are empty, if solved."""
    if state.walls_remaining != (0, 0):
        return None
    outcome, _ = exact_pawn_race(state)
    if outcome == 0:
        return None
    return state.to_play if outcome > 0 else 1 - state.to_play


def _paired_opening(base: State, pair: int, config: ArenaConfig) -> State:
    """Play a uniform-random legal prefix shared by both games of a color pair.

    Openings are rule-derived diversity only: both members of the pair start
    from the identical position with swapped colors, so neither model gains
    information the other lacks.
    """
    if not config.opening_random_plies:
        return base
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x0A11, pair]))
    state = base
    for _ in range(config.opening_random_plies):
        if state.terminal_value() is not None:
            break
        legal = state.legal_actions()
        if not legal:
            break
        state = state.play(int(rng.choice(legal)))
    return state if state.terminal_value() is None else base


def evaluate_candidate(
    candidate: Evaluator,
    incumbent: Evaluator,
    mcts_config: MCTSConfig,
    config: ArenaConfig,
    *,
    progress: Callable[[ArenaResult], None] | None = None,
    initial_state: State | None = None,
) -> ArenaResult:
    """Pit two frozen networks against each other with equally swapped colors."""
    result = ArenaResult()
    seed_sequence = np.random.SeedSequence(config.seed)
    seeds = iter(seed_sequence.spawn(config.games))
    launched = 0

    base_state = initial_state or State.initial()
    while launched < config.games:
        wave_size = min(config.parallel_games, config.games - launched)
        games = [
            _ArenaGame(
                tree=SearchTree.from_state(
                    _paired_opening(base_state, (launched + offset) // 2, config)
                ),
                candidate_player=(launched + offset) % 2,
                rng=np.random.default_rng(next(seeds)),
            )
            for offset in range(wave_size)
        ]
        launched += wave_size

        while games:
            candidate_games = [
                game
                for game in games
                if game.tree.state.to_play == game.candidate_player
            ]
            incumbent_games = [
                game
                for game in games
                if game.tree.state.to_play != game.candidate_player
            ]
            if candidate_games:
                run_batched_search(
                    [game.tree for game in candidate_games],
                    candidate,
                    mcts_config,
                    add_noise=False,
                    rngs=[game.rng for game in candidate_games],
                )
            if incumbent_games:
                run_batched_search(
                    [game.tree for game in incumbent_games],
                    incumbent,
                    mcts_config,
                    add_noise=False,
                    rngs=[game.rng for game in incumbent_games],
                )

            finished: list[tuple[_ArenaGame, int | None, bool]] = []
            for game in games:
                action = game.tree.select_action(0.0, game.rng)
                game.tree.advance(action)
                state = game.tree.state
                game.repetitions[state.position_key] += 1
                repeated = (
                    game.repetitions[state.position_key] >= config.repetition_limit
                )
                terminal = state.terminal_value(mcts_config.max_plies)
                if repeated or terminal is not None:
                    winner = None if repeated else state.winner
                    finished.append((game, winner, repeated))
                    continue
                adjudicated = _adjudicated_winner(state)
                if adjudicated is not None:
                    result.adjudications += 1
                    finished.append((game, adjudicated, False))

            for game, winner, repeated in finished:
                games.remove(game)
                if winner is None:
                    result.draws += 1
                    if repeated:
                        result.repetition_draws += 1
                    else:
                        result.max_ply_draws += 1
                elif winner == game.candidate_player:
                    result.candidate_wins += 1
                else:
                    result.incumbent_wins += 1
            if finished and progress is not None:
                progress(result)

    return result
