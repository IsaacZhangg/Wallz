"""Independent strength evaluation, separate from the promotion gate."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from wallzero.arena import ArenaConfig, ArenaResult, evaluate_candidate
from wallzero.mcts import Evaluator, MCTSConfig, UniformEvaluator
from wallzero.parallel import (
    PlanesEvaluator,
    evaluate_candidate_parallel,
    uniform_planes_evaluator,
)


def _as_planes_evaluator(evaluator: Evaluator) -> PlanesEvaluator:
    method = getattr(evaluator, "evaluate_planes", None)
    if method is not None:
        return method
    if isinstance(evaluator, UniformEvaluator):
        return uniform_planes_evaluator
    raise TypeError(f"no planes evaluator available for {type(evaluator).__name__}")


@dataclass(frozen=True, slots=True)
class MatchReport:
    label: str
    games: int
    wins: int
    losses: int
    draws: int
    repetition_draws: int
    max_ply_draws: int
    adjudications: int
    score_rate: float
    ci_low: float
    ci_high: float
    simulations: int
    opening_random_plies: int
    seed: int
    leaf_batch: int = 1


def wilson_interval(
    successes: float, trials: int, z: float = 1.959964
) -> tuple[float, float]:
    """Two-sided 95% Wilson score interval for a (possibly half-point) score."""
    if trials == 0:
        return 0.0, 1.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def run_match(
    label: str,
    subject: Evaluator,
    opponent: Evaluator,
    *,
    games: int,
    simulations: int,
    seed: int,
    parallel_games: int = 20,
    opening_random_plies: int = 6,
    max_plies: int = 256,
    leaf_batch: int = 1,
    workers: int = 1,
    lcb_selection: bool = False,
) -> MatchReport:
    """Play a color-balanced, paired-opening match and report a 95% CI."""
    config = ArenaConfig(
        games=games,
        parallel_games=min(games, parallel_games),
        opening_random_plies=opening_random_plies,
        seed=seed,
    )
    mcts_config = MCTSConfig(
        simulations=simulations,
        dirichlet_fraction=0.0,
        max_plies=max_plies,
        leaf_batch=leaf_batch,
        lcb_selection=lcb_selection,
    )
    if workers > 1:
        result: ArenaResult = evaluate_candidate_parallel(
            _as_planes_evaluator(subject),
            _as_planes_evaluator(opponent),
            mcts_config,
            config,
            workers=workers,
        )
    else:
        result = evaluate_candidate(subject, opponent, mcts_config, config)
    score = result.candidate_wins + 0.5 * result.draws
    low, high = wilson_interval(score, result.games)
    return MatchReport(
        label=label,
        games=result.games,
        wins=result.candidate_wins,
        losses=result.incumbent_wins,
        draws=result.draws,
        repetition_draws=result.repetition_draws,
        max_ply_draws=result.max_ply_draws,
        adjudications=result.adjudications,
        score_rate=result.score_rate,
        ci_low=low,
        ci_high=high,
        simulations=simulations,
        opening_random_plies=opening_random_plies,
        seed=seed,
        leaf_batch=leaf_batch,
    )


def report_dict(report: MatchReport) -> dict[str, object]:
    return {"schema": "wallzero.match-report.v1", **asdict(report)}
