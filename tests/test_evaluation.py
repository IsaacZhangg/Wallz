from __future__ import annotations

from wallzero.evaluation import run_match, wilson_interval
from wallzero.mcts import UniformEvaluator


def test_wilson_interval_brackets_the_observed_proportion() -> None:
    low, high = wilson_interval(60, 100)
    assert low < 0.6 < high
    assert 0.49 < low < 0.51
    assert 0.68 < high < 0.70


def test_wilson_interval_handles_extremes() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    low, high = wilson_interval(0, 20)
    assert low == 0.0
    assert high < 0.25
    low, high = wilson_interval(20, 20)
    assert low > 0.75
    assert high == 1.0


def test_run_match_reports_consistent_counts() -> None:
    report = run_match(
        "uniform-self-match",
        UniformEvaluator(),
        UniformEvaluator(),
        games=2,
        simulations=2,
        seed=31,
        max_plies=32,
    )

    assert report.games == 2
    assert report.wins + report.losses + report.draws == 2
    assert 0.0 <= report.ci_low <= report.score_rate <= report.ci_high <= 1.0
    assert report.label == "uniform-self-match"
