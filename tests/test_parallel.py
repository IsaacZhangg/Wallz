from __future__ import annotations

import numpy as np
import pytest
import torch

from wallzero.game import State
from wallzero.mcts import (
    MCTSConfig,
    SearchTree,
    UniformEvaluator,
    _backup,
    run_batched_search,
)
from wallzero.network import NetworkConfig, PolicyValueNet, TorchEvaluator
from wallzero.parallel import generate_self_play_parallel
from wallzero.selfplay import SelfPlayConfig, generate_self_play


def test_leaf_batch_must_be_positive() -> None:
    with pytest.raises(ValueError):
        MCTSConfig(leaf_batch=0)


@pytest.mark.parametrize(("simulations", "leaf_batch"), [(48, 8), (200, 6), (33, 3)])
def test_leaf_batched_search_spends_the_exact_budget_with_no_virtual_residue(
    simulations: int, leaf_batch: int
) -> None:
    # Every real backup passes through the root exactly once, so any leaked
    # virtual loss or budget miscount shows up as a wrong root visit count.
    config = MCTSConfig(simulations=simulations, leaf_batch=leaf_batch, max_plies=64)
    tree = SearchTree.from_state(State.initial())
    run_batched_search(
        [tree],
        UniformEvaluator(),
        config,
        add_noise=False,
        rngs=[np.random.default_rng(3)],
    )
    assert tree.root.visit_count == simulations
    assert tree.root.children is not None
    assert (
        sum(child.visit_count for child in tree.root.children.values()) == simulations
    )


def _reference_search(
    tree: SearchTree, evaluator: TorchEvaluator, config: MCTSConfig
) -> None:
    """The pre-leaf-batch search loop, kept verbatim as a behavioral oracle."""
    if not tree.root.expanded:
        logits, _ = evaluator([tree.root.materialized_state])
        tree.root.expand(logits[0])
    for _ in range(config.simulations):
        item = tree.select_leaf(config)
        if item is None:
            continue
        logits, values = evaluator([item.node.materialized_state])
        item.node.expand(logits[0])
        _backup(item.path, float(values[0]))


def test_leaf_batch_one_matches_the_original_search_exactly() -> None:
    torch.manual_seed(13)
    model = PolicyValueNet(NetworkConfig(channels=16, blocks=1, value_hidden=32))
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    state = State.initial()
    for _ in range(3):
        state = state.play(sorted(state.legal_actions())[0])
    config = MCTSConfig(simulations=24, max_plies=64, leaf_batch=1)

    reference_tree = SearchTree.from_state(state)
    _reference_search(reference_tree, evaluator, config)

    tree = SearchTree.from_state(state)
    run_batched_search(
        [tree],
        evaluator,
        config,
        add_noise=False,
        rngs=[np.random.default_rng(7)],
    )
    assert np.array_equal(tree.policy(), reference_tree.policy())
    assert tree.root.visit_count == reference_tree.root.visit_count


def test_leaf_batched_self_play_produces_valid_games() -> None:
    examples, stats = generate_self_play(
        UniformEvaluator(),
        MCTSConfig(simulations=16, max_plies=48, leaf_batch=4),
        SelfPlayConfig(games=2, parallel_games=2, temperature_moves=6, seed=11),
    )
    assert stats.games == 2
    assert stats.positions == len(examples)
    for example in examples:
        assert float(example.policy.sum()) == pytest.approx(1.0, abs=1e-4)


def test_parallel_self_play_matches_requested_volume_and_valid_policies() -> None:
    torch.manual_seed(17)
    model = PolicyValueNet(NetworkConfig(channels=16, blocks=1, value_hidden=32))
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    examples, stats = generate_self_play_parallel(
        evaluator.evaluate_planes,
        MCTSConfig(simulations=8, max_plies=32, leaf_batch=4),
        SelfPlayConfig(games=4, parallel_games=2, temperature_moves=6, seed=23),
        workers=2,
    )
    assert stats.games == 4
    assert stats.positions == len(examples)
    assert stats.player_one_wins + stats.player_two_wins + stats.draws == 4
    for example in examples:
        assert example.policy.shape == (209,)
        assert float(example.policy.sum()) == pytest.approx(1.0, abs=1e-4)
        assert -1.0 <= example.value <= 1.0


def test_parallel_self_play_rejects_single_worker() -> None:
    model = PolicyValueNet(NetworkConfig(channels=16, blocks=1, value_hidden=32))
    evaluator = TorchEvaluator(model, torch.device("cpu"))
    with pytest.raises(ValueError):
        generate_self_play_parallel(
            evaluator.evaluate_planes,
            MCTSConfig(simulations=4),
            SelfPlayConfig(games=2, parallel_games=2),
            workers=1,
        )


def _broken_evaluator(
    planes: np.typing.NDArray[np.float32],
) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
    raise RuntimeError("evaluator exploded")


def test_parallel_self_play_surfaces_evaluator_failures() -> None:
    with pytest.raises(RuntimeError, match="distributed search failed"):
        generate_self_play_parallel(
            _broken_evaluator,
            MCTSConfig(simulations=4, max_plies=32),
            SelfPlayConfig(games=2, parallel_games=1, seed=31),
            workers=2,
        )


def test_partition_pairs_keeps_color_pairs_whole() -> None:
    from wallzero.parallel import _partition_pairs

    assert _partition_pairs(10, 4) == [4, 2, 2, 2]
    assert _partition_pairs(4, 8) == [2, 2]
    assert _partition_pairs(60, 10) == [6] * 10
    for split in (_partition_pairs(10, 4), _partition_pairs(60, 7)):
        assert all(count % 2 == 0 and count > 0 for count in split)


def test_parallel_arena_plays_all_games_color_balanced() -> None:
    from wallzero.arena import ArenaConfig
    from wallzero.parallel import evaluate_candidate_parallel

    torch.manual_seed(29)
    candidate = PolicyValueNet(NetworkConfig(channels=16, blocks=1, value_hidden=32))
    incumbent = PolicyValueNet(NetworkConfig(channels=16, blocks=1, value_hidden=32))
    result = evaluate_candidate_parallel(
        TorchEvaluator(candidate, torch.device("cpu")).evaluate_planes,
        TorchEvaluator(incumbent, torch.device("cpu")).evaluate_planes,
        MCTSConfig(simulations=8, max_plies=48, leaf_batch=4, dirichlet_fraction=0.0),
        ArenaConfig(games=4, parallel_games=2, promotion_threshold=0.5, seed=41),
        workers=2,
    )
    assert result.games == 4
    assert result.candidate_wins + result.incumbent_wins + result.draws == 4


def test_run_match_parallel_with_uniform_opponent() -> None:
    from wallzero.evaluation import run_match

    torch.manual_seed(31)
    model = PolicyValueNet(NetworkConfig(channels=16, blocks=1, value_hidden=32))
    subject = TorchEvaluator(model, torch.device("cpu"))
    report = run_match(
        "test-parallel-match",
        subject,
        UniformEvaluator(),
        games=4,
        simulations=8,
        seed=43,
        max_plies=48,
        leaf_batch=4,
        workers=2,
    )
    assert report.games == 4
    assert 0.0 <= report.ci_low <= report.ci_high <= 1.0
