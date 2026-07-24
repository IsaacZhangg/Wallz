from __future__ import annotations

import numpy as np

from wallzero.constants import ACTION_SIZE
from wallzero.game import State, cell
from wallzero.mcts import MCTSConfig, SearchTree, UniformEvaluator, run_batched_search


class WinningEvaluator:
    def __call__(self, states: list[State]) -> tuple[np.ndarray, np.ndarray]:
        logits = np.zeros((len(states), ACTION_SIZE), dtype=np.float32)
        for index, state in enumerate(states):
            if state.to_play == 0 and state.pawns[0] // 9 == 7:
                logits[index, cell(4, 8)] = 10.0
        return logits, np.zeros(len(states), dtype=np.float32)


def test_mcts_finds_an_immediate_terminal_win() -> None:
    state = State(pawns=(cell(4, 7), cell(4, 1)), to_play=0)
    tree = SearchTree.from_state(state)
    rng = np.random.default_rng(7)

    run_batched_search(
        [tree],
        WinningEvaluator(),
        MCTSConfig(simulations=32, dirichlet_fraction=0.0),
        add_noise=False,
        rngs=[rng],
    )

    assert tree.select_action(0.0, rng) == cell(4, 8)
    assert np.isclose(tree.policy().sum(), 1.0)
    assert tree.root_value > 0.0


def test_batched_search_runs_multiple_independent_roots() -> None:
    trees = [SearchTree.from_state(State.initial()) for _ in range(4)]
    rngs = [np.random.default_rng(seed) for seed in range(4)]
    config = MCTSConfig(simulations=4, dirichlet_fraction=0.25)

    run_batched_search(
        trees,
        UniformEvaluator(),
        config,
        add_noise=True,
        rngs=rngs,
    )

    assert all(tree.root.visit_count == config.simulations for tree in trees)
    assert all(np.isclose(tree.policy().sum(), 1.0) for tree in trees)
    assert (
        len(
            {
                tree.select_action(1.0, rng)
                for tree, rng in zip(trees, rngs, strict=True)
            }
        )
        > 1
    )


def test_exact_pawn_race_leaf_oracle_converts_without_network_help() -> None:
    state = State(
        pawns=(cell(4, 7), cell(4, 2)),
        walls_remaining=(0, 0),
        to_play=0,
    )
    tree = SearchTree.from_state(state)
    rng = np.random.default_rng(99)

    run_batched_search(
        [tree],
        UniformEvaluator(),
        MCTSConfig(simulations=16, dirichlet_fraction=0.0),
        add_noise=False,
        rngs=[rng],
    )

    assert tree.select_action(0.0, rng) == cell(4, 8)
