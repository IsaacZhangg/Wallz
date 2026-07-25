"""KataGo search-side ports: LCB move selection and distance-margin utility."""

from __future__ import annotations

import numpy as np
import pytest

from wallzero.game import State
from wallzero.mcts import (
    MCTSConfig,
    Node,
    SearchTree,
    UniformEvaluator,
    blended_leaf_value,
    run_batched_search,
)


def test_config_rejects_invalid_search_port_settings() -> None:
    with pytest.raises(ValueError):
        MCTSConfig(lcb_z=-0.1)
    with pytest.raises(ValueError):
        MCTSConfig(lcb_min_visit_fraction=0.0)
    with pytest.raises(ValueError):
        MCTSConfig(distance_utility_weight=1.5)
    with pytest.raises(ValueError):
        MCTSConfig(distance_utility_scale=0.0)


def test_value_statistics_survive_virtual_loss_round_trip() -> None:
    config = MCTSConfig(simulations=64, leaf_batch=8, max_plies=48)
    tree = SearchTree.from_state(State.initial())
    run_batched_search(
        [tree],
        UniformEvaluator(),
        config,
        add_noise=False,
        rngs=[np.random.default_rng(5)],
    )

    def check(node: Node) -> None:
        # Squared values can never be negative or exceed the visit count,
        # which is what a leaked virtual-loss adjustment would produce.
        assert node.value_sq_sum >= -1e-9
        assert node.value_sq_sum <= node.visit_count + 1e-9
        assert node.value_standard_error >= 0.0
        for child in (node.children or {}).values():
            if child.expanded:
                check(child)

    check(tree.root)


def test_lcb_prefers_the_reliable_child_over_a_lucky_one() -> None:
    tree = SearchTree.from_state(State.initial())
    config = MCTSConfig(lcb_selection=True, lcb_z=1.28)
    steady = Node(state=None, prior=0.5)
    steady.visit_count = 100
    steady.value_sum = -20.0  # mean -0.2 from the child's view => +0.2 for us
    steady.value_sq_sum = 20.0
    lucky = Node(state=None, prior=0.5)
    lucky.visit_count = 40
    lucky.value_sum = -10.0  # mean -0.25 => +0.25 for us, but noisy
    lucky.value_sq_sum = 40.0
    tree.root.children = {10: steady, 20: lucky}

    rng = np.random.default_rng(0)
    assert tree.select_action(0.0, rng, config=config) == 10
    # Without LCB the most-visited child also wins here; make the lucky child
    # the most visited to prove the two rules genuinely differ.
    lucky.visit_count = 200
    assert tree.select_action(0.0, rng, config=config) == 10
    assert tree.select_action(0.0, rng) == 20


def test_lcb_falls_back_when_no_child_is_sufficiently_visited() -> None:
    tree = SearchTree.from_state(State.initial())
    config = MCTSConfig(lcb_selection=True)
    only = Node(state=None, prior=1.0)
    only.visit_count = 1
    only.value_sum = -0.5
    only.value_sq_sum = 0.25
    tree.root.children = {7: only}
    assert tree.select_action(0.0, np.random.default_rng(1), config=config) == 7


def test_distance_utility_rewards_the_closer_player() -> None:
    state = State.initial()
    neutral = MCTSConfig(distance_utility_weight=0.0)
    shaped = MCTSConfig(distance_utility_weight=0.15, distance_utility_scale=6.0)
    # Symmetric start: the margin is zero, so a zero value stays zero.
    assert blended_leaf_value(state, 0.0, shaped) == pytest.approx(0.0, abs=1e-9)
    assert blended_leaf_value(state, 0.7, neutral) == 0.7

    advanced = state
    for _ in range(3):
        mover = advanced.to_play
        forward = min(
            advanced.legal_pawn_actions(),
            key=lambda action: abs(action - advanced.pawns[mover]),
        )
        advanced = advanced.play(forward)
    own = advanced.shortest_distance(advanced.to_play)
    opponent = advanced.shortest_distance(1 - advanced.to_play)
    blended = blended_leaf_value(advanced, 0.0, shaped)
    if own < opponent:
        assert blended > 0.0
    elif own > opponent:
        assert blended < 0.0
    assert -1.0 <= blended <= 1.0


def test_distance_utility_changes_backed_up_values() -> None:
    base = MCTSConfig(simulations=32, max_plies=48, leaf_batch=4)
    shaped = MCTSConfig(
        simulations=32,
        max_plies=48,
        leaf_batch=4,
        distance_utility_weight=0.5,
    )
    plain_tree = SearchTree.from_state(State.initial())
    shaped_tree = SearchTree.from_state(State.initial())
    for tree, config in ((plain_tree, base), (shaped_tree, shaped)):
        run_batched_search(
            [tree],
            UniformEvaluator(),
            config,
            add_noise=False,
            rngs=[np.random.default_rng(11)],
        )
    assert plain_tree.root.visit_count == shaped_tree.root.visit_count == 32
    # The uniform evaluator returns zero values, so any nonzero root value can
    # only come from the distance-margin term.
    assert plain_tree.root.value_sum == pytest.approx(0.0, abs=1e-9)
    assert shaped_tree.root.value_sum != pytest.approx(0.0, abs=1e-9)
