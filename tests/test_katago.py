from __future__ import annotations

from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch

from wallzero.campaign import run_self_play_chunk, run_training_round
from wallzero.game import State
from wallzero.mcts import MCTSConfig, SearchTree, UniformEvaluator, run_batched_search
from wallzero.network import load_checkpoint
from wallzero.pipeline import preset
from wallzero.replay import load_shard, save_shard
from wallzero.selfplay import SelfPlayConfig, generate_self_play

_KATA_MCTS = MCTSConfig(
    simulations=48,
    max_plies=48,
    leaf_batch=4,
    forced_playout_scale=2.0,
    root_policy_temperature=1.2,
    dirichlet_concentration=10.83,
)


def test_forced_playouts_and_pruned_policy_stay_normalized() -> None:
    tree = SearchTree.from_state(State.initial())
    run_batched_search(
        [tree],
        UniformEvaluator(),
        _KATA_MCTS,
        add_noise=True,
        rngs=[np.random.default_rng(3)],
    )
    pruned = tree.pruned_policy(_KATA_MCTS)
    raw = tree.policy()
    assert pruned.sum() == np.float32(1.0) or abs(float(pruned.sum()) - 1.0) < 1e-5
    # Pruning removes exploration mass: never more spread than the raw target.
    assert np.count_nonzero(pruned) <= np.count_nonzero(raw)
    assert tree.root.visit_count == 48


def test_playout_cap_randomization_mixes_policy_weights() -> None:
    examples, stats = generate_self_play(
        UniformEvaluator(),
        _KATA_MCTS,
        SelfPlayConfig(
            games=2,
            parallel_games=2,
            temperature_moves=6,
            seed=11,
            full_search_probability=0.5,
            fast_simulations=8,
            surprise_weighting=True,
        ),
    )
    assert stats.games == 2
    weights = {example.policy_weight for example in examples}
    assert weights <= {0.0, 1.0} and len(weights) == 2
    for example in examples:
        assert example.weight >= 0.0
        assert example.own_distance >= 0
        assert example.opp_distance >= 0
        if example.policy_weight == 1.0:
            assert abs(float(example.policy.sum()) - 1.0) < 1e-4


def test_replay_v2_round_trip_preserves_kata_fields(tmp_path: Path) -> None:
    examples, _ = generate_self_play(
        UniformEvaluator(),
        _KATA_MCTS,
        SelfPlayConfig(
            games=1,
            parallel_games=1,
            temperature_moves=6,
            seed=17,
            full_search_probability=0.5,
            fast_simulations=8,
            surprise_weighting=True,
        ),
    )
    shard = tmp_path / "chunk-00000.npz"
    save_shard(shard, examples)
    restored = load_shard(shard)
    for left, right in zip(examples, restored, strict=True):
        assert left.state == right.state
        assert left.policy_weight == right.policy_weight
        assert abs(left.weight - right.weight) < 1e-2
        assert left.own_distance == right.own_distance
        assert left.opp_distance == right.opp_distance


def test_gateless_round_adopts_candidate_with_distance_head(tmp_path: Path) -> None:
    config = dc_replace(preset("smoke"), replay_window=500)
    output = tmp_path / "campaign"
    run_self_play_chunk(output, config, games=2, simulations=4, device_name="cpu")
    incumbent, _ = load_checkpoint(Path(output) / "best.pt")
    head_config = dc_replace(config.network, distance_head=True)
    run_training_round(
        output,
        config,
        training_steps=2,
        arena_games=0,
        candidate_network=head_config,
        warm_start=True,
        device_name="cpu",
    )
    best, payload = load_checkpoint(Path(output) / "best.pt")
    assert best.config.distance_head is True
    assert payload["metadata"]["status"] == "promoted-gateless"
    # Warm start keeps the incumbent trunk (modulo the two training steps);
    # a fresh random init would differ by orders of magnitude more.
    assert torch.allclose(
        best.state_dict()["stem.0.weight"],
        incumbent.state_dict()["stem.0.weight"],
        atol=0.05,
    )
