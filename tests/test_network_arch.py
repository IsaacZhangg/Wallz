from __future__ import annotations

from pathlib import Path

import torch

from wallzero.network import (
    GlobalPoolBias,
    NetworkConfig,
    PolicyValueNet,
    load_checkpoint,
    save_checkpoint,
)

_FIXED_SCALE = NetworkConfig(
    blocks=6,
    channels=32,
    value_hidden=64,
    distance_head=True,
    norm_kind="fixscaleonenorm",
    gpool_blocks=(3, 5),
    gpool_channels=8,
)
_LEGACY = NetworkConfig(blocks=2, channels=16, value_hidden=32, distance_head=True)


def test_fixed_scale_forward_shapes_and_value_range() -> None:
    net = PolicyValueNet(_FIXED_SCALE).eval()
    policy, value = net(torch.randn(4, _FIXED_SCALE.input_planes, 9, 9))
    assert policy.shape == (4, _FIXED_SCALE.action_size)
    assert value.shape == (4,)
    assert bool((value.abs() <= 1.0).all())


def test_gpool_blocks_land_on_configured_positions() -> None:
    net = PolicyValueNet(_FIXED_SCALE)
    pooled = [
        index + 1
        for index, block in enumerate(net.tower)
        if isinstance(getattr(block, "gpool", None), GlobalPoolBias)
    ]
    assert tuple(pooled) == _FIXED_SCALE.gpool_blocks


def test_trunk_scale_follows_the_variance_budget() -> None:
    net = PolicyValueNet(_FIXED_SCALE)
    # Block i reads a trunk of idealised variance i+1 and must undo it.
    for index, block in enumerate(net.tower):
        assert block.norm1.scale == 1.0 / ((index + 1.0) ** 0.5)
        assert block.norm2.scale == 1.0


def test_dual_heads_split_the_loss_and_put_inference_last() -> None:
    net = PolicyValueNet(_FIXED_SCALE)
    heads = net.forward_train_heads(torch.randn(3, _FIXED_SCALE.input_planes, 9, 9))
    assert len(heads) == 2
    assert sum(weight for weight, *_ in heads) == 1.0
    assert heads[0][0] == _FIXED_SCALE.bn_head_loss_weight
    # The inference path is last, and matches what forward() serves.
    net.eval()
    inputs = torch.randn(3, _FIXED_SCALE.input_planes, 9, 9)
    with torch.no_grad():
        served_policy, served_value = net(inputs)
        _, train_policy, train_value, _ = net.forward_train_heads(inputs)[-1]
    assert torch.allclose(served_policy, train_policy, atol=1e-6)
    assert torch.allclose(served_value, train_value, atol=1e-6)


def test_legacy_net_still_trains_through_a_single_head() -> None:
    net = PolicyValueNet(_LEGACY)
    heads = net.forward_train_heads(torch.randn(2, _LEGACY.input_planes, 9, 9))
    assert len(heads) == 1
    assert heads[0][0] == 1.0


def test_legacy_state_dict_keys_are_unchanged() -> None:
    # Guards the deployed b24c256 checkpoints: adding the fixed-scale
    # architecture must not rename or add anything on the legacy path.
    keys = set(PolicyValueNet(_LEGACY).state_dict())
    assert any(key.startswith("policy_head.") for key in keys)
    assert any(key.startswith("tower.0.bn1.") for key in keys)
    assert not any(key.startswith(("bn_heads.", "inference_heads.")) for key in keys)


def test_distance_head_receives_gradient_on_both_architectures() -> None:
    # The auxiliary distance target was silently skipped on the fixed-scale
    # net when training probed for a top-level distance_output attribute.
    import numpy as np

    from wallzero.constants import ACTION_SIZE
    from wallzero.game import State
    from wallzero.replay import TrainingExample
    from wallzero.training import TrainConfig, train_candidate

    rng = np.random.default_rng(0)
    policy = rng.random(ACTION_SIZE).astype("float32")
    examples = [
        TrainingExample(
            state=State.initial(),
            policy=policy / policy.sum(),
            value=1.0,
            policy_weight=1.0,
            weight=1.0,
            own_distance=8,
            opp_distance=8,
        )
    ] * 8

    for config in (_FIXED_SCALE, _LEGACY):
        net = PolicyValueNet(config)
        train_candidate(
            net,
            examples,
            torch.device("cpu"),
            TrainConfig(steps=2, batch_size=4, seed=1),
        )
        distance_params = [
            parameter
            for name, parameter in net.named_parameters()
            if "distance_output" in name
        ]
        assert distance_params, config.norm_kind
        assert all(parameter.grad is not None for parameter in distance_params), (
            config.norm_kind
        )


def test_checkpoint_round_trip_preserves_outputs(tmp_path: Path) -> None:
    for config in (_FIXED_SCALE, _LEGACY):
        net = PolicyValueNet(config).eval()
        inputs = torch.randn(2, config.input_planes, 9, 9)
        with torch.no_grad():
            before = net(inputs)
        destination = tmp_path / f"{config.norm_kind}.pt"
        save_checkpoint(destination, net)
        restored, _ = load_checkpoint(destination)
        restored.eval()
        with torch.no_grad():
            after = restored(inputs)
        assert restored.config == config
        assert torch.allclose(before[0], after[0], atol=1e-6)
        assert torch.allclose(before[1], after[1], atol=1e-6)
