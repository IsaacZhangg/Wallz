from __future__ import annotations

import numpy as np

from wallzero.constants import ACTION_SIZE
from wallzero.game import State
from wallzero.replay import TrainingExample, load_shard, save_shard


def test_replay_round_trip_preserves_raw_state_policy_and_outcome(tmp_path) -> None:
    state = State.initial().play(13)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[3] = 0.25
    policy[5] = 0.75
    examples = [TrainingExample(state, policy, -1.0)]

    path = save_shard(tmp_path / "replay.npz", examples)
    restored = load_shard(path)

    assert restored[0].state == state
    assert np.allclose(restored[0].policy, policy)
    assert restored[0].value == -1.0


def test_replay_window_ignores_hidden_metadata_files(tmp_path) -> None:
    import numpy as np

    from wallzero.game import State
    from wallzero.replay import TrainingExample, load_replay_window, save_shard

    example = TrainingExample(
        state=State.initial(),
        policy=np.full(209, 1.0 / 209, dtype=np.float32),
        value=1.0,
    )
    save_shard(tmp_path / "chunk-00000.npz", [example])
    (tmp_path / "._chunk-00000.npz").write_bytes(b"appledouble junk")

    window = load_replay_window(tmp_path, max_samples=10)
    assert len(window) == 1
