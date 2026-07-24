"""Measure self-play throughput across worker counts and leaf batch sizes.

Run locally before spending metered GPU time, e.g.:

    uv run python scripts/bench_selfplay.py --device cpu --games 8 \
        --simulations 96 --workers 1 4 8 --leaf-batch 1 8
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from wallzero.mcts import MCTSConfig
from wallzero.network import (
    NetworkConfig,
    PolicyValueNet,
    TorchEvaluator,
    select_device,
)
from wallzero.parallel import generate_self_play_parallel
from wallzero.selfplay import SelfPlayConfig, generate_self_play


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--leaf-batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = select_device(args.device)
    torch.manual_seed(args.seed)
    model = PolicyValueNet(NetworkConfig(channels=args.channels, blocks=args.blocks))
    evaluator = TorchEvaluator(model, device)
    for workers in args.workers:
        for leaf_batch in args.leaf_batch:
            mcts_config = MCTSConfig(
                simulations=args.simulations, leaf_batch=leaf_batch
            )
            self_play_config = SelfPlayConfig(
                games=args.games,
                parallel_games=args.parallel_games,
                temperature_moves=24,
                seed=args.seed,
            )
            started = time.monotonic()
            if workers > 1:
                _, stats = generate_self_play_parallel(
                    evaluator.evaluate_planes,
                    mcts_config,
                    self_play_config,
                    workers=workers,
                )
            else:
                _, stats = generate_self_play(evaluator, mcts_config, self_play_config)
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "device": str(device),
                        "workers": workers,
                        "leaf_batch": leaf_batch,
                        "games": stats.games,
                        "positions": stats.positions,
                        "elapsed_seconds": round(elapsed, 2),
                        "positions_per_second": round(stats.positions / elapsed, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
