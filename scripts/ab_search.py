"""A/B test search-side settings with one fixed network, locally and free.

Both sides use the same checkpoint, so any score difference is caused purely
by the search configuration under test.

    uv run python scripts/ab_search.py --checkpoint path.pt --games 40 \
        --simulations 200 --lcb --distance-weight 0.15
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

from wallzero.arena import ArenaConfig, evaluate_candidate
from wallzero.mcts import MCTSConfig
from wallzero.network import TorchEvaluator, load_checkpoint, select_device


def wilson(successes: float, trials: int, z: float = 1.959964) -> tuple[float, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--leaf-batch", type=int, default=8)
    parser.add_argument("--parallel-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=90_001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lcb", action="store_true")
    parser.add_argument("--distance-weight", type=float, default=0.0)
    parser.add_argument("--distance-scale", type=float, default=6.0)
    parser.add_argument("--subtree-bias", type=float, default=0.0)
    parser.add_argument("--subtree-alpha", type=float, default=0.8)
    # Exploration constants: inherited from AlphaZero and never tuned for
    # Quoridor's branching factor. Both sides share the network, so a sweep
    # here is a pure search measurement.
    parser.add_argument("--pb-c-init", type=float, default=None)
    parser.add_argument("--fpu", type=float, default=None)
    args = parser.parse_args()

    device = select_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    evaluator = TorchEvaluator(model, device)

    baseline = MCTSConfig(
        simulations=args.simulations,
        dirichlet_fraction=0.0,
        leaf_batch=args.leaf_batch,
    )
    variant = replace(
        baseline,
        lcb_selection=args.lcb,
        distance_utility_weight=args.distance_weight,
        distance_utility_scale=args.distance_scale,
        subtree_bias_lambda=args.subtree_bias,
        subtree_bias_alpha=args.subtree_alpha,
        pb_c_init=args.pb_c_init if args.pb_c_init is not None else baseline.pb_c_init,
        fpu_reduction=args.fpu if args.fpu is not None else baseline.fpu_reduction,
    )

    started = time.monotonic()
    result = evaluate_candidate(
        evaluator,
        evaluator,
        variant,
        ArenaConfig(
            games=args.games,
            parallel_games=min(args.games, args.parallel_games),
            promotion_threshold=0.5,
            seed=args.seed,
        ),
        incumbent_mcts_config=baseline,
    )
    score = result.candidate_wins + 0.5 * result.draws
    low, high = wilson(score, result.games)
    print(
        json.dumps(
            {
                "schema": "wallzero.ab-search.v1",
                "checkpoint": args.checkpoint,
                "simulations": args.simulations,
                "variant": {
                    "lcb_selection": args.lcb,
                    "distance_utility_weight": args.distance_weight,
                    "subtree_bias_lambda": args.subtree_bias,
                    "pb_c_init": variant.pb_c_init,
                    "fpu_reduction": variant.fpu_reduction,
                },
                "games": result.games,
                "variant_wins": result.candidate_wins,
                "baseline_wins": result.incumbent_wins,
                "draws": result.draws,
                "score": round(score / result.games, 4),
                "ci95": [round(low, 4), round(high, 4)],
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
