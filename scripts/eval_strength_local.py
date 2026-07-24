"""Resumable local strength suite for the frozen best checkpoint.

Each match appends one durable JSON line; rerunning skips finished labels.
Inputs are the locally restored A100 campaign checkpoints, so results are
independent of any live Colab session.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wallzero.evaluation import report_dict, run_match
from wallzero.mcts import Evaluator, UniformEvaluator
from wallzero.network import TorchEvaluator, load_checkpoint, select_device

SIMULATIONS = 192
SEED = 977_001


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--opponents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    best_model, payload = load_checkpoint(args.best, device=device)
    best = TorchEvaluator(best_model, device)
    done = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["label"])

    matches: list[tuple[str, Evaluator, int]] = [
        ("best-vs-uniform-mcts-equal-budget", UniformEvaluator(), SEED + 1),
    ]
    for index, name in enumerate(("random-init", "round-00000", "round-00001")):
        path = args.opponents / f"{name}.pt"
        if path.exists():
            model, _ = load_checkpoint(path, device=device)
            matches.append(
                (f"best-vs-{name}", TorchEvaluator(model, device), SEED + 10 + index)
            )

    print(
        json.dumps(
            {
                "event": "eval-start",
                "device": str(device),
                "best_metadata": payload.get("metadata", {}).get("status"),
                "pending": [label for label, _, _ in matches if label not in done],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for label, opponent, seed in matches:
        if label in done:
            continue
        report = run_match(
            label,
            best,
            opponent,
            games=args.games,
            simulations=SIMULATIONS,
            seed=seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report_dict(report), sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(report_dict(report), sort_keys=True), flush=True)


main()
