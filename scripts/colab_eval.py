"""Independent strength suite for the frozen best checkpoint on the A100.

Opponents are rule-only controls and frozen prior checkpoints uploaded to
/content/wallzero-eval-opponents. This is not the promotion gate; results
feed honest strength claims with 95% Wilson intervals.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from wallzero.evaluation import report_dict, run_match
from wallzero.mcts import Evaluator, UniformEvaluator
from wallzero.network import TorchEvaluator, load_checkpoint, select_device

OUTPUT = Path("/content/wallzero-output")
OPPONENTS = Path("/content/wallzero-eval-opponents")
EVAL_PATH = OUTPUT / "strength-eval.jsonl"
SIMULATIONS = 192
SEED = 977_001

device = select_device("cuda")
best_model, best_payload = load_checkpoint(OUTPUT / "best.pt", device=device)
best = TorchEvaluator(best_model, device)
print(
    json.dumps(
        {
            "event": "eval-start",
            "best_metadata": best_payload.get("metadata", {}),
        },
        sort_keys=True,
        default=str,
    ),
    flush=True,
)


def emit(payload: dict[str, object]) -> None:
    with EVAL_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(payload, sort_keys=True), flush=True)


matches: list[tuple[str, Evaluator, int, int]] = [
    ("best-vs-uniform-mcts-equal-budget", UniformEvaluator(), 200, SEED + 1),
]
for index, name in enumerate(("random-init", "round-00000", "round-00001")):
    path = OPPONENTS / f"{name}.pt"
    if not path.exists():
        print(json.dumps({"event": "opponent-missing", "name": name}), flush=True)
        continue
    model, _ = load_checkpoint(path, device=device)
    matches.append(
        (f"best-vs-{name}", TorchEvaluator(model, device), 100, SEED + 10 + index)
    )

for label, opponent, games, seed in matches:
    report = run_match(
        label,
        best,
        opponent,
        games=games,
        simulations=SIMULATIONS,
        seed=seed,
    )
    emit(report_dict(report))
