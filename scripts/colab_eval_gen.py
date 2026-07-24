"""Pre-declared gate-3 suite: frozen new best vs control and vs prior gate winner.

Declared 2026-07-24 before any generation of this campaign phase was judged:
- subject: frozen /content/wallzero-output/best.pt at evaluation time;
- opponent A: uniform-prior MCTS at equal 192-simulation budget;
- opponent B: the round-14 gate-2 winner (best-gate-passed.pt, sha 48c9b6b9...)
  uploaded to /content/wallzero-eval-opponents/best-gate-passed.pt;
- three fresh seeds x 200 paired-opening games per opponent, pooled;
- claims: CI excluding 0.5 vs A sustains gate 2; CI excluding 0.5 vs B is the
  measured generation-over-generation improvement (gate 3).
Seeds 1_150_xxx have never been used in training, arenas, or prior suites.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from wallzero.evaluation import report_dict, run_match
from wallzero.mcts import UniformEvaluator
from wallzero.network import TorchEvaluator, load_checkpoint, select_device

OUTPUT = Path("/content/wallzero-output")
OPPONENT_PATH = Path("/content/wallzero-eval-opponents/best-gate-passed.pt")
EVAL_PATH = OUTPUT / "strength-eval.jsonl"
SIMULATIONS = 192
LEAF_BATCH = 8
# Seed series are subject-scoped and never reused: 1_150_xxx judged the
# round-22 best (2026-07-24 afternoon); 1_160_xxx pre-declared for the 30M
# round-24 best before any of its evaluation games were played.
SEEDS = (1_160_101, 1_160_202, 1_160_303)

device = select_device("cuda")
best_model, best_payload = load_checkpoint(OUTPUT / "best.pt", device=device)
best = TorchEvaluator(best_model, device)
print(
    json.dumps(
        {"event": "eval-start", "best_metadata": best_payload.get("metadata", {})},
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


opponents = [("uniform-mcts", UniformEvaluator())]
if OPPONENT_PATH.exists():
    prior_model, _ = load_checkpoint(OPPONENT_PATH, device=device)
    opponents.append(("gate2-winner-r14", TorchEvaluator(prior_model, device)))
else:
    print(json.dumps({"event": "opponent-missing", "path": str(OPPONENT_PATH)}))

for name, opponent in opponents:
    for seed in SEEDS:
        report = run_match(
            f"gate3-best-vs-{name}-seed{seed}",
            best,
            opponent,
            games=200,
            simulations=SIMULATIONS,
            seed=seed,
            leaf_batch=LEAF_BATCH,
            workers=10,
        )
        emit(report_dict(report))
