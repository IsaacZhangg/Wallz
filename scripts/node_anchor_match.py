"""Anchor-ladder strength check for the era-3 node pipeline.

Plays the current best.pt against the standing anchor checkpoint under the
deterministic competitive search (192 sims, no noise, leaf_batch 8, fixed
seed) and appends the result to strength-timeline.jsonl. This is the
instrument era 2 lacked: 28 rounds ran with no external measurement and the
flat lineage was only caught by a human playing it.

Outcomes:
  score >= 0.60  -> the candidate becomes the new anchor (old anchors kept)
  score <  0.40  -> writes REGRESSION-ALARM; the flywheel suspends training
                    rounds (generation continues) until a human removes it
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

from wallzero.arena import ArenaConfig, evaluate_candidate
from wallzero.mcts import MCTSConfig
from wallzero.network import TorchEvaluator, load_checkpoint, select_device

GAMES = 100
SIMULATIONS = 192
SEED = 1_200_001
PROMOTE_AT = 0.60
ALARM_AT = 0.40


def wilson(successes: float, trials: int, z: float = 1.959964) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    phat = successes / trials
    denom = 1 + z * z / trials
    center = phat + z * z / (2 * trials)
    margin = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
    return (center - margin) / denom, (center + margin) / denom


if __name__ == "__main__":
    root = Path(os.environ.get("WALLZERO_HOME", Path.home() / "wallzero"))
    output = Path(
        os.environ.get("WALLZERO_OUTPUT", root / "output" / "wallzero-output")
    ).expanduser()
    anchors = root / "anchors"
    anchor_path = anchors / "anchor-current.pt"
    device = select_device(os.environ.get("WALLZERO_DEVICE", "cuda"))

    cand_model, _ = load_checkpoint(output / "best.pt", device=device)
    anchor_model, _ = load_checkpoint(anchor_path, device=device)
    cfg = MCTSConfig(simulations=SIMULATIONS, dirichlet_fraction=0.0, leaf_batch=8)
    started = time.monotonic()
    result = evaluate_candidate(
        TorchEvaluator(cand_model, device),
        TorchEvaluator(anchor_model, device),
        cfg,
        ArenaConfig(
            games=GAMES,
            parallel_games=10,
            promotion_threshold=0.5,
            seed=SEED,
        ),
        incumbent_mcts_config=cfg,
    )
    score = (result.candidate_wins + 0.5 * result.draws) / result.games
    low, high = wilson(result.candidate_wins + 0.5 * result.draws, result.games)

    verdict = "flat"
    if score >= PROMOTE_AT:
        verdict = "promoted"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(output / "best.pt", anchors / f"anchor-{stamp}.pt")
        shutil.copy2(output / "best.pt", anchor_path)
    elif score < ALARM_AT:
        verdict = "REGRESSION"
        (root / "REGRESSION-ALARM").write_text(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')} score {score:.3f} vs anchor\n"
        )

    record = {
        "schema": "wallzero.anchor-match.v1",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "games": result.games,
        "candidate_wins": result.candidate_wins,
        "anchor_wins": result.incumbent_wins,
        "draws": result.draws,
        "score": round(score, 4),
        "ci95": [round(low, 4), round(high, 4)],
        "simulations": SIMULATIONS,
        "seed": SEED,
        "verdict": verdict,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    with open(output / "strength-timeline.jsonl", "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)
