"""Run the whole WallZero flywheel inside one Colab session.

Generation and training alternate in a single long-lived process, so the
session needs no per-step CLI round trips: the driver is started once and
everything after that is reading its log. State is mirrored to Google
Drive after every adopted round, which is also how checkpoints reach the
other machines — a Colab session can die at any moment and the only thing
that must survive is Drive.

Step budget follows KataGo's `MAX_TRAIN_PER_DATA=8` on *fresh* rows, which
is the correct rule when the window is the run's own accumulated output.
An inherited window whose rows never received their 8x has to be paid off
once by a sized run first (see --debt-steps); the fresh-row budget can
never pay that debt down.

    python -m scripts.colab_flywheel --hours 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from wallzero.campaign import run_self_play_chunk, run_training_round
from wallzero.pipeline import preset

OUTPUT = Path("/content/wallzero-output")
DRIVE = Path("/content/drive/MyDrive/wallzero")
BATCH = 512
MAX_TRAIN_PER_DATA = 8
CONSTANT_LR = 3e-4


def log(message: str) -> None:
    print(f"[flywheel] {time.strftime('%Y-%m-%dT%H:%M:%S')} {message}", flush=True)


def fresh_rows(since: int) -> tuple[int, int]:
    """Positions recorded in chunk-metrics after line `since`, and the new count."""
    metrics = OUTPUT / "chunk-metrics.jsonl"
    if not metrics.is_file():
        return 0, since
    lines = metrics.read_text(encoding="utf-8").splitlines()
    total = sum(json.loads(line).get("examples", 0) for line in lines[since:] if line)
    return total, len(lines)


def mirror_to_drive() -> None:
    if not DRIVE.parent.is_dir():
        return
    (DRIVE / "checkpoints").mkdir(parents=True, exist_ok=True)
    (DRIVE / "replay").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT / "best.pt", DRIVE / "checkpoints" / "best-colab.pt")
    for shard in (OUTPUT / "replay").glob("chunk-*.npz"):
        target = DRIVE / "replay" / shard.name
        if not target.exists():
            shutil.copy2(shard, target)
    for name in ("campaign-state.json", "chunk-metrics.jsonl", "round-metrics.jsonl"):
        source = OUTPUT / name
        if source.is_file():
            shutil.copy2(source, DRIVE / name)
    log("mirrored to Drive")


def generate(base, games: int, workers: int, leaf_batch: int) -> None:
    run_self_play_chunk(
        OUTPUT,
        base,
        games=games,
        simulations=800,
        temperature_moves=24,
        endgame_temperature=0.05,
        workers=workers,
        leaf_batch=leaf_batch,
        full_search_probability=0.25,
        fast_simulations=200,
        surprise_weighting=True,
        forced_playout_scale=2.0,
        root_policy_temperature=1.2,
        dirichlet_concentration=10.83,
        device_name="cuda",
    )


def train(base, steps: int) -> None:
    run_training_round(
        OUTPUT,
        replace(
            base,
            replay_window=10_000_000,
            train=replace(
                base.train,
                learning_rate=CONSTANT_LR,
                minimum_learning_rate=CONSTANT_LR,
            ),
        ),
        training_steps=steps,
        arena_games=0,
        device_name="cuda",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=20.0)
    parser.add_argument("--games", type=int, default=192, help="games per chunk")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--leaf-batch", type=int, default=24)
    parser.add_argument("--trigger", type=int, default=40_000, help="fresh positions")
    parser.add_argument(
        "--debt-steps",
        type=int,
        default=0,
        help="one sized training run before generating, for an inherited window",
    )
    args = parser.parse_args()

    # A100 tensor cores: the trunk is fp32 convolutions, and TF32 is a free
    # ~3x on them with no measurable accuracy cost at this width.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "replay").mkdir(exist_ok=True)
    base = preset("colab")
    deadline = time.monotonic() + args.hours * 3600.0
    log(f"start; device {torch.cuda.get_device_name(0)}; until +{args.hours}h")

    if args.debt_steps:
        log(f"paying inherited-window debt: {args.debt_steps} steps")
        train(base, args.debt_steps)
        mirror_to_drive()

    _, baseline = fresh_rows(0)
    rounds = 0
    while time.monotonic() < deadline:
        generate(base, args.games, args.workers, args.leaf_batch)
        fresh, baseline_now = fresh_rows(baseline)
        log(f"{fresh} fresh positions since last round")
        if fresh < args.trigger:
            continue
        steps = max(150, fresh * MAX_TRAIN_PER_DATA // BATCH)
        log(f"training {steps} steps on {fresh} fresh rows")
        train(base, steps)
        baseline = baseline_now
        rounds += 1
        mirror_to_drive()
        log(f"round {rounds} adopted")
    log(f"deadline reached after {rounds} rounds")


if __name__ == "__main__":
    main()
