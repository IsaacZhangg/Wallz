"""Benchmark training-step throughput for the current best network on a node GPU.

Answers one question: can this GPU close the training loop? It times short
bursts of the real `train_candidate` path under each precision mode Pascal
could plausibly use, then projects the wall time of a full kata round
(3,000 steps). Precision is patched around the production code rather than
forked from it, so the measured path is the one a real round would run.

    WALLZERO_OUTPUT=~/wallzero/output/wallzero-output \
    WALLZERO_BENCH_STEPS=30 python scripts/node_train_bench.py

Modes:
  auto - whatever training.py picks on this device (bf16 if reported supported)
  fp16 - force the float16 + GradScaler branch
  fp32 - disable autocast entirely (Pascal's likely winner: fp16 arithmetic is
         crippled on GP104, and bf16 "support" in torch 2.6 may be emulation)
"""

import contextlib
import copy
import json
import os
import time
from pathlib import Path

import torch

from wallzero.network import load_checkpoint
from wallzero.replay import load_replay_window
from wallzero.training import TrainConfig, train_candidate

ROUND_STEPS = 3_000  # a full kata training round


def bench(mode: str, batch_size: int, steps: int, model, examples, device) -> dict:
    candidate = copy.deepcopy(model)
    real_bf16 = torch.cuda.is_bf16_supported
    real_autocast = torch.autocast
    patches = contextlib.ExitStack()
    if mode in ("fp16", "fp32"):
        torch.cuda.is_bf16_supported = lambda *a, **k: False
        patches.callback(setattr, torch.cuda, "is_bf16_supported", real_bf16)
    if mode == "fp32":

        def no_autocast(device_type: str, *args, **kwargs):
            return contextlib.nullcontext()

        torch.autocast = no_autocast
        patches.callback(setattr, torch, "autocast", real_autocast)

    config = TrainConfig(steps=steps, batch_size=batch_size, seed=1)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    try:
        with patches:
            metrics, _ = train_candidate(candidate, examples, device, config)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"mode": mode, "batch_size": batch_size, "error": "oom"}
    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    seconds_per_step = elapsed / steps
    return {
        "mode": mode,
        "batch_size": batch_size,
        "steps": steps,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_step": round(seconds_per_step, 3),
        "projected_round_minutes": round(seconds_per_step * ROUND_STEPS / 60, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
        "policy_loss": round(metrics.policy_loss, 4),
        "value_loss": round(metrics.value_loss, 4),
    }


if __name__ == "__main__":
    output = Path(
        os.environ.get("WALLZERO_OUTPUT", "artifacts/runs/node/wallzero-output")
    ).expanduser()
    steps = int(os.environ.get("WALLZERO_BENCH_STEPS", "30"))
    device = torch.device("cuda")

    properties = torch.cuda.get_device_properties(device)
    print(
        json.dumps(
            {
                "event": "bench-env",
                "gpu": properties.name,
                "sm": f"{properties.major}.{properties.minor}",
                "vram_gb": round(properties.total_memory / 2**30, 1),
                "torch": torch.__version__,
                "bf16_reported": torch.cuda.is_bf16_supported(),
            }
        ),
        flush=True,
    )

    model, _ = load_checkpoint(output / "best.pt", device=device)
    examples = load_replay_window(output / "replay", max_samples=60_000)
    print(
        json.dumps({"event": "bench-data", "replay_samples": len(examples)}),
        flush=True,
    )

    for mode in ("fp32", "fp16", "auto"):
        for batch_size in (256, 512):
            result = bench(mode, batch_size, steps, model, examples, device)
            print(json.dumps({"event": "bench-result", **result}), flush=True)
