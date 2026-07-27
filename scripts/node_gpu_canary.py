"""GPU stability canary for clock-offset steps on the generation node.

First run (at stock clocks) records reference outputs for a fixed batch.
Every later run (at each offset step) re-runs the same kernels and demands
bitwise-identical results, interleaved with 60s of mixed-size stress
forwards. Clocks do not change math — any deviation is a silent compute
error and the offset must come down.

Exit codes: 0 pass, 2 mismatch.
"""

import json
import os
import sys
import time

import numpy as np
import torch

from wallzero.network import TorchEvaluator, load_checkpoint

REF = "/tmp/wz-oc/reference.npz"

model, _ = load_checkpoint("output/wallzero-output/best.pt", device="cuda")
evaluator = TorchEvaluator(model, torch.device("cuda"))
rng = np.random.default_rng(99)
batch = rng.random((512, 13, 9, 9), dtype=np.float32)

for _ in range(3):  # profiling-executor warmup: steady-state kernels only
    evaluator.evaluate_planes(batch)
logits, values = evaluator.evaluate_planes(batch)
os.makedirs("/tmp/wz-oc", exist_ok=True)
if not os.path.exists(REF):
    np.savez(REF, logits=logits, values=values)
    print(json.dumps({"canary": "reference-created"}), flush=True)
    sys.exit(0)

reference = np.load(REF)


def check(tag: str) -> None:
    out_logits, out_values = evaluator.evaluate_planes(batch)
    if not (
        np.array_equal(out_logits, reference["logits"])
        and np.array_equal(out_values, reference["values"])
    ):
        print(json.dumps({"canary": "FAIL", "at": tag}), flush=True)
        sys.exit(2)


check("initial")
sizes = (64, 256, 512, 1024)
deadline = time.monotonic() + 60
iterations = 0
while time.monotonic() < deadline:
    stress = rng.random((sizes[iterations % 4], 13, 9, 9), dtype=np.float32)
    evaluator.evaluate_planes(stress)
    iterations += 1
    if iterations % 25 == 0:
        check(f"stress-{iterations}")
check("final")
print(json.dumps({"canary": "PASS", "iterations": iterations}), flush=True)
