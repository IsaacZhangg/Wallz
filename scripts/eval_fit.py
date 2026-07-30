"""Measure how well checkpoints fit a fixed set of replay shards.

Round metrics report the *mean* loss across a training round, which mixes
the high-loss early steps into the number and so cannot answer "did this
round actually fit the data". This evaluates finished checkpoints against
the same held-fixed positions, which can.

    uv run python scripts/eval_fit.py --shards artifacts/eval-shards \
        --checkpoint a=artifacts/checkpoints/r31.pt b=.../trackA.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wallzero.encoding import encode_state
from wallzero.network import load_checkpoint, select_device
from wallzero.replay import load_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True, help="label=path")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--limit", type=int, default=30_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    examples = []
    for shard in sorted(args.shards.glob("chunk-*.npz")):
        examples.extend(load_shard(shard))
        if len(examples) >= args.limit:
            break
    examples = examples[: args.limit]

    inputs = torch.from_numpy(np.stack([encode_state(e.state) for e in examples])).to(
        device
    )
    targets = torch.tensor([e.value for e in examples], dtype=torch.float32).to(device)
    policies = torch.from_numpy(
        np.stack([np.asarray(e.policy, dtype=np.float32) for e in examples])
    ).to(device)

    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        model, _ = load_checkpoint(Path(path), device=device)
        model.eval()
        policy_total = 0.0
        value_total = 0.0
        seen = 0
        with torch.no_grad():
            for start in range(0, len(examples), args.batch):
                stop = start + args.batch
                logits, value, *_ = model(inputs[start:stop])
                logp = nn.functional.log_softmax(logits, dim=1)
                target = policies[start:stop]
                policy_total += float(-(target * logp).sum(dim=1).sum())
                value_total += float(
                    nn.functional.mse_loss(
                        value.squeeze(-1), targets[start:stop], reduction="sum"
                    )
                )
                seen += stop - start if stop <= len(examples) else len(examples) - start
        print(
            json.dumps(
                {
                    "label": label,
                    "checkpoint": path,
                    "positions": seen,
                    "policy_cross_entropy": round(policy_total / seen, 4),
                    "value_mse": round(value_total / seen, 4),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
