# WallZero

WallZero is a full-size 9×9 Quoridor engine built around the AlphaGo Zero /
AlphaZero loop: an exact rules engine, a residual policy-value network, PUCT
Monte Carlo Tree Search, pure self-play, replay training, and candidate-vs-best
arena promotion.

The project is deliberately split in two:

- `wallzero` is the standalone engine and training system.
- `wallz-coach.user120.js` remains the Wallz.gg practice-board adapter. It is
  not coupled to a particular checkpoint and is not used in ranked play.

The neural engine starts from random weights and receives no human games,
openings, strategy labels, or handcrafted position scores. Legal-action masks,
terminal outcomes, board symmetries, an exact rule-derived wall-free endgame
solver, and the rules themselves are the only non-learned game knowledge in
the training loop.

> Strength is an evaluated artifact, not an architectural claim. A checkpoint
> is only called the best model after it beats the incumbent in a color-balanced
> MCTS arena. “Expert” additionally requires a documented evaluation suite and
> enough independent games to make the estimate credible.

## What improves on the reference

[dorakingx/AlphaQuoridor](https://github.com/dorakingx/AlphaQuoridor) is a clear
teaching implementation of the AlphaZero cycle. Its published example trains a
3×3 variant with a small search. WallZero keeps the useful decomposition while
targeting the actual game:

- exact 9×9 rules and the complete 209-action space;
- compact immutable bitboards and bit-parallel path validation;
- a wall-free retrograde endgame solver that supplies exact optimal root
  actions and policy targets during self-play, adjudicates wall-free arena
  positions, and backs up exact values at search leaves;
- separate reporting of repetition draws and move-limit draws in self-play
  and arena metrics;
- current-player canonicalization plus exact left/right augmentation;
- a configurable residual policy-value network;
- batched GPU inference across concurrent self-play games;
- root Dirichlet exploration, temperature scheduling, and tree reuse;
- virtual-loss leaf batching (several distinct leaves per tree per forward
  pass) and multiprocess actors feeding one central batched inference
  server, so GIL-bound tree/rules work no longer serializes GPU use —
  measured 3.5x self-play throughput on an M4 Pro and 7.2x on an A100
  versus the single-process loop, identical data recipe (`wallzero.parallel`;
  `leaf_batch=1` reproduces the original search exactly and stays the default
  for deterministic runs); the same server drives two-model arenas and
  evaluation matches (requests are tagged per model, whole color-pairs stay
  within one worker), cutting a 60-game arena from ~8.5 to ~3.5 minutes;
- replay shards, resumable checkpoints, mixed precision, and deterministic
  seeds;
- color-balanced arena gates with paired uniform-random openings: both games
  of a color pair start from the same rule-derived random prefix, so the gate
  measures the models rather than one deterministic trajectory;
- an independent strength-evaluation module (`wallzero.evaluation`) reporting
  Wilson 95% intervals, kept separate from the promotion gate;
- a versioned JSON analysis protocol for the eventual userscript bridge.

## Architecture

```text
exact rules ──> PUCT self-play ──> replay shards ──> policy/value training
     │                                                       │
     └──────────── candidate-vs-best arena <─────────────────┘
                               │
                         promoted checkpoint
                               │
                     JSON engine protocol (v1)
                               │
                 practice-only userscript adapter
```

The fixed action layout matches the browser engine:

- `0..80`: pawn destination cell;
- `81..144`: horizontal wall anchor;
- `145..208`: vertical wall anchor.

## Local development

Python dependencies and commands use `uv`:

```bash
uv sync --group dev
uv run pytest
uv run ruff check --fix .
uv run ruff format .
uv run ty check
```

Run the smallest end-to-end learning sanity check:

```bash
uv run wallzero smoke --output artifacts/runs/smoke
```

Run a resumable training preset:

```bash
uv run wallzero train \
  --preset bootstrap \
  --output artifacts/runs/bootstrap \
  --device auto
```

`bootstrap` proves that rules, search, replay, optimization, checkpointing, and
arena evaluation work together. It is not an expert-strength budget. The
`colab` and `expert` presets progressively raise network size, simulations,
self-play volume, and arena confidence.

## Colab A100 workflow

The repository includes `scripts/colab_train.py`, a non-interactive entry point
for a named Colab session. A run writes all durable outputs below its selected
output directory:

```text
best.pt                 current promoted model
candidates/             every trained challenger
replay/                 compressed self-play shards
metrics.jsonl           iteration, losses, arena score, timing
run-state.json          resumable iteration metadata
```

The A100 accelerates neural inference and training. Quoridor legal-wall
generation remains partly CPU-bound, so self-play concurrency and inference
batch size matter more than raw GPU utilization alone.

## Engine protocol and browser boundary

The engine accepts one JSON object per line over stdio, or the same request
via localhost HTTP for the practice userscript:

```bash
uv run wallzero serve --checkpoint artifacts/runs/a100-bootstrap/best-independent.pt --http 8787
```

`POST /analyze` takes the identical JSON body and returns the identical
response; `GET /health` reports engine availability. The server binds
127.0.0.1 only. The userscript probes `/health` at load, verifies the schema,
request id, response fingerprint, and move legality on every reply, discards
stale or illegal replies, and falls back to the built-in engine with a visible
"WallZero offline" note whenever the local server is unreachable.

The stdio form accepts one JSON object per line:

```json
{
  "schema": "wallzero.analyze.v1",
  "id": "position-42",
  "state": {
    "pawns": {"p1": {"x": 4, "y": 0}, "p2": {"x": 4, "y": 8}},
    "turn": "p1",
    "walls": [],
    "wallsRemaining": {"p1": 10, "p2": 10},
    "winner": null
  },
  "options": {"simulations": 800, "topMoves": 5}
}
```

Start the JSON-lines process with:

```bash
uv run wallzero serve --checkpoint artifacts/runs/expert/best.pt
```

The response includes the exact state fingerprint, selected legal move, root
value, visit count, and top policy alternatives. The future userscript bridge
must retain the existing fail-closed computer-practice checks and verify the
live position again before displaying or applying a recommendation.

## Strength gates

Training progress is judged in layers:

1. **Rule correctness:** curated jump/wall fixtures, randomized invariant tests,
   and parity checks against the browser engine.
2. **Learning sanity:** policy loss falls, values separate wins/losses, and the
   model reliably beats a random-policy MCTS control.
3. **Promotion:** a challenger exceeds the configured arena threshold with both
   colors against the prior best checkpoint.
4. **Expert claim:** a frozen checkpoint wins a large, independently seeded
   match set against strong non-training opponents, with confidence intervals
   and reproducible configuration recorded.

Self-play can run for a very long time. The pipeline is intentionally resumable;
ending a compute session does not turn an early checkpoint into an expert one.

## Recorded status (2026-07-24)

A durable A100 campaign (`artifacts/runs/a100-bootstrap/`) has completed
twenty gated rounds with seven promotions; every shard, candidate, decision,
and checkpoint is checksummed locally and the campaign resumed across five
Colab VM losses without state regression (round 4 reproduced bit-identically
from a restored bundle). Since round 15 the campaign runs a hybrid split:
self-play chunks are generated on a local M4 Pro (`scripts/local_chunk_gen.py`,
zero compute-unit cost — self-play is actor-CPU-bound and the laptop's cores
outpace Colab's vCPUs), while the A100 runs training and the multiprocess
arena; each chunk's durable metrics record the generating device. Rounds
15-22 produced four promotions (r15, r19, r21, r22); the current best
checkpoint is the round-22 model. The pre-declared gate-3 suite
(`strength-eval-gate3.jsonl`) records both verdicts honestly: the round-22
best **sustains gate 2** against the uniform control (0.5792, 95% CI
[0.5393, 0.6180], 600 games) but its edge over the frozen gate-2 winner is
**not statistically established** (0.5317, 95% CI [0.4917, 0.5713], 600
games) — arena promotions again outran control-relative evidence. Eight
same-size generations reading as flat at 3.7M parameters points to a
capacity ceiling; the recorded next step is a network scale-up, not more
generations at the current size. All self-play games end decisively — the wall-free
solver eliminated draws entirely. Later generations fixed two data-quality
defects: the cold-start exploration override was poisoning trajectory quality
(now parameterized; deep chunks use the preset temperature schedule) and
trimmed-restore chunk renumbering could hide the freshest shards from the
replay window (indices now continue from the maximum).

Independent evaluation (`strength-eval.jsonl`, `diagnostics-16sim.jsonl`) now
supports exactly one recorded claim: the frozen round-14 checkpoint
(`best-gate-passed.pt`, sha256 `48c9b6b9…`) **reliably beats the uniform-prior
MCTS control at an equal 192-simulation budget** — a pre-declared three-seed
suite of 600 paired-opening games scored 0.575 (95% CI [0.535, 0.614]; per
seed 0.535 / 0.578 / 0.613). That passes strength gate 2, so the userscript's
WallZero bridge (localhost HTTP protocol, fingerprint and legality checks per
reply, stale-reply discard, timeouts, honest outage fallback) now ships
enabled; it stays inert unless a local `wallzero serve --http` process answers
its health probe. Earlier checkpoints failed the same gate (r7 0.5175, r11
0.4725, r12 0.526 over 500 games) and per-seed scores once swung 0.64→0.45,
which is why claims here require multi-seed pooled evidence. No "strong" or
"expert" label is claimed: the expert-scale suite (~1,000 games including the
classical PVS engine as an opponent) and further scaling remain open work.
