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
- KataGo's self-play efficiency package (see `docs/katago-adaptations.md`):
  playout cap randomization, forced playouts with policy target pruning,
  shaped Dirichlet noise, root policy softmax temperature, policy surprise
  weighting, exact shortest-path distance auxiliary targets (the ownership
  analog), and gateless training rounds — all opt-in, with defaults that
  reproduce the original pipeline bit-for-bit;
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

## Recorded status (2026-07-26, node efficiency pass)

Generation on the GTX 1080 measured 4.42 pos/s on a full 96-game
production-recipe chunk (5,225 positions, 19.7 min) against 3.92-4.01
across the five chunks before it — **+11.6%**, attributable to frozen
TorchScript inference (`torch.jit.trace` + `freeze` +
`optimize_for_inference`: batchnorm folding plus NNC elementwise/SE
fusion; +11.4% in the isolated kernel bench at batch 512, max logit
deviation 1.8e-4 — the same order as batchnorm folding alone). The freeze
is gated to fp32 CUDA inference (pre-Ampere) because autocast does not
apply inside TorchScript; the A100 bf16 path and the Mac's MPS path keep
the eager module. The eval server also gained an async submit/collect
pipeline (pinned staging buffers, CUDA events) overlapping queue draining
and reply serialization with GPU compute — measured neutral on top of the
freeze: back-to-back forwards with zero server logic draw the same
~150-160W as live generation, so the server was not leaving meaningful GPU
idle and power draw is not a utilization proxy for this workload (the
wedge-log 176-205W was a different phase mix). Measured and rejected on
this card: fp16 inference (pseudo-half, 7-10% *slower*, 1000x the
deviation — Pascal has no usable fp16), inference-side cuDNN autotune and
eager batchnorm folding (≤1% each), and fp16-vs-fp32 training at batch 512
(now identical at 1.045 vs 1.047 s/step once cuDNN autotune is active; the
07-25 bench that showed fp16 at 57 min predated autotune). Training rounds
stay ~52 min — that is the card's fp32 compute floor for the 3,000-step
recipe. Known remaining headroom: GPU Boost holds the SM at 1670 MHz in P2
(boost table allows 1974); raising it needs Coolbits in xorg.conf plus an
X restart, deferred as a user decision. With ~20-min chunks the flywheel
now trains (52 min) after every ~80 min of generation.

## Recorded status (2026-07-26 overnight)

First full night of the free-fleet flywheel: rounds 32-35 trained, adopted,
and secured (all gateless, 3,000 steps on the 60K window; 52-56 min each on
the GTX 1080), while the fleet generated **91,868 fresh kata positions** —
45,213 from the PC (9 chunks at 4.27 pos/s after the bf16 fix below) and
46,655 from the Mac (10 chunks via a single-process MPS loop; the
multiprocess self-play path deadlocks on MPS and must not be used there).
Mac shards are renumbered into the PC replay sequence at each pre-training
pause. Sanity matches (40 games, 192 sims, paired openings — sanity checks,
not strength claims): r32 vs r31 0.525 [0.375, 0.671]; r35 vs r31 0.500
[0.352, 0.648]. Flat short-horizon sanity results are the expected shape
under the data-volume hypothesis; the decisive pre-declared test
(docs/data-volume-test.md) triggers at >=1M fresh positions (~9% there
after one night). Round times: r32 55.9 min, r33 52.2 (cuDNN autotune
active), r34 103 (an accidental duplicate launch halved throughput — same
seed, so artifacts were identical and unharmed), r35 52.3.

The morning after, the PC cycle became self-driving:
`scripts/node_flywheel.sh` (deployed as `~/wallzero/flywheel.sh`) watches
the replay directory and, every 4 new shards from any source, pauses
generation, trains one gateless round, adopts it, and resumes. Mac and
Colab shards still merge manually (renumbered into the PC sequence while
paused or mid-chunk into free indices); the flywheel counts them like any
other shard. Manual PAUSE files win — the flywheel never acts while one it
didn't create exists, and it retries failed rounds after 10 minutes with
generation running. `train_candidate` also gained a prefetch thread
(batch assembly overlaps GPU compute; verified bit-identical to the old
loop). Round 36 was the live measurement and **falsified the speedup
estimate**: 52.3 min vs the 52.2-52.3 baseline — batch assembly was never
a meaningful fraction of the step time on the 1080 (and the GIL limits
overlap for Python-heavy assembly anyway). The change stays because it is
proven bit-identical and costs nothing, but the 10-20% claim is dead;
round time on this card is GPU compute, full stop.

## Recorded status (2026-07-25)

The KataGo-recipe campaign (rounds 25-31: playout cap randomization, forced
playouts with pruned policy targets, shaped Dirichlet, surprise weighting,
exact-BFS distance head, gateless rounds) moved the classical-engine KPI off
zero for the first time: 0.05 (0-18-2) at an 800-simulation budget. Three
plateau hypotheses were then falsified by measurement — legacy-data dilution
(kata-only retrain scored exactly 0.500), per-round overtraining (a
KataGo-norm 600-step round was *underfit* and scored 0.475), and weak search
(LCB selection, distance-margin utility, and subtree value bias correction
all measured neutral, the last under both a crude and a faithful
pattern-based bucketing; all remain config-gated, default off). The surviving
hypothesis is **data volume**, and `docs/data-volume-test.md` pre-declares
the week-scale test with supported/falsified criteria fixed in advance.

Compute now runs as a three-node fleet: an always-on GTX 1080 generation
node (`scripts/node_chunk_kata.py` under a watchdogged loop — after a silent
CUDA stall wedged a chunk for hours at "100% utilization" and near-idle
power draw, the loop kills anything under 80W for 15 minutes or over 3h
total), the M4 Pro for free A/B and evaluation (`scripts/ab_search.py`), and
Colab A100 reserved for the pre-declared evaluation suites. The 1080 also
closes the training loop: a full 3,000-step round of the 30M network takes
~53-57 min (`scripts/node_train_bench.py`; torch's bf16-"support" on Pascal
is emulation, 9.6x slower, now gated off by compute capability in
`training.py`), so generate → train → adopt runs entirely on free hardware
with round 32 the first PC-trained round (55.9 min live; sanity match vs
round 31: 0.525 [0.375, 0.671] over 40 games).

The "silent CUDA stall" was root-caused the same night
(`docs/node-1080-wedge-log.md`): `TorchEvaluator` autocast sent all CUDA
inference through torch's *emulated* bf16 on Pascal, and driver 580's
emulated-bf16 cublasLt kernels can hang inside `cuLaunchKernel` on sm_61
(py-spy native stack: `cublasLtTSTMatmul` spinning in `sched_yield`).
`network.py` now gates inference autocast on compute capability >= 8, the
generation node runs fp32 self-play at sustained 176-205W, and the loop
script lives at `scripts/node_gen_loop.sh` (watchdog threshold 110W — a
wedge can float at 88W with persistence mode on).

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
same-size generations reading as flat at 3.7M parameters pointed to a
capacity ceiling, so round 23 executed the scale-up: a fresh 29.65M-parameter
network (256 channels x 24 blocks) trained 4,000 steps from random
initialization on the 24K-position clean window and **promoted 36-22-2 over
the 3.7M incumbent** in the standard gate. Round 24 then trained the 30M
network on its own first self-play (generated on the A100 at leaf_batch 16 —
96 games in 251s, the same wall time the 3.7M network needed) and promoted
again (0.575). Independent evaluation immediately deflated both promotions:
the round-24 best sustains gate 2 against the uniform control (0.5542, 95%
CI [0.5142, 0.5935], 600 games) but is dead even with the two-day-old
gate-2 winner (0.5000, 95% CI [0.4601, 0.5399], 600 games) — expected after
a single generation of its own self-play, but not progress. More decisively,
the first direct benchmark against the userscript's classical PVS engine
(`scripts/match_vs_classical.mjs`, evaluation-only, production adaptive
settings) ended **0-20 against WallZero** at a 160-simulation budget. The
uniform control that gates 1-3 measure against is a far lower bar than the
classical engine; the score against the classical engine is now the
project's standing real-strength benchmark, and it starts at zero. All self-play games end decisively — the wall-free
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
