# Adopting KataGo's methodology: sequenced plan

Written 2026-07-29 after audit rounds 6-7 (`docs/katago-training-audit.md`).
The goal stated by the user: make WallZero "super efficient and good like
KataGo". This document is the ordered path, with the reasoning for the
order — which matters more than the list, because several of these items
require a fresh network and should not be spent one at a time.

## The organising insight

Rounds 1-6 closed recipe gaps (window, sample reuse, min rows, LR
continuity, weight decay). Round 7 found the remaining lever is not the
recipe at all: **our net is 7-17× too large for one GTX 1080**, and
KataGo's own run spent its first day at b6c96 and its first week below
b15c192. Everything downstream — every A/B, every anchor match, every
hypothesis test — gets cheaper in proportion to the data rate, so net
size is the multiplier on all future work and goes first.

Second: three separate items (net size, BatchNorm removal, nested
bottleneck blocks) all require building a new network. Doing them
sequentially costs three lineage restarts. **Bundle them into one net
change**, measured against the current lineage on the standing KPIs.

## Phase 0 — done

- All-data replay window (era 3), sample-reuse cap of 8, 100K-row minimum
  before training, constant LR (no per-round cosine restarts), KataGo
  weight-decay groups with per-round trunk-norm logging.

## Phase 1 — the net (one bundled change, highest leverage)

**Status: built and tested (commit 8ba3581), not yet cut over.** The
architecture exists behind `norm_kind="fixscaleonenorm"` with
`gpool_blocks=(5, 8)`; `norm_kind` defaults to `"bnorm"` so the deployed
b24c256 lineage is untouched. Measured b10c128 fixscale+gpool+dual-heads
against the production net: **7.26× eval throughput on MPS, 6.23× on the
GTX 1080** (both nets benchmarked back to back under identical load).
What remains is the cutover decision: starting this lineage ends era 3.


1. **Shrink to b10c128 or b15c192.** b10c128 measures 7.1× our current
   eval throughput, b15c192 2.6×. b10c128 matches where we actually are
   by data volume; b15c192 is the conservative option that still nearly
   triples the data rate.
2. **Replace BatchNorm with fixed-variance init + one trunk-end BN**
   with dual heads (80% loss through the BN path, 20% on the BN-free
   heads used at inference). Removes the train/inference discrepancy and
   the weight-norm drift mechanism measured in round 6 at its source,
   rather than fighting it with weight decay.
3. **Global-pooling blocks in the trunk** — even b6c96 makes 2 of 6
   blocks gpool. Our SE blocks are a weaker cousin (channel gating only).
4. **Nested bottleneck blocks** (optional in this phase): 1×1 down, four
   3×3 at reduced width with inner skips, 1×1 up. Their b18c384nbt runs
   at b40c256 speed for nearly b60c320 strength.

Measurement: the new net trains from scratch (shrinking is not a warm
start). Judge it on the standing KPIs — classical-engine score and the
r31 anchor — not on losses. Keep the current lineage's best.pt as a
frozen comparison opponent. Growth back up to b15c192/b20c256 later uses
the warm-start path we already have, triggered by data volume the way
their schedule was.

## Phase 2 — training targets

Ordered by value per unit of work:

1. **Value loss weight 1.0 → 0.72** (their 1.20 × `value_loss_scale`
   0.6). One constant.
2. **SWA/EMA** of weights, factor 1/8, updated ~every half epoch of
   samples. Open since round 1; KataGo credits it with real strength.
3. **Categorical value head + cross-entropy** (win/loss/draw) replacing
   scalar tanh + MSE. Bundle with Phase 1 if the net is being rebuilt
   anyway — it is a head change.
4. **Short-term (TD) value targets** at λ ≈ 0.545 / 0.773 / 0.913 for our
   ~52-ply games. Needs replay schema v3 to store the search's root value
   per position — data we already compute and discard.
5. **Q-value head** on per-move search values weighted by √visits
   (their weight 1.5), **soft policy target** (policy^0.25, weight 8.0),
   **opponent policy head** (0.15). All schema + head work.
6. **Optimizer-state continuity** across rounds — they run one continuous
   optimizer; we re-initialise AdamW every round, and those transients
   are themselves a norm-inflation source.

## Phase 3 — search

1. **Monte-Carlo Graph Search** (`useGraphSearch`). Quoridor is
   transposition-heavy — wall placements commute, which is why our eval
   cache hits 32% — so sharing search statistics across transpositions
   should pay off disproportionately here. Their `docs/GraphSearch.md`
   documents the pitfalls. Largest search project.
2. **Dynamic variance-scaled cPUCT** — search-only, no net changes,
   free to A/B via `ab_search.py`. Promoted in round 5, still unrun.
3. **Uncertainty-weighted playouts** — requires the error-prediction
   heads from Phase 2.4, so it follows the TD targets.

## Phase 4 — data diversity and compute savers

1. **Game-diversity cluster**: `initGamesWithPolicy` (first ~3 moves
   sampled hot from raw policy, no search), `earlyForkGameProb` 0.04 /
   `forkGameProb` 0.01, `sidePositionProb` 0.02. This cluster exists to
   prevent exactly the opening-diversity collapse era 2 exhibited.
2. **`reduceVisits`** on decided games (≥0.9 winrate for 3 turns → 100
   visits, row weight 0.1): straight compute saving, more chunks/day.
3. **Window taper** once data ≫1M rows (`shuffle.py:414-435`).

## What this does not change

The measurement discipline stays: pre-declared seeds, 3×200-game pooled
confirmations, no strength claim from arena promotions or losses. Three
features have already been killed by that rule after looking good at 40-100
games (LCB, distance utility, subtree bias). Nothing here is exempt.
