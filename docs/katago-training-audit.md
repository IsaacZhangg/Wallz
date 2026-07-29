# KataGo training-loop audit (2026-07-28)

Systematic comparison of WallZero's self-play training loop against the
KataGo reference implementation (github.com/lightvector/KataGo — README,
`SelfplayTraining.md`, `python/selfplay/synchronous_loop.sh`,
`python/shuffle.py`, `cpp/configs/training/selfplay1.cfg`), commissioned
after the era-2 window bug to catch any remaining discrepancies of that
class. The synchronous single-machine loop is the correct reference: it is
KataGo's own recipe for exactly our situation (one box, steps run
sequentially).

## Critical discrepancies found (all now fixed)

| # | Dimension | KataGo reference | WallZero era 2 | Status |
|---|-----------|------------------|----------------|--------|
| 1 | Training window | Power-law growing window (`shuffle.py`; async min 250K rows, sync taper scale 50K). Early in a run the window covers ~all data | Newest 60K positions only — generated data never entered training | **Fixed 2026-07-28** (era 3: all-data window; taper needed only when data ≫1M) |
| 2 | Sample reuse | `MAX_TRAIN_PER_DATA=8` sync ("larger numbers may cause overfitting"), `-max-train-bucket-per-new-data 4` async. Hard-enforced samples-per-data-row cap | ~77 samples per data row (3,000 steps × 512 batch per ~20K fresh) — ~10-20× over the warning line | **Fixed 2026-07-28 late** (steps = 8 × fresh/512 ≈ 330/round; era-3's first "3 epochs of window" formula was still ~73× at the cap and was corrected the same night) |
| 3 | Minimum data before training | `SHUFFLE_MINROWS=100000` — no training at all until 100K rows exist | First rounds trained on ~20K positions | **Fixed 2026-07-28 late** (`FLYWHEEL_MIN_SHARDS=20` gate) |

Consequence of #2 worth recording: with the reuse cap, rounds cost ~6 min
instead of ~50, so the GPU spends ~90% of wall time generating. The
era-2 loop both overfit AND paid 8× too much wall time for the privilege.

## Aligned (verified, no action)

- **Full/cheap search mix**: KataGo 600 visits full / 100 cheap at
  p(full)=0.25; ours 800/200 at p(full)=0.25.
- **Root exploration**: rootNoiseEnabled + temperature schedule (KataGo
  0.75 early → 0.15 late, halflife 19; ours temperature_moves=24, endgame
  0.05, shaped Dirichlet total α=10.83, root softmax temp 1.2 per
  KataGoMethods).
- **Gateless adoption**: KataGo's gatekeeper is optional; they recommend
  *some* strength verification early ("make sure the net is actually
  getting stronger") — that role is now the anchor ladder
  (`node_anchor_match.py`, every 5th round, REGRESSION-ALARM brake).
- **Surprise weighting / pruned policy targets / playout cap
  randomization**: adapted earlier from KataGoMethods.md.

## Queued (real differences, deliberately deferred — one variable at a time)

1. **SWA (stochastic weight averaging)**: KataGo averages weights every
   80K samples and credits it with real strength. We have none. Good
   first candidate after the era-3 window verdict.
2. **Window taper**: once era-3 data ≫1M positions, switch the all-data
   window to KataGo's power-law taper (`shuffle.py
   compute_desired_num_rows`, exponent <1 anchored at taper scale).
3. **Net size vs compute**: KataGo pairs small nets (b6c96/b10c128) with
   small compute and grows the net as data accumulates; our fixed 30M net
   on a single 1080 yields ~7 pos/s. A smaller net would generate data
   several times faster now, at some strength-ceiling cost. Deserves a
   measured decision, not a midnight change.
4. **LR continuity**: KataGo trains one continuous optimizer; we restart
   the cosine schedule (2e-3 → 2e-5) every round. With ~330-step rounds
   this means frequent warm restarts at high LR. Watch training noise; if
   era-3 losses churn, scale per-round LR down before touching anything
   else.
5. **Batch size**: KataGo sync example uses 128; ours 512 (kept for GPU
   efficiency; reuse cap is enforced in samples, not steps, so this is a
   granularity choice, not a correctness one).

## Dynamic cadence status

Phase 1 (live): position-count trigger (~21K fresh positions/cycle) +
per-chunk `surprise_sum`/`surprise_moves` aggregates in chunk metrics.
Phase 2 (implemented, dark): surprise-threshold trigger behind
`FLYWHEEL_SURPRISE` (with a POS_THRESHOLD/2 floor and 2× ceiling so a
mis-set threshold can neither thrash nor stall training). Activate after
calibrating the threshold against a day or two of recorded per-chunk
surprise; note KataGo itself uses no such trigger (its async loop trains
continuously), so this is a WallZero-specific efficiency experiment and
must be evaluated against the fixed-cadence baseline.

## Audit round 2 (2026-07-29 morning): methodology internals

Compared `python/train.py`, `python/katago/train/*`, selfplay1.cfg, and
KataGoMethods.md against `src/wallzero/{training,mcts,selfplay}.py`.

**Fixed immediately:**

- **LR continuity** (the queued item, now confirmed a genuine deviation):
  KataGo trains with a continuous piecewise-constant LR scale across the
  entire run — there is no per-cycle annealing of any kind. Our per-round
  warmup+cosine (2e-3 → 2e-5) was a WallZero invention; with era-3's
  ~330-step rounds it swung the LR 100× every ~7 minutes, a plausible
  source of the churn seen in the round-67 anchor match (0.455) and
  elevated losses. Fixed by collapsing the schedule to warmup-then-flat
  at a constant 3e-4 (`WALLZERO_LR`), the time-weighted average of the
  old cosine. Landed at round ~68, inside the era-3 measurement window —
  noted so the anchor trajectory is read accordingly.

**Verified aligned this pass:**

- FPU reduction 0.2 — exactly KataGo's `fpuReductionMax` default.
- Fast-search moves excluded from the policy loss (policy_weight mask);
  value trains on all rows. Matches KataGo's cheap-search handling.
- Surprise weights drive batch sampling (weighted `rng.choice`) — the
  in-training analog of KataGo's row weighting.
- Mirror augmentation (p=0.5; Quoridor's symmetry group), gradient
  clipping, forced playouts + pruned policy targets, shaped Dirichlet.

**Informed differences, kept deliberately:**

- cPUCT: ours are AlphaZero's published constants (pb_c_init 1.25, base
  19652); KataGo uses cpuct 1.0 + 0.45·log growth. Both are reference
  recipes; ours follows AZ. Not a bug.

**Queued (in rough priority order):**

1. **Value loss weight**: KataGo scales value loss by 0.6 vs policy;
   ours is 1.0. Relevant because era-3 value loss is elevated (~0.23) —
   downweighting is the reference behavior. Cheap, but wait for the
   anchor verdict before another training-loss change.
2. **SWA** (AveragedModel, ~80K-sample period) — KataGo credits it with
   real strength; needs round-spanning averaging design in our loop.
3. **Optimizer state continuity**: KataGo's optimizer runs continuously;
   ours re-initializes AdamW each round (warmup mitigates). Persist
   optimizer state across rounds alongside best.pt.
4. Optimizer family (KataGo: SGD-momentum classically, Muon variants
   now; ours AdamW) and soft-policy auxiliary target (weight 8.0) —
   larger changes, evaluate only if the loop is healthy but slow.
5. Temperature curve: KataGo 0.75 → 0.15 with halflife 19; ours 1.0 for
   24 moves → 0.05. Slightly more early exploration, sharper endgames.
   Minor; align only with measurement.

## Audit round 3 (2026-07-29): full selfplay1.cfg + KataGoMethods fine print

**The headline find — a large implemented-but-dormant feature:** KataGo's
self-play config runs `subtreeValueBiasFactor = 0.30` /
`subtreeValueBiasWeightExponent = 0.8` — the subtree value-bias
correction KataGoMethods credits with **+30-60 Elo**, their largest
single search improvement. WallZero implemented this (config-gated,
`ab_search.py --subtree-bias`) and left it OFF, unmeasured. A/B running
(100 games, KataGo's constants, seed 1210001); enable in self-play and
serve if the pooled evidence confirms.

**Feature gaps found (queued, priority order):**

1. **Game-diversity cluster — we have NONE of it**: `initGamesWithPolicy`
   (first ~3 moves sampled hot from the raw policy, no search),
   `earlyForkGameProb 0.04` / `forkGameProb 0.01` (fork games into
   alternative lines), `sidePositionProb 0.02` (record refutations of
   tempting bad moves). This cluster exists to prevent opening-diversity
   collapse — the exact echo-chamber failure era 2 died of. Highest-value
   feature work after the era-3 verdict.
2. `reduceVisits = true` (≥0.9 winrate for 3 turns → 100 visits, row
   weight 0.1): saves compute on decided games; more chunks/day.
3. `valueSurpriseDataWeight = 0.1` on top of policy surprise (our 0.5
   matches their `policySurpriseDataWeight` exactly).
4. Root symmetry averaging (`rootNumSymmetriesToSample = 4`; Quoridor's
   symmetry group supports 2).

**Verified aligned this pass:** no resignation in their self-play (ours
plays to terminal — correct); `cheapSearchTargetWeight = 0` ↔ our policy
mask; surprise weight split 0.5 exact; `switchNetsMidGame` ↔ our
per-chunk best.pt reload; maxVisits 600 / cheap 100 / p(cheap) 0.75 ↔
our 800/200/0.75; root temp 1.25→1.1 (ours flat 1.2 — within their
range); LCB-in-selfplay is on for them, ours measured neutral (3×200
pooled) and stays off by evidence.

## Audit round 4 (2026-07-29): architecture and search-engine internals

**Verified correct (important negatives):**

- **Search-tree reuse between moves exists** (`SearchTree.advance` re-roots
  after each move in self-play) — no wasted-computation bug.
- **The eval-server position cache (32% hit rate) already implements the
  "low-level mitigation" for transpositions** that GraphSearch.md
  describes — net evals are shared across transposed lines even though
  search statistics are not.
- Value head form (scalar tanh + MSE) matches the AlphaZero reference.

**Improvement paths found (both major projects, post-verdict):**

1. **Monte-Carlo Graph Search (MCGS)**: KataGo searches a DAG
   (`useGraphSearch = true`), sharing search statistics across
   transpositions, with a dedicated doc on the correct formulation
   (docs/GraphSearch.md). Quoridor is transposition-HEAVY — wall
   placements commute, which is exactly why our eval cache hits 32% —
   so MCGS should pay off disproportionately here: same visit budget,
   materially deeper effective search, better training targets.
   Substantial, subtle implementation (KataGo warns of the pitfalls);
   rank as the top search project.
2. **Trunk architecture**: KataGo's nets are BatchNorm-free
   (`norm_kind: fixup`) and interleave **global-pooling blocks** into
   the trunk (pooled features concatenated into conv layers), letting
   every layer see board-wide context. Ours is BN+ReLU+SE — SE gives
   global channel gating (a weaker cousin of gpool). BN in continual RL
   risks running-stats lag under distribution shift (already ranked in
   katago-adaptations.md). Architecture changes mean a new net and a
   warm-start decision — bundle both with any future net-size change.

Net assessment after four dives: scheduling, training loop, self-play
recipe, and data handling are now reference-faithful; remaining
divergences are architecture/search projects with measured upside, not
correctness bugs.
