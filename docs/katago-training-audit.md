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
