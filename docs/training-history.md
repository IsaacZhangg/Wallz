# Training history

[Back to the project README](../README.md)

These campaign notes were moved from the README. They preserve observations,
measurements, and interpretations recorded in July 2026. References to "live",
"current", checkpoints, and deployed node settings describe those dated runs. Paths in
the original notes are relative to the repository root.

## Later campaign decision, July 29, 2026

The subsequent
[era-4 commit](https://github.com/IsaacZhangg/Wallz/commit/ee5adc4fa9c7aaf369273845b2f10553b79fa00c)
records a restart from round 31 and an additional training pass to account for inherited
replay data. That pass improved held-out policy and value fit, but its 0.515 score
against round 31 did not establish a strength gain. The commit also records abandoning
the smaller network lineage. The architecture remains available in code; the earlier
adoption plan and notes below precede this campaign decision.

## Archived README notes

The following entries retain their original wording and newest-first order.

## Recorded status (2026-07-29 night, weight decay and the BN trunk)

Audit round 6 read KataGo's training source directly (see
`docs/katago-training-audit.md`) and found a mechanism, not just a feature gap. Our
trunk is `conv(bias=False) → BatchNorm` throughout, so a conv's weight _scale_ is
invisible to the forward pass; what matters is the relative step `Δw/‖w‖`. Measured on
our own checkpoints, era 2 grew conv weight L2 by **88%** from r31 to r59 — **99 of 99
conv tensors grew**, median 1.71× — so the net was quietly taking ever-smaller relative
steps while its losses kept falling. Era 3 is repeating it (+8% in five rounds).

KataGo scales weight decay by `lr^0.75`, by `sqrt(batch/256)`, and by an adaptive factor
tracking model norm against a baseline, precisely to hold BN effective LR constant;
their AdamW+BN constant is ~100× ours, and they exempt biases and norm betas that we
were decaying.

Shipped at round 69: KataGo's decay groups (`training.py:_weight_decay_groups`) — trunk
weights 0.009, head weights 0.004, BN gammas at a 0.25 factor, biases/betas exempt at
1e-6, all scaled by `sqrt(batch/256)`. `TrainMetrics` now records
`weight_norm_start`/`weight_norm_end` every round, so the norm trajectory lands in the
round logs and the mechanism stays visible.

Honest caveat on magnitude: at our LR the reference constant cancels roughly half of era
2's measured per-step drift, and less of era 3's (short rounds re-initialise AdamW each
time, and those transients are themselves a norm-inflation source KataGo does not have —
they run one continuous optimizer). The per-round norm log is the instrument: if it
keeps climbing, the escalations are optimizer-state continuity across rounds and a
larger decay constant.

## Recorded status (2026-07-29 evening, the shard-load stall)

Era 3 lost ~10 hours to a latent bug that only the all-data window could expose.
`load_shard` held an open `NpzFile` and indexed it _inside_ its per-row loop, so each of
~12 columns was decompressed out of the zip again for every row: **18.4 s per 5K-row
shard**. At 45 shards (r67) a round still finished; at 71 the window load reached **21.7
min** and crossed the flywheel's 15-min power watchdog, which killed the round before it
ever touched the GPU. Twenty-two consecutive rounds died that way (08:54 → 18:32, all
status 137), and because each attempt paused generation for ~17 of every ~29 minutes,
the node was also generating at roughly 40% duty the whole time.

Two fixes, both shipped and deployed:

- **`load_shard` reads each column once**, before the loop. Verified bit-identical to
  the old implementation on a real production shard (5,036 examples: states, policies,
  values, weights and distances all equal, dtypes and Python types preserved). Full
  71-shard window load on the node: **21.7 min → 2.3 s** for 355,133 examples.
- **The flywheel's power watchdog now arms** on the first ≥110W sample or after
  `FLYWHEEL_LOAD_GRACE` minutes (default 30), whichever comes first. A round is no
  longer punished for its CPU-side load phase, while a genuine wedge is still caught
  within 15 min of the GPU going quiet.

Lesson recorded: the watchdogs assume "training round running" implies "GPU busy". Any
pre-GPU phase that grows with the replay window can recreate this, so growth-sensitive
phases need their own grace, not the steady-state rule.

## Recorded status (2026-07-28, era 3: the window fix and the restart)

The data-volume test closed falsified — and the post-mortem found the reason it never
could have succeeded: training only ever consumed the newest 60K positions (~1,100
games, ~26 epochs/round), so generated volume accumulated on disk without ever entering
the training distribution. Full record and measurements in `docs/data-volume-test.md`
(r59 flat vs both r31 and r14 at 100 games each; classical KPI 0-40; verified style
drift to move-1 walls).

Era 3 (live since 19:41): node reset to the frozen r31 tag (equal measured strength,
verified sane style), era-2 shards archived to `replay-era2/` on the node (they encode
the drifted style), and the recipe changed in exactly one place — `node_round_kata.py`
now trains on ALL accumulated shards with steps scaled to ~3 epochs/round (capped at
3,000). New guardrails so a flat or regressing lineage can never again run unmeasured:
`node_anchor_match.py` plays best.pt vs a frozen anchor (100 games, 192 sims, fixed
seed) after every 5th adopted round — promotion at ≥0.60 re-anchors, <0.40 writes
REGRESSION-ALARM which suspends training (generation continues) until a human clears it.
Timeline accumulates in `strength-timeline.jsonl`; anchors live in `~/wallzero/anchors/`
(r31 base + r59 kept as junk-style control). Era-3 prediction, pre-stated: >0.55 vs the
r31 anchor within ~10-15 rounds, or the window hypothesis is falsified too.

## Recorded status (2026-07-26 evening, node watchdog hardening)

The generation node became self-healing end to end. Context: the P2 clock latch (1670
MHz) recurred **spontaneously** the same evening — no `nvidia-smi -pl` involved; the
suspected trigger is idle/load P-state cycling between chunks. Offsets queried as
applied, no throttle reasons active, PowerMizer and offset re-toggles did not clear it;
that episode only cleared on reboot (but see the night-shift addendum below — a later
episode cleared without one). Chunk 130 ran at 6.66 pos/s instead of ~7.05, invisible to
the old power-only watchdog — "slow" is a failure mode distinct from "dead".

Hardening shipped (repo `scripts/node_*.sh` + `scripts/systemd/`, deployed to
`~/wallzero/` and `/etc/systemd/system/` on the node):

- **Clock watchdog** in `gen_loop.sh`: under real load (>=110W), SM clocks below 1700
  MHz for 10 min → re-apply `gpu-oc.sh` once (covers lost offsets after an X restart);
  still latched 10 min later → reboot at the next _chunk boundary_ (no work lost),
  rate-limited to one auto-reboot per 6h and only when systemd autostart is enabled.
- **Fail-safe power reads**: a failed/garbled `nvidia-smi` read now counts as unhealthy
  (the old code defaulted to a healthy 200W, so a dead driver looked fine forever).
- **Group-scoped worker sweeps**: `pkill -9 -g <chunk_pgid>` replaces the broad
  `pkill -f spawn_main` that once killed a healthy training round's loaders. Failed
  chunks escalate: 5-min backoff from the 3rd consecutive failure, last-resort reboot
  from the 5th.
- **Tagged PAUSE**: the flywheel writes `flywheel:<pid>` into PAUSE; a pause whose owner
  died with no round running is removed automatically (by gen_loop after 3 min, by the
  flywheel at startup, by boot prep after a reboot). Manual PAUSE files (any other
  content) are never touched and now survive reboots meaningfully.
- **Flywheel power watchdog**: a wedged training round (same <110W/15-min rule) is
  killed instead of burning its full 3h timeout with generation paused. (Since
  2026-07-29 the rule only arms once the round has reached the GPU, or after
  `FLYWHEEL_LOAD_GRACE` minutes — see the shard-load stall above.)
- **Boot recovery** (`wallzero-prep/gen/flywheel.service`): on every boot, prep waits
  for GPU+X, runs the bitwise canary at stock clocks (re-recording the reference there
  if best.pt changed — stock is always trustworthy), applies the validated OC, and
  re-verifies; canary failure falls back to stock. Generation and flywheel then start
  supervised (`Restart=always`), so a power blip, script crash, or auto-reboot no longer
  needs a human. The old `setsid nohup` launch procedure is obsolete.

### Night-shift addendum (2026-07-26 late): the latch model, revised live

Round 40 settled an open question: it trained in 2980 s — identical to round 39's 2979 s
— while the core read 1670 MHz throughout. **Training speed is unaffected by the latch**
(the +800 mem offset, which survives latches, carries training); a 1670 MHz core reading
during a training round is a non-signal. Do not "fix" it.

Generation then resumed latched for the second time in two post-training resumes — the
trigger is all but confirmed as the training→generation P-state transition, meaning
every flywheel round may re-latch the card. Watching the (new) clock watchdog respond
exposed a live bug: its counters reset per chunk, and a latched chunk only lasts ~13.5
min, so the 10-min re-apply stage fired every chunk while the 10-more-minutes reboot
stage was unreachable. Fixed by persisting the counters across chunk boundaries (a
healthy sample re-arms the ladder).

The fixed ladder then produced a surprise: after the stage-1 offset re-apply (22:33),
clocks recovered to 1898 MHz **without a reboot** ~8 min later, mid-chunk — falsifying
"only a reboot clears it". But the post-round-41 episode showed that self-heal is
unreliable (1 of 2): stage 1 fired at 00:18, clocks stayed latched through a chunk
boundary, stage 2 declared the reboot at 00:29, and at 00:35:45 the **first fully
autonomous recovery ran end to end**: boundary reboot (chunk-140 shard saved first),
28-s boot, prep detected a stale canary reference (best.pt had advanced two rounds),
re-recorded it at stock, applied OC, PASS at 224 iterations, and chunk 141 was
generating at full clocks at 00:37:51 — **under 2 minutes of downtime**, flywheel
baseline intact. Measured economics: the latch costs 0.37 pos/s (6.68 vs 7.05, chunk
134), a full episode ending in reboot ~1150 positions (~2% of a cycle), one that
self-heals ~450. The 6-h reboot rate limit is the right shape: never reboot for a latch
preemptively; let the ladder decide. The fan-80% trigger hypothesis stays untested
(config held constant overnight so post-training resumes stay a clean reproducibility
test; the latch is now 3-for-3 on those resumes); it is a daylight experiment.

Full night's tally (2026-07-26 22:00 → 07-27 06:46, zero human intervention): rounds
40-44 trained and adopted on cadence, every one at ~2980 s regardless of latch state;
losses monotonic (total 1.5820 → 1.5240, value 0.0999 → 0.0878); replay 87 shards /
500,435 positions (376,894 post-chunk-82, ~62% of the pre-declared 1M data-volume
trigger). Two autonomous reboots (00:35 and 06:42 — the second at the first chunk
boundary after the 6-h rate limit expired, ending ~4.5 h of correctly-chosen
degraded-mode generation), both under 2.5 min downtime with the in-flight shard saved
and the canary re-verified. Every latch was triggered by a train↔generate P-state
transition; generation-side latches are now 4-for-4 on those transitions.

**Fault audit (2026-07-27 morning) — the latch is driver-side, not our configuration.**
Every our-fault hypothesis was tested and excluded: (1) we never set clock locks — no
`-lgc`/`-ac` anywhere, and this board reports Applications Clocks unsupported (N/A); (2)
the offset is applied via `GPUGraphicsClockOffsetAllPerformanceLevels` — the same
attribute NVIDIA's own GUI uses, and the documented-correct method for Pascal (offsets
apply to all perf levels); (3) the decisive observation: healthy and latched runs are
BOTH in P2 with identical queried config (offset 125 on all levels) — healthy sustains
1860-2025 MHz, latched pins exactly stock-sustained 1670, so the driver reports one
thing and does another; (4) one trigger was `nvidia-smi -pl` alone, which never touches
the offset path; (5) zero Xid/NVRM/Xorg errors across all boots (580.173.02); (6)
re-applying the identical config sometimes fixes it minutes later (2 of 5 self-heals) —
replaying unchanged config can only matter if hidden driver state is flaky. External
record: silent regressions in this exact subsystem (offsets/fan/ PowerMizer acknowledged
but not honored) recur across driver branches (390→580 reports; a 520-branch report
shows `-lgc` acknowledged and ignored the same way), and Pascal's forced-P2 compute
transitions are notoriously janky (the SETI community built `keepP2` — a tiny always-on
CUDA kernel — specifically because 10-series cards misbehave at compute-load
boundaries). Residual uncertainty: no public report of this exact 1670-latch signature
was found; a VBIOS interaction can't be excluded. Definitive test if ever needed: driver
downgrade. Promising prevention candidate from the audit: a keepP2-style keep-alive
kernel across the train↔generate handoff, so the card never leaves P2 — the latch is
5-for-5 on exactly those transitions. **Tried and falsified 2026-07-27**
(`scripts/node_gpu_keepalive.py` + `wallzero-keepalive.service`, kept in-repo as a
documented dead end, service disabled): with the keep-alive verified holding the card in
P2 continuously — near-idle samples read P2/1670/48 W instead of the old P8/139 MHz
drops — the card **latched anyway** at 11:04-11:05, under ~full load, ~2 min after a
routine chunk boundary, on a freshly rebooted card. So the trigger is NOT the idle
P-state excursion; the hidden driver state can flip while continuously in P2, with
elevated probability around workload rearrangements (post-training resumes remain
5-for-5). Do not re-try keep-alive variants.

**Deep-dig conclusions (2026-07-27 afternoon).** The symptom matches NVIDIA's own
acknowledged bug 5934973 — "when the graphics card is overclocked, GPU voltage may
become capped, preventing it from boosting to expected levels" — which shipped in the
2026 595-era drivers on BOTH Windows (recalled + hotfixed 595.76/78) and Linux
(user-measured on 595.45.04). Our 580.173.02 legacy build is dated June 2026, months
after that regression existed upstream, and legacy branches receive backports. Forensics
with the card latched live: the driver's perf table still advertises nvclockmax 2088
with the mem offset applied (4913) while the governor sits at 1670 — and Pascal exposes
no voltage query to confirm the cap directly. Every remedy short of reboot is now
tested-dead: offset re-apply (2/5), persistence-mode toggle (no), PowerMizer registry
keys (removed from the driver after 525), `-lgc` (Volta+ only), `-ac` (unsupported on
this board). Constraints on driver rollback: 580 is officially the LAST Pascal branch
(support to Aug 2028), apt carries ONLY 580.173.02 for noble, and the node runs kernel
7.0 that older point releases (580.142/570.x) predate — a failed DKMS build would take
down X and with it the OC entirely. Decision: accept-and-manage through the data-volume
milestone. A diagnostic report captured during a live latch was retained privately
for the 580-legacy feedback thread. The raw dump is excluded from the public source
tree because it contains machine and network identifiers.

## Recorded status (2026-07-26, node efficiency pass)

Three stacked, individually measured changes took GTX 1080 generation from 3.96 pos/s
(~22-min 96-game chunks) to **7.05 pos/s (12.8-min chunks, +78%)**, each verified on
full production-recipe chunks:

1. **Frozen TorchScript inference (+11.6%, chunks 121/123 at 4.31-4.42):**
   `torch.jit.trace` + `freeze` + `optimize_for_inference` in `TorchEvaluator` folds
   batchnorm and NNC-fuses the elementwise/SE chains (max logit deviation 1.8e-4, the
   same order as batchnorm folding alone). Gated to fp32 CUDA (pre-Ampere) because
   autocast does not apply inside TorchScript; the A100 bf16 and MPS paths keep the
   eager module.
2. **Position-evaluation cache (+44%, chunk 124 at 6.21):** the eval server keys a
   bounded LRU (400K entries, ~0.5GB, `WALLZERO_EVAL_CACHE`) on a BLAKE2b digest of the
   exact input planes; **32% of all evaluation requests repeat** (shared openings across
   96 concurrent games, transpositions, re-searched subtrees) and now cost a RAM lookup
   instead of a GPU forward. Off by default; hit/miss stats print at chunk end. Idle
   CPU/RAM buying back GPU forwards was the only productive use found for the mostly
   idle i7 — extra workers cannot add throughput at the inference roofline.
3. **GPU clock offsets via Coolbits (+13.6%, chunk 125 at 7.05; final config chunk 129
   at 7.09):** fan at 100% (76→56-63C under load) plus +125 MHz core / +800 MT/s memory
   offsets — the validated maximums: +150 core fails the bitwise canary, so +125 is this
   silicon's ceiling at stock voltage. No overvoltage; stock 198W power limit (full OC
   draws ~195-205W). Sustained compute clocks 1670→1885-1936 MHz, P2 memory 4513→4911
   (still under its own 5005 spec). Every step passed a bitwise-repeatability canary
   (`scripts/node_gpu_canary.py` — clocks do not change math, so any output deviation is
   a silent compute error) and a training-loss sanity bench; round 39 trained in 49.7
   min vs 52.3 stock. Offsets reset on X restart/reboot; re-apply with
   `~/wallzero/gpu-oc.sh` (repo: `scripts/node_gpu_oc.sh`, defaults are the validated
   values). Two traps recorded: (a) the canary reference must be recorded after
   TorchScript profiling-executor warmup and is bound to the best.pt hash (the flywheel
   adopts new weights mid-day, which otherwise poisons the comparison); (b) **never run
   `nvidia-smi -pl` on this node** — changing the power limit (even up) latches the P2
   core boost at 1670 MHz on driver 580 + Pascal and only a reboot clears it; discovered
   when a user-approved 220W cap attempt _pinned_ the clocks the offsets had unlocked.

The eval server also gained an async submit/collect pipeline (pinned staging buffers,
CUDA events) overlapping queue draining and reply serialization with GPU compute —
measured neutral (back-to-back forwards with zero server logic draw the same wattage as
live generation, so the server never left meaningful GPU idle; power draw is not a
utilization proxy for this workload). Measured and rejected on this card: fp16 inference
(pseudo-half, 7-10% _slower_, 1000x the deviation — Pascal has no usable fp16),
inference-side cuDNN autotune and eager batchnorm folding (≤1% each), and fp16-vs-fp32
training at batch 512 (identical at ~1.05 s/step once cuDNN autotune is active; the
07-25 bench showing fp16 at 57 min predated autotune). Training rounds project ~50 min
under the OC — near the card's compute floor for the 3,000-step recipe. With ~13-min
chunks the flywheel now trains (~50 min) after every ~52 min of generation; the
train/generate cadence itself stays fixed per the pre-declared data-volume test design.

## Recorded status (2026-07-26 overnight)

First full night of the free-fleet flywheel: rounds 32-35 trained, adopted, and secured
(all gateless, 3,000 steps on the 60K window; 52-56 min each on the GTX 1080), while the
fleet generated **91,868 fresh kata positions** — 45,213 from the PC (9 chunks at 4.27
pos/s after the bf16 fix below) and 46,655 from the Mac (10 chunks via a single-process
MPS loop; the multiprocess self-play path deadlocks on MPS and must not be used there).
Mac shards are renumbered into the PC replay sequence at each pre-training pause. Sanity
matches (40 games, 192 sims, paired openings — sanity checks, not strength claims): r32
vs r31 0.525 [0.375, 0.671]; r35 vs r31 0.500 [0.352, 0.648]. Flat short-horizon sanity
results are the expected shape under the data-volume hypothesis; the decisive
pre-declared test (docs/data-volume-test.md) triggers at >=1M fresh positions (~9% there
after one night). Round times: r32 55.9 min, r33 52.2 (cuDNN autotune active), r34 103
(an accidental duplicate launch halved throughput — same seed, so artifacts were
identical and unharmed), r35 52.3.

The morning after, the PC cycle became self-driving: `scripts/node_flywheel.sh`
(deployed as `~/wallzero/flywheel.sh`) watches the replay directory and, every 4 new
shards from any source, pauses generation, trains one gateless round, adopts it, and
resumes. Mac and Colab shards still merge manually (renumbered into the PC sequence
while paused or mid-chunk into free indices); the flywheel counts them like any other
shard. Manual PAUSE files win — the flywheel never acts while one it didn't create
exists, and it retries failed rounds after 10 minutes with generation running.
`train_candidate` also gained a prefetch thread (batch assembly overlaps GPU compute;
verified bit-identical to the old loop). Round 36 was the live measurement and
**falsified the speedup estimate**: 52.3 min vs the 52.2-52.3 baseline — batch assembly
was never a meaningful fraction of the step time on the 1080 (and the GIL limits overlap
for Python-heavy assembly anyway). The change stays because it is proven bit-identical
and costs nothing, but the 10-20% claim is dead; round time on this card is GPU compute,
full stop.

## Recorded status (2026-07-25)

The KataGo-recipe campaign (rounds 25-31: playout cap randomization, forced playouts
with pruned policy targets, shaped Dirichlet, surprise weighting, exact-BFS distance
head, gateless rounds) moved the classical-engine KPI off zero for the first time: 0.05
(0-18-2) at an 800-simulation budget. Three plateau hypotheses were then falsified by
measurement — legacy-data dilution (kata-only retrain scored exactly 0.500), per-round
overtraining (a KataGo-norm 600-step round was _underfit_ and scored 0.475), and weak
search (LCB selection, distance-margin utility, and subtree value bias correction all
measured neutral, the last under both a crude and a faithful pattern-based bucketing;
all remain config-gated, default off). The surviving hypothesis is **data volume**, and
`docs/data-volume-test.md` pre-declares the week-scale test with supported/falsified
criteria fixed in advance.

Compute now runs as a three-node fleet: an always-on GTX 1080 generation node
(`scripts/node_chunk_kata.py` under a watchdogged loop — after a silent CUDA stall
wedged a chunk for hours at "100% utilization" and near-idle power draw, the loop kills
anything under 80W for 15 minutes or over 3h total), the M4 Pro for free A/B and
evaluation (`scripts/ab_search.py`), and Colab A100 reserved for the pre-declared
evaluation suites. The 1080 also closes the training loop: a full 3,000-step round of
the 30M network takes ~53-57 min (`scripts/node_train_bench.py`; torch's bf16-"support"
on Pascal is emulation, 9.6x slower, now gated off by compute capability in
`training.py`), so generate → train → adopt runs entirely on free hardware with round 32
the first PC-trained round (55.9 min live; sanity match vs round 31: 0.525 [0.375,
0.671] over 40 games).

The "silent CUDA stall" was root-caused the same night (`docs/node-1080-wedge-log.md`):
`TorchEvaluator` autocast sent all CUDA inference through torch's _emulated_ bf16 on
Pascal, and driver 580's emulated-bf16 cublasLt kernels can hang inside `cuLaunchKernel`
on sm_61 (py-spy native stack: `cublasLtTSTMatmul` spinning in `sched_yield`).
`network.py` now gates inference autocast on compute capability >= 8, the generation
node runs fp32 self-play at sustained 176-205W, and the loop script lives at
`scripts/node_gen_loop.sh` (watchdog threshold 110W — a wedge can float at 88W with
persistence mode on).

## Recorded status (2026-07-24)

A durable A100 campaign (`artifacts/runs/a100-bootstrap/`) has completed twenty gated
rounds with seven promotions; every shard, candidate, decision, and checkpoint is
checksummed locally and the campaign resumed across five Colab VM losses without state
regression (round 4 reproduced bit-identically from a restored bundle). Since round 15
the campaign runs a hybrid split: self-play chunks are generated on a local M4 Pro
(`scripts/local_chunk_gen.py`, zero compute-unit cost — self-play is actor-CPU-bound and
the laptop's cores outpace Colab's vCPUs), while the A100 runs training and the
multiprocess arena; each chunk's durable metrics record the generating device. Rounds
15-22 produced four promotions (r15, r19, r21, r22); the current best checkpoint is the
round-22 model. The pre-declared gate-3 suite (`strength-eval-gate3.jsonl`) records both
verdicts honestly: the round-22 best **sustains gate 2** against the uniform control
(0.5792, 95% CI [0.5393, 0.6180], 600 games) but its edge over the frozen gate-2 winner
is **not statistically established** (0.5317, 95% CI [0.4917, 0.5713], 600 games) —
arena promotions again outran control-relative evidence. Eight same-size generations
reading as flat at 3.7M parameters pointed to a capacity ceiling, so round 23 executed
the scale-up: a fresh 29.65M-parameter network (256 channels x 24 blocks) trained 4,000
steps from random initialization on the 24K-position clean window and **promoted 36-22-2
over the 3.7M incumbent** in the standard gate. Round 24 then trained the 30M network on
its own first self-play (generated on the A100 at leaf_batch 16 — 96 games in 251s, the
same wall time the 3.7M network needed) and promoted again (0.575). Independent
evaluation immediately deflated both promotions: the round-24 best sustains gate 2
against the uniform control (0.5542, 95% CI [0.5142, 0.5935], 600 games) but is dead
even with the two-day-old gate-2 winner (0.5000, 95% CI [0.4601, 0.5399], 600 games) —
expected after a single generation of its own self-play, but not progress. More
decisively, the first direct benchmark against the userscript's classical PVS engine
(`scripts/match_vs_classical.mjs`, evaluation-only, production adaptive settings) ended
**0-20 against WallZero** at a 160-simulation budget. The uniform control that gates 1-3
measure against is a far lower bar than the classical engine; the score against the
classical engine is now the project's standing real-strength benchmark, and it starts at
zero. All self-play games end decisively — the wall-free solver eliminated draws
entirely. Later generations fixed two data-quality defects: the cold-start exploration
override was poisoning trajectory quality (now parameterized; deep chunks use the preset
temperature schedule) and trimmed-restore chunk renumbering could hide the freshest
shards from the replay window (indices now continue from the maximum).

Independent evaluation (`strength-eval.jsonl`, `diagnostics-16sim.jsonl`) now supports
exactly one recorded claim: the frozen round-14 checkpoint (`best-gate-passed.pt`,
sha256 `48c9b6b9…`) **reliably beats the uniform-prior MCTS control at an equal
192-simulation budget** — a pre-declared three-seed suite of 600 paired-opening games
scored 0.575 (95% CI [0.535, 0.614]; per seed 0.535 / 0.578 / 0.613). That passes
strength gate 2, so the userscript's WallZero bridge (localhost HTTP protocol,
fingerprint and legality checks per reply, stale-reply discard, timeouts, honest outage
fallback) now ships enabled; it stays inert unless a local `wallzero serve --http`
process answers its health probe. Earlier checkpoints failed the same gate (r7 0.5175,
r11 0.4725, r12 0.526 over 500 games) and per-seed scores once swung 0.64→0.45, which is
why claims here require multi-seed pooled evidence. No "strong" or "expert" label is
claimed: the expert-scale suite (~1,000 games including the classical PVS engine as an
opponent) and further scaling remain open work.
