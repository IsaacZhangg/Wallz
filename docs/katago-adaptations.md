# KataGo techniques: adaptation status for WallZero

Sources: the KataGo paper (arXiv 1902.10565, "Accelerating Self-Play Learning
in Go") and `docs/KataGoMethods.md` in lightvector/KataGo. KataGo reached
AlphaZero-level strength with roughly 50x less compute using this family of
changes; WallZero adopts every technique that fits a 9x9 Quoridor engine under
the zero-human-data constraint. Defaults preserve the original behavior — the
kata recipe is opt-in per script.

## Implemented (2026-07-24)

| Technique | WallZero adaptation |
| --- | --- |
| Playout cap randomization | `SelfPlayConfig.full_search_probability` / `fast_simulations`: 25% of moves get the full 800-sim search (policy + value targets), 75% get a cheap 200-sim search (value target only, `policy_weight=0`) |
| Forced playouts + policy target pruning | `MCTSConfig.forced_playout_scale` (k=2) forces `sqrt(k * prior * visits)` root playouts per child; `SearchTree.pruned_policy` subtracts the forced allotment from recorded targets so Dirichlet exploration never contaminates training data |
| Shaped/scaled Dirichlet noise | `MCTSConfig.dirichlet_concentration=10.83` total alpha divided across legal moves (KataGo's total-alpha convention; the log-prior shaping half is not adopted) |
| Root policy softmax temperature | `MCTSConfig.root_policy_temperature=1.2` at self-play roots — the SAI/KataGo restoring force toward uniform among near-equal moves |
| Policy surprise weighting | `SelfPlayConfig.surprise_weighting`: half the per-game frequency weight uniform, half proportional to KL(noised prior -> pruned target); training samples proportionally (`TrainingExample.weight`, replay schema v2) |
| Ownership/score auxiliary targets (analog) | Exact BFS shortest-path distances for both players recorded at every position (`own_distance`/`opp_distance`); `NetworkConfig.distance_head` adds a 2-output regression head, loss weight 0.1. This is the Quoridor analog of KataGo's dense ownership supervision: the quantity that decides the game, supervised exactly, at every position |
| Gateless training | `run_training_round(arena_games=0)` always adopts the latest candidate (AlphaZero-style); strength judgment belongs exclusively to independent suites (control gate + classical-engine KPI) |
| Global pooling structures | Already present since round 0: every residual block carries a SqueezeExcite global-pooling gate |
| Warm-start architecture growth | `run_training_round(candidate_network=..., warm_start=True)` keeps every compatible incumbent weight when adding heads or scaling |

## Implemented and measured neutral (2026-07-25)

Both were ported faithfully, are covered by tests, and remain available behind
config flags — but neither produced a measurable strength gain for this
network at this scale, so neither is enabled by default. Evidence:
`artifacts/runs/a100-bootstrap/ab-search-2026-07-25.jsonl`, produced by
`scripts/ab_search.py` (same checkpoint on both sides, so only the search
setting differs; free to run locally).

| Technique | WallZero form | Measured |
| --- | --- | --- |
| LCB move selection | `MCTSConfig.lcb_selection` picks the root child maximizing `-mean - z*stderr` among children with at least 10% of the top visit count; per-node variance tracked through virtual loss | 0.475 over 40 games (exploratory), neutral |
| Score utility | `MCTSConfig.distance_utility_weight` mixes `tanh((opp_dist - own_dist)/scale)` into leaf values — exact BFS margins, the Quoridor analog of KataGo's score utility | Weight 0.10 scored 0.6125 at 40 games, but a pre-declared 3-seed x 200-game confirmation pooled to **0.4975, 95% CI [0.4576, 0.5374]** — the exploratory result was noise |

The lesson is the same one the campaign learned in training: 40-game samples
cannot distinguish a real effect from noise, and every promising exploratory
result must survive a pre-declared multi-seed design before it is believed.

## Identified, not yet implemented (ranked)

1. **Subtree value bias correction** (~30-60 Elo in KataGo): online correction
   of correlated NN evaluation errors, bucketed by local pattern of recent
   moves. Quoridor bucket analog: last wall placement + local wall pattern.
   Search-only change; moderate complexity.
2. **Uncertainty-weighted MCTS playouts + short-term value targets**: needs an
   extra head predicting near-term value error; pairs with dynamic
   variance-scaled cPUCT.
3. **Opponent-next-move auxiliary policy head**: modest paper gains; deferred
   because the current-player canonical frame makes the target subtle to get
   right, and a bug here poisons training silently.
4. **Soft resignation**: value-threshold truncation of hopeless games with
   low-weight fast finishes; our wall-free solver adjudication already covers
   part of this — remaining benefit is midgame truncation.
5. **Optimistic policy / auxiliary soft policy**: later-KataGo refinements,
   evaluate after the core recipe proves itself.
6. **Batch-norm-free architecture (fixed-variance init, one BN)**: 1.6-1.8x
   faster training steps and cleaner multi-device behavior; an architecture
   migration to schedule alongside the next scale-up.
7. **Fork-position curriculum**: start a fraction of self-play games from
   positions sampled out of recent replay (self-generated, rule-derived) for
   opening diversity beyond temperature.
8. **Nested bottleneck residual blocks**: KataGo's current architecture family;
   relevant at the next capacity jump, not at 30M.
9. **Multi-board-size masking**: not applicable — Quoridor is fixed 9x9.

## First kata-recipe campaign scripts

- `scripts/colab_chunk_kata.py`: 192 games, 800/200 sims at 25% full-search
  probability, forced playouts, shaped noise, root temperature, surprise
  weighting, exact distance targets.
- `scripts/colab_round_kata.py`: gateless 3,000-step rounds over a 60K window;
  first round adds the distance head to the 30M incumbent via warm start.
