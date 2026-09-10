#!/bin/bash
# WallZero auto-flywheel for the training node: whenever ~FLYWHEEL_POSITIONS
# fresh positions have been generated (summed from chunk-metrics.jsonl), it
# pauses the generation loop, trains one gateless round, adopts the result,
# and resumes generation.
#
# Runs alongside gen_loop.sh, which owns generation and its watchdogs.
# Respects a PAUSE file it did not create (manual interventions win), and
# skips while any training round is already running. Checkpoint pulls to
# other machines and Mac/Colab merges stay manual by design.
#
# Robustness (2026-07-26): the PAUSE file is tagged "flywheel:<pid>" so
# gen_loop and boot prep can tell a crashed flywheel's stale pause from a
# manual one; a power watchdog kills a wedged training round (same
# 110W/15min rule as generation — a wedge would otherwise burn the full 3h
# timeout with the GPU idle and generation paused).
cd "$HOME/wallzero" || exit 1
OUT="$HOME/wallzero/output/wallzero-output"
# Era-3 trigger: train per ~21K fresh POSITIONS (four 96-game chunks' worth),
# not per shard file — shard sizes have drifted twice, so file count is a
# drifting proxy. Positions are summed from chunk-metrics.jsonl lines
# appended since the last adoption (note: shards merged from other machines
# without metrics lines are invisible to this trigger). Surprise-based
# triggering stays deferred until the era-3 window verdict; the per-chunk
# surprise aggregates now landing in chunk metrics are its calibration data.
METRICS="$OUT/chunk-metrics.jsonl"
POS_THRESHOLD=${FLYWHEEL_POSITIONS:-21000}
# KataGo's synchronous loop refuses to start training before 100K rows
# exist (SHUFFLE_MINROWS); training a big net on a few chunks of data is
# pure overfit. ~20 shards at ~5.1K positions clears it.
MIN_SHARDS_TO_TRAIN=${FLYWHEEL_MIN_SHARDS:-20}
# Surprise trigger (dynamic cadence phase 2): when FLYWHEEL_SURPRISE is a
# positive number, train when accumulated surprise_sum since the last round
# crosses it (with a fresh-position floor of POS_THRESHOLD/2 and a ceiling
# of 2x POS_THRESHOLD so a mis-set threshold can neither thrash nor stall).
# Leave unset/0 for the plain position trigger until calibrated from
# chunk-metrics surprise history.
SURPRISE_THRESHOLD=${FLYWHEEL_SURPRISE:-0}
# Minutes a round may spend below the power watchdog's threshold before it
# has ever reached the GPU (replay window load; see the watchdog below).
LOAD_GRACE=${FLYWHEEL_LOAD_GRACE:-30}
# Era 4 (b10c128, from scratch) starts far below the r31 anchor, so the
# anchor ladder would suspend training on its first check. The match still
# runs and still records to strength-timeline.jsonl — only the automatic
# suspension is held off. RESTORE THIS TO 0.40 once era 4 clears 0.40.
export WALLZERO_ANCHOR_ALARM_AT="${WALLZERO_ANCHOR_ALARM_AT:-0.0}"
PAUSE="$HOME/wallzero/PAUSE"

log() { echo "[flywheel] $(date -Is) $*"; }
count_shards() { ls "$OUT/replay"/chunk-*.npz 2>/dev/null | wc -l; }
metrics_lines() { wc -l <"$METRICS" 2>/dev/null | tr -d " " || echo 0; }
new_positions() {
  tail -n +"$((base_lines + 1))" "$METRICS" 2>/dev/null |
    awk -F'"examples": ' 'NF>1 {split($2,a,","); s+=a[1]} END {printf "%d", s+0}'
}
new_surprise() { # integer sum (x1000) of surprise_sum since baseline
  tail -n +"$((base_lines + 1))" "$METRICS" 2>/dev/null |
    awk -F'"surprise_sum": ' 'NF>1 {split($2,a,","); s+=a[1]} END {printf "%d", s*1000}'
}
is_num() { case "$1" in "" | *[!0-9]*) return 1 ;; *) return 0 ;; esac; }

# A previous flywheel instance may have died mid-round: its tagged pause
# would otherwise stop generation forever. Manual PAUSE files are kept.
if head -1 "$PAUSE" 2>/dev/null | grep -q "^flywheel:"; then
  old=$(head -1 "$PAUSE" | cut -d: -f2)
  if ! kill -0 "$old" 2>/dev/null && ! pgrep -f "node_round_kata[.]py" >/dev/null; then
    log "removing stale PAUSE from dead flywheel $old"
    rm -f "$PAUSE"
  fi
fi

release_pause() { # only remove the pause if it is still ours
  head -1 "$PAUSE" 2>/dev/null | grep -q "^flywheel:$$\$" && rm -f "$PAUSE"
}

# The baseline persists across restarts: without this, every restart
# (crash, reboot, systemd Restart=always) forgot progress toward the next
# round and postponed training by up to a full threshold of fresh data.
LAST_FILE="$HOME/wallzero/.flywheel-baseline-lines"
base_lines=$(cat "$LAST_FILE" 2>/dev/null)
is_num "$base_lines" || base_lines=$(metrics_lines)
[ "$base_lines" -gt "$(metrics_lines)" ] && base_lines=$(metrics_lines) # log rotated
echo "$base_lines" >"$LAST_FILE"
log "started; $(count_shards) shards, metrics baseline line $base_lines, training every ${POS_THRESHOLD} fresh positions"
# Era-3 guardrail: node_anchor_match.py writes REGRESSION-ALARM when the
# current best scores <0.40 against the standing anchor. While it exists,
# training rounds are suspended (generation continues) until a human
# investigates and removes it — the era-2 flat lineage ran 28 rounds with
# no such brake.
ALARM="$HOME/wallzero/REGRESSION-ALARM"
ROUNDS_FILE="$HOME/wallzero/.rounds-since-anchor"

while true; do
  sleep 120
  [ -f "$PAUSE" ] && continue
  pgrep -f "node_round_kata[.]py" >/dev/null && continue
  if [ -f "$ALARM" ]; then
    log "REGRESSION-ALARM present; training suspended (generation continues)"
    sleep 1680
    continue
  fi
  [ "$(count_shards)" -lt "$MIN_SHARDS_TO_TRAIN" ] && continue
  fresh=$(new_positions)
  is_num "$fresh" || fresh=0
  if [ "$SURPRISE_THRESHOLD" -gt 0 ] 2>/dev/null; then
    surprise=$(new_surprise)
    is_num "$surprise" || surprise=0
    if [ "$fresh" -lt $((POS_THRESHOLD / 2)) ]; then
      continue # floor: never train on less than half a normal cycle
    elif [ "$surprise" -lt "$SURPRISE_THRESHOLD" ] &&
      [ "$fresh" -lt $((POS_THRESHOLD * 2)) ]; then
      continue # not stale yet and under the ceiling; keep generating
    fi
    log "$fresh fresh positions, surprise ${surprise}m; pausing generation"
  else
    [ "$fresh" -lt "$POS_THRESHOLD" ] && continue
    log "$fresh fresh positions; pausing generation"
  fi
  echo "flywheel:$$" >"$PAUSE"
  while pgrep -f "node_chunk_kata[.]py" >/dev/null; do sleep 30; done
  log "training round starting"
  setsid timeout --kill-after=60 3h env WALLZERO_OUTPUT="$OUT" WALLZERO_FRESH_ROWS="$fresh" \
    .venv/bin/python scripts/node_round_kata.py >>"$HOME/wallzero/night.log" 2>&1 &
  round_pid=$!
  low=0
  armed=0
  minutes=0
  while kill -0 "$round_pid" 2>/dev/null; do
    sleep 60
    minutes=$((minutes + 1))
    watts=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1)
    is_num "$watts" || watts=0 # failed read = unhealthy
    # A round loads its replay window on the CPU before it ever touches the
    # GPU, so the 110W rule cannot apply yet: on 2026-07-29 that phase grew
    # past 15 min and the watchdog killed 21 healthy rounds in a row. Arm on
    # the first busy sample (normal case) or after LOAD_GRACE minutes, so a
    # true wedge is still caught within 15 min of the GPU going quiet.
    [ "$watts" -ge 110 ] && armed=1
    if [ "$armed" -eq 0 ] && [ "$minutes" -lt "$LOAD_GRACE" ]; then continue; fi
    if [ "$watts" -lt 110 ]; then low=$((low + 1)); else low=0; fi
    if [ "$low" -ge 15 ]; then
      log "power watchdog: ${watts}W for 15 min; killing wedged round group"
      kill -9 -- "-$round_pid" 2>/dev/null
      break
    fi
  done
  wait "$round_pid"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    # Anchor ladder: every 5th adopted round, measure best.pt against the
    # standing anchor while the GPU is still ours (PAUSE held). ~15-25 min;
    # promotes at >=0.60, writes REGRESSION-ALARM at <0.40.
    rounds=$(cat "$ROUNDS_FILE" 2>/dev/null)
    is_num "$rounds" || rounds=0
    rounds=$((rounds + 1))
    if [ "$rounds" -ge 5 ]; then
      log "anchor match starting (5 rounds since last)"
      timeout 60m .venv/bin/python scripts/node_anchor_match.py \
        >>"$HOME/wallzero/anchor.log" 2>&1 ||
        log "anchor match failed (non-fatal); see anchor.log"
      rounds=0
    fi
    echo "$rounds" >"$ROUNDS_FILE"
  fi
  release_pause
  if [ "$rc" -eq 0 ]; then
    log "round adopted; generation resumed"
    base_lines=$(metrics_lines)
    echo "$base_lines" >"$LAST_FILE"
    timeout 30m bash "$HOME/wallzero/backup.sh" >>"$HOME/wallzero/backup.log" 2>&1 ||
      log "backup failed (non-fatal, cron will retry)"
  else
    log "round FAILED (status $rc); generation resumed, retrying in 10 min"
    sleep 600
  fi
done
