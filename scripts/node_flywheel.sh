#!/bin/bash
# WallZero auto-flywheel for the isaac-pc node: whenever FLYWHEEL_CHUNKS new
# replay shards exist (its own generation or manually merged Mac/Colab
# shards — it only counts files), it pauses the generation loop, trains one
# gateless round, adopts the result, and resumes generation.
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
THRESHOLD=${FLYWHEEL_CHUNKS:-4}
PAUSE="$HOME/wallzero/PAUSE"

log() { echo "[flywheel] $(date -Is) $*"; }
count_shards() { ls "$OUT/replay"/chunk-*.npz 2>/dev/null | wc -l; }
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

# The shard baseline persists across restarts: without this, every restart
# (crash, reboot, systemd Restart=always) forgot progress toward the next
# round and postponed training by up to a full threshold of chunks.
LAST_FILE="$HOME/wallzero/.flywheel-last"
last=$(cat "$LAST_FILE" 2>/dev/null)
is_num "$last" || last=$(count_shards)
[ "$last" -gt "$(count_shards)" ] && last=$(count_shards) # shards renumbered/removed
echo "$last" >"$LAST_FILE"
log "started; $(count_shards) shards, baseline $last, training every $THRESHOLD new"
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
  now=$(count_shards)
  [ $((now - last)) -lt "$THRESHOLD" ] && continue

  log "$((now - last)) new shards; pausing generation"
  echo "flywheel:$$" >"$PAUSE"
  while pgrep -f "node_chunk_kata[.]py" >/dev/null; do sleep 30; done
  log "training round starting"
  setsid timeout --kill-after=60 3h env WALLZERO_OUTPUT="$OUT" WALLZERO_STEPS=3000 \
    .venv/bin/python scripts/node_round_kata.py >>"$HOME/wallzero/night.log" 2>&1 &
  round_pid=$!
  low=0
  while kill -0 "$round_pid" 2>/dev/null; do
    sleep 60
    watts=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1)
    is_num "$watts" || watts=0 # failed read = unhealthy
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
    last=$(count_shards)
    echo "$last" >"$LAST_FILE"
    timeout 30m bash "$HOME/wallzero/backup.sh" >>"$HOME/wallzero/backup.log" 2>&1 ||
      log "backup failed (non-fatal, cron will retry)"
  else
    log "round FAILED (status $rc); generation resumed, retrying in 10 min"
    sleep 600
  fi
done
