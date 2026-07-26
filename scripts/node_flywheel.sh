#!/bin/bash
# WallZero auto-flywheel for the isaac-pc node: whenever FLYWHEEL_CHUNKS new
# replay shards exist (its own generation or manually merged Mac/Colab
# shards — it only counts files), it pauses the generation loop, trains one
# gateless round, adopts the result, and resumes generation.
#
# Runs alongside gen_loop.sh, which owns generation and the power watchdog.
# Respects a PAUSE file it did not create (manual interventions win), and
# skips while any training round is already running. Checkpoint pulls to
# other machines and Mac/Colab merges stay manual by design.
cd "$HOME/wallzero" || exit 1
OUT="$HOME/wallzero/output/wallzero-output"
THRESHOLD=${FLYWHEEL_CHUNKS:-4}

count_shards() { ls "$OUT/replay"/chunk-*.npz 2>/dev/null | wc -l; }

last=$(count_shards)
echo "[flywheel] $(date -Is) started; $last shards, training every $THRESHOLD new"
while true; do
  sleep 120
  if [ -f "$HOME/wallzero/PAUSE" ]; then continue; fi
  if pgrep -f node_round_kata.py > /dev/null; then continue; fi
  now=$(count_shards)
  if [ $((now - last)) -lt "$THRESHOLD" ]; then continue; fi

  echo "[flywheel] $(date -Is) $((now - last)) new shards; pausing generation"
  touch "$HOME/wallzero/PAUSE"
  while pgrep -f node_chunk_kata.py > /dev/null; do sleep 30; done
  echo "[flywheel] $(date -Is) training round starting"
  timeout --kill-after=60 3h env WALLZERO_OUTPUT="$OUT" WALLZERO_STEPS=3000 \
    .venv/bin/python scripts/node_round_kata.py >> "$HOME/wallzero/night.log" 2>&1
  rc=$?
  rm -f "$HOME/wallzero/PAUSE"
  if [ "$rc" -eq 0 ]; then
    echo "[flywheel] $(date -Is) round adopted; generation resumed"
    last=$(count_shards)
    timeout 30m bash "$HOME/wallzero/backup.sh" >> "$HOME/wallzero/backup.log" 2>&1 \
      || echo "[flywheel] $(date -Is) backup failed (non-fatal, cron will retry)"
  else
    echo "[flywheel] $(date -Is) round FAILED (status $rc); generation resumed, retrying in 10 min"
    sleep 600
  fi
done
