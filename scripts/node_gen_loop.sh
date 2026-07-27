#!/usr/bin/env bash
# Always-on self-play generation. Each chunk fsyncs its own shard, so a crash
# or reboot costs at most the chunk in flight. 96-game chunks land roughly
# every 20 min on the GTX 1080 (frozen fp32 inference) — small durable
# shards over big ones.
#
# Two watchdogs, both born from the chunk-82 wedge (server thread stuck in a
# CUDA sync spin: 100% utilization at idle power, no Xid, unrecoverable from
# Python). Utilization lies; power draw is the health signal.
#   1. Hard cap: no chunk may run past 3h (timeout --kill-after).
#   2. Power: <80W for 15 consecutive minutes while a chunk runs means the
#      GPU is spinning, not working — kill and restart within ~15 minutes.
cd "$HOME/wallzero"
export WALLZERO_OUTPUT="$HOME/wallzero/output/wallzero-output"
export WALLZERO_WORKERS="${WALLZERO_WORKERS:-12}"
export WALLZERO_LEAF_BATCH="${WALLZERO_LEAF_BATCH:-16}"
export WALLZERO_GAMES="${WALLZERO_GAMES:-96}"
# Position-eval cache in the eval server: dedups repeated positions across
# concurrent games/searches so idle CPU+RAM buy back GPU forwards
# (~400K entries ~= 0.5GB; 0 disables).
export WALLZERO_EVAL_CACHE="${WALLZERO_EVAL_CACHE:-400000}"
export WALLZERO_DEVICE=cuda
# Pause file: touch ~/wallzero/PAUSE to let a training round own the GPU;
# remove it to resume generation without restarting the loop.
while true; do
  if [ -f "$HOME/wallzero/PAUSE" ]; then
    echo "[loop] $(date -Is) paused"
    sleep 60
    continue
  fi
  echo "[loop] $(date -Is) starting chunk"
  setsid timeout --kill-after=60 3h .venv/bin/python scripts/node_chunk_kata.py &
  chunk_pid=$!
  low=0
  wedged=0
  while kill -0 "$chunk_pid" 2>/dev/null; do
    sleep 60
    watts=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | cut -d. -f1)
    if [ "${watts:-200}" -lt 110 ]; then low=$((low + 1)); else low=0; fi
    if [ "$low" -ge 15 ]; then
      echo "[loop] $(date -Is) power watchdog: ${watts}W for 15 min; killing wedged chunk"
      kill -9 -- "-$chunk_pid" 2>/dev/null
      wedged=1
      break
    fi
  done
  wait "$chunk_pid"
  status=$?
  if [ "$wedged" -eq 1 ] || [ "$status" -ne 0 ]; then
    echo "[loop] $(date -Is) chunk failed (status $status); sweeping workers"
    # A killed parent orphans its spawn children; sweep them so the next
    # chunk starts with a free GPU.
    pkill -9 -f "multiprocessing.spawn import spawn_main" 2>/dev/null
    sleep 30
  fi
done
