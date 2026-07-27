#!/usr/bin/env bash
# Always-on self-play generation. Each chunk fsyncs its own shard, so a crash
# or reboot costs at most the chunk in flight. 96-game chunks land roughly
# every 13 min on the GTX 1080 (frozen fp32 inference + eval cache + OC).
#
# Watchdogs (born from the chunk-82 wedge and the 2026-07-26 clock latch):
#   1. Hard cap: no chunk may run past 3h (timeout --kill-after).
#   2. Power: <110W for 15 consecutive minutes while a chunk runs means the
#      GPU is spinning, not working (utilization lies; power is the health
#      signal). Kill the chunk's process group and restart. A failed
#      nvidia-smi read counts as unhealthy, never healthy.
#   3. Clocks: driver 580 + Pascal sometimes latches the P2 boost at
#      1670 MHz — offsets query as applied but are ignored, and only a
#      reboot clears it (seen after nvidia-smi -pl, and once spontaneously
#      after idle/load P-state cycling between chunks). If SM clocks sit
#      below WALLZERO_MIN_SM_MHZ under real load for 10 min, re-apply
#      gpu-oc.sh once; if still latched 10 min later, reboot at the next
#      chunk boundary (rate-limited, and only when systemd autostart is
#      enabled so the pipeline comes back on its own).
#   4. Failures: consecutive chunk failures back off (5 min from the 3rd)
#      and reboot as a last resort from the 5th.
#   5. Stale PAUSE: the flywheel tags its PAUSE file "flywheel:<pid>". If
#      that pid is dead and no training round is running, the flywheel
#      crashed mid-round and its pause is removed after 3 min. A manual
#      PAUSE (empty or any other content) is never touched.
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
# Full OC sustains 1885-1936 MHz; the latch pins 1670. Set to 0 to disable
# the clock watchdog (e.g. when deliberately running stock clocks).
MIN_SM_MHZ="${WALLZERO_MIN_SM_MHZ:-1700}"
REBOOT_MARKER="$HOME/wallzero/.last-auto-reboot"
AUTOSTART="$HOME/wallzero/.autostart-enabled"

log() { echo "[loop] $(date -Is) $*"; }
is_num() { case "$1" in "" | *[!0-9]*) return 1 ;; *) return 0 ;; esac; }
gpu_watts() { nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 | cut -d. -f1; }
gpu_sm_mhz() { nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -1; }

reboot_allowed() {
  [ -f "$AUTOSTART" ] || return 1 # nothing would restart us after boot
  last=$(cat "$REBOOT_MARKER" 2>/dev/null)
  is_num "$last" || last=0
  [ $(($(date +%s) - last)) -ge 21600 ] # at most one auto-reboot per 6h
}

request_reboot() { # $1 = reason
  if reboot_allowed; then
    log "REBOOTING: $1"
    date +%s >"$REBOOT_MARKER"
    sync
    sudo systemctl reboot
    sleep 120 # never fall through into another chunk while shutting down
  else
    log "reboot wanted ($1) but rate-limited or autostart disabled; continuing degraded"
  fi
}

pause_is_stale() {
  head -1 "$HOME/wallzero/PAUSE" 2>/dev/null | grep -q "^flywheel:" || return 1
  fw_pid=$(head -1 "$HOME/wallzero/PAUSE" | cut -d: -f2)
  is_num "$fw_pid" && kill -0 "$fw_pid" 2>/dev/null && return 1
  pgrep -f "node_round_kata[.]py" >/dev/null && return 1
  return 0
}

fails=0
reboot_pending=""
stale_pause=0
# The clock-latch counters live OUTSIDE the per-chunk loop on purpose: a
# latched chunk only lasts ~13.5 min, so a per-chunk counter would fire the
# 10-min offset re-apply and then reset at the boundary — the 10-more-minutes
# reboot stage could never be reached (observed 2026-07-26 after round 40:
# every post-training chunk ran latched at 1670 MHz while the watchdog
# "re-applied" offsets each chunk forever). A healthy fast sample resets both.
slow=0 oc_reapplied=0
log "watchdog config: power<110W/15min kill; clocks<${MIN_SM_MHZ}MHz/10min reapply-then-reboot; autostart=$([ -f "$AUTOSTART" ] && echo on || echo off)"
while true; do
  if [ -f "$HOME/wallzero/PAUSE" ]; then
    if pause_is_stale; then
      stale_pause=$((stale_pause + 1))
      if [ "$stale_pause" -ge 3 ]; then
        log "removing stale flywheel PAUSE (owner dead, no round running)"
        rm -f "$HOME/wallzero/PAUSE"
        stale_pause=0
        continue
      fi
    else
      stale_pause=0
    fi
    log "paused"
    sleep 60
    continue
  fi
  stale_pause=0
  if [ -n "$reboot_pending" ]; then
    request_reboot "$reboot_pending"
    reboot_pending=""
  fi
  log "starting chunk"
  setsid timeout --kill-after=60 3h .venv/bin/python scripts/node_chunk_kata.py &
  chunk_pid=$!
  low=0 wedged=0 ticks=0
  while kill -0 "$chunk_pid" 2>/dev/null; do
    sleep 10
    ticks=$((ticks + 1))
    [ $((ticks % 6)) -eq 0 ] || continue # sample the GPU once a minute
    watts=$(gpu_watts)
    is_num "$watts" || watts=0 # failed read = unhealthy
    if [ "$watts" -lt 110 ]; then
      low=$((low + 1))
      slow=0
    else
      low=0
      mhz=$(gpu_sm_mhz)
      is_num "$mhz" || mhz=$MIN_SM_MHZ
      if [ "$MIN_SM_MHZ" -gt 0 ] && [ "$mhz" -lt "$MIN_SM_MHZ" ]; then
        slow=$((slow + 1))
      else
        slow=0
        oc_reapplied=0 # clocks healthy again; re-arm the two-stage ladder
      fi
    fi
    if [ "$low" -ge 15 ]; then
      log "power watchdog: ${watts}W for 15 min; killing wedged chunk group"
      kill -9 -- "-$chunk_pid" 2>/dev/null
      wedged=1
      break
    fi
    if [ "$slow" -ge 10 ]; then
      slow=0
      if [ "$oc_reapplied" -eq 0 ]; then
        log "clock watchdog: ${mhz}MHz under load for 10 min; re-applying offsets"
        bash "$HOME/wallzero/gpu-oc.sh" >/dev/null 2>&1
        oc_reapplied=1
      elif [ -z "$reboot_pending" ]; then
        log "clock watchdog: still ${mhz}MHz after offset re-apply; P2 latch — reboot at next chunk boundary"
        reboot_pending="P2 clock latch at ${mhz}MHz"
      fi
    fi
  done
  wait "$chunk_pid"
  status=$?
  if [ "$wedged" -eq 1 ] || [ "$status" -ne 0 ]; then
    fails=$((fails + 1))
    log "chunk failed (status $status, consecutive $fails); sweeping chunk process group"
    # Orphaned spawn workers keep the chunk's setsid pgid; kill by group so
    # a concurrent training round's workers are never touched (a broad
    # pkill -f spawn_main once killed a healthy round's loaders).
    pkill -9 -g "$chunk_pid" 2>/dev/null
    sleep 30
    if [ "$fails" -ge 5 ]; then
      request_reboot "$fails consecutive chunk failures"
    elif [ "$fails" -ge 3 ]; then
      log "backing off 5 min"
      sleep 270
    fi
  else
    fails=0
  fi
done
