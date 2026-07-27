#!/bin/bash
# Boot-time recovery for the WallZero node (run by wallzero-prep.service
# before the generation and flywheel services start).
#
# A reboot clears the P2 clock latch but also wipes the clock offsets, so
# every boot re-applies the validated OC and proves it with the bitwise
# canary before generation starts. The canary reference is bound to
# best.pt's sha256; flywheel adoptions change best.pt, so a stale reference
# is re-recorded at stock clocks (always trustworthy — the reference is a
# function of the weights, not the clocks) and the OC verified against it.
#
# Never uses nvidia-smi -pl: changing the power limit latches the P2 boost
# at 1670 MHz on driver 580 + Pascal until the next reboot.
cd "$HOME/wallzero" || exit 1
REF="$HOME/wallzero/gpu-canary-reference.npz"
log() { echo "[prep] $(date -Is) $*"; }
canary() { timeout 5m .venv/bin/python gpu_canary.py; }

for _ in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 2; done
nvidia-smi >/dev/null 2>&1 || { log "GPU never came up; aborting prep"; exit 1; }
for _ in $(seq 60); do sudo test -e /var/run/lightdm/root/:0 && break; sleep 2; done
sudo nvidia-smi -pm 1 >/dev/null 2>&1

# A flywheel-tagged PAUSE from before the reboot would stop generation
# forever (its owner pid is gone). A manual PAUSE is kept — the pause
# intent survives the reboot.
if head -1 PAUSE 2>/dev/null | grep -q "^flywheel:"; then
  log "removing pre-reboot flywheel PAUSE"
  rm -f PAUSE
fi

bash gpu-oc.sh 0 0 80 >/dev/null 2>&1 # validate at stock clocks first
canary
rc=$?
if [ "$rc" -eq 3 ]; then
  log "canary reference stale (new best.pt); re-recording at stock clocks"
  rm -f "$REF"
  canary
  rc=$?
fi
if [ "$rc" -ne 0 ]; then
  log "canary rc=$rc at stock clocks — staying stock, investigate"
  exit 0
fi

bash gpu-oc.sh >/dev/null 2>&1 # validated +125/+800/fan100 defaults
if canary; then
  log "OC applied and canary PASS"
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,fan.speed --format=csv,noheader
else
  log "canary failed at OC; falling back to stock clocks"
  bash gpu-oc.sh 0 0 80 >/dev/null 2>&1
fi
