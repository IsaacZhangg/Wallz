#!/bin/bash
# Apply WallZero GPU clock offsets on the generation node.
# Requires Coolbits 12 in xorg.conf.d and a running X server.
# Offsets shift GPU Boost's stock V/F curve; voltage stays stock, and the
# driver's power (198W) and thermal (83C) limits keep enforcing themselves,
# so an excessive offset fails as a crash or a canary mismatch, not damage.
#
# Usage: gpu-oc.sh <core_offset_mhz> [mem_transfer_offset] [fan_percent]
# Re-apply after every reboot or X restart (nothing persists automatically).
# Defaults are the validated maximums for this card (2026-07-26 canary
# ladder: +125/+800 pass, +150 core fails bitwise).
#
# DO NOT use `nvidia-smi -pl` on this node: changing the power limit latches
# the P2 core boost at 1670 MHz (driver 580 + Pascal) until the next reboot
# — measured 2026-07-26 after -pl 220 pinned the clocks that the offsets had
# unlocked. The stock 198W cap is fine; the full OC draws ~195-205W.
#
# DO NOT use nvidia-smi -pl on this node: changing the power limit latches
# the P2 core boost at 1670 MHz (driver 580 + Pascal) until the next reboot.
# Measured 2026-07-26; the stock 198W cap is fine — full OC runs ~195-205W.
CORE=${1:-125}
MEM=${2:-800}
FAN=${3:-100}
XAUTH=/var/run/lightdm/root/:0
[ -e "$HOME/.Xauthority" ] && sudo test ! -e "$XAUTH" && XAUTH="$HOME/.Xauthority"
run() { sudo env DISPLAY=:0 XAUTHORITY="$XAUTH" nvidia-settings "$@"; }
run -a "[gpu:0]/GPUFanControlState=1" -a "[fan:0]/GPUTargetFanSpeed=$FAN"
run -a "[gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=$CORE" \
  || run -a "[gpu:0]/GPUGraphicsClockOffset[3]=$CORE" -a "[gpu:0]/GPUGraphicsClockOffset[2]=$CORE"
run -a "[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=$MEM" \
  || run -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=$MEM" -a "[gpu:0]/GPUMemoryTransferRateOffset[2]=$MEM"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,fan.speed --format=csv,noheader
