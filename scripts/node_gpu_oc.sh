#!/bin/bash
# Apply WallZero GPU clock offsets on the generation node.
# Requires Coolbits 12 in xorg.conf.d and a running X server.
# Offsets shift GPU Boost's stock V/F curve; voltage stays stock, and the
# driver's power (198W) and thermal (83C) limits keep enforcing themselves,
# so an excessive offset fails as a crash or a canary mismatch, not damage.
#
# Usage: gpu-oc.sh <core_offset_mhz> [mem_transfer_offset] [fan_percent]
# Re-apply after every reboot or X restart (nothing persists automatically).
CORE=${1:-0}
MEM=${2:-0}
FAN=${3:-80}
XAUTH=/var/run/lightdm/root/:0
[ -e "$HOME/.Xauthority" ] && sudo test ! -e "$XAUTH" && XAUTH="$HOME/.Xauthority"
run() { sudo env DISPLAY=:0 XAUTHORITY="$XAUTH" nvidia-settings "$@"; }
run -a "[gpu:0]/GPUFanControlState=1" -a "[fan:0]/GPUTargetFanSpeed=$FAN"
run -a "[gpu:0]/GPUGraphicsClockOffsetAllPerformanceLevels=$CORE" \
  || run -a "[gpu:0]/GPUGraphicsClockOffset[3]=$CORE" -a "[gpu:0]/GPUGraphicsClockOffset[2]=$CORE"
run -a "[gpu:0]/GPUMemoryTransferRateOffsetAllPerformanceLevels=$MEM" \
  || run -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=$MEM" -a "[gpu:0]/GPUMemoryTransferRateOffset[2]=$MEM"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,fan.speed --format=csv,noheader
