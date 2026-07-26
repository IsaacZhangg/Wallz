#!/bin/bash
# WallZero off-site backup for the isaac-pc node -> Google Drive (rclone
# remote "gdrive", scope drive.file). Bounded on both ends by design:
#
#   Drive: all replay shards (~10MB/day), campaign state + logs (KB),
#          best.pt overwritten in place (~356MB steady), and every 5th
#          round's candidate as a lineage snapshot, pruned to the newest
#          SNAPSHOT_KEEP — steady state ~8GB total.
#   Local: candidates pruned to the newest LOCAL_KEEP rounds (~4.3GB);
#          a candidate is only deleted after its presence in Drive is
#          confirmed (snapshot rounds) or it is not a snapshot round.
#
# Idempotent (rclone copy skips existing files); safe to run from the
# flywheel hook and from cron simultaneously-ish, worst case double-copies.
set -u
OUT="$HOME/wallzero/output/wallzero-output"
REMOTE="gdrive:wallzero"
SNAPSHOT_EVERY=5
SNAPSHOT_KEEP=20
LOCAL_KEEP=12

log() { echo "[backup] $(date -Is) $*"; }

rclone mkdir "$REMOTE/replay" "$REMOTE/checkpoints" 2>/dev/null

# 1. Shards, state, logs (cheap, always). Logs are staged to a snapshot
#    copy first: uploading live-appended files fails Drive's checksum
#    (backup.log would literally be uploading itself mid-write).
rclone copy "$OUT/replay" "$REMOTE/replay" --include "chunk-*.npz" -q
rclone copy "$OUT" "$REMOTE" --include "campaign-state.json" --include "chunk-metrics.jsonl" -q
stage=$(mktemp -d)
cp "$HOME/wallzero"/*.log "$stage/" 2>/dev/null
rclone copy "$stage" "$REMOTE/logs" -q
rm -rf "$stage"

# 2. Current best (overwritten in place on Drive).
rclone copyto "$OUT/best.pt" "$REMOTE/checkpoints/best.pt" -q && log "best.pt uploaded"

# 3. Snapshot candidates: every Nth round.
for f in "$OUT"/candidates/round-*.pt; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  round=$((10#$(echo "$base" | tr -dc 0-9)))
  if [ $((round % SNAPSHOT_EVERY)) -eq 0 ]; then
    if ! rclone lsf "$REMOTE/checkpoints/$base" 2>/dev/null | grep -q .; then
      rclone copyto "$f" "$REMOTE/checkpoints/$base" -q && log "snapshot $base uploaded"
    fi
  fi
done

# 4. Prune Drive snapshots beyond the newest SNAPSHOT_KEEP.
rclone lsf "$REMOTE/checkpoints" --include "round-*.pt" | sort | head -n -"$SNAPSHOT_KEEP" | while read -r old; do
  rclone deletefile "$REMOTE/checkpoints/$old" -q && log "pruned Drive snapshot $old"
done

# 5. Prune local candidates beyond the newest LOCAL_KEEP (snapshot rounds
#    only after confirmed on Drive).
ls "$OUT"/candidates/round-*.pt 2>/dev/null | sort | head -n -"$LOCAL_KEEP" | while read -r old; do
  base=$(basename "$old")
  round=$((10#$(echo "$base" | tr -dc 0-9)))
  if [ $((round % SNAPSHOT_EVERY)) -eq 0 ]; then
    if ! rclone lsf "$REMOTE/checkpoints/$base" 2>/dev/null | grep -q .; then
      log "keeping local $base (snapshot not yet confirmed on Drive)"
      continue
    fi
  fi
  rm -f "$old" && log "pruned local $base"
done

log "done; drive usage: $(rclone size "$REMOTE" 2>/dev/null | tr '\n' ' ')"
