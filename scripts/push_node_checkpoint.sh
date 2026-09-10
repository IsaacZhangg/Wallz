#!/usr/bin/env bash
# Publish a freshly trained checkpoint to a generation node so its next chunk
# is played by the newer network. run_self_play_chunk reloads best.pt per
# chunk, so no restart of the generation loop is needed.
set -euo pipefail
CHECKPOINT="${1:?usage: push_node_checkpoint.sh <checkpoint> [node] [remote-dir]}"
NODE="${2:-${WALLZERO_NODE:?set WALLZERO_NODE or pass a node as the second argument}}"
REMOTE="${3:-${WALLZERO_REMOTE_DIR:-wallzero/output/wallzero-output}}"

scp -q "$CHECKPOINT" "$NODE:$REMOTE/best.pt.incoming"
ssh "$NODE" "mv '$REMOTE/best.pt.incoming' '$REMOTE/best.pt'"
echo "published $(basename "$CHECKPOINT") to $NODE"
