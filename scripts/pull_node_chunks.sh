#!/usr/bin/env bash
# Pull finished self-play shards from a generation node into the local mirror.
# Only completed chunks exist on disk (each is fsynced on completion), so this
# is safe to run at any time, including while the node keeps generating.
set -euo pipefail
NODE="${1:-isaac-pc}"
REMOTE="${2:-/home/tzhang/wallzero/output/wallzero-output}"
LOCAL="${3:-artifacts/runs/a100-campaign-local/wallzero-output}"

mkdir -p "$LOCAL/replay"
rsync -av --ignore-existing "$NODE:$REMOTE/replay/chunk-*.npz" "$LOCAL/replay/" | tail -3
rsync -av "$NODE:$REMOTE/chunk-metrics.jsonl" "$LOCAL/node-chunk-metrics.jsonl" | tail -1
echo "local shards: $(ls "$LOCAL"/replay/chunk-*.npz 2>/dev/null | wc -l | tr -d ' ')"
