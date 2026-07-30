"""Start a new network lineage on the node, keeping the replay window.

Swapping net size is the one operation that cannot be a warm start: the
trunk shape changes, so the new net begins from a fresh initialisation. It
does not begin from no data — KataGo keeps its data window across net swaps,
and so do we, which is why this script deliberately leaves `replay/` alone.

The outgoing best.pt is archived into `anchors/` first, so the previous
lineage stays available as a comparison opponent.

    python scripts/node_new_lineage.py --blocks 10 --channels 128 \
        --gpool-blocks 5,8 --label era3-b24c256-final
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from wallzero.network import (
    NetworkConfig,
    PolicyValueNet,
    load_checkpoint,
    save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="~/wallzero/output/wallzero-output")
    parser.add_argument("--anchors", default="~/wallzero/anchors")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=256)
    parser.add_argument("--gpool-blocks", default="5,8")
    parser.add_argument("--gpool-channels", type=int, default=32)
    parser.add_argument("--norm-kind", default="fixscaleonenorm")
    parser.add_argument("--label", required=True, help="archive name for the old best")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser()
    anchors = Path(args.anchors).expanduser()
    best = output / "best.pt"
    archive = anchors / f"{args.label}.pt"

    gpool = tuple(int(x) for x in args.gpool_blocks.split(",") if x.strip())
    config = NetworkConfig(
        blocks=args.blocks,
        channels=args.channels,
        value_hidden=args.value_hidden,
        distance_head=True,
        norm_kind=args.norm_kind,
        gpool_blocks=gpool,
        gpool_channels=args.gpool_channels,
    )
    candidate = PolicyValueNet(config)
    parameters = sum(p.numel() for p in candidate.parameters())

    outgoing, _ = load_checkpoint(best)
    outgoing_parameters = sum(p.numel() for p in outgoing.parameters())
    shards = len(list((output / "replay").glob("chunk-*.npz")))
    state_path = output / "campaign-state.json"
    rounds = json.loads(state_path.read_text(encoding="utf-8"))["completed_rounds"]

    print(
        f"outgoing: {outgoing.config.blocks}b{outgoing.config.channels}c "
        f"{outgoing_parameters / 1e6:.2f}M -> archive {archive}"
    )
    print(
        f"incoming: {config.blocks}b{config.channels}c {parameters / 1e6:.2f}M "
        f"norm={config.norm_kind} gpool={gpool}"
    )
    print(f"replay kept: {shards} shards; round counter continues at {rounds}")
    if args.dry_run:
        print("dry run, nothing written")
        return

    if archive.exists():
        raise SystemExit(f"refusing to overwrite existing archive: {archive}")
    anchors.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, archive)
    save_checkpoint(
        best,
        candidate,
        metadata={
            "status": "lineage-start",
            "round": rounds,
            "previous_lineage": str(archive),
            "zero_human_data": True,
        },
    )
    print(f"wrote fresh {best}")


if __name__ == "__main__":
    main()
