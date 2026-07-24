"""Command-line entry points for training, inspection, and engine serving."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from wallzero.network import load_checkpoint, select_device
from wallzero.pipeline import preset, run_training
from wallzero.protocol import EngineService, serve_http, serve_json_lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wallzero")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="run resumable AlphaZero iterations")
    train.add_argument(
        "--preset",
        choices=("smoke", "bootstrap", "colab", "expert"),
        default="bootstrap",
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--iterations", type=int)

    smoke = commands.add_parser("smoke", help="run the smallest end-to-end loop")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", default="auto")

    serve = commands.add_parser("serve", help="serve one JSON analysis per input line")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--device", default="auto")
    serve.add_argument(
        "--http",
        type=int,
        metavar="PORT",
        help="serve the analyze protocol over localhost HTTP instead of stdio",
    )
    serve.add_argument("--host", default="127.0.0.1")

    analyze = commands.add_parser("analyze", help="analyze one JSON request")
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--request", type=Path)
    analyze.add_argument("--device", default="auto")

    inspect = commands.add_parser("inspect-checkpoint", help="show checkpoint metadata")
    inspect.add_argument("--checkpoint", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command in ("train", "smoke"):
        name = "smoke" if args.command == "smoke" else args.preset
        run_training(
            args.output,
            preset(name),
            device_name=args.device,
            iterations=1 if args.command == "smoke" else args.iterations,
        )
        return
    if args.command == "serve":
        service = EngineService.from_checkpoint(
            args.checkpoint, device_name=args.device
        )
        if args.http is not None:
            serve_http(service, host=args.host, port=args.http)
        else:
            serve_json_lines(service)
        return
    if args.command == "analyze":
        raw = (
            args.request.read_text(encoding="utf-8")
            if args.request
            else sys.stdin.read()
        )
        request = json.loads(raw)
        service = EngineService.from_checkpoint(
            args.checkpoint, device_name=args.device
        )
        print(json.dumps(service.analyze(request), indent=2))
        return
    if args.command == "inspect-checkpoint":
        model, payload = load_checkpoint(args.checkpoint)
        print(
            json.dumps(
                {
                    "network": asdict(model.config),
                    "metadata": payload.get("metadata", {}),
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "torch": torch.__version__,
                    "device": str(select_device()),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
