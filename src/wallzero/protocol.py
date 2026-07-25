"""Versioned JSON analysis protocol for external adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from wallzero.encoding import canonical_action
from wallzero.game import State, action_to_dict, cell, wall_index
from wallzero.mcts import MCTSConfig, SearchTree, run_batched_search
from wallzero.network import TorchEvaluator, load_checkpoint, select_device

REQUEST_SCHEMA = "wallzero.analyze.v1"
RESPONSE_SCHEMA = "wallzero.analysis.v1"


def state_from_wallz(payload: dict[str, Any]) -> State:
    pawns = payload.get("pawns")
    remaining = payload.get("wallsRemaining")
    if not isinstance(pawns, dict) or not isinstance(remaining, dict):
        raise ValueError("state requires pawns and wallsRemaining objects")
    p1 = pawns.get("p1")
    p2 = pawns.get("p2")
    if not isinstance(p1, dict) or not isinstance(p2, dict):
        raise ValueError("state requires p1 and p2 pawn coordinates")
    horizontal = 0
    vertical = 0
    for wall in payload.get("walls", []):
        if not isinstance(wall, dict):
            raise ValueError("every wall must be an object")
        index = wall_index(int(wall["x"]), int(wall["y"]))
        if wall.get("o") == "h":
            horizontal |= 1 << index
        elif wall.get("o") == "v":
            vertical |= 1 << index
        else:
            raise ValueError(f"invalid wall orientation: {wall.get('o')}")
    turn = payload.get("turn")
    if turn not in ("p1", "p2"):
        raise ValueError(f"invalid turn: {turn}")
    state = State(
        pawns=(
            cell(int(p1["x"]), int(p1["y"])),
            cell(int(p2["x"]), int(p2["y"])),
        ),
        walls_remaining=(int(remaining["p1"]), int(remaining["p2"])),
        horizontal=horizontal,
        vertical=vertical,
        to_play=0 if turn == "p1" else 1,
        ply=int(payload.get("ply", 0)),
    )
    if state.shortest_distance(0) < 0 or state.shortest_distance(1) < 0:
        raise ValueError("state contains a path-blocking wall layout")
    return state


def state_fingerprint(state: State) -> str:
    encoded = json.dumps(state.position_key, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(encoded.encode()).hexdigest()[:24]


@dataclass(slots=True)
class EngineService:
    evaluator: TorchEvaluator
    seed: int = 42

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        device_name: str = "auto",
    ) -> EngineService:
        device = select_device(device_name)
        model, _ = load_checkpoint(checkpoint, device=device)
        return cls(TorchEvaluator(model, device))

    def analyze(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("schema") != REQUEST_SCHEMA:
            raise ValueError(f"expected schema {REQUEST_SCHEMA}")
        payload = request.get("state")
        if not isinstance(payload, dict):
            raise ValueError("request requires a state object")
        state = state_from_wallz(payload)
        if state.terminal_value() is not None:
            raise ValueError("cannot analyze a terminal state")
        options = request.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("options must be an object")
        simulations = int(options.get("simulations", 800))
        top_moves = max(1, min(20, int(options.get("topMoves", 5))))
        config = MCTSConfig(
            simulations=simulations,
            dirichlet_fraction=0.0,
        )
        tree = SearchTree.from_state(state)
        rng = np.random.default_rng(self.seed)
        run_batched_search(
            [tree],
            self.evaluator,
            config,
            add_noise=False,
            rngs=[rng],
        )
        action = tree.select_action(0.0, rng)
        policy = tree.policy()
        ranked = sorted(
            tree.root.children.items() if tree.root.children else (),
            key=lambda item: (item[1].visit_count, item[1].prior),
            reverse=True,
        )[:top_moves]
        return {
            "schema": RESPONSE_SCHEMA,
            "id": request.get("id"),
            "fingerprint": state_fingerprint(state),
            "move": action_to_dict(action),
            "action": action,
            "value": tree.root_value,
            "simulations": simulations,
            "topMoves": [
                {
                    "action": candidate_action,
                    "move": action_to_dict(candidate_action),
                    "visits": child.visit_count,
                    "probability": float(
                        policy[canonical_action(state, candidate_action)]
                    ),
                    "q": -child.mean_value,
                }
                for candidate_action, child in ranked
            ],
        }


def serve_json_lines(service: EngineService) -> None:
    for line in __import__("sys").stdin:
        identifier: object = None
        try:
            request = json.loads(line)
            identifier = request.get("id") if isinstance(request, dict) else None
            if not isinstance(request, dict):
                raise ValueError("each line must contain a JSON object")
            response = service.analyze(request)
        except Exception as error:  # Protocol boundary must return structured errors.
            response = {
                "schema": RESPONSE_SCHEMA,
                "id": identifier,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        print(json.dumps(response, separators=(",", ":")), flush=True)


HEALTH_SCHEMA = "wallzero.health.v1"


def _analysis_response(service: EngineService, body: bytes) -> dict[str, Any]:
    identifier: object = None
    try:
        request = json.loads(body)
        identifier = request.get("id") if isinstance(request, dict) else None
        if not isinstance(request, dict):
            raise ValueError("request body must contain a JSON object")
        return service.analyze(request)
    except Exception as error:  # Protocol boundary must return structured errors.
        return {
            "schema": RESPONSE_SCHEMA,
            "id": identifier,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def create_http_server(
    service: EngineService, *, host: str = "127.0.0.1", port: int = 0
):
    """Serve the analyze protocol over local HTTP for the practice userscript.

    Binds localhost by default; the permissive CORS header only exposes the
    practice analysis endpoint on the user's own machine.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            # Chrome Private Network Access: HTTPS pages may only reach
            # localhost when the preflight explicitly allows it.
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send(404, {"schema": HEALTH_SCHEMA, "error": "not found"})
                return
            self._send(200, {"schema": HEALTH_SCHEMA, "status": "ok"})

        def do_POST(self) -> None:
            if self.path not in ("/", "/analyze"):
                self._send(404, {"schema": RESPONSE_SCHEMA, "error": "not found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            response = _analysis_response(service, body)
            self._send(200 if "error" not in response else 400, response)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve_http(
    service: EngineService, *, host: str = "127.0.0.1", port: int = 8787
) -> None:
    server = create_http_server(service, host=host, port=port)
    print(
        json.dumps(
            {
                "schema": HEALTH_SCHEMA,
                "status": "listening",
                "host": host,
                "port": server.server_address[1],
            }
        ),
        flush=True,
    )
    server.serve_forever()
