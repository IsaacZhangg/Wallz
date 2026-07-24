from __future__ import annotations

from wallzero.game import State
from wallzero.protocol import state_fingerprint, state_from_wallz


def test_wallz_protocol_state_matches_engine_initial_position() -> None:
    payload = {
        "pawns": {"p1": {"x": 4, "y": 0}, "p2": {"x": 4, "y": 8}},
        "turn": "p1",
        "walls": [],
        "wallsRemaining": {"p1": 10, "p2": 10},
        "winner": None,
    }
    state = state_from_wallz(payload)

    assert state == State.initial()
    assert state_fingerprint(state) == state_fingerprint(State.initial())


def test_http_server_round_trip_and_error_paths(tmp_path) -> None:
    import json
    import threading
    import urllib.error
    import urllib.request

    import torch

    from wallzero.network import (
        NetworkConfig,
        PolicyValueNet,
        TorchEvaluator,
        save_checkpoint,
    )
    from wallzero.protocol import EngineService, create_http_server

    torch.manual_seed(3)
    model = PolicyValueNet(NetworkConfig(channels=8, blocks=1, value_hidden=16))
    save_checkpoint(tmp_path / "tiny.pt", model)
    service = EngineService(TorchEvaluator(model, torch.device("cpu"), amp=False))
    server = create_http_server(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as reply:
            assert json.load(reply)["status"] == "ok"

        request = {
            "schema": "wallzero.analyze.v1",
            "id": "http-1",
            "state": {
                "pawns": {"p1": {"x": 4, "y": 0}, "p2": {"x": 4, "y": 8}},
                "turn": "p1",
                "walls": [],
                "wallsRemaining": {"p1": 10, "p2": 10},
                "winner": None,
            },
            "options": {"simulations": 4, "topMoves": 3},
        }
        body = json.dumps(request).encode()
        http_request = urllib.request.Request(
            f"{base}/analyze",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http_request, timeout=30) as reply:
            assert reply.headers["Access-Control-Allow-Origin"] == "*"
            response = json.load(reply)
        assert response["schema"] == "wallzero.analysis.v1"
        assert response["id"] == "http-1"
        assert response["move"]["type"] in ("pawn", "wall")
        assert isinstance(response["fingerprint"], str)

        bad = urllib.request.Request(
            f"{base}/analyze",
            data=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(bad, timeout=10)
            raise AssertionError("malformed request should return HTTP 400")
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert "error" in json.load(error)
    finally:
        server.shutdown()
        thread.join(timeout=5)
