"""Multiprocess search phases: CPU actor processes feeding batched evaluators.

Tree search and rule work are GIL-bound Python, so a single process leaves the
GPU idle between tiny batches. Here each actor process runs its own slice of
games and ships pre-encoded input planes to the parent, which coalesces
whatever is queued into per-model forward passes and replies over per-worker
pipes. Self-play uses one model; arenas and evaluation matches use two, with
every request tagged by model index. Actors never import torch; the models
live only in the parent.

Outputs are deterministic per (seed, workers, leaf_batch), but float paths may
vary run-to-run with batch scheduling, so saved artifacts — not regeneration —
remain the durable source of truth.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray

from wallzero.arena import ArenaConfig, ArenaResult, evaluate_candidate
from wallzero.constants import ACTION_SIZE
from wallzero.encoding import encode_state
from wallzero.game import State
from wallzero.mcts import MCTSConfig
from wallzero.replay import TrainingExample
from wallzero.selfplay import SelfPlayConfig, SelfPlayStats, generate_self_play

FloatArray = NDArray[np.float32]
PlanesEvaluator = Callable[[FloatArray], tuple[FloatArray, FloatArray]]

_POLL_SECONDS = 0.05
_STALE_POLL_LIMIT = 200


@dataclass(frozen=True, slots=True)
class _EvalRequest:
    worker_id: int
    model_index: int
    planes: FloatArray


@dataclass(frozen=True, slots=True)
class _WorkerExit:
    worker_id: int


class _RemoteEvaluator:
    """Actor-side stub satisfying the MCTS Evaluator protocol over IPC."""

    def __init__(
        self,
        worker_id: int,
        model_index: int,
        requests: mp.Queue[Any],
        replies: Connection,
    ) -> None:
        self.worker_id = worker_id
        self.model_index = model_index
        self.requests = requests
        self.replies = replies

    def __call__(self, states: list[State]) -> tuple[FloatArray, FloatArray]:
        if not states:
            return (
                np.empty((0, ACTION_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        planes = np.stack([encode_state(state) for state in states])
        self.requests.put(_EvalRequest(self.worker_id, self.model_index, planes))
        logits, values = self.replies.recv()
        return logits, values


def uniform_planes_evaluator(planes: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Rule-only control: zero logits and values, batched server-side."""
    return (
        np.zeros((len(planes), ACTION_SIZE), dtype=np.float32),
        np.zeros((len(planes),), dtype=np.float32),
    )


def _worker_seed(seed: int, worker_id: int) -> int:
    return int(np.random.SeedSequence([seed, worker_id]).generate_state(1)[0])


def _self_play_actor(
    worker_id: int,
    requests: mp.Queue[Any],
    replies: Connection,
    results: mp.Queue[Any],
    payload: tuple[MCTSConfig, SelfPlayConfig, State | None],
) -> None:
    mcts_config, self_play_config, initial_state = payload
    evaluator = _RemoteEvaluator(worker_id, 0, requests, replies)

    def progress(stats: SelfPlayStats) -> None:
        results.put(("progress", worker_id, stats))

    examples, stats = generate_self_play(
        evaluator,
        mcts_config,
        self_play_config,
        progress=progress,
        initial_state=initial_state,
    )
    results.put(("progress", worker_id, stats))
    results.put(("done", worker_id, examples))


def _arena_actor(
    worker_id: int,
    requests: mp.Queue[Any],
    replies: Connection,
    results: mp.Queue[Any],
    payload: tuple[MCTSConfig, ArenaConfig, State | None],
) -> None:
    mcts_config, arena_config, initial_state = payload
    candidate = _RemoteEvaluator(worker_id, 0, requests, replies)
    incumbent = _RemoteEvaluator(worker_id, 1, requests, replies)

    def progress(result: ArenaResult) -> None:
        results.put(("progress", worker_id, result))

    result = evaluate_candidate(
        candidate,
        incumbent,
        mcts_config,
        arena_config,
        progress=progress,
        initial_state=initial_state,
    )
    results.put(("progress", worker_id, result))
    results.put(("done", worker_id, None))


def _actor_main(
    actor: Callable[..., None],
    worker_id: int,
    requests: mp.Queue[Any],
    replies: Connection,
    results: mp.Queue[Any],
    payload: Any,
) -> None:
    try:
        actor(worker_id, requests, replies, results, payload)
    except Exception:
        results.put(("error", worker_id, traceback.format_exc()))
    finally:
        requests.put(_WorkerExit(worker_id))


def _async_planes(evaluator: PlanesEvaluator) -> tuple[Any, Any] | None:
    """Return (submit, collect) when the evaluator supports async batches.

    Callers pass bound ``TorchEvaluator.evaluate_planes`` methods; the owning
    evaluator advertises CUDA-only async support. Plain functions (the uniform
    control, test doubles) keep the synchronous path.
    """
    owner = getattr(evaluator, "__self__", None)
    if owner is None or not getattr(owner, "supports_async_planes", False):
        return None
    return owner.submit_planes, owner.collect_planes


def _serve_evaluations(
    planes_evaluators: Sequence[PlanesEvaluator],
    requests: mp.Queue[Any],
    replies: dict[int, Connection],
    processes: dict[int, BaseProcess],
    server_failure: list[str],
) -> None:
    active = set(replies)
    asyncs = [_async_planes(evaluator) for evaluator in planes_evaluators]
    async_pairs = [pair for pair in asyncs if pair is not None]
    pipelined = len(async_pairs) == len(planes_evaluators)

    def drain(block: bool) -> list[_EvalRequest]:
        batch: list[_EvalRequest] = []
        try:
            message = (
                requests.get(timeout=_POLL_SECONDS) if block else requests.get_nowait()
            )
        except queue.Empty:
            if block:
                for worker_id in list(active):
                    if not processes[worker_id].is_alive():
                        active.discard(worker_id)
            return batch
        while True:
            if isinstance(message, _WorkerExit):
                active.discard(message.worker_id)
            else:
                batch.append(message)
            try:
                message = requests.get_nowait()
            except queue.Empty:
                return batch

    def grouped(batch: list[_EvalRequest]) -> list[tuple[int, list[_EvalRequest]]]:
        by_model: dict[int, list[_EvalRequest]] = {}
        for request in batch:
            by_model.setdefault(request.model_index, []).append(request)
        return sorted(by_model.items())

    def reply(group: list[_EvalRequest], logits: Any, values: Any) -> None:
        offset = 0
        for request in group:
            count = len(request.planes)
            replies[request.worker_id].send(
                (
                    logits[offset : offset + count],
                    values[offset : offset + count],
                )
            )
            offset += count

    def evaluate_sync(batch: list[_EvalRequest]) -> None:
        for model_index, group in grouped(batch):
            planes = np.concatenate([request.planes for request in group])
            logits, values = planes_evaluators[model_index](planes)
            reply(group, logits, values)

    def submit(batch: list[_EvalRequest]) -> list[tuple[int, list[_EvalRequest], Any]]:
        flight = []
        for model_index, group in grouped(batch):
            planes = np.concatenate([request.planes for request in group])
            submit_planes, _ = async_pairs[model_index]
            flight.append((model_index, group, submit_planes(planes)))
        return flight

    def collect(flight: list[tuple[int, list[_EvalRequest], Any]]) -> None:
        for model_index, group, pending in flight:
            _, collect_planes = async_pairs[model_index]
            logits, values = collect_planes(pending)
            reply(group, logits, values)

    try:
        if not pipelined:
            while active:
                batch = drain(block=True)
                if batch:
                    evaluate_sync(batch)
            return
        # Pipelined: while the GPU runs batch N, keep draining requests into
        # batch N+1 and submit it the moment N finishes, so reply
        # serialization and queue work overlap GPU compute instead of
        # leaving the device idle between forward passes.
        in_flight: list[tuple[int, list[_EvalRequest], Any]] | None = None
        backlog: list[_EvalRequest] = []
        while active or in_flight is not None or backlog:
            if in_flight is None:
                backlog.extend(drain(block=not backlog))
                if backlog:
                    in_flight = submit(backlog)
                    backlog = []
                continue
            if not all(pending.ready() for _, _, pending in in_flight):
                gathered = drain(block=False)
                if gathered:
                    backlog.extend(gathered)
                else:
                    time.sleep(0.001)
                continue
            next_flight = submit(backlog) if backlog else None
            backlog = []
            collect(in_flight)
            in_flight = next_flight
    except Exception:
        server_failure.append(traceback.format_exc())
        for connection in replies.values():
            connection.close()


def _run_distributed(
    planes_evaluators: Sequence[PlanesEvaluator],
    actor: Callable[..., None],
    payloads: dict[int, Any],
    on_progress: Callable[[dict[int, Any]], None] | None,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Spawn one actor per payload; return (done payloads, final progress)."""
    context = mp.get_context("spawn")
    requests: mp.Queue[Any] = context.Queue()
    results: mp.Queue[Any] = context.Queue()
    replies: dict[int, Connection] = {}
    processes: dict[int, BaseProcess] = {}
    for worker_id, payload in payloads.items():
        parent_end, child_end = context.Pipe()
        process = context.Process(
            target=_actor_main,
            args=(actor, worker_id, requests, child_end, results, payload),
            daemon=True,
        )
        process.start()
        child_end.close()
        replies[worker_id] = parent_end
        processes[worker_id] = process

    server_failure: list[str] = []
    server = Thread(
        target=_serve_evaluations,
        args=(planes_evaluators, requests, replies, processes, server_failure),
        daemon=True,
    )
    server.start()

    outputs: dict[int, Any] = {}
    latest_progress: dict[int, Any] = {}
    failure: str | None = None
    stale_polls = 0
    while len(outputs) < len(processes) and failure is None:
        try:
            kind, worker_id, payload = results.get(timeout=_POLL_SECONDS)
        except queue.Empty:
            dead = [
                worker_id
                for worker_id, process in processes.items()
                if not process.is_alive() and worker_id not in outputs
            ]
            if dead:
                stale_polls += 1
                if stale_polls > _STALE_POLL_LIMIT:
                    failure = f"workers exited without results: {dead}"
            continue
        stale_polls = 0
        if kind == "error":
            failure = str(payload)
        elif kind == "done":
            outputs[worker_id] = payload
        else:
            latest_progress[worker_id] = payload
            if on_progress is not None:
                on_progress(latest_progress)

    for process in processes.values():
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
    server.join(timeout=30.0)
    for connection in replies.values():
        connection.close()
    if server_failure:
        failure = server_failure[0]
    if failure is not None:
        raise RuntimeError(f"distributed search failed: {failure}")
    return outputs, latest_progress


def _merge_self_play_stats(per_worker: dict[int, SelfPlayStats]) -> SelfPlayStats:
    merged = SelfPlayStats()
    for stats in per_worker.values():
        for field in fields(SelfPlayStats):
            if field.name == "elapsed_seconds":
                continue
            setattr(
                merged,
                field.name,
                getattr(merged, field.name) + getattr(stats, field.name),
            )
    return merged


def _merge_arena_results(per_worker: dict[int, ArenaResult]) -> ArenaResult:
    merged = ArenaResult()
    for result in per_worker.values():
        for field in fields(ArenaResult):
            setattr(
                merged,
                field.name,
                getattr(merged, field.name) + getattr(result, field.name),
            )
    return merged


def generate_self_play_parallel(
    planes_evaluator: PlanesEvaluator,
    mcts_config: MCTSConfig,
    config: SelfPlayConfig,
    *,
    workers: int,
    progress: Callable[[SelfPlayStats], None] | None = None,
    initial_state: State | None = None,
) -> tuple[list[TrainingExample], SelfPlayStats]:
    """Run generate_self_play across worker processes with central batching."""
    if workers < 2:
        raise ValueError("parallel self-play requires at least two workers")
    started = time.monotonic()
    payloads: dict[int, Any] = {}
    remainder = config.games % workers
    for worker_id in range(workers):
        share = config.games // workers + (1 if worker_id < remainder else 0)
        if share == 0:
            continue
        worker_config = replace(
            config,
            games=share,
            parallel_games=min(share, config.parallel_games),
            seed=_worker_seed(config.seed, worker_id),
        )
        payloads[worker_id] = (mcts_config, worker_config, initial_state)

    def on_progress(latest: dict[int, SelfPlayStats]) -> None:
        if progress is not None:
            merged = _merge_self_play_stats(latest)
            merged.elapsed_seconds = time.monotonic() - started
            progress(merged)

    outputs, final_progress = _run_distributed(
        [planes_evaluator], _self_play_actor, payloads, on_progress
    )
    examples = [
        example for worker_id in sorted(outputs) for example in outputs[worker_id]
    ]
    stats = _merge_self_play_stats(final_progress)
    stats.elapsed_seconds = time.monotonic() - started
    return examples, stats


def _partition_pairs(games: int, workers: int) -> list[int]:
    """Split an even game count into even per-worker counts, whole pairs only."""
    pairs = games // 2
    workers = min(workers, pairs)
    return [
        2 * (pairs // workers + (1 if index < pairs % workers else 0))
        for index in range(workers)
    ]


def evaluate_candidate_parallel(
    candidate_planes: PlanesEvaluator,
    incumbent_planes: PlanesEvaluator,
    mcts_config: MCTSConfig,
    config: ArenaConfig,
    *,
    workers: int,
    progress: Callable[[ArenaResult], None] | None = None,
    initial_state: State | None = None,
) -> ArenaResult:
    """Run the color-balanced arena across worker processes.

    Whole opening pairs stay within one worker, so color balance and paired
    openings are preserved; each worker draws its own opening seeds.
    """
    if workers < 2:
        raise ValueError("parallel arena requires at least two workers")

    payloads: dict[int, Any] = {}
    for worker_id, games in enumerate(_partition_pairs(config.games, workers)):
        worker_config = replace(
            config,
            games=games,
            parallel_games=min(games, config.parallel_games),
            seed=_worker_seed(config.seed, worker_id),
        )
        payloads[worker_id] = (mcts_config, worker_config, initial_state)

    def on_progress(latest: dict[int, ArenaResult]) -> None:
        if progress is not None:
            progress(_merge_arena_results(latest))

    _, final_progress = _run_distributed(
        [candidate_planes, incumbent_planes], _arena_actor, payloads, on_progress
    )
    return _merge_arena_results(final_progress)
