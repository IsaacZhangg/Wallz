"""Batched PUCT Monte Carlo Tree Search for neural self-play and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from wallzero.constants import ACTION_SIZE, MAX_GAME_PLIES
from wallzero.encoding import canonical_action
from wallzero.game import State, exact_pawn_race

FloatArray = NDArray[np.float32]


class Evaluator(Protocol):
    def __call__(self, states: list[State]) -> tuple[FloatArray, FloatArray]: ...


@dataclass(frozen=True, slots=True)
class MCTSConfig:
    simulations: int = 200
    pb_c_base: float = 19_652.0
    pb_c_init: float = 1.25
    fpu_reduction: float = 0.2
    dirichlet_alpha: float = 0.15
    dirichlet_fraction: float = 0.25
    max_plies: int = MAX_GAME_PLIES
    leaf_batch: int = 1

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if not 0.0 <= self.dirichlet_fraction <= 1.0:
            raise ValueError("dirichlet_fraction must be between zero and one")
        if self.leaf_batch < 1:
            raise ValueError("leaf_batch must be positive")


@dataclass(slots=True)
class Node:
    state: State | None
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] | None = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def expanded(self) -> bool:
        return self.children is not None

    @property
    def materialized_state(self) -> State:
        if self.state is None:
            raise RuntimeError("child state has not been materialized")
        return self.state

    def expand(self, logits: FloatArray) -> None:
        if self.children is not None:
            return
        if self.state is None:
            raise RuntimeError("cannot expand a child before materializing its state")
        legal = self.state.legal_actions()
        if not legal:
            self.children = {}
            return
        canonical = np.fromiter(
            (canonical_action(self.state, action) for action in legal),
            dtype=np.int64,
            count=len(legal),
        )
        selected = logits[canonical].astype(np.float64)
        selected -= selected.max()
        probabilities = np.exp(selected)
        total = float(probabilities.sum())
        if not math.isfinite(total) or total <= 0.0:
            probabilities.fill(1.0 / len(probabilities))
        else:
            probabilities /= total
        self.children = {
            action: Node(
                state=None,
                prior=float(probability),
            )
            for action, probability in zip(legal, probabilities, strict=True)
        }


@dataclass(slots=True)
class PendingEvaluation:
    node: Node
    path: list[Node]


@dataclass(slots=True)
class SearchTree:
    root: Node
    _noise_applied: bool = field(default=False, init=False)

    @classmethod
    def from_state(cls, state: State) -> SearchTree:
        return cls(root=Node(state=state))

    @property
    def state(self) -> State:
        return self.root.materialized_state

    def add_root_noise(self, config: MCTSConfig, rng: np.random.Generator) -> None:
        if self._noise_applied or not self.root.children:
            return
        children = list(self.root.children.values())
        noise = rng.dirichlet(np.full(len(children), config.dirichlet_alpha))
        fraction = config.dirichlet_fraction
        for child, sample in zip(children, noise, strict=True):
            child.prior = (1.0 - fraction) * child.prior + fraction * float(sample)
        self._noise_applied = True

    def select_leaf(self, config: MCTSConfig) -> PendingEvaluation | None:
        node = self.root
        if node.state is None:
            raise RuntimeError("search root has no state")
        path = [node]
        seen = {node.state.position_key}
        exact_race = node.state.walls_remaining == (0, 0)
        while True:
            terminal = node.state.terminal_value(config.max_plies)
            if terminal is not None:
                _backup(path, terminal)
                return None
            if exact_race and len(path) > 1:
                outcome, distance = exact_pawn_race(node.state)
                magnitude = 1.0 - min(distance, 500) / 1_000
                _backup(path, outcome * magnitude)
                return None
            if not node.expanded:
                return PendingEvaluation(node=node, path=path)
            if not node.children:
                _backup(path, 0.0)
                return None
            action, child = _select_child(node, config)
            if child.state is None:
                child.state = node.state.play(action, validate=False)
            node = child
            path.append(child)
            key = child.state.position_key
            if key in seen:
                _backup(path, 0.0)
                return None
            seen.add(key)

    def policy(self) -> FloatArray:
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        if self.root.state is None:
            raise RuntimeError("search root has no state")
        if not self.root.children:
            return policy
        visits = sum(child.visit_count for child in self.root.children.values())
        if visits:
            for action, child in self.root.children.items():
                policy[canonical_action(self.root.state, action)] = (
                    child.visit_count / visits
                )
            return policy
        for action, child in self.root.children.items():
            policy[canonical_action(self.root.state, action)] = child.prior
        return policy

    def select_action(self, temperature: float, rng: np.random.Generator) -> int:
        if not self.root.children:
            raise ValueError("cannot select an action from an unexpanded root")
        actions = np.fromiter(self.root.children, dtype=np.int64)
        visits = np.fromiter(
            (child.visit_count for child in self.root.children.values()),
            dtype=np.float64,
        )
        if temperature <= 1e-6:
            best = np.flatnonzero(visits == visits.max())
            return int(actions[int(rng.choice(best))])
        if not visits.any():
            weights = np.fromiter(
                (child.prior for child in self.root.children.values()),
                dtype=np.float64,
            )
        else:
            weights = np.power(visits, 1.0 / temperature)
        weights /= weights.sum()
        return int(rng.choice(actions, p=weights))

    def advance(self, action: int) -> None:
        if self.root.state is None:
            raise RuntimeError("search root has no state")
        if not self.root.children or action not in self.root.children:
            self.root = Node(self.root.state.play(action))
        else:
            child = self.root.children[action]
            if child.state is None:
                child.state = self.root.state.play(action, validate=False)
            self.root = child
            self.root.prior = 1.0
        self._noise_applied = False

    @property
    def root_value(self) -> float:
        return self.root.mean_value


def _select_child(parent: Node, config: MCTSConfig) -> tuple[int, Node]:
    if not parent.children:
        raise ValueError("cannot select a child from an unexpanded node")
    parent_visits = max(1, parent.visit_count)
    pb_c = (
        math.log((parent_visits + config.pb_c_base + 1.0) / config.pb_c_base)
        + config.pb_c_init
    ) * math.sqrt(parent_visits)
    visited_prior = sum(
        child.prior for child in parent.children.values() if child.visit_count
    )
    fpu = parent.mean_value - config.fpu_reduction * math.sqrt(visited_prior)

    best_action = -1
    best_score = -math.inf
    best_child: Node | None = None
    for action, child in parent.children.items():
        value = -child.mean_value if child.visit_count else fpu
        score = value + pb_c * child.prior / (child.visit_count + 1)
        if score > best_score or (score == best_score and action < best_action):
            best_action = action
            best_score = score
            best_child = child
    if best_child is None:
        raise RuntimeError("PUCT failed to select a child")
    return best_action, best_child


def _backup(path: list[Node], leaf_value: float) -> None:
    value = leaf_value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _apply_virtual_loss(path: list[Node]) -> None:
    # Values are stored from each node's own to-move perspective and parents
    # maximize -child.mean_value, so raising value_sum discourages every edge
    # on the path until the pending evaluation returns.
    for node in path:
        node.visit_count += 1
        node.value_sum += 1.0


def _revert_virtual_loss(path: list[Node]) -> None:
    for node in path:
        node.visit_count -= 1
        node.value_sum -= 1.0


def run_batched_search(
    trees: list[SearchTree],
    evaluator: Evaluator,
    config: MCTSConfig,
    *,
    add_noise: bool,
    rngs: list[np.random.Generator],
) -> None:
    """Search independent roots while batching leaves across trees on the GPU.

    With leaf_batch == 1 the evaluation batches, selection order, and results
    are identical to the original one-leaf-per-tree loop; larger leaf batches
    use virtual loss to collect several distinct leaves per tree per call.
    """
    if len(trees) != len(rngs):
        raise ValueError("each search tree requires its own random generator")
    unexpanded = [tree.root for tree in trees if not tree.root.expanded]
    if unexpanded:
        states = [node.state for node in unexpanded]
        if any(state is None for state in states):
            raise RuntimeError("search root has no state")
        logits, _ = evaluator([state for state in states if state is not None])
        for node, node_logits in zip(unexpanded, logits, strict=True):
            node.expand(node_logits)
    if add_noise:
        for tree, rng in zip(trees, rngs, strict=True):
            tree.add_root_noise(config, rng)

    remaining = [config.simulations] * len(trees)
    while any(budget > 0 for budget in remaining):
        pending: list[PendingEvaluation] = []
        for index, tree in enumerate(trees):
            collected = 0
            pending_ids: set[int] = set()
            limit = min(config.leaf_batch, remaining[index])
            while collected < limit:
                item = tree.select_leaf(config)
                if item is None:
                    remaining[index] -= 1
                    collected += 1
                    continue
                if id(item.node) in pending_ids:
                    break
                pending_ids.add(id(item.node))
                _apply_virtual_loss(item.path)
                pending.append(item)
                remaining[index] -= 1
                collected += 1
        if not pending:
            continue
        states = [item.node.state for item in pending]
        if any(state is None for state in states):
            raise RuntimeError("pending leaf has no state")
        logits, values = evaluator([state for state in states if state is not None])
        for item, node_logits, value in zip(pending, logits, values, strict=True):
            _revert_virtual_loss(item.path)
            item.node.expand(node_logits)
            _backup(item.path, float(value))


class UniformEvaluator:
    """A rule-only control evaluator useful for tests and baselines."""

    def __call__(self, states: list[State]) -> tuple[FloatArray, FloatArray]:
        return (
            np.zeros((len(states), ACTION_SIZE), dtype=np.float32),
            np.zeros((len(states),), dtype=np.float32),
        )
