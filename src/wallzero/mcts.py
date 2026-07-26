"""Batched PUCT Monte Carlo Tree Search for neural self-play and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from wallzero.constants import ACTION_SIZE, MAX_GAME_PLIES, WALL_GRID_SIZE
from wallzero.encoding import canonical_action
from wallzero.game import (
    State,
    coordinates,
    decode_action,
    exact_pawn_race,
    wall_coordinates,
    wall_index,
)

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
    # KataGo-style knobs; defaults preserve the original search exactly.
    dirichlet_concentration: float | None = None
    root_policy_temperature: float = 1.0
    forced_playout_scale: float = 0.0
    # Lower-confidence-bound move selection for deterministic (match) play.
    lcb_selection: bool = False
    lcb_z: float = 1.28
    lcb_min_visit_fraction: float = 0.1
    # Distance-margin utility: KataGo's score utility adapted to Quoridor, so
    # the search prefers winning by a wider path margin instead of treating
    # every win as equal. Rule-derived (BFS), zero human knowledge.
    distance_utility_weight: float = 0.0
    distance_utility_scale: float = 6.0
    # Subtree value bias correction: online correction of the network's
    # systematic evaluation error for recurring local move patterns, learned
    # within a single search from its own deeper results.
    subtree_bias_lambda: float = 0.0
    subtree_bias_alpha: float = 0.8

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if not 0.0 <= self.dirichlet_fraction <= 1.0:
            raise ValueError("dirichlet_fraction must be between zero and one")
        if self.leaf_batch < 1:
            raise ValueError("leaf_batch must be positive")
        if self.root_policy_temperature <= 0.0:
            raise ValueError("root_policy_temperature must be positive")
        if self.forced_playout_scale < 0.0:
            raise ValueError("forced_playout_scale must be non-negative")
        if self.lcb_z < 0.0:
            raise ValueError("lcb_z must be non-negative")
        if not 0.0 < self.lcb_min_visit_fraction <= 1.0:
            raise ValueError("lcb_min_visit_fraction must be in (0, 1]")
        if not 0.0 <= self.distance_utility_weight <= 1.0:
            raise ValueError("distance_utility_weight must be between zero and one")
        if self.distance_utility_scale <= 0.0:
            raise ValueError("distance_utility_scale must be positive")
        if self.subtree_bias_lambda < 0.0:
            raise ValueError("subtree_bias_lambda must be non-negative")
        if self.subtree_bias_alpha < 0.0:
            raise ValueError("subtree_bias_alpha must be non-negative")


@dataclass(slots=True)
class Node:
    state: State | None
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    value_sq_sum: float = 0.0
    children: dict[int, Node] | None = None
    # Subtree bias bookkeeping: the raw network value for this node, the value
    # actually credited to it, and the (action, local wall pattern, distance
    # bracket) key naming the local pattern that reached it.
    nn_value: float = 0.0
    own_value: float = 0.0
    bucket: tuple[int, ...] | None = None
    bias_error_sum: float = 0.0
    bias_weight: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def value_standard_error(self) -> float:
        """Standard error of this node's backed-up value estimate."""
        if self.visit_count < 2:
            return 1.0
        mean = self.value_sum / self.visit_count
        variance = self.value_sq_sum / self.visit_count - mean * mean
        return math.sqrt(max(variance, 0.0) / self.visit_count)

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


def bias_bucket(state: State, action: int) -> tuple[int, int, int]:
    """Name the local pattern an action created, for subtree bias sharing.

    KataGo buckets by the 5x5 stone pattern around the last move so that the
    same local tactic recurring elsewhere in the tree shares one bias entry.
    The Quoridor analog: the action itself, the 3x3 neighborhood of wall slots
    around it read from the position *after* the move (two bits per slot), and
    the mover's exact-BFS distance bracket so patterns are only shared within
    the same phase of the race. ``state`` is the position the action produced.
    """
    kind, index = decode_action(action)
    if kind == "pawn":
        x, y = coordinates(index)
        anchor_x = min(x, WALL_GRID_SIZE - 1)
        anchor_y = min(y, WALL_GRID_SIZE - 1)
    else:
        anchor_x, anchor_y = wall_coordinates(index)
    pattern = 0
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            pattern <<= 2
            slot_x = anchor_x + offset_x
            slot_y = anchor_y + offset_y
            if not (0 <= slot_x < WALL_GRID_SIZE and 0 <= slot_y < WALL_GRID_SIZE):
                continue
            bit = 1 << wall_index(slot_x, slot_y)
            if state.horizontal & bit:
                pattern |= 1
            if state.vertical & bit:
                pattern |= 2
    mover = 1 - state.to_play
    return action, pattern, state.shortest_distance(mover) // 4


@dataclass(slots=True)
class PendingEvaluation:
    node: Node
    path: list[Node]
    tree: SearchTree | None = None


@dataclass(slots=True)
class SearchTree:
    root: Node
    _noise_applied: bool = field(default=False, init=False)
    _bias: dict[tuple[int, ...], list[float]] = field(default_factory=dict, init=False)

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
        if config.root_policy_temperature != 1.0:
            # A restoring force toward uniform among similarly valued moves
            # (KataGo/SAI root policy softmax temperature), root only.
            powered = np.array(
                [child.prior for child in children], dtype=np.float64
            ) ** (1.0 / config.root_policy_temperature)
            powered /= powered.sum()
            for child, prior in zip(children, powered, strict=True):
                child.prior = float(prior)
        alpha = (
            config.dirichlet_concentration / len(children)
            if config.dirichlet_concentration is not None
            else config.dirichlet_alpha
        )
        noise = rng.dirichlet(np.full(len(children), alpha))
        fraction = config.dirichlet_fraction
        for child, sample in zip(children, noise, strict=True):
            child.prior = (1.0 - fraction) * child.prior + fraction * float(sample)
        self._noise_applied = True

    def _forced_child(self, config: MCTSConfig) -> tuple[int, Node] | None:
        """Return a root child owed forced playouts (KataGo forced playouts)."""
        if config.forced_playout_scale <= 0.0 or not self._noise_applied:
            return None
        children = self.root.children
        if not children:
            return None
        total = sum(child.visit_count for child in children.values())
        if total <= 0:
            return None
        best: tuple[int, Node] | None = None
        best_deficit = 0.0
        for action, child in children.items():
            forced = math.sqrt(config.forced_playout_scale * child.prior * total)
            deficit = forced - child.visit_count
            if deficit > best_deficit:
                best_deficit = deficit
                best = (action, child)
        return best

    def pruned_policy(self, config: MCTSConfig) -> FloatArray:
        """Visit-count policy with forced playouts subtracted (target pruning).

        The most-visited child keeps its full count; every other child loses
        up to its forced-playout allotment so Dirichlet-driven exploration
        does not contaminate the recorded training target.
        """
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        state = self.root.materialized_state
        children = self.root.children
        if not children:
            return policy
        if config.forced_playout_scale <= 0.0:
            return self.policy()
        total = sum(child.visit_count for child in children.values())
        if total <= 0:
            return self.policy()
        top_action = max(children, key=lambda action: children[action].visit_count)
        pruned: dict[int, float] = {}
        for action, child in children.items():
            if action == top_action:
                pruned[action] = float(child.visit_count)
                continue
            forced = math.sqrt(config.forced_playout_scale * child.prior * total)
            kept = max(0.0, child.visit_count - forced)
            if kept >= 1.0:
                pruned[action] = kept
        norm = sum(pruned.values())
        for action, count in pruned.items():
            policy[canonical_action(state, action)] = count / norm
        return policy

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
                return PendingEvaluation(node=node, path=path, tree=self)
            if not node.children:
                _backup(path, 0.0)
                return None
            forced = self._forced_child(config) if node is self.root else None
            action, child = (
                forced if forced is not None else _select_child(node, config)
            )
            if child.state is None:
                child.state = node.state.play(action, validate=False)
                if config.subtree_bias_lambda > 0.0:
                    child.bucket = bias_bucket(child.state, action)
            node = child
            path.append(child)
            key = child.state.position_key
            if key in seen:
                _backup(path, 0.0)
                return None
            seen.add(key)

    def bucket_bias(self, bucket: tuple[int, ...] | None) -> float:
        """Weighted average evaluation error observed for this local pattern."""
        if bucket is None:
            return 0.0
        entry = self._bias.get(bucket)
        if entry is None or entry[1] <= 0.0:
            return 0.0
        return entry[0] / entry[1]

    def update_bias(self, path: list[Node], config: MCTSConfig) -> None:
        """Refresh each path node's contribution to its pattern's bias.

        A node's observed error is its raw network value minus the average its
        own subtree reports, which deeper search makes the better estimate.
        Contributions are replaced rather than accumulated so a bucket stays a
        weighted average over nodes, as in KataGo.
        """
        if config.subtree_bias_lambda <= 0.0:
            return
        for node in path:
            if node.bucket is None or node.visit_count < 2:
                continue
            child_visits = node.visit_count - 1
            children_average = (node.value_sum - node.own_value) / child_visits
            observed_error = node.nn_value - children_average
            weight = float(child_visits) ** config.subtree_bias_alpha
            entry = self._bias.setdefault(node.bucket, [0.0, 0.0])
            entry[0] += observed_error * weight - node.bias_error_sum
            entry[1] += weight - node.bias_weight
            node.bias_error_sum = observed_error * weight
            node.bias_weight = weight

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

    def _lcb_action(self, config: MCTSConfig) -> int | None:
        """Pick the root move with the best lower confidence bound.

        Restricting to reasonably visited children keeps a barely explored
        move with an accidentally flattering mean (and therefore a tiny
        standard error) from winning the comparison.
        """
        children = self.root.children
        if not children:
            return None
        top_visits = max(child.visit_count for child in children.values())
        if top_visits < 2:
            return None
        threshold = max(2.0, config.lcb_min_visit_fraction * top_visits)
        best_action: int | None = None
        best_bound = -math.inf
        for action, child in children.items():
            if child.visit_count < threshold:
                continue
            # Child values are stored from the child's mover's perspective.
            bound = -child.mean_value - config.lcb_z * child.value_standard_error
            if bound > best_bound:
                best_bound = bound
                best_action = action
        return best_action

    def select_action(
        self,
        temperature: float,
        rng: np.random.Generator,
        *,
        config: MCTSConfig | None = None,
    ) -> int:
        if not self.root.children:
            raise ValueError("cannot select an action from an unexpanded root")
        actions = np.fromiter(self.root.children, dtype=np.int64)
        visits = np.fromiter(
            (child.visit_count for child in self.root.children.values()),
            dtype=np.float64,
        )
        if temperature <= 1e-6:
            if config is not None and config.lcb_selection:
                lcb_action = self._lcb_action(config)
                if lcb_action is not None:
                    return lcb_action
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
    squared = leaf_value * leaf_value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        node.value_sq_sum += squared
        value = -value


def blended_leaf_value(state: State, value: float, config: MCTSConfig) -> float:
    """Mix the exact path-distance margin into a network leaf value.

    The margin is measured from the leaf mover's perspective: being closer to
    goal than the opponent is positive. Both distances come from the rules
    engine's shortest-path search, so this adds no learned or human knowledge.
    """
    weight = config.distance_utility_weight
    if weight <= 0.0:
        return value
    own = state.shortest_distance(state.to_play)
    opponent = state.shortest_distance(1 - state.to_play)
    margin = math.tanh((opponent - own) / config.distance_utility_scale)
    return (1.0 - weight) * value + weight * margin


def _apply_virtual_loss(path: list[Node]) -> None:
    # Values are stored from each node's own to-move perspective and parents
    # maximize -child.mean_value, so raising value_sum discourages every edge
    # on the path until the pending evaluation returns.
    for node in path:
        node.visit_count += 1
        node.value_sum += 1.0
        node.value_sq_sum += 1.0


def _revert_virtual_loss(path: list[Node]) -> None:
    for node in path:
        node.visit_count -= 1
        node.value_sum -= 1.0
        node.value_sq_sum -= 1.0


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
            raw = blended_leaf_value(item.node.materialized_state, float(value), config)
            credited = raw
            if config.subtree_bias_lambda > 0.0 and item.tree is not None:
                bias = item.tree.bucket_bias(item.node.bucket)
                credited = min(1.0, max(-1.0, raw - config.subtree_bias_lambda * bias))
            item.node.nn_value = raw
            item.node.own_value = credited
            _backup(item.path, credited)
            if item.tree is not None:
                item.tree.update_bias(item.path, config)


class UniformEvaluator:
    """A rule-only control evaluator useful for tests and baselines."""

    def __call__(self, states: list[State]) -> tuple[FloatArray, FloatArray]:
        return (
            np.zeros((len(states), ACTION_SIZE), dtype=np.float32),
            np.zeros((len(states),), dtype=np.float32),
        )
