// ==UserScript==
// @name         Wallz Practice Coach
// @namespace    https://wallz.gg/
// @version      1.11.0
// @description  Finds and optionally plays strong Wallz moves with adaptive local analysis and an optional local WallZero engine.
// @author       Isaac Zhang
// @match        https://wallz.gg/*
// @match        https://www.wallz.gg/*
// @run-at       document-idle
// @grant        none
// ==/UserScript==

(function loadWallzCoach(factory) {
	const api = factory();

	if (typeof module === "object" && module.exports) {
		module.exports = api;
	}

	if (typeof window !== "undefined" && typeof document !== "undefined") {
		api.boot();
	}
})(function createWallzCoach() {
	const BOARD_SIZE = 9;
	const WALL_SLOTS = 8;
	const BOARD_PIXELS = 636;
	const BOARD_STEP = 72;
	const SVG_NS = "http://www.w3.org/2000/svg";
	const SCRIPT_VERSION = "1.10.0";
	// The WallZero bridge activates only because the recorded independent
	// strength gate passed (README "Strength gates"): the frozen round-14
	// checkpoint beat the uniform-MCTS control at an equal 192-simulation
	// budget over a pre-declared 3-seed, 600-game suite — score 0.575, 95% CI
	// [0.535, 0.614]. Serve it with:
	//   wallzero serve --checkpoint artifacts/runs/a100-bootstrap/best-gate-passed.pt --http 8787
	// The bridge stays inert unless that local server responds to /health.
	// Alt+W toggles it on/off at runtime (persisted in localStorage).
	const WALLZERO_BRIDGE_ENABLED = true;
	const WALLZERO_TOGGLE_KEY = "wallzCoachWallZeroEnabled";

	function wallZeroUserEnabled() {
		try {
			return (localStorage.getItem(WALLZERO_TOGGLE_KEY) ?? "on") !== "off";
		} catch {
			return true;
		}
	}

	function setWallZeroUserEnabled(on) {
		try {
			localStorage.setItem(WALLZERO_TOGGLE_KEY, on ? "on" : "off");
		} catch {
			// Storage may be blocked; the toggle then lasts for this page only.
		}
	}
	const ADAPTIVE_SEARCH_LIMIT = Object.freeze({
		id: "adaptive",
		name: "Adaptive",
		label: "Adaptive · ≤1.5s",
		timeMs: 1_500,
		maxDepth: 25,
	});
	const LOSS_LOG_STORAGE_KEY = "wallz-coach-loss-reports-v1";
	const MAX_STORED_LOSSES = 3;

	function createEngine() {
		const WIN_SCORE = 1_000_000;
		const REPETITION_SCORE = 320;
		const TIMEOUT = Symbol("analysis-timeout");
		const directions = [
			[0, -1],
			[0, 1],
			[1, 0],
			[-1, 0],
		];
		const PLAYER_NAMES = ["p1", "p2"];
		const MAX_WALLS = 10;
		const PAWN_MOVE_COUNT = BOARD_SIZE * BOARD_SIZE;
		const WALL_CODE_COUNT = WALL_SLOTS * WALL_SLOTS * 2;
		const WALL_MOVE_OFFSET = PAWN_MOVE_COUNT;
		const MOVE_CODE_COUNT = PAWN_MOVE_COUNT + WALL_CODE_COUNT;
		const RACE_STATE_COUNT = PAWN_MOVE_COUNT * PAWN_MOVE_COUNT * 2;
		const TT_MAX_ENTRIES = 160_000;
		const FEATURE_CACHE_MAX_ENTRIES = 40_000;
		const MOVE_CACHE_MAX_ENTRIES = 24_000;
		const RACE_TABLE_MAX_ENTRIES = 64;
		const MATE_THRESHOLD = WIN_SCORE - 10_000;
		const PATH_COUNT_CAP = 1_000_000_000;
		const neighborEdges = Array.from(
			{ length: PAWN_MOVE_COUNT },
			(_, index) => {
				const x = index % BOARD_SIZE;
				const y = Math.floor(index / BOARD_SIZE);
				const edges = [];
				if (y > 0) {
					edges.push({
						edge: index - BOARD_SIZE,
						south: true,
						to: index - BOARD_SIZE,
					});
				}
				if (y + 1 < BOARD_SIZE) {
					edges.push({ edge: index, south: true, to: index + BOARD_SIZE });
				}
				if (x > 0) {
					edges.push({ edge: index - 1, south: false, to: index - 1 });
				}
				if (x + 1 < BOARD_SIZE) {
					edges.push({ edge: index, south: false, to: index + 1 });
				}
				return edges;
			},
		);
		const transpositionTable = new Map();
		const routeAnalysisCache = new Map();
		const wallPressureCache = new Map();
		const moveGenerationCache = new Map();
		const pawnRaceTableCache = new Map();
		const zobrist = createZobrist();
		const wallEdgeMasks = createWallEdgeMasks();
		let analysisGeneration = 0;

		function opponent(player) {
			return player === "p1" ? "p2" : "p1";
		}

		function goalRow(player) {
			return player === "p1" ? BOARD_SIZE - 1 : 0;
		}

		function inBounds(position) {
			return (
				Number.isInteger(position.x) &&
				Number.isInteger(position.y) &&
				position.x >= 0 &&
				position.x < BOARD_SIZE &&
				position.y >= 0 &&
				position.y < BOARD_SIZE
			);
		}

		function wallInBounds(wall) {
			return (
				(wall.o === "h" || wall.o === "v") &&
				Number.isInteger(wall.x) &&
				Number.isInteger(wall.y) &&
				wall.x >= 0 &&
				wall.x < WALL_SLOTS &&
				wall.y >= 0 &&
				wall.y < WALL_SLOTS
			);
		}

		function wallKey(wall) {
			return `${wall.o}-${wall.x}-${wall.y}`;
		}

		function wallConflicts(wall, existingWalls) {
			return existingWalls.some((existing) => {
				if (wall.x === existing.x && wall.y === existing.y) {
					return true;
				}
				if (
					wall.o === "h" &&
					existing.o === "h" &&
					wall.y === existing.y &&
					Math.abs(wall.x - existing.x) === 1
				) {
					return true;
				}
				return (
					wall.o === "v" &&
					existing.o === "v" &&
					wall.x === existing.x &&
					Math.abs(wall.y - existing.y) === 1
				);
			});
		}

		function compileBlockedEdges(walls) {
			const south = new Uint8Array(BOARD_SIZE * BOARD_SIZE);
			const east = new Uint8Array(BOARD_SIZE * BOARD_SIZE);

			for (const wall of walls) {
				if (wall.o === "h") {
					const index = wall.y * BOARD_SIZE + wall.x;
					south[index] = 1;
					south[index + 1] = 1;
				} else {
					const index = wall.y * BOARD_SIZE + wall.x;
					east[index] = 1;
					east[index + BOARD_SIZE] = 1;
				}
			}

			return { south, east };
		}

		function edgeBlocked(from, to, blocked) {
			if (to.y === from.y + 1) {
				return blocked.south[from.y * BOARD_SIZE + from.x] === 1;
			}
			if (to.y === from.y - 1) {
				return blocked.south[to.y * BOARD_SIZE + to.x] === 1;
			}
			if (to.x === from.x + 1) {
				return blocked.east[from.y * BOARD_SIZE + from.x] === 1;
			}
			return blocked.east[to.y * BOARD_SIZE + to.x] === 1;
		}

		function shortestPathCells(start, targetRow, walls, compiled) {
			if (start.y === targetRow) {
				return [{ x: start.x, y: start.y }];
			}

			const blocked = compiled || compileBlockedEdges(walls);
			const seen = new Uint8Array(BOARD_SIZE * BOARD_SIZE);
			const previous = new Int16Array(BOARD_SIZE * BOARD_SIZE);
			previous.fill(-1);
			const queue = new Int16Array(BOARD_SIZE * BOARD_SIZE);
			let head = 0;
			let tail = 0;
			const startIndex = start.y * BOARD_SIZE + start.x;
			seen[startIndex] = 1;
			queue[tail++] = startIndex;
			let finishIndex = -1;

			while (head < tail && finishIndex === -1) {
				const index = queue[head++];
				const x = index % BOARD_SIZE;
				const y = Math.floor(index / BOARD_SIZE);
				const from = { x, y };

				for (const [dx, dy] of directions) {
					const to = { x: x + dx, y: y + dy };
					if (!inBounds(to) || edgeBlocked(from, to, blocked)) {
						continue;
					}
					const nextIndex = to.y * BOARD_SIZE + to.x;
					if (seen[nextIndex]) {
						continue;
					}
					seen[nextIndex] = 1;
					previous[nextIndex] = index;
					if (to.y === targetRow) {
						finishIndex = nextIndex;
						break;
					}
					queue[tail++] = nextIndex;
				}
			}

			if (finishIndex === -1) {
				return null;
			}

			const path = [];
			let cursor = finishIndex;
			while (cursor !== -1) {
				const x = cursor % BOARD_SIZE;
				path.push({ x, y: Math.floor(cursor / BOARD_SIZE) });
				cursor = previous[cursor];
			}
			path.reverse();
			return path;
		}

		function shortestDistance(state, player, compiled) {
			const path = shortestPathCells(
				state.pawns[player],
				goalRow(player),
				state.walls,
				compiled,
			);
			return path ? path.length - 1 : -1;
		}

		function samePosition(left, right) {
			return left.x === right.x && left.y === right.y;
		}

		function legalPawnMoves(state, player) {
			const pawn = state.pawns[player];
			const otherPawn = state.pawns[opponent(player)];
			const blocked = compileBlockedEdges(state.walls);
			const moves = [];

			for (const [dx, dy] of directions) {
				const adjacent = { x: pawn.x + dx, y: pawn.y + dy };
				if (!inBounds(adjacent) || edgeBlocked(pawn, adjacent, blocked)) {
					continue;
				}

				if (!samePosition(adjacent, otherPawn)) {
					moves.push(adjacent);
					continue;
				}

				const beyond = { x: adjacent.x + dx, y: adjacent.y + dy };
				if (inBounds(beyond) && !edgeBlocked(adjacent, beyond, blocked)) {
					moves.push(beyond);
					continue;
				}

				const sideDirections =
					dx === 0
						? [
								[1, 0],
								[-1, 0],
							]
						: [
								[0, 1],
								[0, -1],
							];

				for (const [sideX, sideY] of sideDirections) {
					const diagonal = {
						x: adjacent.x + sideX,
						y: adjacent.y + sideY,
					};
					if (inBounds(diagonal) && !edgeBlocked(adjacent, diagonal, blocked)) {
						moves.push(diagonal);
					}
				}
			}

			const unique = new Map();
			for (const move of moves) {
				unique.set(`${move.x},${move.y}`, move);
			}
			return Array.from(unique.values());
		}

		function stateAfterPawnMove(state, player, to) {
			const pawns = {
				p1: { ...state.pawns.p1 },
				p2: { ...state.pawns.p2 },
				[player]: { ...to },
			};
			const winner = to.y === goalRow(player) ? player : null;
			return {
				...state,
				pawns,
				turn: opponent(player),
				winner,
			};
		}

		function stateAfterWall(state, player, wall) {
			return {
				...state,
				walls: state.walls.concat({ ...wall }),
				wallsRemaining: {
					...state.wallsRemaining,
					[player]: state.wallsRemaining[player] - 1,
				},
				turn: opponent(player),
			};
		}

		function applyMove(state, move) {
			if (state.winner) {
				return { ok: false, error: "game_over" };
			}

			const player = state.turn;
			if (move.type === "pawn") {
				const legal = legalPawnMoves(state, player).some((candidate) =>
					samePosition(candidate, move.to),
				);
				return legal
					? { ok: true, state: stateAfterPawnMove(state, player, move.to) }
					: { ok: false, error: "illegal_pawn_move" };
			}

			if (!wallInBounds(move.wall)) {
				return { ok: false, error: "wall_out_of_bounds" };
			}
			if (state.wallsRemaining[player] <= 0) {
				return { ok: false, error: "no_walls_remaining" };
			}
			if (wallConflicts(move.wall, state.walls)) {
				return { ok: false, error: "wall_conflict" };
			}

			const next = stateAfterWall(state, player, move.wall);
			const blocked = compileBlockedEdges(next.walls);
			if (
				shortestDistance(next, "p1", blocked) === -1 ||
				shortestDistance(next, "p2", blocked) === -1
			) {
				return { ok: false, error: "path_blocked" };
			}

			return { ok: true, state: next };
		}

		function createZobrist() {
			const mask = (1n << 64n) - 1n;
			let seed = 0x6a09e667f3bcc909n;
			function next() {
				seed = (seed + 0x9e3779b97f4a7c15n) & mask;
				let value = seed;
				value = ((value ^ (value >> 30n)) * 0xbf58476d1ce4e5b9n) & mask;
				value = ((value ^ (value >> 27n)) * 0x94d049bb133111ebn) & mask;
				return (value ^ (value >> 31n)) & mask;
			}

			return {
				pawns: PLAYER_NAMES.map(() =>
					Array.from({ length: PAWN_MOVE_COUNT }, next),
				),
				walls: Array.from({ length: WALL_CODE_COUNT }, next),
				remaining: PLAYER_NAMES.map(() =>
					Array.from({ length: MAX_WALLS + 1 }, next),
				),
				turn: PLAYER_NAMES.map(next),
				routeSalt: PLAYER_NAMES.map(next),
				pressureSalt: PLAYER_NAMES.map(next),
			};
		}

		function createWallEdgeMasks() {
			const masks = Array.from({ length: WALL_CODE_COUNT }, () => ({
				east: 0n,
				south: 0n,
			}));
			for (let slot = 0; slot < WALL_SLOTS * WALL_SLOTS; slot += 1) {
				const x = slot % WALL_SLOTS;
				const y = Math.floor(slot / WALL_SLOTS);
				const southIndex = y * BOARD_SIZE + x;
				masks[slot].south =
					(1n << BigInt(southIndex)) | (1n << BigInt(southIndex + 1));
				const eastIndex = y * BOARD_SIZE + x;
				masks[slot + WALL_SLOTS * WALL_SLOTS].east =
					(1n << BigInt(eastIndex)) | (1n << BigInt(eastIndex + BOARD_SIZE));
			}
			return masks;
		}

		function playerIndex(player) {
			return player === "p1" ? 0 : 1;
		}

		function encodedWall(wall) {
			const slot = wall.y * WALL_SLOTS + wall.x;
			return wall.o === "v" ? slot + WALL_SLOTS * WALL_SLOTS : slot;
		}

		function decodedWall(code) {
			const vertical = code >= WALL_SLOTS * WALL_SLOTS;
			const slot = vertical ? code - WALL_SLOTS * WALL_SLOTS : code;
			return {
				x: slot % WALL_SLOTS,
				y: Math.floor(slot / WALL_SLOTS),
				o: vertical ? "v" : "h",
			};
		}

		function decodedMove(code) {
			if (code < PAWN_MOVE_COUNT) {
				return {
					type: "pawn",
					to: { x: code % BOARD_SIZE, y: Math.floor(code / BOARD_SIZE) },
				};
			}
			return { type: "wall", wall: decodedWall(code - WALL_MOVE_OFFSET) };
		}

		function encodedMove(move) {
			return move.type === "pawn"
				? move.to.y * BOARD_SIZE + move.to.x
				: WALL_MOVE_OFFSET + encodedWall(move.wall);
		}

		function applyWallEdges(position, code, delta) {
			const vertical = code >= WALL_SLOTS * WALL_SLOTS;
			const slot = vertical ? code - WALL_SLOTS * WALL_SLOTS : code;
			const x = slot % WALL_SLOTS;
			const y = Math.floor(slot / WALL_SLOTS);
			const index = y * BOARD_SIZE + x;
			if (vertical) {
				position.east[index] += delta;
				position.east[index + BOARD_SIZE] += delta;
			} else {
				position.south[index] += delta;
				position.south[index + 1] += delta;
			}
		}

		function mutableHash(position) {
			let hash = position.geometryHash ^ zobrist.turn[position.turn];
			for (let player = 0; player < PLAYER_NAMES.length; player += 1) {
				hash ^= zobrist.remaining[player][position.wallsRemaining[player]];
			}
			return hash;
		}

		function mutableGeometryHash(position) {
			let hash = position.wallHash;
			for (let player = 0; player < PLAYER_NAMES.length; player += 1) {
				hash ^= zobrist.pawns[player][position.pawns[player]];
			}
			return hash;
		}

		function createMutablePosition(state) {
			const position = {
				pawns: Int16Array.of(
					state.pawns.p1.y * BOARD_SIZE + state.pawns.p1.x,
					state.pawns.p2.y * BOARD_SIZE + state.pawns.p2.x,
				),
				wallsRemaining: Int8Array.of(
					state.wallsRemaining.p1,
					state.wallsRemaining.p2,
				),
				turn: playerIndex(state.turn),
				winner: state.winner ? playerIndex(state.winner) : -1,
				wallList: [],
				wallUsed: new Uint8Array(WALL_CODE_COUNT),
				south: new Uint8Array(PAWN_MOVE_COUNT),
				east: new Uint8Array(PAWN_MOVE_COUNT),
				pathCache: [null, null],
				wallHash: 0n,
				geometryHash: 0n,
				hash: 0n,
			};
			for (const wall of state.walls) {
				const code = encodedWall(wall);
				position.wallList.push(code);
				position.wallUsed[code] = 1;
				position.wallHash ^= zobrist.walls[code];
				applyWallEdges(position, code, 1);
			}
			position.geometryHash = mutableGeometryHash(position);
			position.hash = mutableHash(position);
			return position;
		}

		function mutableEdgeBlocked(position, from, to) {
			const difference = to - from;
			if (difference === BOARD_SIZE) {
				return position.south[from] !== 0;
			}
			if (difference === -BOARD_SIZE) {
				return position.south[to] !== 0;
			}
			if (difference === 1) {
				return position.east[from] !== 0;
			}
			return position.east[to] !== 0;
		}

		function pathRecord(cells) {
			let southMask = 0n;
			let eastMask = 0n;
			for (let index = 0; index + 1 < cells.length; index += 1) {
				const from = cells[index];
				const to = cells[index + 1];
				const difference = to - from;
				if (difference === BOARD_SIZE) {
					southMask |= 1n << BigInt(from);
				} else if (difference === -BOARD_SIZE) {
					southMask |= 1n << BigInt(to);
				} else if (difference === 1) {
					eastMask |= 1n << BigInt(from);
				} else {
					eastMask |= 1n << BigInt(to);
				}
			}
			return {
				cells,
				distance: cells.length - 1,
				southMask,
				eastMask,
			};
		}

		function computeMutablePath(position, player, context) {
			if (context) {
				context.bfsCalls += 1;
			}
			const start = position.pawns[player];
			const targetRow = player === 0 ? BOARD_SIZE - 1 : 0;
			if (Math.floor(start / BOARD_SIZE) === targetRow) {
				return pathRecord([start]);
			}

			const previous = context?.pathPrevious ?? new Int16Array(PAWN_MOVE_COUNT);
			previous.fill(-1);
			const seen = context?.pathSeen ?? new Uint8Array(PAWN_MOVE_COUNT);
			seen.fill(0);
			const queue = context?.pathQueue ?? new Int16Array(PAWN_MOVE_COUNT);
			let head = 0;
			let tail = 0;
			let finish = -1;
			seen[start] = 1;
			queue[tail++] = start;
			while (head < tail && finish === -1) {
				const current = queue[head++];
				for (const edge of neighborEdges[current]) {
					if ((edge.south ? position.south : position.east)[edge.edge]) {
						continue;
					}
					const next = edge.to;
					if (seen[next]) {
						continue;
					}
					seen[next] = 1;
					previous[next] = current;
					if (Math.floor(next / BOARD_SIZE) === targetRow) {
						finish = next;
						break;
					}
					queue[tail++] = next;
				}
			}

			if (finish === -1) {
				return { cells: null, distance: -1, southMask: 0n, eastMask: 0n };
			}
			const cells = [];
			for (let cursor = finish; cursor !== -1; cursor = previous[cursor]) {
				cells.push(cursor);
			}
			cells.reverse();
			return pathRecord(cells);
		}

		function mutablePath(position, player, context) {
			if (position.pathCache[player]) {
				if (context) {
					context.pathCacheHits += 1;
				}
				return position.pathCache[player];
			}
			const path = computeMutablePath(position, player, context);
			position.pathCache[player] = path;
			return path;
		}

		function wallCutsCachedPath(code, path) {
			if (!path?.cells) {
				return true;
			}
			const masks = wallEdgeMasks[code];
			return (
				(masks.south & path.southMask) !== 0n ||
				(masks.east & path.eastMask) !== 0n
			);
		}

		function advancedPath(path, destination) {
			if (!path?.cells) {
				return null;
			}
			const offset = path.cells.indexOf(destination);
			return offset === -1 ? null : pathRecord(path.cells.slice(offset));
		}

		function wallsConflictByCode(left, right) {
			const leftWall = decodedWall(left);
			const rightWall = decodedWall(right);
			if (leftWall.x === rightWall.x && leftWall.y === rightWall.y) {
				return true;
			}
			if (
				leftWall.o === "h" &&
				rightWall.o === "h" &&
				leftWall.y === rightWall.y
			) {
				return Math.abs(leftWall.x - rightWall.x) === 1;
			}
			return (
				leftWall.o === "v" &&
				rightWall.o === "v" &&
				leftWall.x === rightWall.x &&
				Math.abs(leftWall.y - rightWall.y) === 1
			);
		}

		function mutableWallConflict(position, code) {
			if (position.wallUsed[code]) {
				return true;
			}
			const wall = decodedWall(code);
			const crossingCode =
				code >= WALL_SLOTS * WALL_SLOTS
					? code - WALL_SLOTS * WALL_SLOTS
					: code + WALL_SLOTS * WALL_SLOTS;
			if (position.wallUsed[crossingCode]) {
				return true;
			}
			if (wall.o === "h") {
				return (
					(wall.x > 0 && position.wallUsed[code - 1]) ||
					(wall.x + 1 < WALL_SLOTS && position.wallUsed[code + 1])
				);
			}
			return (
				(wall.y > 0 && position.wallUsed[code - WALL_SLOTS]) ||
				(wall.y + 1 < WALL_SLOTS && position.wallUsed[code + WALL_SLOTS])
			);
		}

		function placeProbeWall(position, code) {
			const undo = {
				geometryHash: position.geometryHash,
				pathCache: position.pathCache.slice(),
				wallHash: position.wallHash,
			};
			position.wallList.push(code);
			position.wallUsed[code] = 1;
			position.wallHash ^= zobrist.walls[code];
			position.geometryHash ^= zobrist.walls[code];
			applyWallEdges(position, code, 1);
			for (let player = 0; player < PLAYER_NAMES.length; player += 1) {
				if (wallCutsCachedPath(code, position.pathCache[player])) {
					position.pathCache[player] = null;
				}
			}
			return undo;
		}

		function undoProbeWall(position, code, undo) {
			applyWallEdges(position, code, -1);
			position.wallUsed[code] = 0;
			position.wallList.pop();
			position.pathCache = undo.pathCache;
			position.wallHash = undo.wallHash;
			position.geometryHash = undo.geometryHash;
		}

		function pawnMoveCodes(position, pawn, otherPawn) {
			const moves = [];
			const seen = new Uint8Array(PAWN_MOVE_COUNT);
			const x = pawn % BOARD_SIZE;
			const y = Math.floor(pawn / BOARD_SIZE);

			function addMove(code) {
				if (!seen[code]) {
					seen[code] = 1;
					moves.push(code);
				}
			}

			for (const [dx, dy] of directions) {
				const adjacentX = x + dx;
				const adjacentY = y + dy;
				if (
					adjacentX < 0 ||
					adjacentX >= BOARD_SIZE ||
					adjacentY < 0 ||
					adjacentY >= BOARD_SIZE
				) {
					continue;
				}
				const adjacent = adjacentY * BOARD_SIZE + adjacentX;
				if (mutableEdgeBlocked(position, pawn, adjacent)) {
					continue;
				}
				if (adjacent !== otherPawn) {
					addMove(adjacent);
					continue;
				}

				const beyondX = adjacentX + dx;
				const beyondY = adjacentY + dy;
				if (
					beyondX >= 0 &&
					beyondX < BOARD_SIZE &&
					beyondY >= 0 &&
					beyondY < BOARD_SIZE
				) {
					const beyond = beyondY * BOARD_SIZE + beyondX;
					if (!mutableEdgeBlocked(position, adjacent, beyond)) {
						addMove(beyond);
						continue;
					}
				}

				const sideDirections =
					dx === 0
						? [
								[1, 0],
								[-1, 0],
							]
						: [
								[0, 1],
								[0, -1],
							];
				for (const [sideX, sideY] of sideDirections) {
					const diagonalX = adjacentX + sideX;
					const diagonalY = adjacentY + sideY;
					if (
						diagonalX < 0 ||
						diagonalX >= BOARD_SIZE ||
						diagonalY < 0 ||
						diagonalY >= BOARD_SIZE
					) {
						continue;
					}
					const diagonal = diagonalY * BOARD_SIZE + diagonalX;
					if (!mutableEdgeBlocked(position, adjacent, diagonal)) {
						addMove(diagonal);
					}
				}
			}
			return moves;
		}

		function legalPawnMoveCodes(position, player) {
			return pawnMoveCodes(
				position,
				position.pawns[player],
				position.pawns[1 - player],
			);
		}

		function doEncodedMove(position, code) {
			const actor = position.turn;
			const undo = {
				actor,
				geometryHash: position.geometryHash,
				hash: position.hash,
				pathCache: position.pathCache.slice(),
				wallHash: position.wallHash,
				winner: position.winner,
			};
			if (code < PAWN_MOVE_COUNT) {
				undo.pawn = position.pawns[actor];
				position.hash ^=
					zobrist.pawns[actor][undo.pawn] ^ zobrist.pawns[actor][code];
				position.geometryHash ^=
					zobrist.pawns[actor][undo.pawn] ^ zobrist.pawns[actor][code];
				position.pawns[actor] = code;
				position.pathCache[actor] = advancedPath(
					position.pathCache[actor],
					code,
				);
				const destinationRow = Math.floor(code / BOARD_SIZE);
				if (destinationRow === (actor === 0 ? BOARD_SIZE - 1 : 0)) {
					position.winner = actor;
				}
			} else {
				const wallCode = code - WALL_MOVE_OFFSET;
				undo.wallCode = wallCode;
				undo.wallsRemaining = position.wallsRemaining[actor];
				position.hash ^= zobrist.walls[wallCode];
				position.geometryHash ^= zobrist.walls[wallCode];
				position.wallHash ^= zobrist.walls[wallCode];
				position.hash ^=
					zobrist.remaining[actor][undo.wallsRemaining] ^
					zobrist.remaining[actor][undo.wallsRemaining - 1];
				position.wallsRemaining[actor] -= 1;
				position.wallList.push(wallCode);
				position.wallUsed[wallCode] = 1;
				applyWallEdges(position, wallCode, 1);
				for (let player = 0; player < PLAYER_NAMES.length; player += 1) {
					if (wallCutsCachedPath(wallCode, position.pathCache[player])) {
						position.pathCache[player] = null;
					}
				}
			}
			position.hash ^= zobrist.turn[actor] ^ zobrist.turn[1 - actor];
			position.turn = 1 - actor;
			return undo;
		}

		function undoEncodedMove(position, code, undo) {
			position.turn = undo.actor;
			position.winner = undo.winner;
			if (code < PAWN_MOVE_COUNT) {
				position.pawns[undo.actor] = undo.pawn;
			} else {
				applyWallEdges(position, undo.wallCode, -1);
				position.wallUsed[undo.wallCode] = 0;
				position.wallList.pop();
				position.wallsRemaining[undo.actor] = undo.wallsRemaining;
			}
			position.pathCache = undo.pathCache;
			position.wallHash = undo.wallHash;
			position.geometryHash = undo.geometryHash;
			position.hash = undo.hash;
		}

		function wallCodesBlockingEdge(from, to) {
			const codes = [];
			const difference = to - from;
			if (Math.abs(difference) === BOARD_SIZE) {
				const top = Math.min(from, to);
				const x = top % BOARD_SIZE;
				const y = Math.floor(top / BOARD_SIZE);
				if (x < WALL_SLOTS && y < WALL_SLOTS) {
					codes.push(y * WALL_SLOTS + x);
				}
				if (x > 0 && y < WALL_SLOTS) {
					codes.push(y * WALL_SLOTS + x - 1);
				}
			} else {
				const left = Math.min(from, to);
				const x = left % BOARD_SIZE;
				const y = Math.floor(left / BOARD_SIZE);
				const verticalOffset = WALL_SLOTS * WALL_SLOTS;
				if (x < WALL_SLOTS && y < WALL_SLOTS) {
					codes.push(verticalOffset + y * WALL_SLOTS + x);
				}
				if (x < WALL_SLOTS && y > 0) {
					codes.push(verticalOffset + (y - 1) * WALL_SLOTS + x);
				}
			}
			return codes;
		}

		function collectPathWallCodes(target, path) {
			if (!path?.cells) {
				return;
			}
			for (let index = 0; index + 1 < path.cells.length; index += 1) {
				for (const code of wallCodesBlockingEdge(
					path.cells[index],
					path.cells[index + 1],
				)) {
					target.add(code);
				}
			}
		}

		function addWallCode(target, x, y, orientation) {
			if (x < 0 || x >= WALL_SLOTS || y < 0 || y >= WALL_SLOTS) {
				return;
			}
			const slot = y * WALL_SLOTS + x;
			target.add(orientation === "v" ? slot + WALL_SLOTS * WALL_SLOTS : slot);
		}

		function wallTouchesExisting(position, code) {
			const wall = decodedWall(code);
			return position.wallList.some((existingCode) => {
				const existing = decodedWall(existingCode);
				return (
					Math.abs(wall.x - existing.x) <= 2 &&
					Math.abs(wall.y - existing.y) <= 2
				);
			});
		}

		function focusedWallCodes(position, context) {
			const candidates = new Set();
			for (let player = 0; player < PLAYER_NAMES.length; player += 1) {
				collectPathWallCodes(
					candidates,
					mutablePath(position, player, context),
				);
				const pawn = position.pawns[player];
				const pawnX = pawn % BOARD_SIZE;
				const pawnY = Math.floor(pawn / BOARD_SIZE);
				for (let y = pawnY - 2; y <= pawnY + 1; y += 1) {
					for (let x = pawnX - 2; x <= pawnX + 1; x += 1) {
						addWallCode(candidates, x, y, "h");
						addWallCode(candidates, x, y, "v");
					}
				}
			}

			for (const existingCode of position.wallList) {
				const existing = decodedWall(existingCode);
				for (let dy = -2; dy <= 2; dy += 1) {
					for (let dx = -2; dx <= 2; dx += 1) {
						if (Math.abs(dx) + Math.abs(dy) > 3) {
							continue;
						}
						addWallCode(candidates, existing.x + dx, existing.y + dy, "h");
						addWallCode(candidates, existing.x + dx, existing.y + dy, "v");
					}
				}
			}
			return Array.from(candidates);
		}

		function allEncodedWalls() {
			return Array.from({ length: WALL_CODE_COUNT }, (_, code) => code);
		}

		function adaptiveWallLimit(
			position,
			rootNode,
			actorDistance,
			victimDistance,
		) {
			const actorPawn = position.pawns[position.turn];
			const victimPawn = position.pawns[1 - position.turn];
			const pawnSeparation =
				Math.abs((actorPawn % BOARD_SIZE) - (victimPawn % BOARD_SIZE)) +
				Math.abs(
					Math.floor(actorPawn / BOARD_SIZE) -
						Math.floor(victimPawn / BOARD_SIZE),
				);
			let tension = 0;
			if (Math.min(actorDistance, victimDistance) <= 5) {
				tension += 7;
			}
			if (Math.abs(actorDistance - victimDistance) <= 1) {
				tension += 3;
			}
			if (pawnSeparation <= 3) {
				tension += 5;
			}
			if (position.wallList.length >= 6) {
				tension += 5;
			}
			if (position.wallsRemaining[0] + position.wallsRemaining[1] <= 6) {
				tension += 3;
			}
			return rootNode ? Math.min(64, 36 + tension) : Math.min(28, 6 + tension);
		}

		function scoreWallMove(
			position,
			actorPath,
			victimPath,
			wallCode,
			nextActorDistance,
			nextVictimDistance,
			touchesExisting,
		) {
			const actorDistance = actorPath.distance;
			const victimDistance = victimPath.distance;
			const actorDelay = nextActorDistance - actorDistance;
			const victimDelay = nextVictimDistance - victimDistance;
			const netDelay = victimDelay - actorDelay;
			const earlyPosition =
				position.wallList.length <= 3 &&
				actorDistance >= 6 &&
				victimDistance >= 6;
			const openingPenalty =
				earlyPosition && victimDelay < 2 ? 520 + (2 - victimDelay) * 220 : 0;
			const wastePenalty =
				victimDelay === 0 && position.wallsRemaining[position.turn] > 3
					? 220
					: 0;
			const routeLead = victimDistance - actorDistance;
			const secureRace =
				position.wallsRemaining[1 - position.turn] === 0 && routeLead > 1;
			const racePenalty = secureRace ? 320 + routeLead * 35 : 0;
			const strongDelayBonus = victimDelay >= 2 ? (victimDelay - 1) * 170 : 0;
			const urgentDefense =
				victimDistance <= 3 && victimDelay > 0 && actorDelay <= victimDelay;
			const recoveringTempo =
				victimDelay === 1 &&
				actorDelay === 0 &&
				actorDistance > victimDistance &&
				position.wallList.length > 0 &&
				(victimDistance <= 5 || touchesExisting);
			const createsTempo =
				victimDelay >= 2 && victimDelay > actorDelay && actorDelay <= 1;
			return {
				actorDelay,
				priority:
					1_000 +
					victimDelay * 600 -
					actorDelay * 720 +
					strongDelayBonus +
					(wallCutsCachedPath(wallCode, victimPath) ? 80 : 0) +
					(touchesExisting ? 36 : 0) -
					openingPenalty -
					wastePenalty -
					racePenalty,
				tactical:
					victimDelay > 0 &&
					actorDelay <= victimDelay &&
					(victimDelay >= 2 || victimDistance <= 3),
				tempoRecovery: recoveringTempo,
				victimDelay,
				worthwhile: urgentDefense || recoveringTempo || createsTempo,
				netDelay,
			};
		}

		function pollDeadline(context) {
			context.operations += 1;
			if (
				(context.operations & 63) === 0 &&
				performance.now() >= context.deadline
			) {
				throw TIMEOUT;
			}
		}

		function distanceMap(position, sources, context) {
			if (context) {
				context.bfsCalls += 1;
			}
			const distances =
				context?.distanceBuffer ?? new Int16Array(PAWN_MOVE_COUNT);
			distances.fill(-1);
			const queue = context?.distanceQueue ?? new Int16Array(PAWN_MOVE_COUNT);
			let head = 0;
			let tail = 0;
			for (const source of sources) {
				if (distances[source] !== -1) {
					continue;
				}
				distances[source] = 0;
				queue[tail++] = source;
			}
			while (head < tail) {
				const current = queue[head++];
				for (const edge of neighborEdges[current]) {
					if ((edge.south ? position.south : position.east)[edge.edge]) {
						continue;
					}
					const next = edge.to;
					if (distances[next] !== -1) {
						continue;
					}
					distances[next] = distances[current] + 1;
					queue[tail++] = next;
				}
			}
			return distances;
		}

		function minimumLayerWallCover(position, edges) {
			if (edges.length === 0) {
				return 6;
			}
			const wallMasks = new Map();
			const options = [];
			for (let index = 0; index < edges.length; index += 1) {
				const edgeOptions = wallCodesBlockingEdge(
					edges[index].from,
					edges[index].to,
				).filter((code) => !mutableWallConflict(position, code));
				if (edgeOptions.length === 0) {
					return 6;
				}
				options.push(edgeOptions);
				for (const code of edgeOptions) {
					wallMasks.set(
						code,
						(wallMasks.get(code) || 0n) | (1n << BigInt(index)),
					);
				}
			}

			const target = (1n << BigInt(edges.length)) - 1n;
			let best = 6;
			function visit(mask, selected) {
				if (mask === target) {
					best = Math.min(best, selected.length);
					return;
				}
				if (selected.length + 1 >= best) {
					return;
				}
				let chosenIndex = -1;
				let chosenOptions = null;
				for (let index = 0; index < edges.length; index += 1) {
					if ((mask & (1n << BigInt(index))) !== 0n) {
						continue;
					}
					if (!chosenOptions || options[index].length < chosenOptions.length) {
						chosenIndex = index;
						chosenOptions = options[index];
					}
				}
				if (chosenIndex === -1) {
					return;
				}
				for (const code of chosenOptions) {
					if (
						selected.some((existing) => wallsConflictByCode(existing, code))
					) {
						continue;
					}
					visit(mask | wallMasks.get(code), selected.concat(code));
				}
			}
			visit(0n, []);
			return best;
		}

		function routeAnalysis(position, player, context) {
			const key =
				position.wallHash ^
				zobrist.pawns[player][position.pawns[player]] ^
				zobrist.routeSalt[player];
			const cached = routeAnalysisCache.get(key);
			if (cached) {
				cached.generation = analysisGeneration;
				if (context) {
					context.routeCacheHits += 1;
				}
				return cached;
			}

			const start = position.pawns[player];
			const targetRow = player === 0 ? BOARD_SIZE - 1 : 0;
			const goals = Array.from(
				{ length: BOARD_SIZE },
				(_, x) => targetRow * BOARD_SIZE + x,
			);
			const fromStart = distanceMap(position, [start], context);
			const baseDistance = mutablePath(position, player, context).distance;
			if (baseDistance < 0) {
				const blockedResult = {
					distance: -1,
					generation: analysisGeneration,
					robustness: 0,
					singleCutCodes: [],
				};
				routeAnalysisCache.set(key, blockedResult);
				return blockedResult;
			}

			const layers = Array.from({ length: baseDistance }, () => []);
			for (let from = 0; from < PAWN_MOVE_COUNT; from += 1) {
				const layer = fromStart[from];
				if (layer < 0 || layer >= baseDistance) {
					continue;
				}
				for (const edge of neighborEdges[from]) {
					if ((edge.south ? position.south : position.east)[edge.edge]) {
						continue;
					}
					const to = edge.to;
					if (fromStart[to] === layer + 1) {
						layers[layer].push({ from, to });
					}
				}
			}

			const waysFrom = new Float64Array(PAWN_MOVE_COUNT);
			const waysTo = new Float64Array(PAWN_MOVE_COUNT);
			waysFrom[start] = 1;
			for (let layer = 0; layer < baseDistance; layer += 1) {
				for (const edge of layers[layer]) {
					waysFrom[edge.to] = Math.min(
						PATH_COUNT_CAP,
						waysFrom[edge.to] + waysFrom[edge.from],
					);
				}
			}
			for (const goal of goals) {
				if (fromStart[goal] === baseDistance) {
					waysTo[goal] = 1;
				}
			}
			for (let layer = baseDistance - 1; layer >= 0; layer -= 1) {
				for (const edge of layers[layer]) {
					waysTo[edge.from] = Math.min(
						PATH_COUNT_CAP,
						waysTo[edge.from] + waysTo[edge.to],
					);
				}
			}
			for (let layer = 0; layer < layers.length; layer += 1) {
				layers[layer] = layers[layer].filter((edge) => waysTo[edge.to] > 0);
			}

			const totalPaths = waysTo[start];
			const wallCoverage = new Map();
			for (const layer of layers) {
				for (const edge of layer) {
					const throughEdge = Math.min(
						PATH_COUNT_CAP,
						waysFrom[edge.from] * waysTo[edge.to],
					);
					for (const code of wallCodesBlockingEdge(edge.from, edge.to)) {
						if (mutableWallConflict(position, code)) {
							continue;
						}
						wallCoverage.set(
							code,
							Math.min(
								PATH_COUNT_CAP,
								(wallCoverage.get(code) || 0) + throughEdge,
							),
						);
					}
				}
			}

			let robustness = 6;
			for (const layer of layers) {
				robustness = Math.min(
					robustness,
					minimumLayerWallCover(position, layer),
				);
			}
			const result = {
				distance: baseDistance,
				generation: analysisGeneration,
				robustness,
				singleCutCodes: Array.from(wallCoverage)
					.filter(([, coverage]) => coverage >= totalPaths)
					.map(([code]) => code),
			};
			routeAnalysisCache.set(key, result);
			return result;
		}

		function wallPressure(position, attacker, victim, victimRoute, context) {
			if (position.wallsRemaining[attacker] <= 0) {
				return 0;
			}
			const key = position.hash ^ zobrist.pressureSalt[attacker];
			const cached = wallPressureCache.get(key);
			if (cached) {
				cached.generation = analysisGeneration;
				return cached.threat;
			}

			const attackerPath = mutablePath(position, attacker, context);
			const victimPawn = position.pawns[victim];
			const victimX = victimPawn % BOARD_SIZE;
			const victimY = Math.floor(victimPawn / BOARD_SIZE);
			const candidates = victimRoute.singleCutCodes
				.slice()
				.sort((left, right) => {
					function priority(code) {
						const wall = decodedWall(code);
						const proximity =
							Math.abs(wall.x - victimX) + Math.abs(wall.y - victimY);
						return (wallTouchesExisting(position, code) ? 20 : 0) - proximity;
					}
					return priority(right) - priority(left);
				});

			let threat = 0;
			let strongest = null;
			for (const code of candidates) {
				if (mutableWallConflict(position, code)) {
					continue;
				}
				strongest ??= code;
				const selfCutPenalty = wallCutsCachedPath(code, attackerPath) ? 3 : 0;
				const trapBonus = wallTouchesExisting(position, code) ? 3 : 0;
				threat = Math.max(threat, 4 + trapBonus - selfCutPenalty);
			}
			if (
				strongest !== null &&
				(wallTouchesExisting(position, strongest) || victimRoute.distance <= 5)
			) {
				const attackerDistance = attackerPath.distance;
				const undo = placeProbeWall(position, strongest);
				try {
					const nextVictimDistance = mutablePath(
						position,
						victim,
						context,
					).distance;
					const nextAttackerDistance = mutablePath(
						position,
						attacker,
						context,
					).distance;
					if (nextVictimDistance >= 0 && nextAttackerDistance >= 0) {
						threat = Math.max(
							threat,
							(nextVictimDistance - victimRoute.distance) * 4 -
								Math.max(0, nextAttackerDistance - attackerDistance) * 3,
						);
					}
				} finally {
					undoProbeWall(position, strongest, undo);
				}
			}
			wallPressureCache.set(key, {
				generation: analysisGeneration,
				threat: Math.max(0, threat),
			});
			return Math.max(0, threat);
		}

		function terminalScore(position, ply) {
			return position.winner === position.turn
				? WIN_SCORE - ply
				: -WIN_SCORE + ply;
		}

		function distanceRaceScore(position, context, ply) {
			const actor = position.turn;
			const actorDistance = mutablePath(position, actor, context).distance;
			const victimDistance = mutablePath(position, 1 - actor, context).distance;
			const actorArrival = actorDistance * 2 - 1;
			const victimArrival = victimDistance * 2;
			return actorArrival < victimArrival
				? WIN_SCORE - ply - actorDistance
				: -WIN_SCORE + ply + victimDistance;
		}

		function raceStateCode(p1Pawn, p2Pawn, turn) {
			return (p1Pawn * PAWN_MOVE_COUNT + p2Pawn) * 2 + turn;
		}

		function raceWallKey(position) {
			return position.wallList
				.slice()
				.sort((left, right) => left - right)
				.join(",");
		}

		function createPawnRaceTable(position) {
			const outcomes = new Int8Array(RACE_STATE_COUNT);
			const distances = new Int16Array(RACE_STATE_COUNT);
			const resolvedSuccessors = new Uint8Array(RACE_STATE_COUNT);
			const successors = new Array(RACE_STATE_COUNT);
			const predecessors = new Array(RACE_STATE_COUNT);
			const buckets = [null, []];

			for (let p1Pawn = 0; p1Pawn < PAWN_MOVE_COUNT; p1Pawn += 1) {
				if (Math.floor(p1Pawn / BOARD_SIZE) === BOARD_SIZE - 1) {
					continue;
				}
				for (let p2Pawn = 0; p2Pawn < PAWN_MOVE_COUNT; p2Pawn += 1) {
					if (p1Pawn === p2Pawn || Math.floor(p2Pawn / BOARD_SIZE) === 0) {
						continue;
					}
					for (let turn = 0; turn < PLAYER_NAMES.length; turn += 1) {
						const code = raceStateCode(p1Pawn, p2Pawn, turn);
						const pawns = [p1Pawn, p2Pawn];
						const nextStates = [];
						let immediateWin = false;
						for (const move of pawnMoveCodes(
							position,
							pawns[turn],
							pawns[1 - turn],
						)) {
							const destinationRow = Math.floor(move / BOARD_SIZE);
							if (destinationRow === (turn === 0 ? BOARD_SIZE - 1 : 0)) {
								immediateWin = true;
								continue;
							}
							const nextP1 = turn === 0 ? move : p1Pawn;
							const nextP2 = turn === 1 ? move : p2Pawn;
							nextStates.push(raceStateCode(nextP1, nextP2, 1 - turn));
						}
						successors[code] = nextStates;
						for (const nextCode of nextStates) {
							if (!predecessors[nextCode]) {
								predecessors[nextCode] = [];
							}
							predecessors[nextCode].push(code);
						}
						if (immediateWin) {
							outcomes[code] = 1;
							distances[code] = 1;
							buckets[1].push(code);
						}
					}
				}
			}

			for (let distance = 1; distance < buckets.length; distance += 1) {
				for (const code of buckets[distance] ?? []) {
					for (const previousCode of predecessors[code] ?? []) {
						if (outcomes[previousCode] !== 0) {
							continue;
						}
						if (outcomes[code] === -1) {
							outcomes[previousCode] = 1;
							distances[previousCode] = distance + 1;
							if (!buckets[distance + 1]) {
								buckets[distance + 1] = [];
							}
							buckets[distance + 1].push(previousCode);
							continue;
						}
						resolvedSuccessors[previousCode] += 1;
						if (
							resolvedSuccessors[previousCode] ===
							successors[previousCode].length
						) {
							outcomes[previousCode] = -1;
							distances[previousCode] = distance + 1;
							if (!buckets[distance + 1]) {
								buckets[distance + 1] = [];
							}
							buckets[distance + 1].push(previousCode);
						}
					}
				}
			}

			return { distances, outcomes };
		}

		function pawnRaceTable(position) {
			const key = raceWallKey(position);
			let table = pawnRaceTableCache.get(key);
			if (table) {
				return table;
			}
			table = createPawnRaceTable(position);
			if (pawnRaceTableCache.size >= RACE_TABLE_MAX_ENTRIES) {
				pawnRaceTableCache.delete(pawnRaceTableCache.keys().next().value);
			}
			pawnRaceTableCache.set(key, table);
			return table;
		}

		function exactRaceScore(position, context, ply) {
			const table = pawnRaceTable(position);
			const code = raceStateCode(
				position.pawns[0],
				position.pawns[1],
				position.turn,
			);
			const outcome = table.outcomes[code];
			const distance = table.distances[code];
			if (outcome === 1) {
				return WIN_SCORE - ply - distance;
			}
			if (outcome === -1) {
				return -WIN_SCORE + ply + distance;
			}
			return distanceRaceScore(position, context, ply);
		}

		function heldWallValue(position) {
			return 24 + Math.min(32, position.wallList.length * 2);
		}

		function wallReserveBonus(position, player) {
			const remaining = position.wallsRemaining[player];
			if (remaining <= 0) {
				return 0;
			}
			const opponentRemaining = position.wallsRemaining[1 - player];
			const scarcity = remaining <= 2 ? (3 - remaining) * 18 : 0;
			const leverageGap = Math.max(0, opponentRemaining - remaining) * 5;
			return scarcity + leverageGap;
		}

		function wallResourceScore(position, player) {
			const other = 1 - player;
			return (
				(position.wallsRemaining[player] - position.wallsRemaining[other]) *
					heldWallValue(position) +
				wallReserveBonus(position, player) -
				wallReserveBonus(position, other)
			);
		}

		function quickEvaluation(position, perspective, context) {
			const other = 1 - perspective;
			const ownDistance = mutablePath(position, perspective, context).distance;
			const otherDistance = mutablePath(position, other, context).distance;
			const ownArrival =
				ownDistance * 2 - (position.turn === perspective ? 1 : 0);
			const otherArrival =
				otherDistance * 2 - (position.turn === other ? 1 : 0);
			return (
				(otherArrival - ownArrival) * 60 +
				wallResourceScore(position, perspective)
			);
		}

		function evaluateMutable(position, context) {
			const actor = position.turn;
			const victim = 1 - actor;
			const actorDistance = mutablePath(position, actor, context).distance;
			const victimDistance = mutablePath(position, victim, context).distance;
			if (actorDistance < 0) {
				return -WIN_SCORE / 2;
			}
			if (victimDistance < 0) {
				return WIN_SCORE / 2;
			}

			const actorArrival = actorDistance * 2 - 1;
			const victimArrival = victimDistance * 2;
			const raceMargin = victimArrival - actorArrival;
			const actorY = Math.floor(position.pawns[actor] / BOARD_SIZE);
			const victimY = Math.floor(position.pawns[victim] / BOARD_SIZE);
			const actorProgress = actor === 0 ? actorY : BOARD_SIZE - 1 - actorY;
			const victimProgress = victim === 0 ? victimY : BOARD_SIZE - 1 - victimY;
			const actorMoves = legalPawnMoveCodes(position, actor);
			const victimMoves = legalPawnMoveCodes(position, victim);
			const mobility = actorMoves.length - victimMoves.length;
			function hasForwardJump(player, moves) {
				const pawnY = Math.floor(position.pawns[player] / BOARD_SIZE);
				return moves.some((code) => {
					const destinationY = Math.floor(code / BOARD_SIZE);
					return player === 0
						? destinationY - pawnY >= 2
						: pawnY - destinationY >= 2;
				});
			}
			const jumpBalance =
				Number(hasForwardJump(actor, actorMoves)) -
				Number(hasForwardJump(victim, victimMoves));
			let score =
				raceMargin * 60 +
				wallResourceScore(position, actor) +
				(actorProgress - victimProgress) * 2 +
				mobility +
				jumpBalance * 28;
			const routeLead = victimDistance - actorDistance;
			const secureLead =
				position.wallsRemaining[victim] === 0 ? routeLead - 1 : 0;
			if (secureLead > 0) {
				score += 45 * secureLead;
			}
			const pawnSeparation =
				Math.abs(
					(position.pawns[actor] % BOARD_SIZE) -
						(position.pawns[victim] % BOARD_SIZE),
				) + Math.abs(actorY - victimY);
			const needsStrategicDetail =
				position.wallList.length >= 4 ||
				Math.min(actorDistance, victimDistance) <= 5 ||
				pawnSeparation <= 3 ||
				position.wallsRemaining[0] + position.wallsRemaining[1] <= 12;
			if (!needsStrategicDetail) {
				context.lazyEvaluations += 1;
				return score;
			}
			const actorRoute = routeAnalysis(position, actor, context);
			const victimRoute = routeAnalysis(position, victim, context);
			const actorRobustness =
				position.wallsRemaining[victim] >= actorRoute.robustness
					? actorRoute.robustness
					: 6;
			const victimRobustness =
				position.wallsRemaining[actor] >= victimRoute.robustness
					? victimRoute.robustness
					: 6;
			const actorThreat = wallPressure(
				position,
				actor,
				victim,
				victimRoute,
				context,
			);
			const danger = wallPressure(position, victim, actor, actorRoute, context);

			score +=
				(actorRobustness - victimRobustness) * 9 + (actorThreat - danger) * 7;
			if (
				position.wallsRemaining[victim] === 0 &&
				actorArrival < victimArrival
			) {
				score += 90;
			}
			if (
				position.wallsRemaining[actor] === 0 &&
				actorArrival > victimArrival
			) {
				score -= 90;
			}
			return score;
		}

		function checkTime(context) {
			context.nodes += 1;
			pollDeadline(context);
		}

		function moveCacheMode(rootNode, wallCapOverride, searchDepth) {
			if (rootNode) {
				return `root:${wallCapOverride ?? "auto"}`;
			}
			if (wallCapOverride !== undefined) {
				return `cap:${wallCapOverride}`;
			}
			return `depth:${Math.max(0, Math.min(4, searchDepth ?? 0))}`;
		}

		function cachedMoves(position, context, mode) {
			const entry = moveGenerationCache.get(position.hash);
			const moves = entry?.variants.get(mode);
			if (!moves) {
				return null;
			}
			entry.generation = analysisGeneration;
			context.moveCacheHits += 1;
			return moves;
		}

		function cacheMoves(position, mode, moves) {
			let entry = moveGenerationCache.get(position.hash);
			if (!entry) {
				entry = { generation: analysisGeneration, variants: new Map() };
				moveGenerationCache.set(position.hash, entry);
			}
			entry.generation = analysisGeneration;
			entry.variants.set(mode, moves);
			return moves;
		}

		function depthAdjustedWallLimit(baseLimit, searchDepth) {
			if (searchDepth === undefined || searchDepth === null) {
				return baseLimit;
			}
			if (searchDepth <= 1) {
				return Math.min(baseLimit, 10);
			}
			if (searchDepth === 2) {
				return Math.min(baseLimit, 16);
			}
			if (searchDepth >= 4) {
				return Math.min(32, baseLimit + 4);
			}
			return baseLimit;
		}

		function generateMoves(
			position,
			context,
			rootNode,
			wallCapOverride,
			searchDepth,
		) {
			const cacheMode = moveCacheMode(rootNode, wallCapOverride, searchDepth);
			const reused = cachedMoves(position, context, cacheMode);
			if (reused) {
				return reused;
			}

			const actor = position.turn;
			const victim = 1 - actor;
			const moves = [];
			const actorPath = mutablePath(position, actor, context);
			const victimPath = mutablePath(position, victim, context);
			const actorDistance = actorPath.distance;
			const victimDistance = victimPath.distance;
			const actorY = Math.floor(position.pawns[actor] / BOARD_SIZE);

			for (const code of legalPawnMoveCodes(position, actor)) {
				const undo = doEncodedMove(position, code);
				let nextDistance;
				try {
					nextDistance = mutablePath(position, actor, context).distance;
				} finally {
					undoEncodedMove(position, code, undo);
				}
				const destinationY = Math.floor(code / BOARD_SIZE);
				const destinationX = code % BOARD_SIZE;
				const actorX = position.pawns[actor] % BOARD_SIZE;
				const progress =
					actor === 0 ? destinationY - actorY : actorY - destinationY;
				moves.push({
					code,
					priority: 1_200 - nextDistance * 40 + progress * 80,
					tactical:
						Math.abs(destinationX - actorX) + Math.abs(destinationY - actorY) >
						1,
				});
			}

			if (position.wallsRemaining[actor] <= 0) {
				moves.sort(
					(left, right) =>
						right.priority - left.priority || left.code - right.code,
				);
				return cacheMoves(position, cacheMode, moves);
			}

			const rawCandidates = rootNode
				? allEncodedWalls()
				: focusedWallCodes(position, context);
			const baseWallLimit =
				wallCapOverride ??
				adaptiveWallLimit(position, rootNode, actorDistance, victimDistance);
			const wallLimit =
				rootNode || wallCapOverride !== undefined
					? baseWallLimit
					: depthAdjustedWallLimit(baseWallLimit, searchDepth);
			const candidatePool = [];
			for (const wallCode of rawCandidates) {
				if (mutableWallConflict(position, wallCode)) {
					continue;
				}
				const cutsVictimPath = wallCutsCachedPath(wallCode, victimPath);
				const cutsActorPath = wallCutsCachedPath(wallCode, actorPath);
				const touchesExisting = wallTouchesExisting(position, wallCode);
				candidatePool.push({
					preScore:
						(cutsVictimPath ? 320 : 0) -
						(cutsActorPath ? 260 : 0) +
						(touchesExisting ? 70 : 0),
					touchesExisting,
					wallCode,
				});
			}
			candidatePool.sort(
				(left, right) =>
					right.preScore - left.preScore || left.wallCode - right.wallCode,
			);
			const wallsToProbe = rootNode
				? candidatePool
				: candidatePool.slice(0, wallLimit + 6);
			const wallMoves = [];

			for (const candidate of wallsToProbe) {
				pollDeadline(context);
				const { touchesExisting, wallCode } = candidate;
				const undo = placeProbeWall(position, wallCode);
				let nextActorDistance;
				let nextVictimDistance;
				try {
					nextActorDistance = mutablePath(position, actor, context).distance;
					nextVictimDistance = mutablePath(position, victim, context).distance;
				} finally {
					undoProbeWall(position, wallCode, undo);
				}
				if (nextActorDistance === -1 || nextVictimDistance === -1) {
					continue;
				}

				const scoredWall = scoreWallMove(
					position,
					actorPath,
					victimPath,
					wallCode,
					nextActorDistance,
					nextVictimDistance,
					touchesExisting,
				);
				if (rootNode && !scoredWall.worthwhile) {
					continue;
				}
				wallMoves.push({
					code: WALL_MOVE_OFFSET + wallCode,
					priority: scoredWall.priority,
					tactical: scoredWall.tactical,
					tempoRecovery: scoredWall.tempoRecovery,
				});
			}

			wallMoves.sort(
				(left, right) =>
					right.priority - left.priority || left.code - right.code,
			);
			const selectedWalls = wallMoves.slice(0, wallLimit);
			const selectedCodes = new Set(selectedWalls.map((move) => move.code));
			for (const wallMove of wallMoves) {
				if (
					!wallMove.tactical ||
					selectedCodes.has(wallMove.code) ||
					selectedWalls.length >= wallLimit + 6
				) {
					continue;
				}
				selectedWalls.push(wallMove);
				selectedCodes.add(wallMove.code);
			}
			moves.push(...selectedWalls);
			moves.sort(
				(left, right) =>
					right.priority - left.priority || left.code - right.code,
			);
			return cacheMoves(position, cacheMode, moves);
		}

		function scoreToTable(score, ply) {
			if (score > MATE_THRESHOLD) {
				return score + ply;
			}
			if (score < -MATE_THRESHOLD) {
				return score - ply;
			}
			return score;
		}

		function scoreFromTable(score, ply) {
			if (score > MATE_THRESHOLD) {
				return score - ply;
			}
			if (score < -MATE_THRESHOLD) {
				return score + ply;
			}
			return score;
		}

		function moveOrderScore(
			candidate,
			position,
			context,
			ply,
			tableMove,
			preferredMove,
			mctsStats,
		) {
			let score = candidate.priority;
			if (candidate.code === tableMove) {
				score += 2_000_000_000;
			}
			if (candidate.code === preferredMove) {
				score += 1_500_000_000;
			}
			const killers = context.killers[ply];
			if (candidate.code === killers?.[0]) {
				score += 1_000_000;
			} else if (candidate.code === killers?.[1]) {
				score += 750_000;
			}
			score += context.historyScores[position.turn][candidate.code];
			if (context.stockfishFeatures && ply > 0) {
				const previousMove = context.moveStack[ply - 1];
				if (previousMove >= 0) {
					score +=
						context.continuationHistory[
							previousMove * MOVE_CODE_COUNT + candidate.code
						];
					if (context.counterMoves[previousMove] === candidate.code) {
						score += 600_000;
					}
				}
			}
			const monteCarlo = mctsStats?.get(candidate.code);
			if (monteCarlo) {
				const mean = monteCarlo.value / Math.max(1, monteCarlo.visits);
				score += Math.log2(monteCarlo.visits + 1) * 20_000 + mean * 8_000;
			}
			return score;
		}

		function orderMoves(
			candidates,
			position,
			context,
			ply,
			tableMove,
			preferredMove,
			mctsStats,
		) {
			return candidates
				.slice()
				.sort(
					(left, right) =>
						moveOrderScore(
							right,
							position,
							context,
							ply,
							tableMove,
							preferredMove,
							mctsStats,
						) -
						moveOrderScore(
							left,
							position,
							context,
							ply,
							tableMove,
							preferredMove,
							mctsStats,
						),
				);
		}

		function updateHistoryScore(table, index, bonus) {
			const limit = 1_000_000;
			const current = table[index];
			const adjustment =
				bonus - Math.trunc((current * Math.abs(bonus)) / limit);
			table[index] = Math.max(-limit, Math.min(limit, current + adjustment));
		}

		function recordCutoff(
			context,
			player,
			candidate,
			depth,
			ply,
			searchedMoves,
		) {
			const code = candidate.code;
			if (!context.killers[ply]) {
				context.killers[ply] = [];
			}
			const killers = context.killers[ply];
			if (killers[0] !== code) {
				killers[1] = killers[0];
				killers[0] = code;
			}
			if (!context.stockfishFeatures) {
				context.historyScores[player][code] = Math.min(
					1_000_000,
					context.historyScores[player][code] + depth * depth * 32,
				);
				return;
			}

			const bonus = Math.min(96_000, depth * depth * 1_500);
			updateHistoryScore(context.historyScores[player], code, bonus);
			const previousMove = ply > 0 ? context.moveStack[ply - 1] : -1;
			if (previousMove >= 0) {
				const continuationIndex = previousMove * MOVE_CODE_COUNT + code;
				updateHistoryScore(
					context.continuationHistory,
					continuationIndex,
					bonus,
				);
				context.counterMoves[previousMove] = code;
			}

			const malus = -Math.min(24_000, depth * depth * 300);
			for (const searched of searchedMoves) {
				if (searched.tactical) {
					continue;
				}
				updateHistoryScore(context.historyScores[player], searched.code, malus);
				if (previousMove >= 0) {
					updateHistoryScore(
						context.continuationHistory,
						previousMove * MOVE_CODE_COUNT + searched.code,
						malus,
					);
				}
			}
		}

		function storeTable(
			position,
			depth,
			score,
			flag,
			bestMove,
			ply,
			staticEval,
		) {
			const existing = transpositionTable.get(position.hash);
			if (existing) {
				const existingIsStronger =
					existing.depth > depth ||
					(existing.depth === depth &&
						existing.flag === "exact" &&
						flag !== "exact");
				if (existingIsStronger) {
					existing.generation = analysisGeneration;
					if (existing.staticEval == null && Number.isFinite(staticEval)) {
						existing.staticEval = staticEval;
					}
					return;
				}
			}
			transpositionTable.set(position.hash, {
				bestMove,
				depth,
				flag,
				generation: analysisGeneration,
				score: scoreToTable(score, ply),
				staticEval: Number.isFinite(staticEval)
					? staticEval
					: (existing?.staticEval ?? null),
			});
		}

		function lateMoveReduction(
			depth,
			moveIndex,
			candidate,
			rootNode,
			context,
			ply,
		) {
			if (
				depth < 4 ||
				candidate.code < WALL_MOVE_OFFSET ||
				candidate.tactical ||
				candidate.priority >= 1_400 ||
				moveIndex < (rootNode ? 10 : 6)
			) {
				return 0;
			}
			let reduction = depth >= 7 && moveIndex >= (rootNode ? 24 : 14) ? 2 : 1;
			if (context.stockfishFeatures) {
				let history =
					context.historyScores[positionPlayer(context, ply)][candidate.code];
				const previousMove = ply > 0 ? context.moveStack[ply - 1] : -1;
				if (previousMove >= 0) {
					history +=
						context.continuationHistory[
							previousMove * MOVE_CODE_COUNT + candidate.code
						];
				}
				if (history > 180_000) {
					reduction -= 1;
				} else if (history < -180_000 && depth >= 6) {
					reduction += 1;
				}
			}
			return Math.max(0, Math.min(depth - 2, reduction));
		}

		function positionPlayer(context, ply) {
			return context.rootPlayer ^ (ply & 1);
		}

		function candidateFromCode(position, context, code) {
			if (!Number.isInteger(code) || code < 0 || code >= MOVE_CODE_COUNT) {
				return null;
			}
			const actor = position.turn;
			const victim = 1 - actor;
			if (code < PAWN_MOVE_COUNT) {
				if (!legalPawnMoveCodes(position, actor).includes(code)) {
					return null;
				}
				const actorPawn = position.pawns[actor];
				const actorX = actorPawn % BOARD_SIZE;
				const actorY = Math.floor(actorPawn / BOARD_SIZE);
				const undo = doEncodedMove(position, code);
				let nextDistance;
				try {
					nextDistance = mutablePath(position, actor, context).distance;
				} finally {
					undoEncodedMove(position, code, undo);
				}
				const destinationX = code % BOARD_SIZE;
				const destinationY = Math.floor(code / BOARD_SIZE);
				const progress =
					actor === 0 ? destinationY - actorY : actorY - destinationY;
				return {
					code,
					priority: 1_200 - nextDistance * 40 + progress * 80,
					tactical:
						Math.abs(destinationX - actorX) + Math.abs(destinationY - actorY) >
						1,
				};
			}

			if (position.wallsRemaining[actor] <= 0) {
				return null;
			}
			const wallCode = code - WALL_MOVE_OFFSET;
			if (mutableWallConflict(position, wallCode)) {
				return null;
			}
			pollDeadline(context);
			const actorPath = mutablePath(position, actor, context);
			const victimPath = mutablePath(position, victim, context);
			const touchesExisting = wallTouchesExisting(position, wallCode);
			const undo = placeProbeWall(position, wallCode);
			let nextActorDistance;
			let nextVictimDistance;
			try {
				nextActorDistance = mutablePath(position, actor, context).distance;
				nextVictimDistance = mutablePath(position, victim, context).distance;
			} finally {
				undoProbeWall(position, wallCode, undo);
			}
			if (nextActorDistance < 0 || nextVictimDistance < 0) {
				return null;
			}
			const scoredWall = scoreWallMove(
				position,
				actorPath,
				victimPath,
				wallCode,
				nextActorDistance,
				nextVictimDistance,
				touchesExisting,
			);
			return {
				code,
				priority: scoredWall.priority,
				tactical: scoredWall.tactical,
				tempoRecovery: scoredWall.tempoRecovery,
			};
		}

		function tacticallyVolatile(position) {
			const p1 = position.pawns[0];
			const p2 = position.pawns[1];
			const p1Rows = BOARD_SIZE - 1 - Math.floor(p1 / BOARD_SIZE);
			const p2Rows = Math.floor(p2 / BOARD_SIZE);
			const separation =
				Math.abs((p1 % BOARD_SIZE) - (p2 % BOARD_SIZE)) +
				Math.abs(Math.floor(p1 / BOARD_SIZE) - Math.floor(p2 / BOARD_SIZE));
			return Math.min(p1Rows, p2Rows) <= 3 || separation <= 3;
		}

		function needsQuiescence(position) {
			const p1 = position.pawns[0];
			const p2 = position.pawns[1];
			const p1Rows = BOARD_SIZE - 1 - Math.floor(p1 / BOARD_SIZE);
			const p2Rows = Math.floor(p2 / BOARD_SIZE);
			return Math.min(p1Rows, p2Rows) <= 1;
		}

		function hasImmediateWinningPawnMove(position, player) {
			const winningRow = player === 0 ? BOARD_SIZE - 1 : 0;
			return legalPawnMoveCodes(position, player).some(
				(code) => Math.floor(code / BOARD_SIZE) === winningRow,
			);
		}

		function forcingMoves(position, context, ply) {
			const actor = position.turn;
			const victim = 1 - actor;
			const candidates = [];
			const underThreat = hasImmediateWinningPawnMove(position, victim);
			const winningRow = actor === 0 ? BOARD_SIZE - 1 : 0;
			for (const code of legalPawnMoveCodes(position, actor)) {
				const wins = Math.floor(code / BOARD_SIZE) === winningRow;
				let resolvesThreat = false;
				if (!wins && underThreat) {
					const undo = doEncodedMove(position, code);
					try {
						resolvesThreat = !hasImmediateWinningPawnMove(position, victim);
					} finally {
						undoEncodedMove(position, code, undo);
					}
				}
				if (!wins && !resolvesThreat) {
					continue;
				}
				const candidate = candidateFromCode(position, context, code);
				if (candidate) {
					candidates.push(candidate);
				}
			}

			if (position.wallsRemaining[actor] > 0 && underThreat) {
				const victimPath = mutablePath(position, victim, context);
				const wallCodes = new Set();
				collectPathWallCodes(wallCodes, victimPath);
				for (const wallCode of wallCodes) {
					const candidate = candidateFromCode(
						position,
						context,
						WALL_MOVE_OFFSET + wallCode,
					);
					if (candidate?.tactical) {
						const undo = doEncodedMove(position, candidate.code);
						let resolvesThreat;
						try {
							resolvesThreat = !hasImmediateWinningPawnMove(position, victim);
						} finally {
							undoEncodedMove(position, candidate.code, undo);
						}
						if (resolvesThreat) {
							candidates.push(candidate);
						}
					}
				}
			}

			return orderMoves(
				candidates,
				position,
				context,
				ply,
				transpositionTable.get(position.hash)?.bestMove,
				null,
				null,
			).slice(0, 6);
		}

		function quiescence(
			position,
			alpha,
			beta,
			context,
			ply,
			remainingDepth,
			knownStaticEval,
			countNode,
		) {
			if (countNode) {
				checkTime(context);
			}
			context.quiescenceNodes += 1;
			if (position.winner !== -1) {
				return terminalScore(position, ply);
			}
			if (
				position.wallsRemaining[0] === 0 &&
				position.wallsRemaining[1] === 0
			) {
				return exactRaceScore(position, context, ply);
			}
			const standPat = Number.isFinite(knownStaticEval)
				? knownStaticEval
				: evaluateMutable(position, context);
			const underThreat = hasImmediateWinningPawnMove(
				position,
				1 - position.turn,
			);
			if (remainingDepth <= 0) {
				return underThreat ? standPat - 240 : standPat;
			}
			if (!underThreat && standPat >= beta) {
				return standPat;
			}
			let best = underThreat ? -Infinity : standPat;
			if (!underThreat) {
				alpha = Math.max(alpha, standPat);
			}
			const candidates = forcingMoves(position, context, ply);
			if (underThreat && candidates.length === 0) {
				return -WIN_SCORE + ply + 1;
			}
			for (const candidate of candidates) {
				context.moveStack[ply] = candidate.code;
				const undo = doEncodedMove(position, candidate.code);
				let score;
				try {
					score = -quiescence(
						position,
						-beta,
						-alpha,
						context,
						ply + 1,
						remainingDepth - 1,
						null,
						true,
					);
				} finally {
					undoEncodedMove(position, candidate.code, undo);
				}
				best = Math.max(best, score);
				alpha = Math.max(alpha, best);
				if (alpha >= beta) {
					break;
				}
			}
			return best;
		}

		function searchCandidate(
			position,
			candidate,
			depth,
			alpha,
			beta,
			context,
			ply,
			moveIndex,
			firstMove,
			rootNode,
		) {
			context.moveStack[ply] = candidate.code;
			const undo = doEncodedMove(position, candidate.code);
			try {
				if (firstMove) {
					return -search(position, depth - 1, -beta, -alpha, context, ply + 1);
				}

				const reduction = lateMoveReduction(
					depth,
					moveIndex,
					candidate,
					rootNode,
					context,
					ply,
				);
				if (reduction > 0) {
					context.lmrReductions += 1;
				}
				let score = -search(
					position,
					depth - 1 - reduction,
					-alpha - 1,
					-alpha,
					context,
					ply + 1,
				);
				if (reduction > 0 && score > alpha) {
					context.lmrResearches += 1;
					score = -search(
						position,
						depth - 1,
						-alpha - 1,
						-alpha,
						context,
						ply + 1,
					);
				}
				if (score > alpha && score < beta) {
					score = -search(position, depth - 1, -beta, -alpha, context, ply + 1);
				}
				return score;
			} finally {
				undoEncodedMove(position, candidate.code, undo);
			}
		}

		function search(position, depth, alpha, beta, context, ply) {
			checkTime(context);
			if (position.winner !== -1) {
				return terminalScore(position, ply);
			}
			let repeated = context.recentStateHashes.has(position.hash);
			for (let index = 0; !repeated && index < ply; index += 1) {
				repeated = context.pathHashes[index] === position.hash;
			}
			if (repeated) {
				context.repetitionPrunes += 1;
				return REPETITION_SCORE + Math.min(80, ply * 4);
			}
			context.pathHashes[ply] = position.hash;
			if (
				position.wallsRemaining[0] === 0 &&
				position.wallsRemaining[1] === 0
			) {
				return exactRaceScore(position, context, ply);
			}

			const cached = transpositionTable.get(position.hash);
			if (cached && cached.depth >= depth) {
				cached.generation = analysisGeneration;
				context.ttHits += 1;
				const cachedScore = scoreFromTable(cached.score, ply);
				if (cached.flag === "exact") {
					return cachedScore;
				}
				if (cached.flag === "lower") {
					alpha = Math.max(alpha, cachedScore);
				} else {
					beta = Math.min(beta, cachedScore);
				}
				if (alpha >= beta) {
					return cachedScore;
				}
			}
			if (depth === 0) {
				const staticEval = Number.isFinite(cached?.staticEval)
					? cached.staticEval
					: evaluateMutable(position, context);
				const score =
					context.stockfishFeatures && needsQuiescence(position)
						? quiescence(
								position,
								alpha,
								beta,
								context,
								ply,
								2,
								staticEval,
								false,
							)
						: staticEval;
				const flag =
					score <= alpha ? "upper" : score >= beta ? "lower" : "exact";
				storeTable(position, 0, score, flag, null, ply, staticEval);
				return score;
			}
			if (
				context.stockfishFeatures &&
				beta - alpha <= 1 &&
				depth <= 3 &&
				Number.isFinite(cached?.staticEval) &&
				!tacticallyVolatile(position)
			) {
				const margin = 100 + depth * 150;
				if (cached.staticEval - margin >= beta) {
					context.reverseFutilityPrunes += 1;
					return cached.staticEval;
				}
			}

			const alphaStart = alpha;
			const betaStart = beta;
			const actor = position.turn;
			let best = -Infinity;
			let bestMove = null;
			let firstMove = true;
			let moveIndex = 0;
			const searchedMoves = [];
			let stagedMove = null;
			if (
				context.stockfishFeatures &&
				beta - alpha <= 1 &&
				cached?.bestMove !== null &&
				cached?.bestMove !== undefined
			) {
				stagedMove = candidateFromCode(position, context, cached.bestMove);
			}
			if (stagedMove) {
				context.stagedTtMoves += 1;
				const score = searchCandidate(
					position,
					stagedMove,
					depth,
					alpha,
					beta,
					context,
					ply,
					moveIndex,
					firstMove,
					false,
				);
				best = score;
				bestMove = stagedMove.code;
				alpha = Math.max(alpha, score);
				firstMove = false;
				moveIndex += 1;
				if (alpha >= beta) {
					context.stagedTtCutoffs += 1;
					recordCutoff(context, actor, stagedMove, depth, ply, searchedMoves);
					storeTable(position, depth, best, "lower", bestMove, ply);
					return best;
				}
				searchedMoves.push(stagedMove);
			}

			const moves = orderMoves(
				generateMoves(position, context, false, undefined, depth),
				position,
				context,
				ply,
				cached?.bestMove,
				null,
				null,
			).filter((candidate) => candidate.code !== stagedMove?.code);
			if (moves.length === 0 && !stagedMove) {
				return evaluateMutable(position, context);
			}

			for (const candidate of moves) {
				const score = searchCandidate(
					position,
					candidate,
					depth,
					alpha,
					beta,
					context,
					ply,
					moveIndex,
					firstMove,
					false,
				);
				firstMove = false;
				if (score > best) {
					best = score;
					bestMove = candidate.code;
				}
				alpha = Math.max(alpha, best);
				if (alpha >= beta) {
					recordCutoff(context, actor, candidate, depth, ply, searchedMoves);
					break;
				}
				searchedMoves.push(candidate);
				moveIndex += 1;
			}

			const flag =
				best <= alphaStart ? "upper" : best >= betaStart ? "lower" : "exact";
			storeTable(position, depth, best, flag, bestMove, ply);
			return best;
		}

		function searchRoot(
			position,
			rootMoves,
			depth,
			alpha,
			beta,
			context,
			previousBest,
			mctsStats,
		) {
			const tableEntry = transpositionTable.get(position.hash);
			const candidates = orderMoves(
				rootMoves,
				position,
				context,
				0,
				tableEntry?.bestMove,
				previousBest,
				mctsStats,
			);
			const scored = [];
			const alphaStart = alpha;
			const betaStart = beta;
			let best = -Infinity;
			let bestMove = null;
			let bestCandidate = null;
			let firstMove = true;
			const searchedMoves = [];
			for (let moveIndex = 0; moveIndex < candidates.length; moveIndex += 1) {
				const candidate = candidates[moveIndex];
				let rootPenalty = 0;
				if (candidate.code < PAWN_MOVE_COUNT) {
					const visits = context.rootPawnVisits[candidate.code];
					if (visits > 0) {
						rootPenalty = Math.min(
							220,
							visits * 40 +
								(candidate.code === context.previousPawnSquare ? 90 : 0),
						);
						context.pawnHistoryPenalties += 1;
					}
				} else {
					const actorWalls = position.wallsRemaining[position.turn];
					const opponentWalls = position.wallsRemaining[1 - position.turn];
					if (actorWalls <= 2 && opponentWalls > actorWalls) {
						rootPenalty =
							(actorWalls === 1 ? 40 : 16) + (opponentWalls - actorWalls) * 4;
						context.wallReservePenalties += 1;
					}
				}
				const searchAlpha = Number.isFinite(alpha)
					? alpha + rootPenalty
					: alpha;
				const searchBeta = Number.isFinite(beta) ? beta + rootPenalty : beta;
				let score = searchCandidate(
					position,
					candidate,
					depth,
					searchAlpha,
					searchBeta,
					context,
					0,
					moveIndex,
					firstMove,
					true,
				);
				score -= rootPenalty;
				firstMove = false;
				scored.push({
					code: candidate.code,
					score,
					tempoRecovery: candidate.tempoRecovery === true,
				});
				const candidateRecoversTempo = candidate.tempoRecovery === true;
				const bestRecoversTempo = bestCandidate?.tempoRecovery === true;
				if (
					score > best ||
					(score === best &&
						((candidateRecoversTempo && !bestRecoversTempo) ||
							(candidateRecoversTempo === bestRecoversTempo &&
								candidate.code < PAWN_MOVE_COUNT &&
								bestMove >= PAWN_MOVE_COUNT)))
				) {
					best = score;
					bestMove = candidate.code;
					bestCandidate = candidate;
				}
				alpha = Math.max(alpha, best);
				if (alpha >= beta) {
					recordCutoff(
						context,
						position.turn,
						candidate,
						depth,
						0,
						searchedMoves,
					);
					break;
				}
				searchedMoves.push(candidate);
			}
			scored.sort(
				(left, right) =>
					right.score - left.score ||
					Number(right.tempoRecovery) - Number(left.tempoRecovery) ||
					Number(right.code < PAWN_MOVE_COUNT) -
						Number(left.code < PAWN_MOVE_COUNT),
			);
			const flag =
				best <= alphaStart ? "upper" : best >= betaStart ? "lower" : "exact";
			if (!context.hasPawnHistory && context.wallReservePenalties === 0) {
				storeTable(position, depth, best, flag, bestMove, 0);
			}
			return { bestMove, score: best, scored };
		}

		function monteCarloOutcome(position, rootPlayer, context) {
			if (position.winner !== -1) {
				return position.winner === rootPlayer ? 1 : -1;
			}
			if (
				position.wallsRemaining[0] === 0 &&
				position.wallsRemaining[1] === 0
			) {
				const currentWins = exactRaceScore(position, context, 0) > 0;
				const winner = currentWins ? position.turn : 1 - position.turn;
				return winner === rootPlayer ? 1 : -1;
			}
			return null;
		}

		function guidedRolloutMove(position, context, random) {
			const actor = position.turn;
			const victim = 1 - actor;
			const actorPath = mutablePath(position, actor, context);
			const victimPath = mutablePath(position, victim, context);
			if (
				position.wallsRemaining[actor] > 0 &&
				actorPath.distance >= victimPath.distance &&
				random() < 0.32
			) {
				const wallCodes = new Set();
				collectPathWallCodes(wallCodes, victimPath);
				const candidates = Array.from(wallCodes).sort(
					(left, right) =>
						Number(wallTouchesExisting(position, right)) -
						Number(wallTouchesExisting(position, left)),
				);
				let bestWall = null;
				let bestWallValue = 0;
				let probes = 0;
				for (const wallCode of candidates) {
					if (probes >= 3 || mutableWallConflict(position, wallCode)) {
						continue;
					}
					probes += 1;
					const undo = placeProbeWall(position, wallCode);
					try {
						const nextActorDistance = mutablePath(
							position,
							actor,
							context,
						).distance;
						const nextVictimDistance = mutablePath(
							position,
							victim,
							context,
						).distance;
						if (nextActorDistance < 0 || nextVictimDistance < 0) {
							continue;
						}
						const value =
							(nextVictimDistance - victimPath.distance) * 3 -
							(nextActorDistance - actorPath.distance) * 2;
						if (value > bestWallValue) {
							bestWallValue = value;
							bestWall = WALL_MOVE_OFFSET + wallCode;
						}
					} finally {
						undoProbeWall(position, wallCode, undo);
					}
				}
				if (bestWall !== null) {
					return bestWall;
				}
			}

			const pawnMoves = legalPawnMoveCodes(position, actor);
			let bestMove = pawnMoves[0] ?? null;
			let bestScore = -Infinity;
			for (const code of pawnMoves) {
				const undo = doEncodedMove(position, code);
				let distance;
				try {
					distance = mutablePath(position, actor, context).distance;
				} finally {
					undoEncodedMove(position, code, undo);
				}
				const destinationY = Math.floor(code / BOARD_SIZE);
				const progress =
					actor === 0 ? destinationY : BOARD_SIZE - 1 - destinationY;
				const score = -distance * 20 + progress * 3 + random() * 2;
				if (score > bestScore) {
					bestScore = score;
					bestMove = code;
				}
			}
			return bestMove;
		}

		function monteCarloRollout(
			position,
			rootPlayer,
			context,
			random,
			undos,
			deadline,
		) {
			for (let ply = 0; ply < 32; ply += 1) {
				const outcome = monteCarloOutcome(position, rootPlayer, context);
				if (outcome !== null) {
					return outcome;
				}
				if (performance.now() >= deadline) {
					break;
				}
				const code = guidedRolloutMove(position, context, random);
				if (code === null) {
					break;
				}
				undos.push({ code, undo: doEncodedMove(position, code) });
			}
			return Math.tanh(quickEvaluation(position, rootPlayer, context) / 180);
		}

		function runHybridMcts(position, rootMoves, context, settings) {
			if (!settings.useMcts || rootMoves.length < 2) {
				return new Map();
			}
			const now = performance.now();
			const remaining = context.deadline - now;
			const budget = Math.min(
				settings.mctsMaxMs,
				remaining * settings.mctsFraction,
			);
			const deadline = Math.min(context.deadline - 12, now + budget);
			if (deadline <= now + 2) {
				return new Map();
			}

			let randomState =
				(Number(position.hash & 0xffffffffn) ^ analysisGeneration) >>> 0;
			function random() {
				randomState ^= randomState << 13;
				randomState ^= randomState >>> 17;
				randomState ^= randomState << 5;
				return (randomState >>> 0) / 4_294_967_296;
			}
			function createNode(hash) {
				return {
					children: new Map(),
					hash,
					moves: null,
					unexpanded: null,
					value: 0,
					visits: 0,
				};
			}

			const rootPlayer = position.turn;
			const nodes = new Map();
			const rootNode = createNode(position.hash);
			rootNode.moves = rootMoves;
			rootNode.unexpanded = rootMoves.map((candidate) => candidate.code);
			nodes.set(position.hash, rootNode);
			while (
				performance.now() < deadline &&
				context.mctsSimulations < settings.mctsSimulationLimit
			) {
				const path = [rootNode];
				const undos = [];
				const visited = new Set([position.hash]);
				let node = rootNode;
				let outcome = null;
				try {
					for (let treeDepth = 0; treeDepth < 10; treeDepth += 1) {
						outcome = monteCarloOutcome(position, rootPlayer, context);
						if (outcome !== null) {
							break;
						}
						if (node.moves === null) {
							node.moves = generateMoves(position, context, false, 8);
							node.unexpanded = node.moves.map((candidate) => candidate.code);
						}
						if (node.unexpanded.length > 0) {
							const choiceCount = Math.min(3, node.unexpanded.length);
							const choice = Math.floor(random() * choiceCount);
							const code = node.unexpanded.splice(choice, 1)[0];
							undos.push({ code, undo: doEncodedMove(position, code) });
							let child = nodes.get(position.hash);
							if (!child) {
								child = createNode(position.hash);
								nodes.set(position.hash, child);
							}
							node.children.set(code, child);
							node = child;
							path.push(node);
							break;
						}
						if (node.children.size === 0) {
							break;
						}

						const maximize = position.turn === rootPlayer;
						let selectedCode = null;
						let selectedNode = null;
						let selectedValue = -Infinity;
						for (const [code, child] of node.children) {
							const mean = child.value / Math.max(1, child.visits);
							const exploration =
								1.35 *
								Math.sqrt(
									Math.log(node.visits + 2) / Math.max(1, child.visits),
								);
							const value = (maximize ? mean : -mean) + exploration;
							if (value > selectedValue) {
								selectedValue = value;
								selectedCode = code;
								selectedNode = child;
							}
						}
						if (selectedCode === null) {
							break;
						}
						undos.push({
							code: selectedCode,
							undo: doEncodedMove(position, selectedCode),
						});
						if (visited.has(position.hash)) {
							break;
						}
						visited.add(position.hash);
						node = selectedNode;
						path.push(node);
					}

					if (outcome === null) {
						outcome = monteCarloRollout(
							position,
							rootPlayer,
							context,
							random,
							undos,
							deadline,
						);
					}
					for (const visitedNode of path) {
						visitedNode.visits += 1;
						visitedNode.value += outcome;
					}
					context.mctsSimulations += 1;
				} finally {
					for (let index = undos.length - 1; index >= 0; index -= 1) {
						undoEncodedMove(position, undos[index].code, undos[index].undo);
					}
				}
			}

			context.mctsNodes = nodes.size;
			const statistics = new Map();
			for (const [code, child] of rootNode.children) {
				statistics.set(code, { value: child.value, visits: child.visits });
			}
			return statistics;
		}

		function moveKey(move) {
			return move.type === "pawn"
				? `p-${move.to.x}-${move.to.y}`
				: `w-${wallKey(move.wall)}`;
		}

		function createSearchContext(
			deadline,
			stockfishFeatures = true,
			recentStateHashes = new Set(),
			recentPawnSquares = [],
		) {
			const counterMoves = new Int16Array(MOVE_CODE_COUNT);
			counterMoves.fill(-1);
			const moveStack = new Int16Array(128);
			moveStack.fill(-1);
			const rootPawnVisits = new Uint8Array(PAWN_MOVE_COUNT);
			for (const code of recentPawnSquares) {
				rootPawnVisits[code] = Math.min(255, rootPawnVisits[code] + 1);
			}
			return {
				aspirationResearches: 0,
				bfsCalls: 0,
				continuationHistory: new Int32Array(MOVE_CODE_COUNT * MOVE_CODE_COUNT),
				counterMoves,
				deadline,
				distanceBuffer: new Int16Array(PAWN_MOVE_COUNT),
				distanceQueue: new Int16Array(PAWN_MOVE_COUNT),
				historyScores: [
					new Int32Array(MOVE_CODE_COUNT),
					new Int32Array(MOVE_CODE_COUNT),
				],
				hasPawnHistory: recentPawnSquares.length > 0,
				killers: [],
				lazyEvaluations: 0,
				lmrReductions: 0,
				lmrResearches: 0,
				mctsNodes: 0,
				mctsSimulations: 0,
				moveCacheHits: 0,
				moveStack,
				nodes: 0,
				operations: 0,
				pathPrevious: new Int16Array(PAWN_MOVE_COUNT),
				pathHashes: new Array(128),
				pathQueue: new Int16Array(PAWN_MOVE_COUNT),
				pathSeen: new Uint8Array(PAWN_MOVE_COUNT),
				pathCacheHits: 0,
				pawnHistoryPenalties: 0,
				previousPawnSquare: recentPawnSquares.at(-1) ?? -1,
				quiescenceNodes: 0,
				recentStateHashes,
				repetitionPrunes: 0,
				reverseFutilityPrunes: 0,
				rootPawnVisits,
				rootPlayer: 0,
				routeCacheHits: 0,
				stagedTtCutoffs: 0,
				stagedTtMoves: 0,
				stockfishFeatures,
				ttHits: 0,
				wallReservePenalties: 0,
			};
		}

		function trimGenerationCache(cache, maximum) {
			if (cache.size <= maximum) {
				return;
			}
			for (const [key, value] of cache) {
				if (value.generation + 2 < analysisGeneration) {
					cache.delete(key);
				}
			}
			const target = Math.floor(maximum * 0.8);
			for (const key of cache.keys()) {
				if (cache.size <= target) {
					break;
				}
				cache.delete(key);
			}
		}

		function trimCaches() {
			trimGenerationCache(transpositionTable, TT_MAX_ENTRIES);
			trimGenerationCache(routeAnalysisCache, FEATURE_CACHE_MAX_ENTRIES);
			trimGenerationCache(wallPressureCache, FEATURE_CACHE_MAX_ENTRIES);
			trimGenerationCache(moveGenerationCache, MOVE_CACHE_MAX_ENTRIES);
		}

		function exactRootRace(position, rootMoves, context) {
			const scored = [];
			for (const candidate of rootMoves) {
				const undo = doEncodedMove(position, candidate.code);
				let score;
				try {
					score =
						position.winner !== -1
							? -terminalScore(position, 1)
							: -exactRaceScore(position, context, 1);
				} finally {
					undoEncodedMove(position, candidate.code, undo);
				}
				scored.push({ code: candidate.code, score });
			}
			scored.sort((left, right) => right.score - left.score);
			return scored;
		}

		function fallbackPawnCandidate(position, context) {
			const actor = position.turn;
			const actorY = Math.floor(position.pawns[actor] / BOARD_SIZE);
			let best = null;
			for (const code of legalPawnMoveCodes(position, actor)) {
				const undo = doEncodedMove(position, code);
				let distance;
				try {
					distance = mutablePath(position, actor, context).distance;
				} finally {
					undoEncodedMove(position, code, undo);
				}
				const destinationY = Math.floor(code / BOARD_SIZE);
				const progress =
					actor === 0 ? destinationY - actorY : actorY - destinationY;
				const priority = 1_200 - distance * 40 + progress * 80;
				if (!best || priority > best.priority) {
					best = { code, priority };
				}
			}
			return best;
		}

		function analyze(state, rootPlayer, options) {
			const recentStateHashes = new Set();
			for (const value of options?.recentStateHashes ?? []) {
				try {
					recentStateHashes.add(BigInt(value));
				} catch {
					// Ignore malformed external history entries.
				}
			}
			const recentPawnSquares = [];
			for (const value of options?.recentPawnSquares ?? []) {
				const code = Number(value);
				if (Number.isInteger(code) && code >= 0 && code < PAWN_MOVE_COUNT) {
					recentPawnSquares.push(code);
				}
			}
			const settings = {
				maxDepth: Math.max(1, options?.maxDepth ?? 25),
				mctsFraction: Math.max(0, options?.mctsFraction ?? 0.14),
				mctsMaxMs: Math.max(0, options?.mctsMaxMs ?? 120),
				mctsSimulationLimit: Math.max(
					0,
					options?.mctsSimulations ?? Number.POSITIVE_INFINITY,
				),
				timeMs: Math.max(10, options?.timeMs ?? 1_500),
				stockfishFeatures: options?.stockfishFeatures !== false,
				useMcts: options?.useMcts === true,
			};
			analysisGeneration += 1;
			const startedAt = performance.now();
			const deadlineReserveMs = Math.min(
				60,
				Math.max(1, settings.timeMs * 0.04),
			);
			const context = createSearchContext(
				startedAt + settings.timeMs - deadlineReserveMs,
				settings.stockfishFeatures,
				recentStateHashes,
				recentPawnSquares,
			);
			const position = createMutablePosition(state);
			context.rootPlayer = position.turn;
			context.pathHashes[0] = position.hash;
			const requestedPlayer = playerIndex(rootPlayer);
			const scoreSign = requestedPlayer === position.turn ? 1 : -1;
			let completedDepth = 0;
			let exactEndgame = false;
			let internalScoredMoves = [];
			let previousBest = null;
			let previousScore = 0;
			let rootMoves = [];
			let mctsStats = new Map();

			try {
				rootMoves = generateMoves(position, context, true);
				if (
					position.wallsRemaining[0] === 0 &&
					position.wallsRemaining[1] === 0
				) {
					exactEndgame = true;
					internalScoredMoves = exactRootRace(position, rootMoves, context);
					completedDepth = internalScoredMoves.length > 0 ? 1 : 0;
				} else {
					mctsStats = runHybridMcts(position, rootMoves, context, settings);
					for (let depth = 1; depth <= settings.maxDepth; depth += 1) {
						let window = depth === 1 ? Number.POSITIVE_INFINITY : 72;
						let alpha = Number.isFinite(window)
							? previousScore - window
							: Number.NEGATIVE_INFINITY;
						let beta = Number.isFinite(window)
							? previousScore + window
							: Number.POSITIVE_INFINITY;
						let iteration = null;
						for (;;) {
							iteration = searchRoot(
								position,
								rootMoves,
								depth,
								alpha,
								beta,
								context,
								previousBest,
								mctsStats,
							);
							if (
								!Number.isFinite(window) ||
								(iteration.score > alpha && iteration.score < beta)
							) {
								break;
							}
							context.aspirationResearches += 1;
							if (settings.stockfishFeatures) {
								if (iteration.score <= alpha) {
									beta = alpha;
									alpha = Math.max(
										Number.NEGATIVE_INFINITY,
										iteration.score - window,
									);
								} else {
									alpha = Math.max(alpha, beta - window);
									beta = Math.min(
										Number.POSITIVE_INFINITY,
										iteration.score + window,
									);
								}
								window += Math.max(16, Math.ceil((window * 3) / 8));
							} else {
								window *= 2;
								if (window > WIN_SCORE * 2) {
									window = Number.POSITIVE_INFINITY;
								}
								alpha = Number.isFinite(window)
									? previousScore - window
									: Number.NEGATIVE_INFINITY;
								beta = Number.isFinite(window)
									? previousScore + window
									: Number.POSITIVE_INFINITY;
							}
						}
						if (!iteration || iteration.scored.length === 0) {
							break;
						}
						internalScoredMoves = iteration.scored;
						completedDepth = depth;
						previousBest = iteration.bestMove;
						previousScore = iteration.score;
						if (performance.now() >= context.deadline) {
							break;
						}
					}
				}
			} catch (error) {
				if (error !== TIMEOUT) {
					throw error;
				}
			}
			if (rootMoves.length === 0 && position.winner === -1) {
				const fallback = fallbackPawnCandidate(position, context);
				if (fallback) {
					rootMoves = [fallback];
				}
			}

			if (internalScoredMoves.length === 0 && rootMoves.length > 0) {
				const ordered = orderMoves(
					rootMoves,
					position,
					context,
					0,
					transpositionTable.get(position.hash)?.bestMove,
					previousBest,
					mctsStats,
				);
				const fallback = ordered[0];
				const undo = doEncodedMove(position, fallback.code);
				let score;
				try {
					score =
						position.winner !== -1
							? -terminalScore(position, 1)
							: quickEvaluation(position, 1 - position.turn, context);
				} finally {
					undoEncodedMove(position, fallback.code, undo);
				}
				internalScoredMoves = [{ code: fallback.code, score }];
			}

			const scoredMoves = internalScoredMoves.map((candidate) => ({
				move: decodedMove(candidate.code),
				score: candidate.score * scoreSign,
			}));
			const ownDistance = mutablePath(
				position,
				requestedPlayer,
				context,
			).distance;
			const opponentDistance = mutablePath(
				position,
				1 - requestedPlayer,
				context,
			).distance;
			trimCaches();
			return {
				bestMove: scoredMoves[0]?.move || null,
				score: scoredMoves[0]?.score ?? 0,
				alternatives: scoredMoves.slice(0, 3),
				aspirationResearches: context.aspirationResearches,
				bfsCalls: context.bfsCalls,
				depth: completedDepth,
				exactEndgame,
				nodes: context.nodes,
				elapsedMs: Math.round(performance.now() - startedAt),
				mctsNodes: context.mctsNodes,
				mctsSimulations: context.mctsSimulations,
				moveCacheHits: context.moveCacheHits,
				lmrReductions: context.lmrReductions,
				lmrResearches: context.lmrResearches,
				lazyEvaluations: context.lazyEvaluations,
				opponentDistance,
				ownDistance,
				pathCacheHits: context.pathCacheHits,
				pawnHistoryPenalties: context.pawnHistoryPenalties,
				quiescenceNodes: context.quiescenceNodes,
				reverseFutilityPrunes: context.reverseFutilityPrunes,
				repetitionPrunes: context.repetitionPrunes,
				searchMode: settings.useMcts ? "hybrid" : "pvs",
				stagedTtCutoffs: context.stagedTtCutoffs,
				stagedTtMoves: context.stagedTtMoves,
				ttHits: context.ttHits,
				ttSize: transpositionTable.size,
				wallReservePenalties: context.wallReservePenalties,
			};
		}

		function inspectCandidateMoves(state, options) {
			const context = createSearchContext(Number.POSITIVE_INFINITY);
			const position = createMutablePosition(state);
			const wallCap = Object.hasOwn(options || {}, "wallCap")
				? options.wallCap
				: undefined;
			return generateMoves(
				position,
				context,
				Boolean(options?.rootNode),
				wallCap,
			).map((candidate) => ({
				move: decodedMove(candidate.code),
				priority: candidate.priority,
			}));
		}

		function inspectWallLimit(state, rootNode) {
			const context = createSearchContext(Number.POSITIVE_INFINITY);
			const position = createMutablePosition(state);
			const actorDistance = mutablePath(
				position,
				position.turn,
				context,
			).distance;
			const victimDistance = mutablePath(
				position,
				1 - position.turn,
				context,
			).distance;
			return adaptiveWallLimit(
				position,
				Boolean(rootNode),
				actorDistance,
				victimDistance,
			);
		}

		function inspectWallImpact(state, wall) {
			const context = createSearchContext(Number.POSITIVE_INFINITY);
			const position = createMutablePosition(state);
			const code = encodedWall(wall);
			if (mutableWallConflict(position, code)) {
				return { conflict: true };
			}
			const before = [
				mutablePath(position, 0, context),
				mutablePath(position, 1, context),
			];
			const bfsBefore = context.bfsCalls;
			const intersects = before.map((path) => wallCutsCachedPath(code, path));
			const undo = placeProbeWall(position, code);
			let after;
			try {
				after = [
					mutablePath(position, 0, context).distance,
					mutablePath(position, 1, context).distance,
				];
			} finally {
				undoProbeWall(position, code, undo);
			}
			return {
				additionalBfsCalls: context.bfsCalls - bfsBefore,
				conflict: false,
				distances: after,
				intersects,
			};
		}

		function verifyMoveRoundTrip(state, move) {
			const position = createMutablePosition(state);
			const before = {
				east: Array.from(position.east),
				geometryHash: position.geometryHash,
				hash: position.hash,
				pawns: Array.from(position.pawns),
				south: Array.from(position.south),
				turn: position.turn,
				wallHash: position.wallHash,
				wallList: position.wallList.slice(),
				wallsRemaining: Array.from(position.wallsRemaining),
				winner: position.winner,
			};
			const code = encodedMove(move);
			const undo = doEncodedMove(position, code);
			undoEncodedMove(position, code, undo);
			return (
				position.hash === before.hash &&
				position.geometryHash === before.geometryHash &&
				position.wallHash === before.wallHash &&
				position.turn === before.turn &&
				position.winner === before.winner &&
				JSON.stringify(Array.from(position.pawns)) ===
					JSON.stringify(before.pawns) &&
				JSON.stringify(Array.from(position.wallsRemaining)) ===
					JSON.stringify(before.wallsRemaining) &&
				JSON.stringify(position.wallList) === JSON.stringify(before.wallList) &&
				JSON.stringify(Array.from(position.south)) ===
					JSON.stringify(before.south) &&
				JSON.stringify(Array.from(position.east)) ===
					JSON.stringify(before.east)
			);
		}

		function measurePathRobustness(state, player) {
			const context = createSearchContext(Number.POSITIVE_INFINITY);
			return routeAnalysis(
				createMutablePosition(state),
				playerIndex(player),
				context,
			).robustness;
		}

		function measureWallThreat(state, attacker, victim) {
			const context = createSearchContext(Number.POSITIVE_INFINITY);
			const position = createMutablePosition(state);
			const victimRoute = routeAnalysis(position, playerIndex(victim), context);
			return wallPressure(
				position,
				playerIndex(attacker),
				playerIndex(victim),
				victimRoute,
				context,
			);
		}

		function cacheInfo() {
			return {
				moveGenerationEntries: moveGenerationCache.size,
				pawnRaceEntries: pawnRaceTableCache.size,
				routeAnalysisEntries: routeAnalysisCache.size,
				transpositionEntries: transpositionTable.size,
				wallPressureEntries: wallPressureCache.size,
			};
		}

		function clearSearchCaches() {
			transpositionTable.clear();
			routeAnalysisCache.clear();
			wallPressureCache.clear();
			moveGenerationCache.clear();
			pawnRaceTableCache.clear();
		}

		function stateHash(state) {
			return createMutablePosition(state).hash;
		}

		return {
			analyze,
			applyMove,
			cacheInfo,
			clearSearchCaches,
			compileBlockedEdges,
			goalRow,
			inspectCandidateMoves,
			inspectWallImpact,
			inspectWallLimit,
			legalPawnMoves,
			measurePathRobustness,
			measureWallThreat,
			moveKey,
			opponent,
			shortestDistance,
			shortestPathCells,
			stateHash,
			verifyMoveRoundTrip,
			wallConflicts,
			wallInBounds,
			wallKey,
		};
	}

	function selectAdaptiveSearch(engine, state, player) {
		const other = engine.opponent(player);
		const ownDistance = engine.shortestDistance(state, player);
		const opponentDistance = engine.shortestDistance(state, other);
		const nearestGoal = Math.min(ownDistance, opponentDistance);
		const pathGap = Math.abs(ownDistance - opponentDistance);
		const pawnSeparation =
			Math.abs(state.pawns.p1.x - state.pawns.p2.x) +
			Math.abs(state.pawns.p1.y - state.pawns.p2.y);
		const wallsPlaced = state.walls.length;
		const wallsRemaining = state.wallsRemaining.p1 + state.wallsRemaining.p2;
		const legalMoves = engine.legalPawnMoves(state, player);
		const immediateWin = legalMoves.some(
			(move) => move.y === engine.goalRow(player),
		);
		const base = { ...ADAPTIVE_SEARCH_LIMIT };

		if (state.wallsRemaining.p1 === 0 && state.wallsRemaining.p2 === 0) {
			return {
				...base,
				profile: "exact",
				reason: "The fixed-wall race table solves this position exactly.",
				timeMs: 100,
			};
		}
		if (immediateWin) {
			return {
				...base,
				profile: "forced",
				reason: "An immediate legal move reaches the goal.",
				timeMs: 120,
			};
		}

		const calmOpening =
			wallsPlaced <= 2 &&
			wallsRemaining >= 18 &&
			ownDistance >= 7 &&
			opponentDistance >= 7 &&
			pawnSeparation >= 6;
		if (calmOpening) {
			return {
				...base,
				profile: "opening",
				reason: "The open board favors a simple advance and conserving walls.",
				timeMs: 500,
			};
		}

		const routeLead = opponentDistance - ownDistance;
		const opponentWalls = state.wallsRemaining[other];
		if (opponentWalls === 0 && routeLead > 1) {
			return {
				...base,
				profile: "conversion",
				reason:
					"The lead exceeds their remaining wall leverage, so race directly.",
				timeMs: 350,
			};
		}

		let pressure = 0;
		if (nearestGoal <= 2) {
			pressure += 6;
		} else if (nearestGoal <= 3) {
			pressure += 3;
		}
		if (nearestGoal <= 5 && pathGap <= 1) {
			pressure += 2;
		}
		if (pawnSeparation <= 2) {
			pressure += 2;
		} else if (pawnSeparation <= 4) {
			pressure += 1;
		}
		if (wallsPlaced >= 10) {
			pressure += 2;
		} else if (wallsPlaced >= 5) {
			pressure += 1;
		}
		if (wallsRemaining <= 6) {
			pressure += 2;
		} else if (wallsRemaining <= 12) {
			pressure += 1;
		}
		if (legalMoves.length <= 2) {
			pressure += 1;
		}
		if (pressure >= 5) {
			return {
				...base,
				pressure,
				profile: "tactical",
				reason:
					"Goal threats, close pawns, or constrained routes need the full budget.",
			};
		}
		return {
			...base,
			pressure,
			profile: "strategic",
			reason: "The balanced position receives the full adaptive search budget.",
		};
	}

	function coordinate(position) {
		return String.fromCharCode(97 + position.x) + String(position.y + 1);
	}

	function formatMove(move) {
		if (!move) {
			return "No legal move";
		}
		if (move.type === "pawn") {
			return `Move ${coordinate(move.to)}`;
		}
		const wall = move.wall;
		const end =
			wall.o === "h"
				? { x: wall.x + 1, y: wall.y }
				: { x: wall.x, y: wall.y + 1 };
		return (
			(wall.o === "h" ? "H wall " : "V wall ") +
			coordinate(wall) +
			"–" +
			coordinate(end)
		);
	}

	function decodeBoardGeometry(geometry) {
		function numeric(value) {
			const parsed = Number(value);
			return Number.isFinite(parsed) ? parsed : null;
		}

		function findPawn(player) {
			const color = `--color-${player}`;
			const circle = geometry.circles.find((candidate) => {
				if (numeric(candidate.r) !== 20) {
					return false;
				}
				return (
					String(candidate.fill || "").includes(color) ||
					String(candidate.stroke || "").includes(color)
				);
			});
			if (!circle) {
				return null;
			}
			const cx = numeric(circle.cx);
			const cy = numeric(circle.cy);
			if (cx === null || cy === null) {
				return null;
			}
			return {
				x: Math.round((cx - 30) / 72),
				y: Math.round((cy - 30) / 72),
			};
		}

		const pawns = { p1: findPawn("p1"), p2: findPawn("p2") };
		if (
			!pawns.p1 ||
			!pawns.p2 ||
			pawns.p1.x < 0 ||
			pawns.p1.x >= BOARD_SIZE ||
			pawns.p1.y < 0 ||
			pawns.p1.y >= BOARD_SIZE ||
			pawns.p2.x < 0 ||
			pawns.p2.x >= BOARD_SIZE ||
			pawns.p2.y < 0 ||
			pawns.p2.y >= BOARD_SIZE
		) {
			return null;
		}

		const wallsByKey = new Map();
		for (const rectangle of geometry.rects) {
			if (rectangle.stroke) {
				continue;
			}
			const width = numeric(rectangle.width);
			const height = numeric(rectangle.height);
			const fill = String(rectangle.fill || "");
			if (
				!fill.includes("--color-wall") &&
				!fill.includes("--color-p1") &&
				!fill.includes("--color-p2")
			) {
				continue;
			}

			let wall = null;
			if (width === 132 && height === 12) {
				wall = {
					x: Math.round(numeric(rectangle.x) / 72),
					y: Math.round((numeric(rectangle.y) - 60) / 72),
					o: "h",
				};
			} else if (width === 12 && height === 132) {
				wall = {
					x: Math.round((numeric(rectangle.x) - 60) / 72),
					y: Math.round(numeric(rectangle.y) / 72),
					o: "v",
				};
			}
			if (
				wall &&
				wall.x >= 0 &&
				wall.x < WALL_SLOTS &&
				wall.y >= 0 &&
				wall.y < WALL_SLOTS
			) {
				wallsByKey.set(`${wall.o}-${wall.x}-${wall.y}`, wall);
			}
		}

		const bottomPlayer = geometry.rotated ? "p1" : "p2";
		const otherPlayer = bottomPlayer === "p1" ? "p2" : "p1";
		const activeRing = geometry.circles.find(
			(circle) =>
				numeric(circle.r) === 22 &&
				String(circle.stroke || "").includes("--color-p"),
		);
		let turn = null;
		if (String(activeRing?.stroke || "").includes("--color-p1")) {
			turn = "p1";
		} else if (String(activeRing?.stroke || "").includes("--color-p2")) {
			turn = "p2";
		} else if (/\byour turn\b/i.test(geometry.statusText || "")) {
			turn = bottomPlayer;
		} else {
			turn = otherPlayer;
		}

		const wallCounts = geometry.wallCounts || [];
		const wallsRemaining = { p1: 10, p2: 10 };
		if (wallCounts.length >= 2) {
			wallsRemaining[otherPlayer] = wallCounts[0];
			wallsRemaining[bottomPlayer] = wallCounts[1];
		} else {
			for (const rectangle of geometry.rects) {
				const fill = String(rectangle.fill || "");
				if (
					(Number(rectangle.width) === 132 &&
						Number(rectangle.height) === 12) ||
					(Number(rectangle.width) === 12 && Number(rectangle.height) === 132)
				) {
					if (fill.includes("--color-p1")) {
						wallsRemaining.p1 -= 1;
					} else if (fill.includes("--color-p2")) {
						wallsRemaining.p2 -= 1;
					}
				}
			}
		}
		let winner = null;
		if (pawns.p1.y === BOARD_SIZE - 1) {
			winner = "p1";
		} else if (pawns.p2.y === 0) {
			winner = "p2";
		} else {
			winner = winnerFromStatusText(geometry.statusText, bottomPlayer);
		}

		return {
			bottomPlayer,
			state: {
				pawns,
				walls: Array.from(wallsByKey.values()),
				wallsRemaining,
				turn,
				winner,
			},
		};
	}

	function winnerFromStatusText(statusText, bottomPlayer) {
		if (
			/\b(?:you lost|you lose|defeat|computer wins?)\b/i.test(statusText || "")
		) {
			return bottomPlayer === "p1" ? "p2" : "p1";
		}
		if (/\b(?:you won|you win|victory)\b/i.test(statusText || "")) {
			return bottomPlayer;
		}
		return null;
	}

	function readBoardState(svg) {
		const attributes = (element) => ({
			cx: element.getAttribute("cx"),
			cy: element.getAttribute("cy"),
			r: element.getAttribute("r"),
			fill: element.getAttribute("fill"),
			stroke: element.getAttribute("stroke"),
			x: element.getAttribute("x"),
			y: element.getAttribute("y"),
			width: element.getAttribute("width"),
			height: element.getAttribute("height"),
		});
		const aside =
			document.querySelector("main aside") || document.querySelector("aside");
		const statusText = aside?.textContent || "";
		const wallCounts = Array.from(aside?.querySelectorAll("span") || [])
			.map((element) => (element.textContent || "").trim())
			.map((text) => text.match(/^walls\s*·\s*(\d+)$/i))
			.filter(Boolean)
			.map((match) => Number(match[1]));
		const rotated = Array.from(svg.children).some(
			(element) =>
				element.tagName.toLowerCase() === "g" &&
				String(element.getAttribute("transform") || "").includes("rotate(180"),
		);

		return decodeBoardGeometry({
			circles: Array.from(svg.querySelectorAll("circle")).map(attributes),
			rects: Array.from(svg.querySelectorAll("rect")).map(attributes),
			rotated,
			statusText,
			wallCounts,
		});
	}

	function makeSvgElement(tag, attributes) {
		const element = document.createElementNS(SVG_NS, tag);
		for (const [name, value] of Object.entries(attributes)) {
			element.setAttribute(name, String(value));
		}
		return element;
	}

	function clearBoardHint(svg) {
		svg?.querySelector("[data-wallz-coach-overlay]")?.remove();
	}

	function renderBoardHint(svg, move, state, rotated) {
		clearBoardHint(svg);
		if (!move) {
			return;
		}

		const overlay = makeSvgElement("g", {
			"data-wallz-coach-overlay": "true",
			"pointer-events": "none",
		});
		if (rotated) {
			overlay.setAttribute("transform", "rotate(180 318 318)");
		}

		if (move.type === "pawn") {
			const from = state.pawns[state.turn];
			const fromX = from.x * 72 + 30;
			const fromY = from.y * 72 + 30;
			const toX = move.to.x * 72 + 30;
			const toY = move.to.y * 72 + 30;
			const line = makeSvgElement("line", {
				x1: fromX,
				y1: fromY,
				x2: toX,
				y2: toY,
				stroke: "#eaff6a",
				"stroke-width": 8,
				"stroke-linecap": "round",
				"stroke-dasharray": "7 10",
				opacity: 0.86,
				style: "filter: drop-shadow(0 0 8px rgba(234,255,106,.7))",
			});
			const target = makeSvgElement("circle", {
				cx: toX,
				cy: toY,
				r: 27,
				fill: "rgba(234,255,106,.28)",
				stroke: "#eaff6a",
				"stroke-width": 5,
				style: "filter: drop-shadow(0 0 12px rgba(234,255,106,.75))",
			});
			const pulse = makeSvgElement("circle", {
				cx: toX,
				cy: toY,
				r: 32,
				fill: "none",
				stroke: "#eaff6a",
				"stroke-width": 2,
				opacity: 0.7,
			});
			pulse.append(
				makeSvgElement("animate", {
					attributeName: "r",
					values: "29;38;29",
					dur: "1.5s",
					repeatCount: "indefinite",
				}),
				makeSvgElement("animate", {
					attributeName: "opacity",
					values: ".75;.12;.75",
					dur: "1.5s",
					repeatCount: "indefinite",
				}),
			);
			overlay.append(line, pulse, target);
		} else {
			const wall = move.wall;
			const rectangle =
				wall.o === "h"
					? {
							x: wall.x * 72,
							y: wall.y * 72 + 60,
							width: 132,
							height: 12,
						}
					: {
							x: wall.x * 72 + 60,
							y: wall.y * 72,
							width: 12,
							height: 132,
						};
			const glow = makeSvgElement("rect", {
				...rectangle,
				rx: 6,
				fill: "#eaff6a",
				stroke: "#10140e",
				"stroke-width": 3,
				style: "filter: drop-shadow(0 0 13px rgba(234,255,106,.85))",
			});
			glow.append(
				makeSvgElement("animate", {
					attributeName: "opacity",
					values: "1;.55;1",
					dur: "1.25s",
					repeatCount: "indefinite",
				}),
			);
			overlay.append(glow);
		}

		svg.append(overlay);
	}

	function createPanel(actions) {
		const host = document.createElement("div");
		host.id = "wallz-practice-coach";
		host.style.display = "none";
		document.documentElement.append(host);
		const shadow = host.attachShadow({ mode: "open" });
		const style = document.createElement("style");
		style.textContent = [
			":host{all:initial;position:fixed;right:18px;bottom:18px;z-index:2147483646;color:#f4f4e8;font-family:'Avenir Next Condensed','DIN Alternate','Helvetica Neue',sans-serif}",
			"*{box-sizing:border-box}",
			".panel{position:relative;width:min(344px,calc(100vw - 24px));overflow:hidden;border:1px solid rgba(234,255,106,.2);border-radius:22px;background:linear-gradient(145deg,rgba(22,25,20,.97),rgba(8,10,8,.98));box-shadow:0 24px 70px rgba(0,0,0,.52),inset 0 1px rgba(255,255,255,.06);backdrop-filter:blur(18px)}",
			".panel:before{content:'';position:absolute;inset:0;background:radial-gradient(circle at 88% 5%,rgba(234,255,106,.12),transparent 33%),repeating-linear-gradient(135deg,transparent 0 18px,rgba(255,255,255,.018) 18px 19px);pointer-events:none}",
			".top,.body{position:relative}",
			".top{display:flex;align-items:center;justify-content:space-between;padding:15px 16px 13px;border-bottom:1px solid rgba(255,255,255,.07)}",
			".brand{display:flex;align-items:center;gap:11px}",
			".signal{width:11px;height:11px;border-radius:50%;background:#eaff6a;box-shadow:0 0 0 5px rgba(234,255,106,.09),0 0 18px rgba(234,255,106,.55)}",
			".panel[data-tone='busy'] .signal{animation:signal 1s ease-in-out infinite}",
			".panel[data-tone='error'] .signal{background:#ff866f;box-shadow:0 0 0 5px rgba(255,134,111,.08)}",
			".brand-copy{display:flex;flex-direction:column;line-height:1}",
			".kicker{font-size:9px;letter-spacing:.22em;color:#a0a696;text-transform:uppercase}",
			".brand strong{margin-top:5px;font-size:16px;letter-spacing:.035em;font-weight:700}",
			"button{font:inherit;color:inherit}",
			".icon{display:grid;place-items:center;width:30px;height:30px;border:1px solid rgba(255,255,255,.09);border-radius:10px;background:rgba(255,255,255,.035);cursor:pointer;transition:background .18s,border-color .18s,transform .18s}",
			".icon:hover{background:rgba(234,255,106,.08);border-color:rgba(234,255,106,.28)}",
			".icon:active{transform:scale(.94)}",
			".body{padding:15px 16px 14px}",
			".status-row{display:flex;align-items:center;justify-content:space-between;gap:12px}",
			".status{font-size:9px;font-weight:800;letter-spacing:.19em;color:#eaff6a;text-transform:uppercase}",
			".engine-stats{font-family:'SF Mono','Cascadia Mono',monospace;font-size:9px;color:#858b7d}",
			".move{margin-top:9px;font-size:31px;line-height:1.05;letter-spacing:-.035em;font-weight:750;color:#fcfff1;text-wrap:balance}",
			".reason{min-height:36px;margin:8px 0 13px;color:#afb5a7;font-size:12px;line-height:1.45}",
			".routes{display:grid;grid-template-columns:1fr 1fr 1fr;border:1px solid rgba(255,255,255,.07);border-radius:14px;overflow:hidden;background:rgba(255,255,255,.022)}",
			".route{padding:9px 10px;border-right:1px solid rgba(255,255,255,.065)}",
			".route:last-child{border-right:0}",
			".route span{display:block;font-size:8px;letter-spacing:.16em;color:#747b70;text-transform:uppercase}",
			".route strong{display:block;margin-top:4px;font-family:'SF Mono','Cascadia Mono',monospace;font-size:16px;color:#eef0e9}",
			".alternatives-title{margin:13px 0 6px;font-size:8px;letter-spacing:.18em;color:#747b70;text-transform:uppercase}",
			".alternatives{display:flex;gap:6px;min-height:27px;margin:0;padding:0;list-style:none;overflow:hidden}",
			".alternatives li{min-width:0;padding:6px 8px;border:1px solid rgba(255,255,255,.07);border-radius:9px;background:rgba(255,255,255,.028);font-family:'SF Mono','Cascadia Mono',monospace;font-size:9px;color:#b9bfb3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
			".controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:13px}",
			".control{height:35px;border:1px solid rgba(255,255,255,.08);border-radius:11px;background:rgba(255,255,255,.035);font-size:10px;font-weight:700;letter-spacing:.04em;cursor:pointer;transition:.18s}",
			".control:hover{border-color:rgba(234,255,106,.28);background:rgba(234,255,106,.07)}",
			".control:disabled{opacity:.38;cursor:not-allowed}",
			".control:focus-visible{outline:2px solid #eaff6a;outline-offset:2px}",
			".control[aria-pressed='true']{color:#eaff6a;border-color:rgba(234,255,106,.4);background:rgba(234,255,106,.1)}",
			".control.autoplay[aria-pressed='true']{color:#ffd36a;border-color:rgba(255,211,106,.48);background:rgba(255,211,106,.1);box-shadow:inset 0 0 18px rgba(255,211,106,.05)}",
			".control.autoplay[data-state='playing']{animation:armed 1s ease-in-out infinite}",
			".control.autoplay[data-state='blocked']{color:#ff9b85;border-color:rgba(255,155,133,.42);background:rgba(255,155,133,.09)}",
			".control.primary{color:#11150d;background:#eaff6a;border-color:#eaff6a}",
			".control.primary:hover{background:#f1ff9b}",
			".control.export{grid-column:1/-1}",
			".guard{margin:10px 0 0;text-align:center;font-size:8px;letter-spacing:.12em;color:#62685e;text-transform:uppercase}",
			".panel[data-collapsed='true'] .body{display:none}",
			".panel[data-collapsed='true'] .top{border-bottom:0}",
			"@keyframes signal{50%{opacity:.38;transform:scale(.78)}}",
			"@keyframes armed{50%{border-color:rgba(255,211,106,.9);background:rgba(255,211,106,.17)}}",
			"@media(max-width:640px){:host{right:10px;bottom:10px}.panel{width:min(318px,calc(100vw - 20px))}.move{font-size:27px}}",
			"@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}",
		].join("");
		const panel = document.createElement("section");
		panel.className = "panel";
		panel.dataset.tone = "idle";
		panel.dataset.collapsed = "false";
		panel.innerHTML = [
			"<header class='top'>",
			"<div class='brand'><span class='signal'></span><div class='brand-copy'><span class='kicker'>Wallz / coach</span><strong>Routefinder</strong></div></div>",
			"<button class='icon' id='collapse' type='button' aria-label='Collapse coach'>−</button>",
			"</header>",
			"<div class='body'>",
			"<div class='status-row'><span class='status' id='status'>Ready</span><span class='engine-stats' id='stats'>local engine</span></div>",
			"<div class='move' id='move'>Waiting for a board</div>",
			"<p class='reason' id='reason'>Recommendations appear when it is your turn.</p>",
			"<div class='routes'>",
			"<div class='route'><span>Your route</span><strong id='own-route'>—</strong></div>",
			"<div class='route'><span>Their route</span><strong id='their-route'>—</strong></div>",
			"<div class='route'><span>Path lead</span><strong id='lead'>—</strong></div>",
			"</div>",
			"<p class='alternatives-title'>Candidate line</p><ol class='alternatives' id='alternatives'><li>Waiting for board</li></ol>",
			"<div class='controls'>",
			"<button class='control primary' id='refresh' type='button'>Recalculate</button>",
			"<button class='control autoplay' id='autoplay' type='button' aria-pressed='false' data-state='off' title='Play fresh engine recommendations automatically'>Autoplay · Off</button>",
			"<button class='control wallzero' id='wallzero-toggle' type='button' aria-pressed='false' title='Toggle the local WallZero engine bridge'>WallZero · …</button>",
			"<button class='control export' id='export-loss' type='button' disabled title='A local diagnostic report becomes available after a loss'>Export latest loss</button>",
			"</div>",
			"<p class='guard'>Autoplay stays off until armed · Alt+W to hide</p>",
			"</div>",
		].join("");
		shadow.append(style, panel);

		const elements = {
			status: panel.querySelector("#status"),
			stats: panel.querySelector("#stats"),
			move: panel.querySelector("#move"),
			reason: panel.querySelector("#reason"),
			ownRoute: panel.querySelector("#own-route"),
			theirRoute: panel.querySelector("#their-route"),
			lead: panel.querySelector("#lead"),
			alternatives: panel.querySelector("#alternatives"),
			autoplay: /** @type {HTMLButtonElement} */ (
				panel.querySelector("#autoplay")
			),
			exportLoss: /** @type {HTMLButtonElement} */ (
				panel.querySelector("#export-loss")
			),
			refresh: /** @type {HTMLButtonElement} */ (
				panel.querySelector("#refresh")
			),
			wallZeroToggle: /** @type {HTMLButtonElement} */ (
				panel.querySelector("#wallzero-toggle")
			),
			collapse: /** @type {HTMLButtonElement} */ (
				panel.querySelector("#collapse")
			),
		};
		function setTone(tone) {
			panel.dataset.tone = tone;
		}

		function setAlternatives(labels) {
			elements.alternatives.replaceChildren();
			for (const label of labels.length ? labels : ["—"]) {
				const item = document.createElement("li");
				item.textContent = label;
				elements.alternatives.append(item);
			}
		}

		function setRoutes(ownDistance, theirDistance) {
			elements.ownRoute.textContent =
				Number.isFinite(ownDistance) && ownDistance >= 0
					? String(ownDistance)
					: "—";
			elements.theirRoute.textContent =
				Number.isFinite(theirDistance) && theirDistance >= 0
					? String(theirDistance)
					: "—";
			if (
				Number.isFinite(ownDistance) &&
				Number.isFinite(theirDistance) &&
				ownDistance >= 0 &&
				theirDistance >= 0
			) {
				const difference = theirDistance - ownDistance;
				elements.lead.textContent =
					difference > 0 ? `+${difference}` : String(difference);
			} else {
				elements.lead.textContent = "—";
			}
		}

		elements.collapse.addEventListener("click", () => {
			const collapsed = panel.dataset.collapsed !== "true";
			panel.dataset.collapsed = String(collapsed);
			elements.collapse.textContent = collapsed ? "+" : "−";
			elements.collapse.setAttribute(
				"aria-label",
				collapsed ? "Expand coach" : "Collapse coach",
			);
		});
		elements.refresh.addEventListener("click", actions.onRefresh);
		elements.wallZeroToggle.addEventListener("click", () => {
			actions.onWallZeroToggle?.();
		});
		elements.exportLoss.addEventListener("click", actions.onExportLoss);
		elements.autoplay.addEventListener("click", () => {
			actions.onAutoplayChange(
				elements.autoplay.getAttribute("aria-pressed") !== "true",
			);
		});

		return {
			hide() {
				host.style.display = "none";
			},
			setWallZeroState(label, active) {
				elements.wallZeroToggle.textContent = `WallZero · ${label}`;
				elements.wallZeroToggle.setAttribute(
					"aria-pressed",
					active ? "true" : "false",
				);
			},
			show() {
				host.style.display = "block";
			},
			setAnalyzing(ownDistance, theirDistance, plan) {
				setTone("busy");
				elements.status.textContent = "Calculating";
				elements.stats.textContent = `${plan.profile} · ≤${plan.timeMs}ms`;
				elements.move.textContent = "Reading the lanes…";
				elements.reason.textContent = plan.reason;
				setRoutes(ownDistance, theirDistance);
				setAlternatives(["searching"]);
			},
			setError(message) {
				setTone("error");
				elements.status.textContent = "Needs attention";
				elements.stats.textContent = "board changed";
				elements.move.textContent = "Could not read this position";
				elements.reason.textContent = message;
				setRoutes(null, null);
				setAlternatives([]);
			},
			setAutoplay(state, message) {
				const enabled = state === "armed" || state === "playing";
				elements.autoplay.setAttribute("aria-pressed", String(enabled));
				elements.autoplay.dataset.state = state;
				elements.autoplay.textContent =
					state === "playing"
						? "Autoplay · Moving"
						: state === "armed"
							? "Autoplay · Armed"
							: state === "blocked"
								? "Autoplay · Paused"
								: "Autoplay · Off";
				elements.autoplay.title =
					message ||
					(enabled
						? "Play fresh engine recommendations automatically"
						: "Autoplay is off");
			},
			setLossExportAvailable(available, message) {
				elements.exportLoss.disabled = !available;
				elements.exportLoss.textContent = available
					? "Export latest loss · JSON"
					: "Export latest loss";
				elements.exportLoss.title =
					message ||
					(available
						? "Download the local diagnostic report to attach in Codex"
						: "A local diagnostic report becomes available after a loss");
			},
			setGameOver(won) {
				setTone(won ? "ready" : "error");
				elements.status.textContent = won ? "Game won" : "Loss recorded";
				elements.stats.textContent = won ? "finished" : "report ready";
				elements.move.textContent = won
					? "Nice conversion"
					: "Export this loss";
				elements.reason.textContent = won
					? "The next game will start a fresh adaptive log."
					: "The report includes every observed position, move, recommendation, and search statistic.";
				setAlternatives(
					won ? ["New game ready"] : ["Attach the JSON in Codex"],
				);
			},
			setResult(result, reason) {
				setTone("ready");
				elements.status.textContent =
					result.engine === "wallzero"
						? "WallZero engine"
						: result.exactEndgame
							? "Exact endgame"
							: "Best practical move";
				elements.stats.textContent =
					result.engine === "wallzero"
						? `${result.simulations.toLocaleString()} sims · ${result.elapsedMs}ms · local server`
						: result.exactEndgame
							? `perfect race · ${result.elapsedMs}ms`
							: "d" +
								result.depth +
								" · " +
								result.nodes.toLocaleString() +
								" nodes · " +
								result.elapsedMs +
								"ms";
				elements.move.textContent = formatMove(result.bestMove);
				elements.reason.textContent = reason;
				setRoutes(result.ownDistance, result.opponentDistance);
				setAlternatives(
					result.alternatives.map((candidate) => formatMove(candidate.move)),
				);
			},
			setWaiting(ownDistance, theirDistance) {
				setTone("idle");
				elements.status.textContent = "Opponent thinking";
				elements.stats.textContent = "standing by";
				elements.move.textContent = "Waiting for your turn";
				elements.reason.textContent =
					"The recommendation will refresh after your opponent moves.";
				setRoutes(ownDistance, theirDistance);
				setAlternatives(["Position will update automatically"]);
			},
			toggle() {
				const hiding = host.style.display !== "none";
				host.style.display = hiding ? "none" : "block";
				if (hiding) {
					actions.onHide();
				}
			},
		};
	}

	function createAnalysisWorker() {
		if (
			typeof Worker === "undefined" ||
			typeof Blob === "undefined" ||
			typeof URL === "undefined"
		) {
			return null;
		}

		const source = [
			'"use strict";',
			"const BOARD_SIZE = 9;",
			"const WALL_SLOTS = 8;",
			`const createEngine = ${createEngine.toString()};`,
			"const engine = createEngine();",
			"self.onmessage = function(event) {",
			"  try {",
			"    const result = engine.analyze(event.data.state, event.data.player, event.data.options);",
			"    self.postMessage({ ok: true, requestId: event.data.requestId, result: result });",
			"  } catch (error) {",
			"    self.postMessage({ ok: false, requestId: event.data.requestId, error: String(error && error.message || error) });",
			"  }",
			"};",
		].join("\n");
		const url = URL.createObjectURL(
			new Blob([source], { type: "text/javascript" }),
		);
		return { busy: false, worker: new Worker(url), url };
	}

	function describeRecommendation(engine, state, player, result) {
		const move = result.bestMove;
		if (!move) {
			return "No legal move was found.";
		}
		const applied = engine.applyMove(state, move);
		if (!applied.ok) {
			return "The board changed; recalculate before playing.";
		}

		const other = engine.opponent(player);
		const nextState = applied.state;
		const nextOwnDistance = engine.shortestDistance(nextState, player);
		const nextOtherDistance = engine.shortestDistance(nextState, other);

		if (move.type === "wall") {
			const opponentDelay = nextOtherDistance - result.opponentDistance;
			const selfDelay = nextOwnDistance - result.ownDistance;
			if (opponentDelay > 0 && selfDelay <= 0) {
				return (
					"Adds " +
					opponentDelay +
					(opponentDelay === 1 ? " move" : " moves") +
					" to their route without lengthening yours."
				);
			}
			if (opponentDelay > selfDelay) {
				return (
					"Trades " +
					Math.max(0, selfDelay) +
					" of your path for " +
					opponentDelay +
					" of theirs."
				);
			}
			return "Cuts their preferred lane while preserving the best race tempo.";
		}

		const from = state.pawns[player];
		const travel = Math.abs(move.to.x - from.x) + Math.abs(move.to.y - from.y);
		const lead = nextOtherDistance - nextOwnDistance;
		if (travel > 1) {
			return "Use the jump and gain two squares in a single turn.";
		}
		if (lead > 0) {
			const currentLead = result.opponentDistance - result.ownDistance;
			const change = lead > currentLead ? "create" : "keep";
			return `Advance on the shortest route and ${change} a ${lead}-move lead.`;
		}
		if (lead === 0) {
			return "Advance on the shortest route and keep the race level.";
		}
		return "Advance now; spending a wall here loses more tempo.";
	}

	function stateFingerprint(state, bottomPlayer) {
		return [
			bottomPlayer,
			state.turn,
			state.winner || "-",
			state.pawns.p1.x,
			state.pawns.p1.y,
			state.pawns.p2.x,
			state.pawns.p2.y,
			state.wallsRemaining.p1,
			state.wallsRemaining.p2,
			state.walls
				.map((wall) => `${wall.o}-${wall.x}-${wall.y}`)
				.sort()
				.join(","),
		].join("|");
	}

	function cloneGameState(state) {
		return {
			pawns: {
				p1: { ...state.pawns.p1 },
				p2: { ...state.pawns.p2 },
			},
			turn: state.turn,
			walls: state.walls.map((wall) => ({ ...wall })),
			wallsRemaining: { ...state.wallsRemaining },
			winner: state.winner || null,
		};
	}

	function inferredTransitions(previous, current) {
		if (!previous || !current) {
			return [];
		}
		const actors = [previous.turn, previous.turn === "p1" ? "p2" : "p1"];
		const actions = { p1: [], p2: [] };
		for (const actor of actors) {
			if (
				previous.pawns[actor].x !== current.pawns[actor].x ||
				previous.pawns[actor].y !== current.pawns[actor].y
			) {
				actions[actor].push({
					actor,
					move: { type: "pawn", to: { ...current.pawns[actor] } },
				});
			}
		}
		const previousWalls = new Set(
			previous.walls.map((wall) => `${wall.o}-${wall.x}-${wall.y}`),
		);
		const addedWalls = current.walls.filter(
			(wall) => !previousWalls.has(`${wall.o}-${wall.x}-${wall.y}`),
		);
		const wallActors = [];
		for (const actor of actors) {
			const count = Math.max(
				0,
				previous.wallsRemaining[actor] - current.wallsRemaining[actor],
			);
			for (let index = 0; index < count; index += 1) {
				wallActors.push(actor);
			}
		}
		for (let index = 0; index < addedWalls.length; index += 1) {
			const actor = wallActors[index] ?? actors[index % actors.length];
			actions[actor].push({
				actor,
				move: { type: "wall", wall: { ...addedWalls[index] } },
			});
		}
		return actors.flatMap((actor) => actions[actor]);
	}

	function diagnosticAnalysis(plan, result) {
		return {
			plan: {
				maxDepth: plan.maxDepth,
				profile: plan.profile,
				reason: plan.reason,
				timeMs: plan.timeMs,
			},
			result: {
				alternatives: result.alternatives,
				aspirationResearches: result.aspirationResearches,
				bestMove: result.bestMove,
				bfsCalls: result.bfsCalls,
				depth: result.depth,
				elapsedMs: result.elapsedMs,
				exactEndgame: result.exactEndgame,
				lazyEvaluations: result.lazyEvaluations,
				lmrReductions: result.lmrReductions,
				lmrResearches: result.lmrResearches,
				moveCacheHits: result.moveCacheHits,
				nodes: result.nodes,
				opponentDistance: result.opponentDistance,
				ownDistance: result.ownDistance,
				pawnHistoryPenalties: result.pawnHistoryPenalties,
				quiescenceNodes: result.quiescenceNodes,
				repetitionPrunes: result.repetitionPrunes,
				reverseFutilityPrunes: result.reverseFutilityPrunes,
				score: result.score,
				stagedTtCutoffs: result.stagedTtCutoffs,
				stagedTtMoves: result.stagedTtMoves,
				ttHits: result.ttHits,
				wallReservePenalties: result.wallReservePenalties,
			},
		};
	}

	function createGameRecorder(engine, options = {}) {
		const now = options.now || (() => Date.now());
		const onLoss = options.onLoss || (() => {});
		let current = null;
		let latestLoss = null;

		function begin(parsed) {
			const startedAt = new Date(now()).toISOString();
			current = {
				bottomPlayer: parsed.bottomPlayer,
				engine: {
					maxDepth: ADAPTIVE_SEARCH_LIMIT.maxDepth,
					timeLimitMs: ADAPTIVE_SEARCH_LIMIT.timeMs,
				},
				gameId: `${startedAt}-${parsed.bottomPlayer}`,
				opponent: "computer",
				positions: [],
				schema: "wallz-loss-report-v2",
				scriptVersion: SCRIPT_VERSION,
				startedAt,
			};
		}

		function finalize(winner) {
			if (!current || current.result || (winner !== "p1" && winner !== "p2")) {
				return null;
			}
			current.endedAt = new Date(now()).toISOString();
			current.result = winner === current.bottomPlayer ? "win" : "loss";
			current.winner = winner;
			if (current.result !== "loss") {
				return null;
			}
			latestLoss = JSON.parse(JSON.stringify(current));
			onLoss(latestLoss);
			return latestLoss;
		}

		function observe(parsed) {
			const fingerprint = stateFingerprint(parsed.state, parsed.bottomPlayer);
			if (
				!current ||
				current.bottomPlayer !== parsed.bottomPlayer ||
				(current.result && parsed.state.winner === null)
			) {
				begin(parsed);
			}
			const previousEntry = current.positions.at(-1);
			if (previousEntry?.fingerprint === fingerprint) {
				return { fingerprint, lossReport: null, newPosition: false };
			}
			const transitions = inferredTransitions(
				previousEntry?.state,
				parsed.state,
			);
			for (const transition of transitions) {
				if (
					transition.actor === current.bottomPlayer &&
					previousEntry?.analysis?.result.bestMove
				) {
					transition.followedRecommendation =
						engine.moveKey(transition.move) ===
						engine.moveKey(previousEntry.analysis.result.bestMove);
				}
			}
			const repeatedPosition = current.positions.some(
				(entry) => entry.fingerprint === fingerprint,
			);
			const other = engine.opponent(parsed.bottomPlayer);
			const entry = {
				fingerprint,
				index: current.positions.length,
				metrics: {
					opponentDistance: engine.shortestDistance(parsed.state, other),
					ownDistance: engine.shortestDistance(
						parsed.state,
						parsed.bottomPlayer,
					),
				},
				observedAt: new Date(now()).toISOString(),
				repeatedPosition,
				state: cloneGameState(parsed.state),
				transition: transitions[0] ?? null,
				transitions,
			};
			current.positions.push(entry);

			const lossReport = parsed.state.winner
				? finalize(parsed.state.winner)
				: null;
			return { fingerprint, lossReport, newPosition: true };
		}

		function latestEntry(fingerprint) {
			for (
				let index = (current?.positions.length ?? 0) - 1;
				index >= 0;
				index -= 1
			) {
				if (current.positions[index].fingerprint === fingerprint) {
					return current.positions[index];
				}
			}
			return null;
		}

		function recordAnalysis(fingerprint, plan, result) {
			const entry = latestEntry(fingerprint);
			if (entry) {
				entry.analysis = diagnosticAnalysis(plan, result);
			}
		}

		function markAutoplay(fingerprint, move) {
			const entry = latestEntry(fingerprint);
			if (entry) {
				entry.autoplay = { attempted: true, move };
			}
		}

		function recentStateHashes(limit = 24) {
			return (current?.positions ?? [])
				.slice(-limit)
				.map((entry) => engine.stateHash(entry.state).toString());
		}

		function recentPawnSquares(player, limit = 12) {
			const squares = [];
			let previous = -1;
			for (const entry of current?.positions ?? []) {
				const pawn = entry.state.pawns[player];
				const code = pawn.y * BOARD_SIZE + pawn.x;
				if (code !== previous) {
					squares.push(code);
					previous = code;
				}
			}
			return squares.slice(-(limit + 1), -1);
		}

		return {
			finalize,
			getCurrent: () => current,
			getLatestLoss: () => latestLoss,
			markAutoplay,
			observe,
			recentPawnSquares,
			recentStateHashes,
			recordAnalysis,
		};
	}

	function loadLossReports(storage) {
		try {
			const parsed = JSON.parse(storage?.getItem(LOSS_LOG_STORAGE_KEY) || "[]");
			return Array.isArray(parsed) ? parsed : [];
		} catch {
			return [];
		}
	}

	function persistLossReport(storage, report) {
		const reports = [report, ...loadLossReports(storage)].slice(
			0,
			MAX_STORED_LOSSES,
		);
		try {
			storage?.setItem(LOSS_LOG_STORAGE_KEY, JSON.stringify(reports));
		} catch {
			// Export still works from memory when browser storage is unavailable.
		}
		return reports;
	}

	function downloadLossReport(report) {
		if (!report || typeof document === "undefined") {
			return false;
		}
		const blob = new Blob([JSON.stringify(report, null, 2)], {
			type: "application/json",
		});
		const url = URL.createObjectURL(blob);
		const link = document.createElement("a");
		link.href = url;
		link.download = `wallz-loss-${report.gameId.replaceAll(":", "-")}.json`;
		document.body.append(link);
		link.click();
		link.remove();
		URL.revokeObjectURL(url);
		return true;
	}

	function recommendationIsCurrent(engine, parsed, expectedFingerprint, move) {
		return Boolean(
			parsed &&
				move &&
				parsed.state.turn === parsed.bottomPlayer &&
				stateFingerprint(parsed.state, parsed.bottomPlayer) ===
					expectedFingerprint &&
				engine.applyMove(parsed.state, move).ok,
		);
	}

	const WALLZERO_REQUEST_SCHEMA = "wallzero.analyze.v1";
	const WALLZERO_RESPONSE_SCHEMA = "wallzero.analysis.v1";
	const WALLZERO_HEALTH_SCHEMA = "wallzero.health.v1";

	function wallZeroMoveShapeIsValid(move) {
		if (!move || typeof move !== "object") {
			return false;
		}
		if (move.type === "pawn") {
			return (
				Number.isInteger(move.to?.x) &&
				Number.isInteger(move.to?.y) &&
				move.to.x >= 0 &&
				move.to.x < BOARD_SIZE &&
				move.to.y >= 0 &&
				move.to.y < BOARD_SIZE
			);
		}
		if (move.type === "wall") {
			return (
				(move.wall?.o === "h" || move.wall?.o === "v") &&
				Number.isInteger(move.wall?.x) &&
				Number.isInteger(move.wall?.y) &&
				move.wall.x >= 0 &&
				move.wall.x < WALL_SLOTS &&
				move.wall.y >= 0 &&
				move.wall.y < WALL_SLOTS
			);
		}
		return false;
	}

	function createWallZeroClient(options = {}) {
		const endpoint = (options.endpoint || "http://127.0.0.1:8787").replace(
			/\/$/,
			"",
		);
		const fetchImpl =
			options.fetch ||
			(typeof fetch === "function" ? fetch.bind(globalThis) : null);
		const timeoutMs = options.timeoutMs ?? 2_500;
		const healthTimeoutMs = options.healthTimeoutMs ?? 900;
		const simulations = options.simulations ?? 800;

		async function fetchJson(url, init, budgetMs) {
			if (!fetchImpl) {
				throw new Error("wallzero-offline: fetch is unavailable");
			}
			const controller =
				typeof AbortController === "function" ? new AbortController() : null;
			const timer = controller
				? setTimeout(() => controller.abort(), budgetMs)
				: null;
			try {
				const reply = await fetchImpl(
					url,
					controller ? { ...init, signal: controller.signal } : init,
				);
				if (!reply.ok) {
					throw new Error(`wallzero-offline: HTTP ${reply.status}`);
				}
				return await reply.json();
			} catch (error) {
				if (error?.name === "AbortError") {
					throw new Error(`wallzero-timeout: no reply within ${budgetMs}ms`);
				}
				throw error;
			} finally {
				if (timer !== null) {
					clearTimeout(timer);
				}
			}
		}

		async function health() {
			const payload = await fetchJson(
				`${endpoint}/health`,
				{ method: "GET" },
				healthTimeoutMs,
			);
			return (
				payload?.schema === WALLZERO_HEALTH_SCHEMA && payload?.status === "ok"
			);
		}

		async function analyze(state, fingerprint) {
			const payload = await fetchJson(
				`${endpoint}/analyze`,
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						schema: WALLZERO_REQUEST_SCHEMA,
						id: fingerprint,
						state,
						options: { simulations, topMoves: 4 },
					}),
				},
				timeoutMs,
			);
			if (payload?.error) {
				throw new Error(
					`wallzero-protocol: ${payload.error.message || payload.error.type}`,
				);
			}
			if (payload?.schema !== WALLZERO_RESPONSE_SCHEMA) {
				throw new Error("wallzero-protocol: unexpected response schema");
			}
			if (payload.id !== fingerprint) {
				throw new Error(
					"wallzero-stale: reply does not match the requested position",
				);
			}
			if (!wallZeroMoveShapeIsValid(payload.move)) {
				throw new Error("wallzero-protocol: malformed move");
			}
			return payload;
		}

		return { analyze, endpoint, health };
	}

	function createWallZeroBridge(client) {
		let enabled = false;
		let lastError = "";

		return {
			get enabled() {
				return enabled;
			},
			get lastError() {
				return lastError;
			},
			async probe() {
				try {
					enabled = await client.health();
					lastError = enabled ? "" : "wallzero-offline: health check failed";
				} catch (error) {
					enabled = false;
					lastError = String(error?.message || error);
				}
				return enabled;
			},
			reportOutage(error) {
				enabled = false;
				lastError = String(error?.message || error);
			},
		};
	}

	function moveTargetPoint(move) {
		if (move?.type === "pawn") {
			return {
				x: move.to.x * BOARD_STEP + 30,
				y: move.to.y * BOARD_STEP + 30,
			};
		}
		if (move?.type === "wall") {
			return {
				x: move.wall.x * BOARD_STEP + 66,
				y: move.wall.y * BOARD_STEP + 66,
			};
		}
		return null;
	}

	function orientBoardPoint(point, rotated) {
		return rotated
			? { x: BOARD_PIXELS - point.x, y: BOARD_PIXELS - point.y }
			: { x: point.x, y: point.y };
	}

	function boardClientPoint(svg, point, rotated) {
		const matrix = svg.getScreenCTM?.();
		if (!matrix || typeof svg.createSVGPoint !== "function") {
			return null;
		}
		const oriented = orientBoardPoint(point, rotated);
		const svgPoint = svg.createSVGPoint();
		svgPoint.x = oriented.x;
		svgPoint.y = oriented.y;
		const clientPoint = svgPoint.matrixTransform(matrix);
		if (!Number.isFinite(clientPoint.x) || !Number.isFinite(clientPoint.y)) {
			return null;
		}
		return { x: clientPoint.x, y: clientPoint.y };
	}

	function emitPointer(target, type, point, pointerId, buttons) {
		if (typeof PointerEvent !== "function") {
			throw new Error("This browser does not support pointer events.");
		}
		target.dispatchEvent(
			new PointerEvent(type, {
				bubbles: true,
				button: 0,
				buttons,
				cancelable: true,
				clientX: point.x,
				clientY: point.y,
				composed: true,
				isPrimary: true,
				pointerId,
				pointerType: "mouse",
				view: window,
			}),
		);
	}

	function waitForInterface() {
		return new Promise((resolve) => window.setTimeout(resolve, 24));
	}

	function bypassSyntheticPointerCapture(element) {
		const replacements = {
			hasPointerCapture: () => true,
			releasePointerCapture: () => {},
			setPointerCapture: () => {},
		};
		const originals = new Map();

		for (const [name, replacement] of Object.entries(replacements)) {
			originals.set(name, {
				descriptor: Object.getOwnPropertyDescriptor(element, name),
				hadOwnProperty: Object.hasOwn(element, name),
			});
			Object.defineProperty(element, name, {
				configurable: true,
				value: replacement,
			});
		}

		return () => {
			for (const [name, original] of originals) {
				if (original.hadOwnProperty) {
					Object.defineProperty(element, name, original.descriptor);
				} else {
					delete element[name];
				}
			}
		};
	}

	function findWallDragButton(orientation) {
		const pattern =
			orientation === "h" ? /drag a horizontal wall/i : /drag a vertical wall/i;
		const buttons = /** @type {NodeListOf<HTMLButtonElement>} */ (
			document.querySelectorAll("button[aria-label]")
		);
		return Array.from(buttons).find((button) =>
			pattern.test(button.getAttribute("aria-label") || ""),
		);
	}

	async function playMoveOnBoard(svg, move, rotated, pointerId) {
		const target = moveTargetPoint(move);
		const clientTarget = target ? boardClientPoint(svg, target, rotated) : null;
		if (!clientTarget) {
			throw new Error("The board moved before autoplay could aim.");
		}

		if (move.type === "pawn") {
			emitPointer(svg, "pointerdown", clientTarget, pointerId, 1);
			emitPointer(svg, "pointerup", clientTarget, pointerId, 0);
			return;
		}

		const button = findWallDragButton(move.wall.o);
		if (!button || button.disabled) {
			throw new Error("The matching wall tray control is not available.");
		}
		const bounds = button.getBoundingClientRect();
		const source = {
			x: bounds.left + bounds.width / 2,
			y: bounds.top + bounds.height / 2,
		};
		const restorePointerCapture = bypassSyntheticPointerCapture(button);

		try {
			emitPointer(button, "pointerdown", source, pointerId, 1);
			await waitForInterface();
			emitPointer(button, "pointermove", clientTarget, pointerId, 1);
			await waitForInterface();
			emitPointer(button, "pointermove", clientTarget, pointerId, 1);
			await waitForInterface();
			emitPointer(button, "pointerup", clientTarget, pointerId, 0);
		} catch (error) {
			emitPointer(button, "pointercancel", clientTarget, pointerId, 0);
			throw error;
		} finally {
			restorePointerCapture();
		}
	}

	function boot() {
		if (
			window.top !== window.self ||
			document.querySelector("#wallz-practice-coach")
		) {
			return;
		}

		const engine = createEngine();
		const wallZeroClient = createWallZeroClient();
		const wallZeroBridge = createWallZeroBridge(wallZeroClient);
		let latestSvg = null;
		let lastFingerprint = "";
		let analysisRequest = 0;
		let analysisWorker = null;
		let inspectTimer = null;
		let autoplayEnabled = false;
		let autoplayTimer = null;
		let autoplayConfirmationTimer = null;
		let latestRecommendation = null;
		let lastAutoPlayedFingerprint = "";
		let nextPointerId = 7_000;
		let panel = null;
		let browserStorage = null;
		try {
			browserStorage = window.localStorage;
		} catch {
			// In-memory export remains available when storage is blocked.
		}
		let latestLossReport = loadLossReports(browserStorage)[0] || null;
		const gameRecorder = createGameRecorder(engine, {
			onLoss(report) {
				latestLossReport = report;
				persistLossReport(browserStorage, report);
				panel?.setLossExportAvailable(
					true,
					"Loss recorded locally. Export the JSON and attach it in Codex.",
				);
			},
		});

		function disposeWorker() {
			if (!analysisWorker) {
				return;
			}
			analysisWorker.worker.terminate();
			URL.revokeObjectURL(analysisWorker.url);
			analysisWorker = null;
		}

		function cancelPendingAutoplay() {
			if (autoplayTimer !== null) {
				window.clearTimeout(autoplayTimer);
				autoplayTimer = null;
			}
			if (autoplayConfirmationTimer !== null) {
				window.clearTimeout(autoplayConfirmationTimer);
				autoplayConfirmationTimer = null;
			}
			if (autoplayEnabled) {
				panel?.setAutoplay("armed");
			}
		}

		function disableAutoplay(message, blocked = false) {
			autoplayEnabled = false;
			cancelPendingAutoplay();
			panel?.setAutoplay(blocked ? "blocked" : "off", message);
		}

		function invalidateAnalysis(resetWorker = false) {
			analysisRequest += 1;
			latestRecommendation = null;
			cancelPendingAutoplay();
			if (resetWorker || analysisWorker?.busy) {
				disposeWorker();
			}
		}

		panel = createPanel({
			onExportLoss() {
				if (!downloadLossReport(latestLossReport)) {
					panel.setLossExportAvailable(
						false,
						"No completed loss report is available yet.",
					);
				}
			},
			onRefresh() {
				invalidateAnalysis(true);
				lastFingerprint = "";
				if (WALLZERO_BRIDGE_ENABLED && wallZeroUserEnabled()) {
					void wallZeroBridge.probe().finally(() => scheduleInspect(true));
				} else {
					scheduleInspect(true);
				}
			},
			onAutoplayChange(enabled) {
				if (!enabled) {
					disableAutoplay("Autoplay is off.");
					return;
				}
				autoplayEnabled = true;
				panel.setAutoplay("armed");
				if (latestRecommendation) {
					scheduleAutoplay(latestRecommendation);
				}
			},
			onHide() {
				disableAutoplay("Autoplay was turned off when the coach was hidden.");
			},
			onWallZeroToggle() {
				toggleWallZero();
			},
		});

		function syncWallZeroButton() {
			if (!panel) {
				return;
			}
			if (!WALLZERO_BRIDGE_ENABLED) {
				panel.setWallZeroState("Unavailable", false);
				return;
			}
			if (!wallZeroUserEnabled()) {
				panel.setWallZeroState("Off", false);
				return;
			}
			panel.setWallZeroState(
				wallZeroBridge.enabled ? "Connected" : "On · offline",
				true,
			);
		}

		function toggleWallZero() {
			const next = !wallZeroUserEnabled();
			setWallZeroUserEnabled(next);
			if (next && WALLZERO_BRIDGE_ENABLED) {
				void wallZeroBridge.probe().finally(() => {
					syncWallZeroButton();
					scheduleInspect(true);
				});
			} else {
				wallZeroBridge.reportOutage(
					new Error("wallzero-disabled: toggled off"),
				);
				scheduleInspect(true);
			}
			syncWallZeroButton();
		}
		panel.setLossExportAvailable(Boolean(latestLossReport));

		function scheduleAutoplay(recommendation) {
			cancelPendingAutoplay();
			if (
				!autoplayEnabled ||
				!recommendation ||
				recommendation.fingerprint === lastAutoPlayedFingerprint
			) {
				return;
			}
			autoplayTimer = window.setTimeout(() => {
				autoplayTimer = null;
				void performAutoplay(recommendation);
			}, 240);
		}

		async function performAutoplay(recommendation) {
			if (!autoplayEnabled || recommendation.requestId !== analysisRequest) {
				return;
			}
			if (document.visibilityState !== "visible") {
				disableAutoplay(
					"Autoplay paused because the Wallz tab is not visible.",
					true,
				);
				return;
			}

			const svg = document.querySelector(
				'svg[role="grid"][aria-label="Wallz board"]',
			);
			const parsed = svg ? readBoardState(svg) : null;
			if (
				!svg ||
				!recommendationIsCurrent(
					engine,
					parsed,
					recommendation.fingerprint,
					recommendation.move,
				)
			) {
				latestRecommendation = null;
				lastFingerprint = "";
				scheduleInspect(false);
				return;
			}

			gameRecorder.markAutoplay(
				recommendation.fingerprint,
				recommendation.move,
			);
			lastAutoPlayedFingerprint = recommendation.fingerprint;
			latestRecommendation = null;
			nextPointerId += 1;
			panel.setAutoplay("playing");

			try {
				await playMoveOnBoard(
					svg,
					recommendation.move,
					parsed.bottomPlayer === "p1",
					nextPointerId,
				);
			} catch (error) {
				lastAutoPlayedFingerprint = "";
				disableAutoplay(String(error?.message || error), true);
				return;
			}

			if (!autoplayEnabled) {
				return;
			}
			autoplayConfirmationTimer = window.setTimeout(() => {
				autoplayConfirmationTimer = null;
				const currentSvg = document.querySelector(
					'svg[role="grid"][aria-label="Wallz board"]',
				);
				const current = currentSvg ? readBoardState(currentSvg) : null;
				if (
					current &&
					stateFingerprint(current.state, current.bottomPlayer) ===
						recommendation.fingerprint
				) {
					lastAutoPlayedFingerprint = "";
					disableAutoplay(
						"Wallz did not accept the automatic input, so autoplay paused.",
						true,
					);
					return;
				}
				if (autoplayEnabled) {
					panel.setAutoplay("armed");
				}
			}, 900);
		}

		function finishAnalysis(requestId, svg, result, searchPlan, fingerprint) {
			if (requestId !== analysisRequest) {
				return;
			}
			if (!result.bestMove) {
				clearBoardHint(svg);
				panel.setError("No legal move was available in this position.");
				return;
			}
			const currentSvg = document.querySelector(
				'svg[role="grid"][aria-label="Wallz board"]',
			);
			const current = currentSvg ? readBoardState(currentSvg) : null;
			if (
				currentSvg !== svg ||
				!recommendationIsCurrent(engine, current, fingerprint, result.bestMove)
			) {
				lastFingerprint = "";
				scheduleInspect(false);
				return;
			}
			gameRecorder.recordAnalysis(fingerprint, searchPlan, result);
			const outageNote =
				result.engine !== "wallzero" && wallZeroBridge.lastError
					? " · WallZero offline — built-in engine."
					: "";
			panel.setResult(
				result,
				describeRecommendation(
					engine,
					current.state,
					current.bottomPlayer,
					result,
				) + outageNote,
			);
			renderBoardHint(
				svg,
				result.bestMove,
				current.state,
				current.bottomPlayer === "p1",
			);
			latestRecommendation = {
				fingerprint,
				move: result.bestMove,
				requestId,
			};
			scheduleAutoplay(latestRecommendation);
		}

		function startAnalysis(parsed, svg) {
			invalidateAnalysis(true);
			const requestId = analysisRequest;
			const fingerprint = stateFingerprint(parsed.state, parsed.bottomPlayer);
			const searchPlan = selectAdaptiveSearch(
				engine,
				parsed.state,
				parsed.bottomPlayer,
			);
			const ownDistance = engine.shortestDistance(
				parsed.state,
				parsed.bottomPlayer,
			);
			const theirDistance = engine.shortestDistance(
				parsed.state,
				engine.opponent(parsed.bottomPlayer),
			);
			panel.setAnalyzing(ownDistance, theirDistance, searchPlan);
			clearBoardHint(svg);

			if (wallZeroBridge.enabled) {
				const startedAt = Date.now();
				wallZeroClient
					.analyze(cloneGameState(parsed.state), fingerprint)
					.then((payload) => {
						if (requestId !== analysisRequest) {
							return;
						}
						const move = payload.move;
						if (!engine.applyMove(parsed.state, move).ok) {
							throw new Error(
								"wallzero-illegal: engine rejected the recommended move",
							);
						}
						const bestKey = engine.moveKey(move);
						const alternatives = (payload.topMoves || [])
							.map((entry) => ({ move: entry.move }))
							.filter(
								(entry) =>
									wallZeroMoveShapeIsValid(entry.move) &&
									engine.moveKey(entry.move) !== bestKey &&
									engine.applyMove(parsed.state, entry.move).ok,
							)
							.slice(0, 3);
						finishAnalysis(
							requestId,
							svg,
							{
								alternatives,
								bestMove: move,
								depth: 0,
								elapsedMs: Date.now() - startedAt,
								engine: "wallzero",
								exactEndgame: false,
								nodes: payload.simulations ?? 0,
								opponentDistance: theirDistance,
								ownDistance,
								score: Math.round((payload.value ?? 0) * 1000),
								simulations: payload.simulations ?? 0,
								value: payload.value ?? 0,
							},
							searchPlan,
							fingerprint,
						);
					})
					.catch((error) => {
						if (requestId !== analysisRequest) {
							return;
						}
						wallZeroBridge.reportOutage(error);
						runLocalAnalysis();
					});
				return;
			}

			runLocalAnalysis();

			function runLocalAnalysis() {
				if (!analysisWorker) {
					try {
						analysisWorker = createAnalysisWorker();
					} catch {
						analysisWorker = null;
					}
				}

				if (analysisWorker) {
				const workerHandle = analysisWorker;
				workerHandle.busy = true;
				workerHandle.worker.onmessage = (event) => {
					if (analysisWorker !== workerHandle) {
						return;
					}
					workerHandle.busy = false;
					if (event.data.requestId !== analysisRequest) {
						return;
					}
					if (!event.data.ok) {
						disposeWorker();
						panel.setError(event.data.error || "The analysis worker stopped.");
						return;
					}
					finishAnalysis(
						requestId,
						svg,
						event.data.result,
						searchPlan,
						fingerprint,
					);
				};
				workerHandle.worker.onerror = () => {
					if (analysisWorker !== workerHandle) {
						return;
					}
					if (requestId === analysisRequest) {
						disposeWorker();
						panel.setError("The browser blocked the local analysis worker.");
					}
				};
				workerHandle.worker.postMessage({
					requestId,
					state: parsed.state,
					player: parsed.bottomPlayer,
					options: {
						maxDepth: searchPlan.maxDepth,
						recentPawnSquares: gameRecorder.recentPawnSquares(
							parsed.bottomPlayer,
						),
						recentStateHashes: gameRecorder.recentStateHashes(),
						timeMs: searchPlan.timeMs,
					},
				});
				return;
			}

			window.setTimeout(() => {
				if (requestId !== analysisRequest) {
					return;
				}
				try {
					const result = engine.analyze(parsed.state, parsed.bottomPlayer, {
						timeMs: Math.min(searchPlan.timeMs, 350),
						maxDepth: Math.min(searchPlan.maxDepth, 4),
						recentPawnSquares: gameRecorder.recentPawnSquares(
							parsed.bottomPlayer,
						),
						recentStateHashes: gameRecorder.recentStateHashes(),
					});
					finishAnalysis(requestId, svg, result, searchPlan, fingerprint);
				} catch (error) {
					panel.setError(String(error?.message || error));
				}
			}, 0);
			}
		}

		function inspect(force) {
			inspectTimer = null;
			const svg = document.querySelector(
				'svg[role="grid"][aria-label="Wallz board"]',
			);
			if (!svg) {
				let finishedGame = false;
				if (latestSvg) {
					clearBoardHint(latestSvg);
					const currentGame = gameRecorder.getCurrent();
					const statusText =
						(document.querySelector("main") || document.body)?.textContent ||
						"";
					const winner = currentGame
						? winnerFromStatusText(statusText, currentGame.bottomPlayer)
						: null;
					if (winner) {
						gameRecorder.finalize(winner);
						panel.show();
						panel.setGameOver(winner === currentGame.bottomPlayer);
						finishedGame = true;
					}
				}
				latestSvg = null;
				lastFingerprint = "";
				invalidateAnalysis(true);
				disableAutoplay("Autoplay is off because the match board closed.");
				if (!finishedGame) {
					panel.hide();
				}
				return;
			}

			panel.show();
			latestSvg = svg;

			const parsed = readBoardState(svg);
			if (!parsed) {
				invalidateAnalysis(true);
				if (autoplayEnabled) {
					disableAutoplay(
						"Autoplay paused because the board could not be read safely.",
						true,
					);
				}
				clearBoardHint(svg);
				panel.setError(
					"Wallz rendered an unfamiliar board. Reload the match and try again.",
				);
				return;
			}
			const observation = gameRecorder.observe(parsed);
			const fingerprint = observation.fingerprint;
			if (
				lastAutoPlayedFingerprint &&
				fingerprint !== lastAutoPlayedFingerprint
			) {
				lastAutoPlayedFingerprint = "";
			}
			if (parsed.state.winner) {
				invalidateAnalysis(true);
				clearBoardHint(svg);
				lastFingerprint = fingerprint;
				panel.setGameOver(parsed.state.winner === parsed.bottomPlayer);
				return;
			}
			const theirPlayer = engine.opponent(parsed.bottomPlayer);
			const ownDistance = engine.shortestDistance(
				parsed.state,
				parsed.bottomPlayer,
			);
			const theirDistance = engine.shortestDistance(parsed.state, theirPlayer);

			if (parsed.state.turn !== parsed.bottomPlayer) {
				invalidateAnalysis();
				clearBoardHint(svg);
				lastFingerprint = fingerprint;
				panel.setWaiting(ownDistance, theirDistance);
				return;
			}

			if (!force && fingerprint === lastFingerprint) {
				return;
			}
			lastFingerprint = fingerprint;
			startAnalysis(parsed, svg);
		}

		function scheduleInspect(force) {
			if (inspectTimer !== null) {
				window.clearTimeout(inspectTimer);
			}
			inspectTimer = window.setTimeout(() => inspect(Boolean(force)), 120);
		}

		const observer = new MutationObserver((mutations) => {
			const relevant = mutations.some((mutation) => {
				const target = mutation.target;
				return !(
					target instanceof Element &&
					target.closest("[data-wallz-coach-overlay]")
				);
			});
			if (relevant) {
				scheduleInspect(false);
			}
		});
		observer.observe(document.documentElement, {
			subtree: true,
			childList: true,
			characterData: true,
			attributes: true,
			attributeFilter: [
				"cx",
				"cy",
				"x",
				"y",
				"width",
				"height",
				"fill",
				"stroke",
				"transform",
			],
		});

		window.addEventListener("keydown", (event) => {
			if (event.altKey && event.key.toLowerCase() === "w") {
				event.preventDefault();
				panel.toggle();
			}
		});
		window.addEventListener(
			"beforeunload",
			() => {
				observer.disconnect();
				invalidateAnalysis(true);
			},
			{ once: true },
		);

		if (WALLZERO_BRIDGE_ENABLED && wallZeroUserEnabled()) {
			void wallZeroBridge.probe().finally(() => {
				syncWallZeroButton();
				scheduleInspect(false);
			});
		} else {
			syncWallZeroButton();
			scheduleInspect(false);
		}

		document.addEventListener("keydown", (event) => {
			if (!event.altKey || event.code !== "KeyW") {
				return;
			}
			toggleWallZero();
		});
	}

	return {
		boot,
		coordinate,
		createEngine,
		createGameRecorder,
		createWallZeroBridge,
		createWallZeroClient,
		decodeBoardGeometry,
		formatMove,
		loadLossReports,
		moveTargetPoint,
		orientBoardPoint,
		persistLossReport,
		readBoardState,
		recommendationIsCurrent,
		selectAdaptiveSearch,
		stateFingerprint,
		wallZeroMoveShapeIsValid,
	};
});
