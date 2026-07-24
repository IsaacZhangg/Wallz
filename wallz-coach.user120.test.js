const assert = require("node:assert/strict");
const test = require("node:test");

const {
	createEngine,
	createGameRecorder,
	decodeBoardGeometry,
	loadLossReports,
	moveTargetPoint,
	orientBoardPoint,
	persistLossReport,
	recommendationIsCurrent,
	selectAdaptiveSearch,
	stateFingerprint,
} = require("./wallz-coach.user120.js");

function initialState() {
	return {
		pawns: {
			p1: { x: 4, y: 0 },
			p2: { x: 4, y: 8 },
		},
		turn: "p1",
		walls: [],
		wallsRemaining: { p1: 10, p2: 10 },
		winner: null,
	};
}

function developedState() {
	return {
		pawns: {
			p1: { x: 6, y: 1 },
			p2: { x: 3, y: 6 },
		},
		turn: "p1",
		walls: [
			{ o: "v", x: 2, y: 1 },
			{ o: "v", x: 4, y: 1 },
			{ o: "h", x: 4, y: 4 },
			{ o: "h", x: 1, y: 7 },
			{ o: "v", x: 6, y: 1 },
			{ o: "v", x: 3, y: 0 },
			{ o: "h", x: 7, y: 6 },
			{ o: "h", x: 0, y: 3 },
		],
		wallsRemaining: { p1: 7, p2: 5 },
		winner: null,
	};
}

test("move application round-trips the mutable search position", () => {
	const engine = createEngine();
	const state = initialState();

	assert.equal(
		engine.verifyMoveRoundTrip(state, {
			type: "pawn",
			to: { x: 4, y: 1 },
		}),
		true,
	);
	assert.equal(
		engine.verifyMoveRoundTrip(state, {
			type: "wall",
			wall: { o: "h", x: 3, y: 0 },
		}),
		true,
	);
});

test("the default analysis is deterministic PVS, not a random MCTS pre-pass", () => {
	const engine = createEngine();
	const result = engine.analyze(initialState(), "p1", {
		maxDepth: 1,
		timeMs: 100,
	});

	assert.equal(result.searchMode, "pvs");
	assert.equal(result.mctsSimulations, 0);
	assert.deepEqual(result.bestMove, {
		type: "pawn",
		to: { x: 4, y: 1 },
	});
});

test("Stockfish-inspired staging preserves the fixed-depth result with less pathfinding", () => {
	const options = { maxDepth: 4, timeMs: 2_000, useMcts: false };
	const baseline = createEngine().analyze(developedState(), "p1", {
		...options,
		stockfishFeatures: false,
	});
	const improved = createEngine().analyze(developedState(), "p1", options);

	assert.equal(baseline.depth, 4);
	assert.equal(improved.depth, 4);
	assert.deepEqual(improved.bestMove, baseline.bestMove);
	assert.equal(improved.score, baseline.score);
	assert.ok(improved.stagedTtMoves > 0);
	assert.ok(improved.stagedTtCutoffs > 0);
	assert.ok(improved.reverseFutilityPrunes > 0);
	assert.ok(improved.bfsCalls < baseline.bfsCalls);
});

test("forcing leaf search resolves an immediate goal threat", () => {
	const engine = createEngine();
	const state = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 7 },
			p2: { x: 4, y: 1 },
		},
	};
	const result = engine.analyze(state, "p1", {
		maxDepth: 1,
		timeMs: 500,
	});

	assert.ok(result.quiescenceNodes > 0);
	assert.deepEqual(result.bestMove, {
		type: "pawn",
		to: { x: 4, y: 8 },
	});
	assert.ok(result.score > 900_000);
});

test("the exact pawn-race table converts a former forced-loss recommendation into a win", () => {
	const engine = createEngine();
	const state = {
		pawns: {
			p1: { x: 3, y: 1 },
			p2: { x: 2, y: 7 },
		},
		turn: "p2",
		walls: [
			{ o: "v", x: 2, y: 5 },
			{ o: "v", x: 5, y: 3 },
			{ o: "v", x: 6, y: 6 },
			{ o: "v", x: 1, y: 3 },
			{ o: "v", x: 0, y: 6 },
			{ o: "h", x: 0, y: 4 },
			{ o: "v", x: 1, y: 7 },
			{ o: "h", x: 2, y: 7 },
			{ o: "h", x: 2, y: 2 },
			{ o: "h", x: 7, y: 2 },
		],
		wallsRemaining: { p1: 0, p2: 0 },
		winner: null,
	};
	const result = engine.analyze(state, "p2", {
		maxDepth: 1,
		timeMs: 100,
	});

	assert.equal(result.exactEndgame, true);
	assert.ok(result.score > 0);
	assert.deepEqual(result.bestMove, {
		type: "pawn",
		to: { x: 3, y: 7 },
	});
	assert.ok(result.alternatives[1].score < 0);
});

test("an expired root search still returns a legal pawn fallback", () => {
	const engine = createEngine();
	const state = initialState();
	const result = engine.analyze(state, "p1", {
		maxDepth: 20,
		timeMs: 10,
	});

	assert.ok(result.bestMove);
	assert.equal(engine.applyMove(state, result.bestMove).ok, true);
});

test("forcing defensive walls survive an aggressive quiet-wall cap", () => {
	const engine = createEngine();
	const state = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 7 },
			p2: { x: 4, y: 1 },
		},
	};
	const candidates = engine.inspectCandidateMoves(state, { wallCap: 0 });

	assert.ok(candidates.some((candidate) => candidate.move.type === "wall"));
});

test("a completed search remains stable after its transposition table is warmed", {
	timeout: 10_000,
}, () => {
	const engine = createEngine();
	const state = developedState();
	const options = { maxDepth: 6, timeMs: 4_000, useMcts: false };
	const cold = engine.analyze(state, "p1", options);
	const warm = engine.analyze(state, "p1", options);

	assert.equal(cold.depth, 6);
	assert.equal(warm.depth, 6);
	assert.deepEqual(warm.bestMove, cold.bestMove);
	assert.equal(warm.score, cold.score);
	assert.ok(cold.moveCacheHits > 0);
	assert.ok(cold.lmrReductions > 0);
	assert.ok(engine.cacheInfo().moveGenerationEntries > 0);
	engine.clearSearchCaches();
	assert.equal(engine.cacheInfo().moveGenerationEntries, 0);
});

test("the single adaptive mode caps every search at 1.5 seconds", () => {
	const engine = createEngine();
	const opening = selectAdaptiveSearch(engine, initialState(), "p1");
	assert.equal(opening.id, "adaptive");
	assert.equal(opening.profile, "opening");
	assert.equal(opening.maxDepth, 25);
	assert.equal(opening.timeMs, 500);

	const race = initialState();
	race.wallsRemaining = { p1: 0, p2: 0 };
	const exact = selectAdaptiveSearch(engine, race, "p1");
	assert.equal(exact.id, "adaptive");
	assert.equal(exact.profile, "exact");
	assert.equal(exact.timeMs, 100);
	assert.match(exact.reason, /exact/i);
});

test("adaptive search only shortens a conversion after the opponent runs out of walls", () => {
	const engine = createEngine();
	const critical = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 4 },
			p2: { x: 4, y: 1 },
		},
	};
	const contestedLead = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 5 },
			p2: { x: 4, y: 7 },
		},
		wallsRemaining: { p1: 10, p2: 1 },
	};
	const safeConversion = {
		...contestedLead,
		wallsRemaining: { p1: 10, p2: 0 },
	};

	const tactical = selectAdaptiveSearch(engine, critical, "p1");
	assert.equal(tactical.profile, "tactical");
	assert.equal(tactical.maxDepth, 25);
	assert.equal(tactical.timeMs, 1_500);

	const contested = selectAdaptiveSearch(engine, contestedLead, "p1");
	assert.notEqual(contested.profile, "conversion");
	assert.equal(contested.timeMs, 1_500);

	const race = selectAdaptiveSearch(engine, safeConversion, "p1");
	assert.equal(race.profile, "conversion");
	assert.equal(race.timeMs, 350);
	assert.ok(tactical.timeMs <= 1_500);
	assert.ok(race.timeMs <= 1_500);
});

test("recent game history prevents the engine from cycling back to the same position", () => {
	const engine = createEngine();
	const state = initialState();
	const repeatedMove = { type: "pawn", to: { x: 4, y: 1 } };
	const repeatedState = engine.applyMove(state, repeatedMove).state;
	const result = engine.analyze(state, "p1", {
		maxDepth: 2,
		recentStateHashes: [engine.stateHash(repeatedState).toString()],
		timeMs: 1_000,
	});

	assert.notDeepEqual(result.bestMove, repeatedMove);
	assert.ok(result.repetitionPrunes > 0);
});

test("pawn-square history discourages backtracking without hiding a forced win", () => {
	const engine = createEngine();
	const result = engine.analyze(initialState(), "p1", {
		maxDepth: 2,
		recentPawnSquares: [13],
		timeMs: 1_000,
	});

	assert.notDeepEqual(result.bestMove, {
		type: "pawn",
		to: { x: 4, y: 1 },
	});
	assert.ok(result.pawnHistoryPenalties > 0);

	const winningState = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 7 },
			p2: { x: 3, y: 3 },
		},
	};
	assert.deepEqual(
		engine.analyze(winningState, "p1", {
			maxDepth: 2,
			recentPawnSquares: [76, 76, 76],
			timeMs: 1_000,
		}).bestMove,
		{ type: "pawn", to: { x: 4, y: 8 } },
	);
});

test("jump tempo, urgent defense, and secure races are handled tactically", () => {
	const engine = createEngine();
	const jump = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 3 },
			p2: { x: 4, y: 4 },
		},
	};
	assert.deepEqual(
		engine.analyze(jump, "p1", { maxDepth: 3, timeMs: 1_000 }).bestMove,
		{ type: "pawn", to: { x: 4, y: 5 } },
	);

	const defense = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 4 },
			p2: { x: 4, y: 1 },
		},
	};
	const defensiveResult = engine.analyze(defense, "p1", {
		maxDepth: 3,
		timeMs: 1_000,
	});
	assert.equal(defensiveResult.bestMove.type, "wall");
	const defended = engine.applyMove(defense, defensiveResult.bestMove).state;
	assert.equal(
		engine
			.legalPawnMoves(defended, "p2")
			.some((move) => move.y === engine.goalRow("p2")),
		false,
	);

	const secureRace = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 5 },
			p2: { x: 4, y: 7 },
		},
		wallsRemaining: { p1: 10, p2: 1 },
	};
	assert.deepEqual(
		engine.analyze(secureRace, "p1", { maxDepth: 3, timeMs: 1_000 }).bestMove,
		{ type: "pawn", to: { x: 4, y: 6 } },
	);
});

test("the opening move order conserves walls that do not buy enough tempo", () => {
	const candidates = createEngine().inspectCandidateMoves(initialState(), {
		rootNode: true,
	});

	assert.equal(candidates[0].move.type, "pawn");
	assert.deepEqual(candidates[0].move.to, { x: 4, y: 1 });
});

test("a trailing player spends a self-safe wall to recover tempo", {
	timeout: 5_000,
}, () => {
	const engine = createEngine();
	const state = {
		pawns: {
			p1: { x: 4, y: 3 },
			p2: { x: 4, y: 5 },
		},
		turn: "p2",
		walls: [{ o: "h", x: 4, y: 2 }],
		wallsRemaining: { p1: 9, p2: 10 },
		winner: null,
	};
	const result = engine.analyze(state, "p2", {
		maxDepth: 6,
		timeMs: 2_000,
		useMcts: false,
	});

	assert.equal(result.depth, 6);
	assert.equal(result.bestMove.type, "wall");
	const next = engine.applyMove(state, result.bestMove);
	assert.equal(next.ok, true);
	assert.equal(
		engine.shortestDistance(next.state, "p2"),
		engine.shortestDistance(state, "p2"),
	);
	assert.equal(
		engine.shortestDistance(next.state, "p1"),
		engine.shortestDistance(state, "p1") + 1,
	);
});

test("loss-report regressions reject walls that waste tempo or damage the engine's lane", () => {
	const engine = createEngine();
	const cases = [
		{
			state: {
				...initialState(),
				pawns: {
					p1: { x: 4, y: 3 },
					p2: { x: 4, y: 5 },
				},
			},
			wall: { o: "h", x: 4, y: 2 },
		},
		{
			state: {
				pawns: {
					p1: { x: 4, y: 2 },
					p2: { x: 4, y: 7 },
				},
				turn: "p2",
				walls: [
					{ o: "h", x: 3, y: 2 },
					{ o: "h", x: 4, y: 6 },
				],
				wallsRemaining: { p1: 9, p2: 9 },
				winner: null,
			},
			wall: { o: "v", x: 2, y: 1 },
		},
		{
			state: {
				pawns: {
					p1: { x: 7, y: 1 },
					p2: { x: 4, y: 7 },
				},
				turn: "p2",
				walls: [
					{ o: "h", x: 3, y: 2 },
					{ o: "h", x: 4, y: 6 },
					{ o: "v", x: 2, y: 1 },
					{ o: "h", x: 2, y: 6 },
					{ o: "h", x: 5, y: 2 },
					{ o: "h", x: 7, y: 2 },
					{ o: "h", x: 3, y: 0 },
					{ o: "h", x: 5, y: 0 },
					{ o: "h", x: 6, y: 6 },
					{ o: "h", x: 1, y: 0 },
					{ o: "h", x: 5, y: 7 },
					{ o: "h", x: 7, y: 7 },
				],
				wallsRemaining: { p1: 7, p2: 1 },
				winner: null,
			},
			wall: { o: "h", x: 0, y: 6 },
		},
	];

	for (const { state, wall } of cases) {
		const candidates = engine.inspectCandidateMoves(state, { rootNode: true });
		assert.equal(
			candidates.some(
				(candidate) =>
					candidate.move.type === "wall" &&
					engine.wallKey(candidate.move.wall) === engine.wallKey(wall),
			),
			false,
		);
	}
});

test("a completed loss records positions, recommendations, and local exports", () => {
	const engine = createEngine();
	let clock = Date.UTC(2026, 6, 22, 12, 0, 0);
	let emitted = null;
	const recorder = createGameRecorder(engine, {
		now: () => {
			clock += 1_000;
			return clock;
		},
		onLoss: (report) => {
			emitted = report;
		},
	});
	const before = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 4 },
			p2: { x: 4, y: 1 },
		},
		turn: "p2",
	};
	const first = recorder.observe({ bottomPlayer: "p1", state: before });
	recorder.recordAnalysis(
		first.fingerprint,
		selectAdaptiveSearch(engine, before, "p1"),
		engine.analyze(before, "p1", { maxDepth: 1, timeMs: 200 }),
	);
	const lost = {
		...before,
		pawns: { ...before.pawns, p2: { x: 4, y: 0 } },
		turn: "p1",
		winner: "p2",
	};
	const terminal = recorder.observe({ bottomPlayer: "p1", state: lost });

	assert.equal(terminal.lossReport, emitted);
	assert.equal(emitted.schema, "wallz-loss-report-v2");
	assert.equal(emitted.result, "loss");
	assert.equal(emitted.positions.length, 2);
	assert.deepEqual(emitted.positions[1].transition, {
		actor: "p2",
		move: { type: "pawn", to: { x: 4, y: 0 } },
	});
	assert.deepEqual(emitted.positions[1].transitions, [
		emitted.positions[1].transition,
	]);
	assert.ok(emitted.positions[0].analysis.result.bestMove);

	const values = new Map();
	const storage = {
		getItem: (key) => values.get(key) ?? null,
		setItem: (key, value) => values.set(key, value),
	};
	persistLossReport(storage, emitted);
	assert.deepEqual(loadLossReports(storage), [emitted]);
});

test("loss reports preserve both moves when one DOM observation spans a full turn", () => {
	const recorder = createGameRecorder(createEngine());
	recorder.observe({ bottomPlayer: "p1", state: initialState() });
	recorder.observe({
		bottomPlayer: "p1",
		state: {
			...initialState(),
			pawns: {
				p1: { x: 4, y: 1 },
				p2: { x: 4, y: 7 },
			},
		},
	});

	assert.deepEqual(recorder.getCurrent().positions[1].transitions, [
		{ actor: "p1", move: { type: "pawn", to: { x: 4, y: 1 } } },
		{ actor: "p2", move: { type: "pawn", to: { x: 4, y: 7 } } },
	]);
});

test("a loss can still be finalized after Wallz removes the board", () => {
	const engine = createEngine();
	let emitted = null;
	const recorder = createGameRecorder(engine, {
		onLoss: (report) => {
			emitted = report;
		},
	});
	const state = {
		...initialState(),
		pawns: {
			p1: { x: 4, y: 4 },
			p2: { x: 4, y: 1 },
		},
		turn: "p2",
	};
	recorder.observe({ bottomPlayer: "p1", state });

	const report = recorder.finalize("p2");
	assert.equal(report, emitted);
	assert.equal(report.result, "loss");
	assert.equal(report.positions.length, 1);
	assert.equal(recorder.finalize("p2"), null);
});

test("board status text identifies a loss before exporting the report", () => {
	const geometry = {
		circles: [
			{ cx: 318, cy: 318, fill: "var(--color-p1)", r: 20 },
			{ cx: 318, cy: 102, fill: "var(--color-p2)", r: 20 },
			{ r: 22, stroke: "var(--color-p1)" },
		],
		rects: [],
		rotated: true,
		statusText: "Defeat — computer wins",
		wallCounts: [10, 10],
	};

	const parsed = decodeBoardGeometry(geometry);
	assert.equal(parsed.bottomPlayer, "p1");
	assert.equal(parsed.state.winner, "p2");
});

test("autoplay only accepts a legal recommendation for the exact live turn", () => {
	const engine = createEngine();
	const state = initialState();
	const parsed = { bottomPlayer: "p1", state };
	const fingerprint = stateFingerprint(state, "p1");
	const move = { type: "pawn", to: { x: 4, y: 1 } };

	assert.equal(
		recommendationIsCurrent(engine, parsed, fingerprint, move),
		true,
	);
	assert.equal(
		recommendationIsCurrent(engine, parsed, `${fingerprint}-stale`, move),
		false,
	);
	assert.equal(
		recommendationIsCurrent(
			engine,
			{ ...parsed, state: { ...state, turn: "p2" } },
			fingerprint,
			move,
		),
		false,
	);
});

test("autoplay aims at Wallz pawn centers and forced wall snap centers", () => {
	const pawn = moveTargetPoint({ type: "pawn", to: { x: 4, y: 1 } });
	const wall = moveTargetPoint({
		type: "wall",
		wall: { o: "h", x: 3, y: 2 },
	});

	assert.deepEqual(pawn, { x: 318, y: 102 });
	assert.deepEqual(wall, { x: 282, y: 210 });
	assert.deepEqual(orientBoardPoint(wall, true), { x: 354, y: 426 });
});

const {
	createWallZeroBridge,
	createWallZeroClient,
	wallZeroMoveShapeIsValid,
} = require("./wallz-coach.user120.js");

function jsonResponse(payload, ok = true, status = 200) {
	return {
		ok,
		status,
		json: async () => payload,
	};
}

function analysisPayload(overrides = {}) {
	return {
		schema: "wallzero.analysis.v1",
		id: "fp-1",
		fingerprint: "abc",
		move: { type: "pawn", to: { x: 4, y: 1 } },
		value: 0.25,
		simulations: 800,
		topMoves: [],
		...overrides,
	};
}

test("wallzero client posts the analyze schema and returns a valid reply", async () => {
	const requests = [];
	const client = createWallZeroClient({
		fetch: async (url, init) => {
			requests.push({ url, init });
			return jsonResponse(analysisPayload());
		},
	});

	const payload = await client.analyze(initialState(), "fp-1");
	assert.equal(payload.move.type, "pawn");
	assert.equal(requests.length, 1);
	assert.match(requests[0].url, /\/analyze$/);
	const body = JSON.parse(requests[0].init.body);
	assert.equal(body.schema, "wallzero.analyze.v1");
	assert.equal(body.id, "fp-1");
	assert.equal(body.state.turn, "p1");
});

test("wallzero client rejects replies for a different position as stale", async () => {
	const client = createWallZeroClient({
		fetch: async () => jsonResponse(analysisPayload({ id: "fp-other" })),
	});

	await assert.rejects(
		client.analyze(initialState(), "fp-1"),
		/wallzero-stale/,
	);
});

test("wallzero client rejects malformed moves and protocol errors", async () => {
	const malformed = createWallZeroClient({
		fetch: async () =>
			jsonResponse(analysisPayload({ move: { type: "pawn", to: { x: 11, y: 0 } } })),
	});
	await assert.rejects(
		malformed.analyze(initialState(), "fp-1"),
		/wallzero-protocol/,
	);

	const errored = createWallZeroClient({
		fetch: async () =>
			jsonResponse({
				schema: "wallzero.analysis.v1",
				id: "fp-1",
				error: { type: "ValueError", message: "bad state" },
			}),
	});
	await assert.rejects(
		errored.analyze(initialState(), "fp-1"),
		/wallzero-protocol: bad state/,
	);
});

test("wallzero client times out and reports the budget", async () => {
	const client = createWallZeroClient({
		timeoutMs: 30,
		fetch: (url, init) =>
			new Promise((resolve, reject) => {
				init.signal.addEventListener("abort", () => {
					const error = new Error("aborted");
					error.name = "AbortError";
					reject(error);
				});
			}),
	});

	await assert.rejects(
		client.analyze(initialState(), "fp-1"),
		/wallzero-timeout/,
	);
});

test("wallzero bridge fails closed when the engine is unreachable", async () => {
	const bridge = createWallZeroBridge({
		health: async () => {
			throw new Error("connection refused");
		},
	});

	assert.equal(bridge.enabled, false);
	assert.equal(await bridge.probe(), false);
	assert.equal(bridge.enabled, false);
	assert.match(bridge.lastError, /connection refused/);
});

test("wallzero bridge enables on a healthy probe and disables on outage", async () => {
	let healthy = true;
	const bridge = createWallZeroBridge({ health: async () => healthy });

	assert.equal(await bridge.probe(), true);
	assert.equal(bridge.enabled, true);
	assert.equal(bridge.lastError, "");

	bridge.reportOutage(new Error("wallzero-timeout: no reply within 2500ms"));
	assert.equal(bridge.enabled, false);
	assert.match(bridge.lastError, /wallzero-timeout/);

	healthy = false;
	assert.equal(await bridge.probe(), false);
	assert.equal(bridge.enabled, false);
});

test("wallzero move shape validation rejects out-of-space moves", () => {
	assert.equal(
		wallZeroMoveShapeIsValid({ type: "pawn", to: { x: 0, y: 8 } }),
		true,
	);
	assert.equal(
		wallZeroMoveShapeIsValid({ type: "wall", wall: { o: "h", x: 7, y: 7 } }),
		true,
	);
	assert.equal(
		wallZeroMoveShapeIsValid({ type: "wall", wall: { o: "h", x: 8, y: 0 } }),
		false,
	);
	assert.equal(
		wallZeroMoveShapeIsValid({ type: "pawn", to: { x: 4, y: 9 } }),
		false,
	);
	assert.equal(wallZeroMoveShapeIsValid(null), false);
	assert.equal(
		wallZeroMoveShapeIsValid({ type: "wall", wall: { o: "x", x: 1, y: 1 } }),
		false,
	);
});

test("wallzero recommendations still pass the fail-closed board check", () => {
	const engine = createEngine();
	const state = initialState();
	const parsed = { state, bottomPlayer: "p1" };
	const fingerprint = stateFingerprint(state, "p1");

	assert.equal(
		recommendationIsCurrent(engine, parsed, fingerprint, {
			type: "pawn",
			to: { x: 4, y: 1 },
		}),
		true,
	);
	assert.equal(
		recommendationIsCurrent(engine, parsed, fingerprint, {
			type: "pawn",
			to: { x: 4, y: 5 },
		}),
		false,
	);
	assert.equal(
		recommendationIsCurrent(engine, parsed, "different-fingerprint", {
			type: "pawn",
			to: { x: 4, y: 1 },
		}),
		false,
	);
});
