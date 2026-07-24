// Head-to-head: WallZero (local HTTP serve) vs the userscript's classical
// engine at its production adaptive-search settings. Evaluation only — the
// classical engine never produces training data. Usage:
//   node scripts/match_vs_classical.mjs [games] [simulations] [port]

const api = await import("../wallz-coach.user120.js").then(
	(m) => m.default ?? m,
);
const games = Number(process.argv[2] ?? 20);
const simulations = Number(process.argv[3] ?? 160);
const port = Number(process.argv[4] ?? 8791);
const maxPlies = 240;

const engine = api.createEngine();

function initialState() {
	return {
		pawns: { p1: { x: 4, y: 0 }, p2: { x: 4, y: 8 } },
		turn: "p1",
		walls: [],
		wallsRemaining: { p1: 10, p2: 10 },
		winner: null,
	};
}

async function wallZeroMove(state, id) {
	const response = await fetch(`http://127.0.0.1:${port}/analyze`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({
			schema: "wallzero.analyze.v1",
			id,
			state,
			options: { simulations, topMoves: 1 },
		}),
	});
	if (!response.ok) {
		throw new Error(`wallzero http ${response.status}`);
	}
	const payload = await response.json();
	if (!payload.move) {
		throw new Error(`wallzero returned no move: ${JSON.stringify(payload)}`);
	}
	return payload.move;
}

function classicalMove(state, player) {
	const plan = api.selectAdaptiveSearch(engine, state, player);
	const analysis = engine.analyze(state, player, plan);
	if (!analysis?.bestMove) {
		throw new Error("classical engine returned no move");
	}
	return analysis.bestMove;
}

let wallZeroWins = 0;
let classicalWins = 0;
let draws = 0;

for (let game = 0; game < games; game += 1) {
	const wallZeroPlayer = game % 2 === 0 ? "p1" : "p2";
	let state = initialState();
	let winner = null;
	for (let ply = 0; ply < maxPlies; ply += 1) {
		const mover = state.turn;
		const move =
			mover === wallZeroPlayer
				? await wallZeroMove(state, `match-g${game}-p${ply}`)
				: classicalMove(state, mover);
		const applied = engine.applyMove(state, move);
		if (!applied.ok) {
			throw new Error(
				`illegal move by ${mover === wallZeroPlayer ? "wallzero" : "classical"} at game ${game} ply ${ply}: ${JSON.stringify(move)}`,
			);
		}
		state = applied.state;
		if (state.winner) {
			winner = state.winner;
			break;
		}
	}
	if (winner === wallZeroPlayer) {
		wallZeroWins += 1;
	} else if (winner === null) {
		draws += 1;
	} else {
		classicalWins += 1;
	}
	const tag = winner === null ? "draw" : winner === wallZeroPlayer ? "WZ" : "CL";
	console.log(
		JSON.stringify({
			game,
			wallZeroPlayer,
			winner: tag,
			tally: `${wallZeroWins}-${classicalWins}-${draws}`,
		}),
	);
}

function wilson(successes, trials, z = 1.959964) {
	const p = successes / trials;
	const d = 1 + (z * z) / trials;
	const center = (p + (z * z) / (2 * trials)) / d;
	const margin =
		(z *
			Math.sqrt(
				(p * (1 - p)) / trials + (z * z) / (4 * trials * trials),
			)) /
		d;
	return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

const score = wallZeroWins + 0.5 * draws;
const [low, high] = wilson(score, games);
console.log(
	JSON.stringify({
		schema: "wallzero.classical-match.v1",
		games,
		simulations,
		wallZeroWins,
		classicalWins,
		draws,
		score: score / games,
		ci95: [Number(low.toFixed(4)), Number(high.toFixed(4))],
	}),
);
