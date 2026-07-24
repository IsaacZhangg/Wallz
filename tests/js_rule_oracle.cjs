const fs = require("node:fs");

const { createEngine } = require("../wallz-coach.user120.js");

const engine = createEngine();
const states = JSON.parse(fs.readFileSync(0, "utf8"));

function enumerate(state) {
	const player = state.turn;
	const actions = engine
		.legalPawnMoves(state, player)
		.map((to) => to.y * 9 + to.x);
	for (const orientation of ["h", "v"]) {
		for (let y = 0; y < 8; y += 1) {
			for (let x = 0; x < 8; x += 1) {
				const result = engine.applyMove(state, {
					type: "wall",
					wall: { o: orientation, x, y },
				});
				if (result.ok) {
					const offset = orientation === "h" ? 81 : 145;
					actions.push(offset + y * 8 + x);
				}
			}
		}
	}
	return actions.sort((left, right) => left - right);
}

process.stdout.write(JSON.stringify(states.map(enumerate)));
