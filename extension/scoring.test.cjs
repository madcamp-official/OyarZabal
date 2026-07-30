const assert = require("node:assert/strict");

require("./scoring.js");

const { createScore, metrics, normalizeScore, outcome, recordRound } =
  globalThis.OyarzabalScoring;

assert.deepEqual(outcome("FOUR_SEAM", "FOUR_SEAM"), {
  points: 3,
  exact: true,
  family: true,
});
assert.deepEqual(outcome("FOUR_SEAM", "MOVING_FASTBALL"), {
  points: 1,
  exact: false,
  family: true,
});
assert.equal(outcome("FOUR_SEAM", "CURVE").points, 0);

const score = createScore();
assert.equal(
  recordRound(score, "round-1", "FOUR_SEAM", "CURVE", "MOVING_FASTBALL"),
  true,
);
assert.equal(recordRound(score, "round-1", "FOUR_SEAM", "CURVE", "CURVE"), false);
assert.deepEqual(score.user, { points: 1, exactHits: 0, familyHits: 1 });
assert.deepEqual(score.model, { points: 0, exactHits: 0, familyHits: 0 });
assert.deepEqual(metrics(score.user, score.attempts), {
  exact: 0,
  family: 1,
  hierarchical: 0.5,
});

const unselected = createScore();
recordRound(unselected, "round-missed", null, "CURVE", "SLIDER");
assert.deepEqual(unselected.user, {
  points: 0,
  exactHits: 0,
  familyHits: 0,
});
assert.equal(metrics(unselected.user, unselected.attempts).family, 0);
assert.equal(
  metrics({ exactHits: 0, familyHits: Number.NaN }, 1).family,
  0,
);

assert.deepEqual(normalizeScore({ user: 2, model: 1, seen: ["a", "b"] }), {
  version: 2,
  attempts: 2,
  user: { points: 6, exactHits: 2, familyHits: 2 },
  model: { points: 3, exactHits: 1, familyHits: 1 },
  seen: ["a", "b"],
});

console.log("extension scoring checks passed");
