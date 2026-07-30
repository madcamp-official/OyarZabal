(() => {
  const families = [
    {
      key: "FASTBALL",
      label: "패스트볼 계열",
      groups: [
        ["FOUR_SEAM", "포심"],
        ["MOVING_FASTBALL", "무빙 패스트볼"],
      ],
    },
    {
      key: "BREAKING",
      label: "브레이킹볼 계열",
      groups: [
        ["SLIDER", "슬라이더 계열"],
        ["CURVE", "커브 계열"],
      ],
    },
    {
      key: "OFFSPEED",
      label: "오프스피드 계열",
      groups: [
        ["CHANGEUP", "체인지업"],
        ["SPLITTER_FORK", "스플리터·포크"],
      ],
    },
  ];
  const groupFamilies = Object.fromEntries(
    families.flatMap(({ key, groups }) => groups.map(([group]) => [group, key])),
  );

  function createScore() {
    return {
      version: 2,
      attempts: 0,
      user: { points: 0, exactHits: 0, familyHits: 0 },
      model: { points: 0, exactHits: 0, familyHits: 0 },
      seen: [],
    };
  }

  function safeCount(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
  }

  function normalizePlayer(value, attempts) {
    const exactHits = Math.min(safeCount(value?.exactHits), attempts);
    const familyHits = Math.min(
      Math.max(safeCount(value?.familyHits), exactHits),
      attempts,
    );
    return {
      points: safeCount(value?.points),
      exactHits,
      familyHits,
    };
  }

  function normalizeScore(value) {
    if (!value || typeof value !== "object") return createScore();
    const seen = Array.isArray(value.seen)
      ? value.seen.filter((id) => typeof id === "string").slice(-100)
      : [];
    if (value.version === 2) {
      const attempts = safeCount(value.attempts);
      return {
        version: 2,
        attempts,
        user: normalizePlayer(value.user, attempts),
        model: normalizePlayer(value.model, attempts),
        seen,
      };
    }

    const userExact = safeCount(value.user);
    const modelExact = safeCount(value.model);
    const attempts = Math.max(seen.length, userExact, modelExact);
    return {
      version: 2,
      attempts,
      user: {
        points: userExact * 3,
        exactHits: userExact,
        familyHits: userExact,
      },
      model: {
        points: modelExact * 3,
        exactHits: modelExact,
        familyHits: modelExact,
      },
      seen,
    };
  }

  function outcome(pick, actual) {
    const exact = pick === actual;
    const family =
      exact ||
      Boolean(
        groupFamilies[pick] &&
          groupFamilies[pick] === groupFamilies[actual],
      );
    return { points: exact ? 3 : family ? 1 : 0, exact, family };
  }

  function addOutcome(player, result) {
    player.points += result.points;
    player.exactHits += Number(result.exact);
    player.familyHits += Number(result.family);
  }

  function recordRound(score, roundId, userPick, modelPick, actual) {
    if (score.seen.includes(roundId)) return false;
    score.seen = [...score.seen.slice(-99), roundId];
    score.attempts += 1;
    addOutcome(score.user, outcome(userPick, actual));
    addOutcome(score.model, outcome(modelPick, actual));
    return true;
  }

  function metrics(player, attempts) {
    const total = safeCount(attempts);
    if (!total) return { exact: 0, family: 0, hierarchical: 0 };
    const normalized = normalizePlayer(player, total);
    const exact = normalized.exactHits / total;
    const family = normalized.familyHits / total;
    return { exact, family, hierarchical: (exact + family) / 2 };
  }

  globalThis.OyarzabalScoring = {
    families,
    groups: families.flatMap(({ groups }) => groups),
    groupFamilies,
    createScore,
    normalizeScore,
    outcome,
    recordRound,
    metrics,
  };
})();
