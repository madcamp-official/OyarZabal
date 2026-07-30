const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");

function element() {
  return {
    hidden: false,
    disabled: false,
    textContent: "",
    dataset: {},
    className: "",
    children: [],
    listeners: {},
    attributes: {},
    classList: { add() {}, remove() {}, toggle() {} },
    style: { setProperty() {} },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
    addEventListener(name, listener) {
      this.listeners[name] = listener;
    },
    append(...children) {
      this.children.push(...children);
    },
    replaceChildren(...children) {
      this.children = [...children];
    },
    click() {
      if (!this.disabled) this.listeners.click?.();
    },
  };
}

function snapshot(gamePk, mode = "live") {
  return {
    mode,
    game: {
      gamePk,
      status: "Final",
      detailedStatus: "Final",
      awayTeam: "AWAY",
      homeTeam: "HOME",
    },
    serverTime: "2026-07-29T00:00:00Z",
    lastResult: null,
    round: { status: "final" },
  };
}

function activeSnapshot(gamePk, mode, inning) {
  const rates = {
    FOUR_SEAM: 0.4,
    MOVING_FASTBALL: 0.2,
    SLIDER: 0.15,
    CURVE: 0.1,
    CHANGEUP: 0.1,
    SPLITTER_FORK: 0.05,
  };
  return {
    ...snapshot(gamePk, mode),
    round: {
      id: `${mode}-${gamePk}-${inning}`,
      mode,
      status: "open",
      answer: null,
      deadline: mode === "live" ? "2026-07-29T00:00:06Z" : null,
      prediction: null,
      context: {
        inning,
        half: "Top",
        count: { balls: 0, strikes: 0 },
        outs: 0,
        bases: [false, false, false],
        score: { away: 0, home: 0 },
        pitcher: { name: "Pitcher" },
        batter: { name: "Batter", stand: "L" },
        insights: {
          repertoire: ["FOUR_SEAM", "SLIDER", "CHANGEUP"],
          rates: {
            season: rates,
            recent100: rates,
            game: rates,
            count: rates,
            stand: rates,
          },
          support: { count: 42, stand: 318 },
          workload: {
            gamePitches: 74,
            timesFaced: 3,
            plateAppearancePitches: 4,
          },
        },
      },
    },
  };
}

async function main() {
  assert.doesNotMatch(
    fs.readFileSync("extension/sidepanel.html", "utf8"),
    /id="submit"/,
    "the manual submit button must be removed",
  );
  let activeUrl =
    "https://www.mlb.com/gameday/test-game/1001/final";
  const storage = {
    oyarzabalScore: {
      version: 2,
      attempts: 1,
      user: { points: 3, exactHits: 1, familyHits: 1 },
      model: { points: 0, exactHits: 0, familyHits: 0 },
      seen: ["legacy-round"],
    },
  };
  const fetchCalls = [];
  const syncLogs = [];
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element,
  };
  const chrome = {
    storage: {
      local: {
        async get(keys) {
          return Object.fromEntries(
            keys.filter((key) => key in storage).map((key) => [key, storage[key]]),
          );
        },
        async set(values) {
          Object.assign(storage, values);
        },
      },
    },
    tabs: {
      async query() {
        return [{ url: activeUrl }];
      },
      onActivated: { addListener() {} },
      onUpdated: { addListener() {} },
    },
  };
  const fetch = async (url, options = {}) => {
    fetchCalls.push({ url, options });
    if (url.includes("/rounds/") && url.endsWith("/answers")) {
      const body = JSON.parse(options.body);
      return {
        ok: true,
        async json() {
          return {
            roundId: url.match(/rounds\/([^/]+)/)?.[1],
            selectedPitch: body.selectedPitch,
            status: "draft",
          };
        },
      };
    }
    const mode = url.includes("/api/replay/") ? "replay" : "live";
    const gamePk = Number(
      url.match(/games\/(\d+)/)?.[1] ??
        JSON.parse(options.body ?? "{}").gamePk,
    );
    return {
      ok: true,
      async json() {
        return snapshot(gamePk, mode);
      },
    };
  };
  const context = vm.createContext({
    assert,
    chrome,
    console: {
      log: console.log,
      error: console.error,
      info(...values) {
        syncLogs.push(values);
      },
    },
    crypto: webcrypto,
    document,
    fetch,
    setInterval: () => 1,
    clearInterval() {},
    URL,
  });
  context.window = context;
  context.window.addEventListener = () => {};
  context.OYARZABAL_API_ORIGIN = "https://api.test";

  vm.runInContext(
    fs.readFileSync("extension/scoring.js", "utf8"),
    context,
  );
  vm.runInContext(
    fs.readFileSync("extension/sidepanel.js", "utf8"),
    context,
  );
  await new Promise(setImmediate);
  await new Promise(setImmediate);
  assert.equal(elements.get("user-exact").textContent, "0.0%");
  assert.equal(elements.get("user-family").textContent, "0.0%");
  assert.equal(
    vm.runInContext("score.user.points", context),
    0,
    "an unscoped legacy score must not leak into the active game",
  );
  vm.runInContext(
    `renderResult({
      roundId: "game-1001-live-round",
      context: { pitchNumber: 2 },
      selectedPitch: "FOUR_SEAM",
      actualPitch: "FOUR_SEAM",
      prediction: { topPitch: "CURVE" },
    })`,
    context,
  );
  assert.equal(vm.runInContext("score.user.points", context), 3);
  assert.equal(elements.get("result-title").textContent, "2구째 · 실제 포심");
  const attemptsBeforeMiss = vm.runInContext("score.attempts", context);
  vm.runInContext(
    `renderResult({
      roundId: "game-1001-live-round-unselected",
      context: { pitchNumber: 3 },
      selectedPitch: null,
      actualPitch: "SLIDER",
      prediction: { topPitch: "CURVE" },
    })`,
    context,
  );
  assert.equal(
    vm.runInContext("score.attempts", context),
    attemptsBeforeMiss + 1,
    "an unselected pitch must still count toward Accuracy",
  );
  assert.equal(elements.get("user-family").textContent, "50.0%");
  assert.match(elements.get("result-detail").textContent, /나 미선택 \+0점/);

  await vm.runInContext("startReplay()", context);
  assert.equal(
    vm.runInContext("score.user.points", context),
    0,
    "starting a replay must begin with a fresh score",
  );
  activeUrl = "https://example.com/";
  await vm.runInContext("syncTab()", context);
  activeUrl = "https://www.mlb.com/gameday/test-game/1001/final";
  await vm.runInContext("syncTab()", context);
  const resumedUrl = fetchCalls.at(-1).url;

  vm.runInContext(
    `renderResult({
      roundId: "game-1001-replay-round-1",
      selectedPitch: "FOUR_SEAM",
      actualPitch: "FOUR_SEAM",
      prediction: { topPitch: "CURVE" },
    })`,
    context,
  );
  assert.equal(vm.runInContext("score.user.points", context), 3);
  activeUrl = "https://www.mlb.com/gameday/another-game/1002/final";
  await vm.runInContext("syncTab()", context);
  const nextGamePoints = vm.runInContext("score.user.points", context);
  assert.equal(
    nextGamePoints,
    0,
    "a different game must start with its own score",
  );
  assert.match(
    resumedUrl,
    /\/api\/replay\/games\/1001/,
    "returning to the same game must resume its replay endpoint",
  );

  vm.runInContext(
    `renderResult({
      roundId: "game-1002-round-1",
      selectedPitch: "FOUR_SEAM",
      actualPitch: "FOUR_SEAM",
      prediction: { topPitch: "FOUR_SEAM" },
    })`,
    context,
  );
  assert.equal(vm.runInContext("score.user.points", context), 3);
  await vm.runInContext("startReplay(true)", context);
  assert.equal(
    vm.runInContext("score.user.points", context),
    0,
    "restarting a replay must reset only the current game's score",
  );
  activeUrl = "https://www.mlb.com/gameday/test-game/1001/final";
  await vm.runInContext("syncTab()", context);
  assert.equal(
    vm.runInContext("score.user.points", context),
    3,
    "returning to a previous game must restore that game's score",
  );

  vm.runInContext(
    "selectedFamily = null; selectedPitch = null; renderChoices(true)",
    context,
  );
  let choiceRoot = elements.get("choices");
  let familyButtons = choiceRoot.children[0].children[1].children;
  assert.equal(familyButtons.length, 3);
  assert.equal(choiceRoot.children[1].children[1].className, "choice-hint");
  familyButtons[0].click();
  assert.equal(vm.runInContext("selectedFamily", context), "FASTBALL");
  assert.equal(vm.runInContext("selectedPitch", context), null);
  choiceRoot = elements.get("choices");
  let detailButtons = choiceRoot.children[1].children[1].children;
  assert.deepEqual(
    detailButtons.map((button) => button.dataset.group),
    ["FOUR_SEAM", "MOVING_FASTBALL"],
  );
  detailButtons[1].click();
  assert.equal(vm.runInContext("selectedPitch", context), "MOVING_FASTBALL");
  choiceRoot = elements.get("choices");
  familyButtons = choiceRoot.children[0].children[1].children;
  familyButtons[1].click();
  assert.equal(vm.runInContext("selectedFamily", context), "BREAKING");
  assert.equal(vm.runInContext("selectedPitch", context), null);

  context.testSnapshot = activeSnapshot(1001, "live", 2);
  vm.runInContext("renderSnapshot(testSnapshot)", context);
  assert.equal(
    elements.get("game").hidden,
    true,
    "the situation panel must stay hidden during a live game",
  );
  assert.equal(elements.get("sync").hidden, false);
  assert.equal(elements.get("sync-inning").textContent, "▲ 2회");
  assert.equal(elements.get("sync-count").textContent, "0B 0S");
  assert.equal(elements.get("sync-batter").textContent, "Batter");
  assert.equal(elements.get("sync-pitch").textContent, "5구째");
  assert.equal(syncLogs.length, 1);
  assert.deepEqual(JSON.parse(JSON.stringify(syncLogs[0][1])), {
    gamePk: 1001,
    inning: "초 2회",
    outs: 0,
    count: "0B 0S",
    pitchInAtBat: 5,
    pitcherPitch: 75,
    stateKey: "live-1001-2",
  });
  choiceRoot = elements.get("choices");
  familyButtons = choiceRoot.children[0].children[1].children;
  familyButtons[0].click();
  choiceRoot = elements.get("choices");
  detailButtons = choiceRoot.children[1].children[1].children;
  detailButtons[1].click();
  await new Promise(setImmediate);
  assert.equal(
    JSON.parse(fetchCalls.at(-1).options.body).selectedPitch,
    "MOVING_FASTBALL",
    "a detail choice must be saved immediately",
  );
  assert.equal(
    elements.get("selection-status").textContent,
    "선택됨 · 변경 가능",
  );
  choiceRoot = elements.get("choices");
  choiceRoot.children[1].children[1].children[0].click();
  await new Promise(setImmediate);
  assert.equal(
    JSON.parse(fetchCalls.at(-1).options.body).selectedPitch,
    "FOUR_SEAM",
    "the saved draft must remain editable before the deadline",
  );

  context.testSnapshot.round.status = "locked";
  context.testSnapshot.round.answer = {
    selectedPitch: "FOUR_SEAM",
    receivedAt: "2026-07-29T00:00:03Z",
  };
  context.testSnapshot.round.prediction = {
    modelVersion: "test",
    topPitch: "FOUR_SEAM",
    confidence: 0.4,
    probabilities: {
      FOUR_SEAM: 0.4,
      MOVING_FASTBALL: 0.2,
      SLIDER: 0.15,
      CURVE: 0.1,
      CHANGEUP: 0.1,
      SPLITTER_FORK: 0.05,
    },
  };
  vm.runInContext("renderSnapshot(testSnapshot)", context);
  assert.equal(elements.get("selection-status").textContent, "LOCK");
  assert.equal(elements.get("prediction").hidden, false);
  assert.equal(
    elements.get("choices").children[0].children[1].children[0].disabled,
    true,
  );
  vm.runInContext("renderSnapshot(testSnapshot)", context);
  assert.equal(syncLogs.length, 1, "unchanged states must not spam the console");
  const attemptsBeforeNotice = vm.runInContext("score.attempts", context);
  context.testSnapshot.roundNotice = {
    type: "unsupportedPitch",
    rawCode: "KN",
  };
  vm.runInContext("renderSnapshot(testSnapshot)", context);
  assert.equal(elements.get("result").hidden, false);
  assert.equal(
    elements.get("result-title").textContent,
    "지원 외 구종 KN · 이번 라운드 무효",
  );
  assert.equal(
    elements.get("result-detail").textContent,
    "점수와 Accuracy에 포함되지 않습니다.",
  );
  assert.equal(
    vm.runInContext("score.attempts", context),
    attemptsBeforeNotice,
  );
  context.testSnapshot.roundNotice = { type: "stateChanged" };
  vm.runInContext("renderSnapshot(testSnapshot)", context);
  assert.equal(
    elements.get("result-title").textContent,
    "주자·경기 상황 변경 · 예측 재시작",
  );
  context.testSnapshot = activeSnapshot(1001, "replay", 2);
  vm.runInContext("renderSnapshot(testSnapshot)", context);
  assert.equal(
    elements.get("game").hidden,
    false,
    "the situation panel must remain visible during replay",
  );
  assert.equal(elements.get("sync").hidden, true);
  assert.equal(elements.get("context-open").hidden, false);
  elements.get("context-open").click();
  assert.equal(elements.get("context-modal").hidden, false);
  assert.equal(
    elements.get("context-repertoire").textContent,
    "포심 · 슬라이더 계열 · 체인지업",
  );
  assert.equal(elements.get("context-game-pitches").textContent, "74구");
  elements.get("context-close").click();
  assert.equal(elements.get("context-modal").hidden, true);

  console.log("extension sidepanel state checks passed");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
