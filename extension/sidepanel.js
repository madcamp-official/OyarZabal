const API_ORIGIN = globalThis.OYARZABAL_API_ORIGIN;
const {
  families: FAMILIES,
  groups: GROUPS,
  groupFamilies: GROUP_FAMILIES,
  createScore,
  normalizeScore,
  outcome,
  recordRound,
  metrics,
} = globalThis.OyarzabalScoring;

const elements = Object.fromEntries(
  [
    "connection",
    "user-score",
    "user-exact",
    "user-family",
    "model-score",
    "model-exact",
    "model-family",
    "replay",
    "replay-status",
    "replay-start",
    "replay-advance",
    "replay-reset",
    "sync",
    "sync-inning",
    "sync-count",
    "sync-batter",
    "sync-pitch",
    "game",
    "away-team",
    "home-team",
    "away-score",
    "home-score",
    "inning",
    "count",
    "bases",
    "pitcher",
    "batter",
    "context-open",
    "context-modal",
    "context-close",
    "context-season",
    "context-recent",
    "context-game-rate",
    "context-repertoire",
    "context-game-pitches",
    "context-times-faced",
    "context-pa-pitches",
    "result",
    "result-title",
    "result-detail",
    "round",
    "timer",
    "choices",
    "selection-status",
    "prediction",
    "model-pick",
    "probabilities",
    "empty",
    "status-label",
    "status-title",
    "status-detail",
    "error",
  ].map((id) => [id, document.getElementById(id)]),
);

let gamePk = null;
let sessionId = null;
let replayMode = false;
let selectedFamily = null;
let selectedPitch = null;
let currentRound = null;
let serverOffset = 0;
let polling = false;
let timerHandle = null;
let lastSyncLogKey = null;
let draftSave = Promise.resolve();
let score = createScore();

function gameStorageKey(targetGamePk, field) {
  return `oyarzabalGame:${targetGamePk}:${field}`;
}

function parseGamePk(url) {
  try {
    const parsed = new URL(url);
    if (parsed.hostname !== "www.mlb.com") return null;
    const match = parsed.pathname.match(
      /^\/gameday\/.+\/(\d+)\/(?:live|preview|final)(?:\/|$)/,
    );
    return match ? Number(match[1]) : null;
  } catch {
    return null;
  }
}

async function activeGamePk() {
  const [tab] = await chrome.tabs.query({
    active: true,
    lastFocusedWindow: true,
  });
  return parseGamePk(tab?.url ?? "");
}

async function initializeStorage() {
  const saved = await chrome.storage.local.get(["oyarzabalSessionId"]);
  sessionId = saved.oyarzabalSessionId ?? crypto.randomUUID();
  await chrome.storage.local.set({ oyarzabalSessionId: sessionId });
  renderScore();
}

async function restoreGameState(targetGamePk) {
  const scoreKey = gameStorageKey(targetGamePk, "score");
  const replayKey = gameStorageKey(targetGamePk, "replay");
  const saved = await chrome.storage.local.get([scoreKey, replayKey]);
  score = normalizeScore(saved[scoreKey]);
  replayMode = saved[replayKey] === true;
  renderScore();
}

function saveCurrentScore() {
  if (!gamePk) return;
  chrome.storage.local.set({
    [gameStorageKey(gamePk, "score")]: score,
  });
}

async function rememberReplay(targetGamePk) {
  await chrome.storage.local.set({
    [gameStorageKey(targetGamePk, "replay")]: true,
  });
}

function renderScore() {
  for (const player of ["user", "model"]) {
    const accuracy = metrics(score[player], score.attempts);
    elements[`${player}-score`].textContent = String(score[player].points);
    elements[`${player}-exact`].textContent =
      `${(accuracy.exact * 100).toFixed(1)}%`;
    elements[`${player}-family`].textContent =
      `${(accuracy.family * 100).toFixed(1)}%`;
  }
}

function setStatus(label, title, detail) {
  elements.empty.hidden = false;
  elements["status-label"].textContent = label;
  elements["status-title"].textContent = title;
  elements["status-detail"].textContent = detail;
}

function hideStatus() {
  elements.empty.hidden = true;
}

function setError(message = "") {
  elements.error.textContent = message;
}

function toggleContext(open) {
  elements["context-modal"].hidden = !open;
  elements["context-open"].setAttribute("aria-expanded", String(open));
}

function resetGame() {
  replayMode = false;
  currentRound = null;
  selectedFamily = null;
  selectedPitch = null;
  lastSyncLogKey = null;
  elements.round.dataset.roundId = "";
  elements.replay.hidden = true;
  elements.sync.hidden = true;
  elements.game.hidden = true;
  elements.result.hidden = true;
  elements.round.hidden = true;
  elements.prediction.hidden = true;
  elements["context-open"].hidden = true;
  toggleContext(false);
  elements.connection.classList.remove("connected");
  setError();
}

function renderReplayControls(snapshot) {
  const available = snapshot.mode === "replay" || snapshot.game.status === "Final";
  elements.replay.hidden = !available;
  if (!available) return;
  const active = snapshot.mode === "replay";
  elements["replay-start"].hidden = active;
  elements["replay-reset"].hidden = !active;
  elements["replay-advance"].hidden =
    !active || !currentRound?.answer || currentRound.status === "final";
  elements["replay-status"].textContent = active
    ? currentRound?.pitchNumber
      ? `테스트 투구 ${currentRound.pitchNumber}`
      : "리플레이 완료"
    : "완료 경기 리플레이";
}

function topRates(rates) {
  const names = Object.fromEntries(GROUPS);
  return Object.entries(rates)
    .sort(([, left], [, right]) => right - left)
    .slice(0, 3)
    .map(([group, value]) => `${names[group] ?? group} ${Math.round(value * 100)}%`)
    .join(" · ");
}

function renderContext(context) {
  const insights = context?.insights;
  elements["context-open"].hidden = !insights;
  if (!insights) {
    toggleContext(false);
    return;
  }
  elements["context-season"].textContent = topRates(insights.rates.season);
  elements["context-recent"].textContent = topRates(insights.rates.recent100);
  elements["context-game-rate"].textContent = topRates(insights.rates.game);
  const names = Object.fromEntries(GROUPS);
  elements["context-repertoire"].textContent =
    (Array.isArray(insights.repertoire) ? insights.repertoire : [])
      .map((group) => names[group] ?? group)
      .join(" · ") || "기록 없음";
  elements["context-game-pitches"].textContent =
    `${insights.workload.gamePitches}구`;
  elements["context-times-faced"].textContent =
    `${insights.workload.timesFaced}번째`;
  elements["context-pa-pitches"].textContent =
    `${insights.workload.plateAppearancePitches}구`;
}

function baseLabel(bases) {
  const occupied = bases
    .map((active, index) => (active ? `${index + 1}루` : null))
    .filter(Boolean);
  return occupied.length ? `주자 ${occupied.join(" · ")}` : "주자 없음";
}

function renderGame(snapshot, context) {
  elements.game.hidden = false;
  elements["away-team"].textContent = snapshot.game.awayTeam;
  elements["home-team"].textContent = snapshot.game.homeTeam;
  elements["away-score"].textContent = String(context.score.away);
  elements["home-score"].textContent = String(context.score.home);
  elements.inning.textContent =
    `${context.half === "Top" ? "▲" : "▼"} ${context.inning}회`;
  elements.count.textContent =
    `${context.count.balls}B ${context.count.strikes}S ${context.outs}O`;
  elements.bases.textContent = baseLabel(context.bases);
  elements.pitcher.textContent = context.pitcher.name;
  elements.batter.textContent = context.batter.name;
}

function renderSync(snapshot, context) {
  elements.sync.hidden = replayMode || !context;
  if (elements.sync.hidden) return;
  elements["sync-inning"].textContent =
    `${context.half === "Top" ? "▲" : "▼"} ${context.inning}회`;
  elements["sync-count"].textContent =
    `${context.count.balls}B ${context.count.strikes}S`;
  elements["sync-batter"].textContent = context.batter.name;
  const pitchNumber =
    context.pitchNumber ??
    (context.insights
      ? context.insights.workload.plateAppearancePitches + 1
      : null);
  elements["sync-pitch"].textContent =
    pitchNumber === null ? "-구째" : `${pitchNumber}구째`;

  const logKey =
    context.stateKey ??
    [
      snapshot.game.gamePk,
      context.inning,
      context.half,
      context.outs,
      context.count.balls,
      context.count.strikes,
      context.pitchNumber,
    ].join(":");
  if (logKey === lastSyncLogKey) return;
  lastSyncLogKey = logKey;
  console.info("[OyarZabal Sync]", {
    gamePk: snapshot.game.gamePk,
    inning: `${context.half === "Top" ? "초" : "말"} ${context.inning}회`,
    outs: context.outs,
    count: `${context.count.balls}B ${context.count.strikes}S`,
    pitchInAtBat:
      context.pitchNumber ?? context.insights?.workload?.plateAppearancePitches + 1,
    pitcherPitch: context.insights?.workload?.gamePitches + 1,
    stateKey: context.stateKey ?? currentRound?.id,
  });
}

function renderChoices(enabled) {
  elements.choices.replaceChildren();

  const familyStep = document.createElement("section");
  familyStep.className = "choice-step";
  const familyLabel = document.createElement("span");
  familyLabel.className = "choice-step-label";
  familyLabel.textContent = "1. 구종 계열 선택";
  const familyChoices = document.createElement("div");
  familyChoices.className = "choice-options family-options";
  familyChoices.setAttribute("role", "group");
  familyChoices.setAttribute("aria-label", "예상 구종 계열");
  for (const family of FAMILIES) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = family.label;
    button.dataset.family = family.key;
    button.disabled = !enabled;
    button.classList.toggle("selected", selectedFamily === family.key);
    button.setAttribute(
      "aria-pressed",
      String(selectedFamily === family.key),
    );
    button.addEventListener("click", () => {
      if (selectedFamily !== family.key) selectedPitch = null;
      selectedFamily = family.key;
      renderChoices(enabled);
      renderSelectionStatus();
    });
    familyChoices.append(button);
  }
  familyStep.append(familyLabel, familyChoices);

  const detailStep = document.createElement("section");
  detailStep.className = "choice-step";
  const detailLabel = document.createElement("span");
  detailLabel.className = "choice-step-label";
  detailLabel.textContent = "2. 상세 구종 선택";
  detailStep.append(detailLabel);
  const family = FAMILIES.find(({ key }) => key === selectedFamily);
  if (!family) {
    const hint = document.createElement("p");
    hint.className = "choice-hint";
    hint.textContent = "계열을 선택하면 상세 구종이 열립니다.";
    detailStep.append(hint);
  } else {
    const detailChoices = document.createElement("div");
    detailChoices.className = "choice-options detail-options";
    detailChoices.setAttribute("role", "group");
    detailChoices.setAttribute("aria-label", `${family.label} 상세 구종`);
    for (const [group, label] of family.groups) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.dataset.group = group;
      button.disabled = !enabled;
      button.classList.toggle("selected", selectedPitch === group);
      button.setAttribute("aria-pressed", String(selectedPitch === group));
      button.addEventListener("click", () => {
        selectedPitch = group;
        renderChoices(enabled);
        renderSelectionStatus();
        saveDraft(group);
      });
      detailChoices.append(button);
    }
    detailStep.append(detailChoices);
  }
  elements.choices.append(familyStep, detailStep);
}

function isRoundEditable() {
  if (!currentRound?.id || currentRound.status !== "open") return false;
  if (currentRound.mode === "replay") return true;
  if (!currentRound.deadline) return false;
  return (
    new Date(currentRound.deadline).getTime() >
    Date.now() + serverOffset
  );
}

function renderSelectionStatus() {
  const editable = isRoundEditable();
  elements["selection-status"].className =
    editable && selectedPitch
      ? "selection-status selected"
      : editable
        ? "selection-status"
        : "selection-status locked";
  elements["selection-status"].textContent = editable
    ? selectedPitch
      ? "선택됨 · 변경 가능"
      : "상세 구종을 선택해주세요"
    : "LOCK";
}

function saveDraft(pitch) {
  if (!currentRound?.id || !isRoundEditable()) return;
  const roundId = currentRound.id;
  const mode = replayMode ? "replay" : "live";
  if (mode === "replay") {
    currentRound.status = "locked";
    renderChoices(false);
    renderSelectionStatus();
  }
  draftSave = draftSave
    .then(async () => {
      const response = await fetch(
        `${API_ORIGIN}/api/${mode}/rounds/` +
          `${encodeURIComponent(roundId)}/answers`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionId,
            selectedPitch: pitch,
          }),
        },
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error ?? "예측을 저장하지 못했습니다.");
      }
      if (currentRound?.id === roundId) {
        currentRound.answer = {
          selectedPitch: payload.selectedPitch,
          receivedAt: payload.receivedAt,
        };
        renderSelectionStatus();
      }
      setError();
    })
    .catch(async (error) => {
      setError(error instanceof Error ? error.message : String(error));
      await poll();
    });
}

function renderPrediction(prediction) {
  elements.prediction.hidden = !prediction;
  if (!prediction) return;
  const names = Object.fromEntries(GROUPS);
  const familyNames = Object.fromEntries(
    FAMILIES.map(({ key, label }) => [key, label]),
  );
  elements["model-pick"].textContent =
    `${prediction.modelVersion} · ${familyNames[GROUP_FAMILIES[prediction.topPitch]]}` +
    ` → ${names[prediction.topPitch]} ` +
    `${Math.round(prediction.confidence * 100)}%`;
  elements.probabilities.replaceChildren();
  const ordered = FAMILIES.map((family) => {
    const children = family.groups
      .map(([group, label]) => [
        group,
        label,
        prediction.probabilities[group] ?? 0,
      ])
      .sort(([, , left], [, , right]) => right - left);
    return {
      ...family,
      children,
      probability: children.reduce((sum, [, , value]) => sum + value, 0),
    };
  }).sort((left, right) => right.probability - left.probability);

  for (const family of ordered) {
    const section = document.createElement("section");
    section.className = "probability-family";
    const familyRow = document.createElement("div");
    familyRow.className = "family-probability";
    const familyLabel = document.createElement("strong");
    familyLabel.textContent = family.label;
    const familyTrack = document.createElement("i");
    familyTrack.style.setProperty("--value", `${family.probability * 100}%`);
    const familyPercentage = document.createElement("b");
    familyPercentage.textContent = `${Math.round(family.probability * 100)}%`;
    familyRow.append(familyLabel, familyTrack, familyPercentage);
    section.append(familyRow);

    for (const [group, label, value] of family.children) {
      const row = document.createElement("div");
      row.className =
        group === prediction.topPitch ? "probability leading" : "probability";
      const childLabel = document.createElement("span");
      childLabel.textContent = label;
      const track = document.createElement("i");
      track.style.setProperty("--value", `${value * 100}%`);
      const percentage = document.createElement("b");
      percentage.textContent = `${Math.round(value * 100)}%`;
      row.append(childLabel, track, percentage);
      section.append(row);
    }
    elements.probabilities.append(section);
  }
}

function updateTimer() {
  if (currentRound?.mode === "replay") {
    elements.timer.textContent =
      currentRound.status === "open" ? "TEST" : "LOCK";
    renderSelectionStatus();
    return;
  }
  if (!currentRound?.deadline || currentRound.status !== "open") {
    elements.timer.textContent = currentRound?.status === "result" ? "결과" : "LOCK";
    renderSelectionStatus();
    return;
  }
  const remaining =
    new Date(currentRound.deadline).getTime() - (Date.now() + serverOffset);
  elements.timer.textContent = `${Math.max(0, remaining / 1000).toFixed(1)}s`;
  if (remaining <= 0) {
    renderChoices(false);
    renderSelectionStatus();
  }
}

function renderResult(result, notice = null) {
  if (notice) {
    elements.result.hidden = false;
    if (notice.type === "unsupportedPitch") {
      elements["result-title"].textContent =
        `지원 외 구종 ${notice.rawCode || ""} · 이번 라운드 무효`;
      elements["result-detail"].textContent =
        "점수와 Accuracy에 포함되지 않습니다.";
    } else {
      elements["result-title"].textContent =
        "주자·경기 상황 변경 · 예측 재시작";
      elements["result-detail"].textContent =
        "변경된 상황을 기준으로 새 예측이 열렸습니다.";
    }
    return;
  }
  if (!result) {
    elements.result.hidden = true;
    return;
  }
  const names = Object.fromEntries(GROUPS);
  const userOutcome = outcome(result.selectedPitch, result.actualPitch);
  const modelOutcome = outcome(result.prediction.topPitch, result.actualPitch);
  const outcomeLabel = (value) =>
    value.exact ? "상세 +3점" : value.family ? "계열 +1점" : "+0점";
  const pitchLabel = Number.isInteger(result.context?.pitchNumber)
    ? `${result.context.pitchNumber}구째 · `
    : "";
  elements.result.hidden = false;
  elements["result-title"].textContent =
    `${pitchLabel}실제 ${names[result.actualPitch] ?? result.actualPitch}`;
  elements["result-detail"].textContent =
    `나 ${result.selectedPitch ? outcomeLabel(userOutcome) : "미선택 +0점"}` +
    ` · 모델 ${outcomeLabel(modelOutcome)}`;
  if (
    recordRound(
      score,
      result.roundId,
      result.selectedPitch,
      result.prediction.topPitch,
      result.actualPitch,
    )
  ) {
    saveCurrentScore();
    renderScore();
  }
}

function renderSnapshot(snapshot) {
  elements.connection.classList.add("connected");
  replayMode = snapshot.mode === "replay";
  serverOffset = new Date(snapshot.serverTime).getTime() - Date.now();
  renderResult(snapshot.lastResult, snapshot.roundNotice);
  currentRound = snapshot.round;
  renderReplayControls(snapshot);
  renderContext(currentRound.context);
  renderSync(snapshot, currentRound.context);

  if (!currentRound.context) {
    elements.game.hidden = true;
    elements.round.hidden = true;
    elements.prediction.hidden = true;
    if (currentRound.status === "final") {
      setStatus(
        replayMode ? "TEST COMPLETE" : "FINAL",
        replayMode ? "리플레이가 끝났습니다" : "종료된 경기입니다",
        replayMode
          ? "처음부터 다시 시작할 수 있습니다."
          : "위 테스트 모드로 지금 바로 대결할 수 있습니다.",
      );
    } else {
      setStatus("SYNCING", "다음 타자를 기다리는 중", snapshot.game.detailedStatus);
    }
    return;
  }

  hideStatus();
  if (replayMode) {
    renderGame(snapshot, currentRound.context);
  } else {
    elements.game.hidden = true;
  }
  const changed = elements.round.dataset.roundId !== currentRound.id;
  if (changed || currentRound.status !== "open") {
    selectedPitch = currentRound.answer?.selectedPitch ?? null;
    selectedFamily = GROUP_FAMILIES[selectedPitch] ?? null;
    elements.round.dataset.roundId = currentRound.id;
  }
  elements.round.hidden = false;
  renderChoices(isRoundEditable());
  renderSelectionStatus();
  renderPrediction(
    ["locked", "result"].includes(currentRound.status)
      ? currentRound.prediction
      : null,
  );
  updateTimer();
}

elements["context-open"].addEventListener("click", () => {
  toggleContext(elements["context-modal"].hidden);
});
elements["context-close"].addEventListener("click", () => toggleContext(false));

async function fetchSnapshot() {
  const mode = replayMode ? "replay" : "live";
  const response = await fetch(
    `${API_ORIGIN}/api/${mode}/games/${gamePk}?sessionId=${encodeURIComponent(sessionId)}`,
    { cache: "no-store" },
  );
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? "경기 정보를 불러오지 못했습니다.");
  renderSnapshot(payload);
}

async function poll() {
  if (polling || !gamePk) return;
  polling = true;
  try {
    await fetchSnapshot();
    setError();
  } catch (error) {
    elements.connection.classList.remove("connected");
    setError(error instanceof Error ? error.message : String(error));
  } finally {
    polling = false;
  }
}

async function syncTab() {
  const nextGamePk = await activeGamePk();
  if (nextGamePk === gamePk) return;
  gamePk = nextGamePk;
  resetGame();
  if (!gamePk) {
    score = createScore();
    renderScore();
    setStatus(
      "READY",
      "Gameday를 열어주세요",
      "MLB Gameday 경기 탭에서 확장 프로그램 아이콘을 눌러 시작합니다.",
    );
    return;
  }
  await restoreGameState(gamePk);
  setStatus("CONNECTING", "경기를 동기화하는 중", `Game PK ${gamePk}`);
  await poll();
}

async function startReplay(resetScore = false) {
  const button = resetScore
    ? elements["replay-reset"]
    : elements["replay-start"];
  button.disabled = true;
  setError();
  try {
    const response = await fetch(`${API_ORIGIN}/api/replay/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sessionId, gamePk }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? "테스트 모드를 시작하지 못했습니다.");
    }
    await rememberReplay(gamePk);
    score = createScore();
    saveCurrentScore();
    renderScore();
    renderSnapshot(payload);
  } catch (error) {
    setError(error instanceof Error ? error.message : String(error));
  } finally {
    button.disabled = false;
  }
}

elements["replay-start"].addEventListener("click", () => startReplay());
elements["replay-reset"].addEventListener("click", () => startReplay(true));
elements["replay-advance"].addEventListener("click", async () => {
  elements["replay-advance"].disabled = true;
  setError();
  try {
    const response = await fetch(
      `${API_ORIGIN}/api/replay/sessions/${encodeURIComponent(sessionId)}/advance`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gamePk }),
      },
    );
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error ?? "다음 투구를 준비하지 못했습니다.");
    }
    renderSnapshot(payload);
  } catch (error) {
    setError(error instanceof Error ? error.message : String(error));
  } finally {
    elements["replay-advance"].disabled = false;
  }
});

chrome.tabs.onActivated.addListener(() => syncTab());
chrome.tabs.onUpdated.addListener((_tabId, change) => {
  if (change.url || change.status === "complete") syncTab();
});

async function start() {
  await initializeStorage();
  await syncTab();
  timerHandle = setInterval(updateTimer, 100);
  setInterval(syncTab, 1000);
  setInterval(poll, 250);
}

window.addEventListener("unload", () => clearInterval(timerHandle));
start().catch((error) => setError(String(error)));
