import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import App from "./App";

const prediction = {
  topPitch: "FOUR_SEAM",
  confidence: 0.7,
  probabilities: {
    FOUR_SEAM: 0.7,
    MOVING_FASTBALL: 0.1,
    SLIDER: 0.1,
    CURVE: 0.03,
    CHANGEUP: 0.02,
    SPLITTER_FORK: 0.05,
  },
};
const manifest = {
  schemaVersion: 8,
  decisionRule: "family-sum-then-child",
  generatedAt: "2026-07-24",
  caveat: "showcase",
  finalModel: "xgboost",
  deploymentStatus: "shadow",
  pitchGroups: {
    FOUR_SEAM: "포심",
    MOVING_FASTBALL: "무빙 패스트볼",
    SLIDER: "슬라이더 계열",
    CURVE: "커브 계열",
    CHANGEUP: "체인지업",
    SPLITTER_FORK: "스플리터·포크",
  },
  pitchFamilies: {
    FASTBALL: "패스트볼 계열",
    BREAKING: "브레이킹볼 계열",
    OFFSPEED: "오프스피드 계열",
  },
  pitchGroupFamilies: {
    FOUR_SEAM: "FASTBALL",
    MOVING_FASTBALL: "FASTBALL",
    SLIDER: "BREAKING",
    CURVE: "BREAKING",
    CHANGEUP: "OFFSPEED",
    SPLITTER_FORK: "OFFSPEED",
  },
  models: {
    final: "균형 XGBoost",
    xgboost: "XGBoost",
    similarity: "Similarity",
    baseline: "기준선",
  },
  validationMetrics: {},
  selection: {
    accepted: true,
    referenceName: "legacy",
    selectedName: "balanced",
    folds: [],
    candidates: [],
  },
  games: [{ id: "1", title: "경기", date: "2024", matchup: "A @ B", pitchCount: 1, path: "/game.json" }],
};
const game = {
  id: "1",
  date: "2024-10-25",
  title: "테스트 경기",
  awayTeam: "NYY",
  homeTeam: "LAD",
  venue: "Park",
  evaluationMode: "historical_showcase",
  caveat: "역사적 쇼케이스",
  pitchCount: 1,
  metrics: Object.fromEntries(
    ["final", "xgboost", "similarity", "baseline"].map((key) => [
      key,
      {
        n: 1,
        accuracy: 1,
        familyAccuracy: 1,
        hierarchicalAccuracy: 1,
        top3Accuracy: 1,
        macroF1: 1,
        logLoss: 0.3,
        actualDistribution: {
          FOUR_SEAM: 1,
          MOVING_FASTBALL: 0,
          SLIDER: 0,
          CURVE: 0,
          CHANGEUP: 0,
          SPLITTER_FORK: 0,
        },
        predictedDistribution: {
          FOUR_SEAM: 1,
          MOVING_FASTBALL: 0,
          SLIDER: 0,
          CURVE: 0,
          CHANGEUP: 0,
          SPLITTER_FORK: 0,
        },
        classShareError: {
          FOUR_SEAM: 0,
          MOVING_FASTBALL: 0,
          SLIDER: 0,
          CURVE: 0,
          CHANGEUP: 0,
          SPLITTER_FORK: 0,
        },
        maxClassShareError: 0,
        totalVariationDistance: 0,
        classCalibrationError: {
          FOUR_SEAM: 0,
          MOVING_FASTBALL: 0,
          SLIDER: 0,
          CURVE: 0,
          CHANGEUP: 0,
          SPLITTER_FORK: 0,
        },
        maxClassCalibrationError: 0,
        perClass: Object.fromEntries(
          ["FOUR_SEAM", "MOVING_FASTBALL", "SLIDER", "CURVE", "CHANGEUP", "SPLITTER_FORK"].map(
            (group) => [
              group,
              { precision: group === "FOUR_SEAM" ? 1 : 0, recall: group === "FOUR_SEAM" ? 1 : 0, f1: group === "FOUR_SEAM" ? 1 : 0, support: group === "FOUR_SEAM" ? 1 : 0 },
            ],
          ),
        ),
        zeroRecallClasses: [],
        majorityPredictionGap: 0,
      },
    ]),
  ),
  pitches: [{
    sequence: 1,
    inning: 1,
    half: "Top",
    atBatNumber: 1,
    pitchNumber: 1,
    pitcher: { id: 1, name: "Pitcher", throws: "R" },
    batter: { id: 2, name: "Batter", stand: "L" },
    count: { balls: 0, strikes: 0 },
    outs: 0,
    bases: [false, false, false],
    score: { away: 0, home: 0 },
    recentPitches: [],
    modelSource: {
      type: "reliability-gated-residual",
      label: "V7 Hierarchical Incremental Residual",
      globalWeight: 0.8,
      specialistWeight: 0.2,
      pitcherReliability: 0.4,
      contextGate: 0.5,
      effectiveScale: 0.2,
      registryTier: "full",
      scaleMultiplier: 1,
      capReason: null,
      hardGateReason: null,
    },
    predictions: { final: prediction, xgboost: prediction, similarity: prediction, baseline: prediction },
    explanations: ["첫 투구"],
    actual: { rawCode: "FF", pitchGroup: "FOUR_SEAM", description: "strike", speed: 95, zone: 5 },
  }],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("keeps the actual pitch hidden until reveal", async () => {
  vi.stubGlobal("fetch", vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => manifest })
    .mockResolvedValueOnce({ ok: true, json: async () => game }));

  render(<App />);
  await waitFor(() => expect(screen.getByText("Pitcher")).toBeInTheDocument());
  expect(screen.queryByText("ACTUAL PITCH")).not.toBeInTheDocument();
  expect(screen.queryByText("MODEL REPORT")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "실제 투구 공개" }));
  expect(screen.getByText("ACTUAL PITCH")).toBeInTheDocument();
  expect(screen.getByText("✓ 정확한 구종 적중")).toBeInTheDocument();
  expect(screen.getByText("MODEL REPORT")).toBeInTheDocument();
  expect(screen.getByText("구종별 진단")).toBeInTheDocument();
  expect(screen.getByText("패스트볼 계열")).toBeInTheDocument();
  expect(screen.getAllByText("Hierarchical Accuracy")).toHaveLength(3);
  expect(screen.getAllByText("균형 XGBoost")).toHaveLength(2);
  expect(
    screen.getByText("V7 Hierarchical Incremental Residual"),
  ).toBeInTheDocument();
  expect(
    screen.getByText(
      "투수 신뢰도 40% · 상황 Gate 50% · Registry full · 선수 배율 100% · 적용 20%",
    ),
  ).toBeInTheDocument();
});

test("awards a family hit when the exact pitch group is different", async () => {
  const familyGame = structuredClone(game);
  familyGame.pitches[0].actual.pitchGroup = "MOVING_FASTBALL";
  vi.stubGlobal("fetch", vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => manifest })
    .mockResolvedValueOnce({ ok: true, json: async () => familyGame }));

  render(<App />);
  await waitFor(() => expect(screen.getByText("Pitcher")).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: "실제 투구 공개" }));

  expect(screen.getByText("△ 계열 적중")).toBeInTheDocument();
  expect(screen.queryByText("✓ 정확한 구종 적중")).not.toBeInTheDocument();
});
