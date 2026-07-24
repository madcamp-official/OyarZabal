import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import App from "./App";

const prediction = {
  topPitch: "FOUR_SEAM",
  confidence: 0.7,
  probabilities: {
    FOUR_SEAM: 0.7,
    SINKER: 0.1,
    CUTTER: 0.05,
    SLIDER: 0.1,
    CURVE: 0.03,
    CHANGEUP: 0.02,
  },
};
const manifest = {
  schemaVersion: 3,
  generatedAt: "2026-07-24",
  caveat: "showcase",
  finalModel: "xgboost",
  pitchGroups: {
    FOUR_SEAM: "포심",
    SINKER: "싱커/투심",
    CUTTER: "커터",
    SLIDER: "슬라이더",
    CURVE: "커브",
    CHANGEUP: "체인지업",
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
        top3Accuracy: 1,
        macroF1: 1,
        logLoss: 0.3,
        actualDistribution: {
          FOUR_SEAM: 1,
          SINKER: 0,
          CUTTER: 0,
          SLIDER: 0,
          CURVE: 0,
          CHANGEUP: 0,
        },
        predictedDistribution: {
          FOUR_SEAM: 1,
          SINKER: 0,
          CUTTER: 0,
          SLIDER: 0,
          CURVE: 0,
          CHANGEUP: 0,
        },
        perClass: Object.fromEntries(
          ["FOUR_SEAM", "SINKER", "CUTTER", "SLIDER", "CURVE", "CHANGEUP"].map(
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
      type: "hybrid",
      label: "Pitcher + Global Hybrid",
      globalWeight: 0.25,
      specialistWeight: 0.75,
    },
    predictions: { final: prediction, xgboost: prediction, similarity: prediction, baseline: prediction },
    explanations: ["첫 투구"],
    actual: { rawCode: "FF", pitchGroup: "FOUR_SEAM", description: "strike", speed: 95, zone: 5 },
  }],
};

afterEach(() => vi.unstubAllGlobals());

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
  expect(screen.getByText("✓ 정확한 예측")).toBeInTheDocument();
  expect(screen.getByText("MODEL REPORT")).toBeInTheDocument();
  expect(screen.getByText("구종별 진단")).toBeInTheDocument();
  expect(screen.getAllByText("균형 XGBoost")).toHaveLength(2);
  expect(screen.getByText("Pitcher + Global Hybrid")).toBeInTheDocument();
  expect(screen.getByText("Global 25% · Specialist 75%")).toBeInTheDocument();
});
