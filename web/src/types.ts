export type ModelKey = "final" | "xgboost" | "similarity" | "baseline";

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface Metrics {
  n: number;
  accuracy: number;
  top3Accuracy: number;
  macroF1: number;
  logLoss: number;
  actualDistribution: Record<string, number>;
  predictedDistribution: Record<string, number>;
  perClass: Record<string, ClassMetrics>;
  zeroRecallClasses: string[];
  majorityPredictionGap: number;
}

export interface Prediction {
  topPitch: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface Pitch {
  sequence: number;
  inning: number;
  half: "Top" | "Bot";
  atBatNumber: number;
  pitchNumber: number;
  pitcher: { id: number; name: string; throws: string };
  batter: { id: number; name: string; stand: string };
  count: { balls: number; strikes: number };
  outs: number;
  bases: boolean[];
  score: { away: number; home: number };
  recentPitches: string[];
  modelSource: {
    type: "global" | "hybrid";
    label: string;
    globalWeight: number;
    specialistWeight: number;
  };
  predictions: Record<ModelKey, Prediction>;
  explanations: string[];
  actual: {
    rawCode: string;
    pitchGroup: string;
    description: string;
    speed: number | null;
    zone: number | null;
  };
}

export interface Game {
  id: string;
  date: string;
  title: string;
  awayTeam: string;
  homeTeam: string;
  venue: string;
  evaluationMode: string;
  caveat: string;
  metrics: Record<ModelKey, Metrics>;
  pitchCount: number;
  pitches: Pitch[];
}

export interface Manifest {
  schemaVersion: 3;
  generatedAt: string;
  caveat: string;
  finalModel: string;
  dataScope?: {
    startDate: string;
    endDate: string;
    pitcherCount: number;
    mlbWide: boolean;
  };
  pitchGroups: Record<string, string>;
  models: Record<ModelKey, string>;
  validationMetrics: Record<string, Metrics>;
  selection: {
    accepted: boolean;
    referenceName: string;
    selectedName: string;
    folds: string[];
    candidates: unknown[];
  };
  games: Array<{
    id: string;
    title: string;
    date: string;
    matchup: string;
    pitchCount: number;
    path: string;
  }>;
}
