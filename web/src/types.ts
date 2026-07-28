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
  familyAccuracy: number;
  hierarchicalAccuracy: number;
  top3Accuracy: number;
  macroF1: number;
  logLoss: number;
  actualDistribution: Record<string, number>;
  predictedDistribution: Record<string, number>;
  classShareError: Record<string, number>;
  maxClassShareError: number;
  totalVariationDistance: number;
  classCalibrationError: Record<string, number>;
  maxClassCalibrationError: number;
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
    type:
      | "global"
      | "hybrid"
      | "pooled-residual"
      | "provisional-residual"
      | "reliability-gated-residual";
    label: string;
    globalWeight: number;
    specialistWeight: number;
    residualScale?: number;
    pitcherReliability?: number;
    contextGate?: number;
    effectiveScale?: number;
    registryTier?: "full" | "limited" | "shadow";
    scaleMultiplier?: number;
    capReason?: string | null;
    hardGateReason?: string | null;
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
  schemaVersion: 8;
  decisionRule: "family-sum-then-child";
  generatedAt: string;
  caveat: string;
  finalModel: string;
  deploymentStatus: "shadow" | "active";
  deploymentBasis: "taxonomy-product-generation" | "candidate-evaluation";
  performanceCertification:
    | "prospective-pending"
    | "prospective-passed"
    | "not-applicable";
  dataScope?: {
    startDate: string;
    endDate: string;
    pitcherCount: number;
    mlbWide: boolean;
  };
  pitchGroups: Record<string, string>;
  pitchFamilies: Record<string, string>;
  pitchGroupFamilies: Record<string, string>;
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
