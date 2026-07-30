import { useEffect, useRef, useState } from "react";

import "./human-benchmark-prototype.css";
import {
  memorableStories,
  type MemorableChapter,
} from "./memorableSets";
type PitchFamily = "FASTBALL" | "BREAKING" | "OFFSPEED";
type PitchGroup =
  | "FOUR_SEAM"
  | "MOVING_FASTBALL"
  | "SLIDER"
  | "CURVE"
  | "CHANGEUP"
  | "SPLITTER_FORK";
type RawPitchCode = "FF" | "FC" | "SI" | "SL" | "SV" | "CU" | "KC" | "CH";

const normalizeNickname = (value: string) => value.trim().replace(/\s+/g, " ");
const nicknamePattern = /^[\p{L}\p{N}_ -]{1,20}$/u;

interface BenchmarkPitch {
  number: number;
  count: string;
  actual: PitchGroup;
  rawLabel: string;
  result: string;
  playId: string;
  modelPrediction?: ModelPrediction | null;
}

interface Answer {
  scenarioId: string;
  pitchNumber: number;
  selectedFamily: PitchFamily;
  selected: PitchGroup;
  actualFamily: PitchFamily;
  actual: PitchGroup;
  familyHit: boolean;
  detailHit: boolean;
  points: 0 | 1 | 3;
  responseMs: number;
}

export interface BenchmarkScenario {
  id: string;
  pitcher: string;
  batter: string;
  date: string;
  inning: string;
  awayTeam: string;
  homeTeam: string;
  awayScore: number;
  homeScore: number;
  outs: number;
  bases: [boolean, boolean, boolean];
  outcome: string;
  gameUrl: string;
  repertoire: Record<PitchFamily, string[]>;
  availableGroups: PitchGroup[];
  pitches: BenchmarkPitch[];
}

type BenchmarkVersion = "v1" | "v2";
type GameMode = BenchmarkVersion | "daily" | "memorable";

interface BenchmarkDefinition {
  label: string;
  year: string;
  description: string;
  scenarios: BenchmarkScenario[];
}

interface DailyChallenge {
  id: string;
  number: number;
  date: string;
  sourceDate: string;
  scenarioCount: number;
  participantCount: number;
}

interface DailyCurrent {
  questionNumber: number;
  scenarioNumber: number;
  scenarioCount: number;
  scenario: Omit<BenchmarkScenario, "pitches">;
  pitch: Pick<BenchmarkPitch, "number" | "count">;
}

export interface ModelPrediction {
  selected: PitchGroup;
  confidence: number;
  probabilities: Partial<Record<PitchGroup, number>>;
  modelVersion: string;
}

interface DailyReveal extends Answer {
  scenarioNumber: number;
  count: string;
  rawLabel: string;
  result: string;
  playId: string;
  modelPrediction: ModelPrediction | null;
}

interface DailyLeaderboardRow {
  rank: number;
  nickname: string;
  points: number;
  detailHits: number;
  familyHits: number;
  isCurrent: boolean;
  isModel: boolean;
}

interface DailyLeaderboard {
  participantCount: number;
  currentRank: number;
  rows: DailyLeaderboardRow[];
}

interface DailyResult {
  points: number;
  maxPoints: number;
  pitchCount: number;
  detailHits: number;
  familyHits: number;
  rank: number;
  streak: number;
  leaderboard: DailyLeaderboard;
}

interface DailySession {
  attemptId: string;
  nickname: string;
  challenge: DailyChallenge;
  status: "playing" | "completed";
  current?: DailyCurrent;
  history: DailyReveal[];
  reveal?: DailyReveal;
  result?: DailyResult;
}

interface MemorableLeaderboard {
  version: string;
  label: string;
  participantCount: number;
  pitchCount: number;
  maxPoints: number;
  rows: Array<{
    rank: number;
    nickname: string;
    points: number;
    detailHits: number;
    familyHits: number;
    averageResponseMs: number | null;
    isModel?: boolean;
  }>;
}

const pitchOptions: Array<{
  key: PitchGroup;
  family: PitchFamily;
  label: string;
  detail: string;
  code: string;
}> = [
  {
    key: "FOUR_SEAM",
    family: "FASTBALL",
    label: "포심",
    detail: "Four-seam",
    code: "FF",
  },
  {
    key: "MOVING_FASTBALL",
    family: "FASTBALL",
    label: "무빙 패스트볼",
    detail: "커터 · 싱커",
    code: "FC / SI",
  },
  {
    key: "SLIDER",
    family: "BREAKING",
    label: "슬라이더 계열",
    detail: "Slider",
    code: "SL",
  },
  {
    key: "CURVE",
    family: "BREAKING",
    label: "커브 계열",
    detail: "Curveball",
    code: "CU",
  },
  {
    key: "CHANGEUP",
    family: "OFFSPEED",
    label: "체인지업",
    detail: "Changeup",
    code: "CH",
  },
  {
    key: "SPLITTER_FORK",
    family: "OFFSPEED",
    label: "스플리터·포크",
    detail: "Splitter · Forkball",
    code: "FS / FO",
  },
];

const familyOptions: Array<{
  key: PitchFamily;
  label: string;
  detail: string;
}> = [
  {
    key: "FASTBALL",
    label: "패스트볼 계열",
    detail: "포심 · 무빙 패스트볼",
  },
  {
    key: "BREAKING",
    label: "브레이킹볼 계열",
    detail: "슬라이더 · 커브",
  },
  {
    key: "OFFSPEED",
    label: "오프스피드 계열",
    detail: "체인지업 · 스플리터·포크",
  },
];

const groupFamily = Object.fromEntries(
  pitchOptions.map((option) => [option.key, option.family]),
) as Record<PitchGroup, PitchFamily>;

const familyByKey = Object.fromEntries(
  familyOptions.map((option) => [option.key, option]),
) as Record<PitchFamily, (typeof familyOptions)[number]>;

const rawPitch: Record<RawPitchCode, { group: PitchGroup; label: string }> = {
  FF: { group: "FOUR_SEAM", label: "포심" },
  FC: { group: "MOVING_FASTBALL", label: "커터" },
  SI: { group: "MOVING_FASTBALL", label: "싱커" },
  SL: { group: "SLIDER", label: "슬라이더" },
  SV: { group: "SLIDER", label: "스위퍼" },
  CU: { group: "CURVE", label: "커브" },
  KC: { group: "CURVE", label: "너클 커브" },
  CH: { group: "CHANGEUP", label: "체인지업" },
};

type PitchRow = [count: string, code: RawPitchCode, result: string, playId: string];

function makePitches(rows: PitchRow[]): BenchmarkPitch[] {
  return rows.map(([count, code, result, playId], index) => ({
    number: index + 1,
    count: count.replace("-", "–"),
    actual: rawPitch[code].group,
    rawLabel: rawPitch[code].label,
    result,
    playId,
  }));
}

const kellyRepertoire: Record<PitchFamily, string[]> = {
  FASTBALL: ["포심", "커터", "싱커"],
  BREAKING: ["슬라이더", "커브"],
  OFFSPEED: ["체인지업"],
};

const brownRepertoire: Record<PitchFamily, string[]> = {
  FASTBALL: ["포심", "커터", "싱커"],
  BREAKING: ["슬라이더", "너클 커브"],
  OFFSPEED: ["체인지업"],
};

const flexenRepertoire: Record<PitchFamily, string[]> = {
  FASTBALL: ["포심", "커터"],
  BREAKING: ["슬라이더", "스위퍼", "커브"],
  OFFSPEED: ["체인지업"],
};

const snellRepertoire: Record<PitchFamily, string[]> = {
  FASTBALL: ["포심"],
  BREAKING: ["슬라이더", "커브"],
  OFFSPEED: ["체인지업"],
};

const petersonRepertoire: Record<PitchFamily, string[]> = {
  FASTBALL: ["포심", "싱커"],
  BREAKING: ["슬라이더", "커브"],
  OFFSPEED: ["체인지업"],
};

const canningRepertoire: Record<PitchFamily, string[]> = {
  FASTBALL: ["포심", "커터", "싱커"],
  BREAKING: ["슬라이더", "너클 커브"],
  OFFSPEED: ["체인지업"],
};

const verlanderRepertoire: Record<PitchFamily, string[]> = {
  FASTBALL: ["포심"],
  BREAKING: ["슬라이더", "스위퍼", "커브"],
  OFFSPEED: ["체인지업"],
};

const allGroups: PitchGroup[] = [
  "FOUR_SEAM",
  "MOVING_FASTBALL",
  "SLIDER",
  "CURVE",
  "CHANGEUP",
];

const kellyScenario: BenchmarkScenario = {
    id: "kelly-cronenworth",
    pitcher: "Merrill Kelly",
    batter: "Jake Cronenworth",
    date: "2025.06.15",
    inning: "4회 초",
    awayTeam: "SD",
    homeTeam: "AZ",
    awayScore: 1,
    homeScore: 0,
    outs: 0,
    bases: [false, true, false],
    outcome: "10구 2점 홈런",
    gameUrl: "https://www.mlb.com/video/game/777491",
    repertoire: kellyRepertoire,
    availableGroups: allGroups,
    pitches: makePitches([
      ["0-0", "FC", "루킹 스트라이크", "fd55d110-d28c-361a-be21-1e513b60b231"],
      ["0-1", "FC", "파울", "1aea2de1-4b90-3501-b2ce-1f56a78badf0"],
      ["0-2", "FF", "파울", "f6430d0e-b5b1-3281-97a3-013daee24b0f"],
      ["0-2", "CH", "파울", "14dffd39-d218-3920-b33f-87aeda928ee6"],
      ["0-2", "FF", "파울", "f02c927f-ed6f-3f94-825a-f692bf55d125"],
      ["0-2", "CU", "파울", "e3eddbec-e6f4-359a-921e-0ecd18d7b6c3"],
      ["0-2", "FC", "볼", "52edd36c-89fb-3af6-9fd0-185544ca9257"],
      ["1-2", "FC", "파울", "5c030906-2f05-370c-9a81-a0940529ae20"],
      ["1-2", "FF", "볼", "00f1d414-7529-3536-92c3-fe0e764f6639"],
      ["2-2", "CH", "2점 홈런", "1a32b39b-51f1-371c-a4b9-31bf02eeecfe"],
    ]),
};

const version1Scenarios: BenchmarkScenario[] = [
  {
    id: "brown-rodriguez",
    pitcher: "Hunter Brown",
    batter: "Julio Rodríguez",
    date: "2025.09.19",
    inning: "1회 초",
    awayTeam: "SEA",
    homeTeam: "HOU",
    awayScore: 0,
    homeScore: 0,
    outs: 2,
    bases: [false, false, false],
    outcome: "3구 홈런",
    gameUrl: "https://www.mlb.com/video/game/776262",
    repertoire: brownRepertoire,
    availableGroups: allGroups,
    pitches: makePitches([
      ["0-0", "KC", "볼", "52cf57c1-8e40-3d47-8a12-0deeac2d862a"],
      ["1-0", "SL", "볼", "73c9f8b1-ef00-3797-afea-bab067cffeee"],
      ["2-0", "SI", "홈런", "cb946579-155c-3f5c-bc46-459d59cf2d09"],
    ]),
  },
  {
    id: "flexen-judge",
    pitcher: "Chris Flexen",
    batter: "Aaron Judge",
    date: "2025.07.11",
    inning: "1회 말",
    awayTeam: "CHC",
    homeTeam: "NYY",
    awayScore: 0,
    homeScore: 0,
    outs: 1,
    bases: [false, false, false],
    outcome: "5구 볼넷",
    gameUrl: "https://www.mlb.com/video/game/777153",
    repertoire: flexenRepertoire,
    availableGroups: allGroups,
    pitches: makePitches([
      ["0-0", "FC", "볼", "986e4c92-6723-37c3-a099-40d6b95b2a76"],
      ["1-0", "FC", "볼", "9014ad03-dcb2-3e84-b16a-572274d70b32"],
      ["2-0", "FF", "볼", "61f60747-5fab-3e01-9f40-94a3bef8019f"],
      ["3-0", "FF", "헛스윙 스트라이크", "aba18eca-7f64-35d0-b140-90ca100ccb5f"],
      ["3-1", "FF", "볼넷", "6c100aac-00d7-31a9-a55b-ba6cfe6e2238"],
    ]),
  },
  {
    id: "snell-tatis",
    pitcher: "Blake Snell",
    batter: "Fernando Tatis Jr.",
    date: "2025.08.16",
    inning: "3회 초",
    awayTeam: "SD",
    homeTeam: "LAD",
    awayScore: 0,
    homeScore: 5,
    outs: 2,
    bases: [false, false, false],
    outcome: "4구 삼진",
    gameUrl: "https://www.mlb.com/video/game/776717",
    repertoire: snellRepertoire,
    availableGroups: ["FOUR_SEAM", "SLIDER", "CURVE", "CHANGEUP"],
    pitches: makePitches([
      ["0-0", "CH", "헛스윙 스트라이크", "ba1da303-1cbf-3cc3-a8d4-cc8c8a9794f9"],
      ["0-1", "FF", "볼", "dfcc2efe-ed10-3bfe-ad1f-c55442605802"],
      ["1-1", "CH", "파울", "ca55dfcc-48c7-3d30-98be-b44169342c93"],
      ["1-2", "CU", "루킹 삼진", "ec1d712f-2ef5-3f68-b732-bec2fcbbd920"],
    ]),
  },
  {
    id: "snell-carroll",
    pitcher: "Blake Snell",
    batter: "Corbin Carroll",
    date: "2025.08.29",
    inning: "6회 초",
    awayTeam: "AZ",
    homeTeam: "LAD",
    awayScore: 2,
    homeScore: 0,
    outs: 0,
    bases: [false, false, false],
    outcome: "3구 2루타",
    gameUrl: "https://www.mlb.com/video/game/776534",
    repertoire: snellRepertoire,
    availableGroups: ["FOUR_SEAM", "SLIDER", "CURVE", "CHANGEUP"],
    pitches: makePitches([
      ["0-0", "CU", "볼", "3b89133d-ebed-3f05-9c76-89ae50ded352"],
      ["1-0", "SL", "볼", "6e427864-20b8-3db8-a2fb-bc96d1ea2b7c"],
      ["2-0", "SL", "2루타", "22fc72c4-17a0-35be-a4c1-6ecf265e9bf4"],
    ]),
  },
  kellyScenario,
];

const version2Scenarios: BenchmarkScenario[] = [
  {
    id: "v2-brown-benintendi",
    pitcher: "Hunter Brown",
    batter: "Andrew Benintendi",
    date: "2026.07.25",
    inning: "7회 말",
    awayTeam: "HOU",
    homeTeam: "CWS",
    awayScore: 3,
    homeScore: 0,
    outs: 2,
    bases: [false, false, false],
    outcome: "5구 헛스윙 삼진",
    gameUrl: "https://www.mlb.com/video/game/824571",
    repertoire: {
      FASTBALL: ["포심", "싱커"],
      BREAKING: ["슬라이더", "너클 커브"],
      OFFSPEED: ["체인지업"],
    },
    availableGroups: allGroups,
    pitches: makePitches([
      ["0-0", "CH", "볼", "c0248f99-7bfe-3381-abc2-66db0281b153"],
      ["1-0", "SI", "볼", "2b92e029-d9e6-3899-9c53-2862051a2233"],
      ["2-0", "SI", "헛스윙 스트라이크", "4ce56d94-f561-3da6-b8df-e6bb28c8839b"],
      ["2-1", "FF", "파울", "8d96c56f-4392-39d2-b564-84e6b9e83536"],
      ["2-2", "KC", "헛스윙 삼진", "68e5507b-af4e-307f-bee3-60fa6bc514c8"],
    ]),
  },
  {
    id: "v2-peterson-alonso",
    pitcher: "David Peterson",
    batter: "Pete Alonso",
    date: "2026.07.09",
    inning: "4회 말",
    awayTeam: "CHC",
    homeTeam: "BAL",
    awayScore: 0,
    homeScore: 1,
    outs: 0,
    bases: [false, false, false],
    outcome: "6구 볼넷",
    gameUrl: "https://www.mlb.com/video/game/824816",
    repertoire: petersonRepertoire,
    availableGroups: allGroups,
    pitches: makePitches([
      ["0-0", "CU", "볼", "1448362a-255e-3dc8-833e-9148d0140251"],
      ["1-0", "CH", "볼", "33b527c2-de46-3d1d-a85b-a202998303b0"],
      ["2-0", "FF", "파울", "0b55b332-b8ed-3023-8e67-e60afdaa41b3"],
      ["2-1", "SL", "볼", "bea1051e-091c-32f0-a243-cbb3bc6eb9e2"],
      ["3-1", "FF", "헛스윙 스트라이크", "98214476-7eab-360f-be4a-4fa7a27cd534"],
      ["3-2", "FF", "볼넷", "d1a12e84-94b0-3d08-96c6-365a36d6e4b5"],
    ]),
  },
  {
    id: "v2-canning-albies",
    pitcher: "Griffin Canning",
    batter: "Ozzie Albies",
    date: "2026.07.23",
    inning: "4회 말",
    awayTeam: "SD",
    homeTeam: "ATL",
    awayScore: 0,
    homeScore: 0,
    outs: 0,
    bases: [false, false, false],
    outcome: "4구 홈런",
    gameUrl: "https://www.mlb.com/video/game/824893",
    repertoire: canningRepertoire,
    availableGroups: allGroups,
    pitches: makePitches([
      ["0-0", "SL", "헛스윙 스트라이크", "8d357201-6eb1-3411-a0c2-15104f722811"],
      ["0-1", "CH", "볼", "0aeb9e12-7966-32f1-9fc1-b1f978f593e7"],
      ["1-1", "SL", "파울", "849226e4-c200-3c1f-98cc-0fd152a9808d"],
      ["1-2", "SI", "홈런", "cefd2c3a-4f7e-33f8-8179-1727d5ac0cf4"],
    ]),
  },
  {
    id: "v2-verlander-arenado",
    pitcher: "Justin Verlander",
    batter: "Nolan Arenado",
    date: "2026.03.30",
    inning: "3회 말",
    awayTeam: "DET",
    homeTeam: "AZ",
    awayScore: 0,
    homeScore: 5,
    outs: 0,
    bases: [false, true, false],
    outcome: "6구 좌익수 뜬공",
    gameUrl: "https://www.mlb.com/video/game/825108",
    repertoire: verlanderRepertoire,
    availableGroups: ["FOUR_SEAM", "SLIDER", "CURVE", "CHANGEUP"],
    pitches: makePitches([
      ["0-0", "SL", "루킹 스트라이크", "4f53c70d-8181-3a01-a992-9b0c2e203d66"],
      ["0-1", "SL", "헛스윙 스트라이크", "1ae6818d-80f1-3289-b0ff-028a83f31071"],
      ["0-2", "SL", "파울", "02a55c7e-271c-3906-91c2-9ed0944854ca"],
      ["0-2", "CU", "파울", "98604fe0-8c56-3a4c-ba06-0134927d7c28"],
      ["0-2", "CH", "볼", "03d2bee0-58ab-3bc7-b4eb-25a204201b41"],
      ["1-2", "FF", "좌익수 뜬공", "870732ef-c899-3692-9565-866a459ac142"],
    ]),
  },
  {
    id: "v2-kelly-wilson",
    pitcher: "Merrill Kelly",
    batter: "Jacob Wilson",
    date: "2026.07.22",
    inning: "3회 초",
    awayTeam: "ATH",
    homeTeam: "AZ",
    awayScore: 0,
    homeScore: 0,
    outs: 1,
    bases: [false, true, false],
    outcome: "4구 우전 안타",
    gameUrl: "https://www.mlb.com/video/game/825055",
    repertoire: kellyRepertoire,
    availableGroups: allGroups,
    pitches: makePitches([
      ["0-0", "SL", "볼", "0e8238d4-2cc2-3644-806f-327ebb73686e"],
      ["1-0", "CH", "파울팁", "ea160849-0035-36d4-a8c5-0bbc9566217d"],
      ["1-1", "SI", "볼", "53571ca2-3d6b-3571-a3f9-4296c9d62cae"],
      ["2-1", "FC", "우전 안타", "cc862cad-9879-3c89-85c3-4027db1c5bc9"],
    ]),
  },
];

const benchmarkVersions: Record<
  BenchmarkVersion,
  BenchmarkDefinition
> = {
  v1: {
    label: "Pitch Test Version 1",
    year: "2025",
    description: "기존 공개 실험 · Brown, Flexen, Snell, Kelly",
    scenarios: version1Scenarios,
  },
  v2: {
    label: "Pitch Test Version 2",
    year: "2026",
    description: "새로운 2026 투구 · Verlander 포함",
    scenarios: version2Scenarios,
  },
};

const optionByKey = Object.fromEntries(
  pitchOptions.map((option) => [option.key, option]),
) as Record<PitchGroup, (typeof pitchOptions)[number]>;

function officialVideoUrl(playId: string) {
  return `https://baseballsavant.mlb.com/sporty-videos?playId=${playId}`;
}

function SituationStrip({
  pitch,
  scenario,
}: {
  pitch: BenchmarkPitch;
  scenario: BenchmarkScenario;
}) {
  const occupiedBases = scenario.bases
    .map((occupied, index) => occupied ? `${index + 1}루` : "")
    .filter(Boolean)
    .join(", ");
  return (
    <section className="hb-situation" aria-label="경기 상황">
      <div><small>INNING</small><strong>{scenario.inning}</strong></div>
      <div>
        <small>SCORE</small>
        <strong>
          {scenario.awayTeam} {scenario.awayScore} <i>—</i>{" "}
          {scenario.homeScore} {scenario.homeTeam}
        </strong>
      </div>
      <div><small>COUNT</small><strong>{pitch.count}</strong></div>
      <div><small>OUT</small><strong>{scenario.outs}</strong></div>
      <div className="hb-base">
        <small>RUNNER</small>
        <span aria-label={occupiedBases ? `${occupiedBases} 주자 있음` : "주자 없음"}>
          {[...scenario.bases].reverse().map((occupied, index) => (
            <i className={occupied ? "occupied" : ""} key={index} />
          ))}
        </span>
      </div>
    </section>
  );
}

function RepertoireStrip({ scenario }: { scenario: BenchmarkScenario }) {
  return (
    <section className="hb-pitcher-repertoire" aria-label={`${scenario.pitcher} 구종`}>
      <div>
        <span>PITCHER REPERTOIRE</span>
        <strong>{scenario.pitcher} 구종</strong>
      </div>
      <ul>
        {familyOptions.map((family) => (
          <li key={family.key}>
            <small>{family.label}</small>
            <b>{scenario.repertoire[family.key].join(" · ")}</b>
          </li>
        ))}
      </ul>
    </section>
  );
}

function PitchChoices({
  availableGroups,
  selectedFamily,
  selected,
  locked,
  onFamilySelect,
  onSelect,
}: {
  availableGroups: PitchGroup[];
  selectedFamily?: PitchFamily;
  selected?: PitchGroup;
  locked: boolean;
  onFamilySelect: (family: PitchFamily) => void;
  onSelect: (pitch: PitchGroup) => void;
}) {
  const detailOptions = selectedFamily
    ? pitchOptions.filter(
        (option) =>
          option.family === selectedFamily &&
          availableGroups.includes(option.key),
      )
    : [];
  return (
    <div className="hb-hierarchical-choice">
      <div className="hb-choice-heading">
        <span>1</span>
        <div>
          <strong>먼저 계열을 고르세요</strong>
          <small>계열 적중 1점</small>
        </div>
      </div>
      <div className="hb-family-choices" role="radiogroup" aria-label="예상 구종 계열">
        {familyOptions.map((option) => (
          <button
            aria-checked={selectedFamily === option.key}
            className={selectedFamily === option.key ? "selected" : ""}
            disabled={locked}
            key={option.key}
            onClick={() => onFamilySelect(option.key)}
            role="radio"
            type="button"
          >
            <strong>{option.label}</strong>
            <small>{option.detail}</small>
          </button>
        ))}
      </div>
      <div className={selectedFamily ? "hb-detail-step open" : "hb-detail-step"}>
        <div className="hb-choice-heading">
          <span>2</span>
          <div>
            <strong>
              {selectedFamily
                ? `${familyByKey[selectedFamily].label}의 상세 구종`
                : "계열을 선택하면 상세 구종이 열립니다"}
            </strong>
            <small>상세 적중 시 총 3점</small>
          </div>
        </div>
        {selectedFamily ? (
          <div className="hb-choices" role="radiogroup" aria-label="예상 상세 구종">
            {detailOptions.map((option, index) => (
              <button
                aria-checked={selected === option.key}
                className={selected === option.key ? "selected" : ""}
                disabled={locked}
                key={option.key}
                onClick={() => onSelect(option.key)}
                role="radio"
                type="button"
              >
                <span>{index + 1}</span>
                <strong>{option.label}</strong>
                <small>{option.detail}</small>
                <code>{option.code}</code>
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function ModelPredictionReveal({
  answer,
  modelPrediction,
  onRevealResult,
}: {
  answer: Answer;
  modelPrediction?: ModelPrediction | null;
  onRevealResult: () => void;
}) {
  const modelProbabilities = modelPrediction
    ? Object.entries(modelPrediction.probabilities)
        .sort(([, left], [, right]) => (right ?? 0) - (left ?? 0))
        .slice(0, 3)
    : [];
  return (
    <section className="hb-reveal" aria-live="polite">
      <p className="hb-reveal-guide">
        답안이 확정되었습니다. 모델의 선택을 확인한 뒤 실제 결과를
        확인해보세요.
      </p>
      <div className="hb-verdict">
        <small>내 선택</small>
        <strong>{optionByKey[answer.selected].label}</strong>
        <span>{familyByKey[answer.selectedFamily].label}</span>
      </div>
      {modelPrediction ? (
        <div className="hb-model-prediction">
          <small>{modelPrediction.modelVersion} MODEL PREDICTION</small>
          <div>
            <span>모델의 선택</span>
            <strong>{optionByKey[modelPrediction.selected].label}</strong>
            <b>{Math.round(modelPrediction.confidence * 100)}%</b>
          </div>
          <ol>
            {modelProbabilities.map(([group, probability]) => (
              <li key={group}>
                <span>{optionByKey[group as PitchGroup].label}</span>
                <strong>{Math.round((probability ?? 0) * 100)}%</strong>
              </li>
            ))}
          </ol>
        </div>
      ) : null}
      <button className="hb-submit" onClick={onRevealResult} type="button">
        결과 확인하기 →
      </button>
    </section>
  );
}

function VideoReveal({
  answer,
  pitch,
  scenario,
  isLastPitch,
  isLastScenario,
  onNext,
}: {
  answer: Answer;
  pitch: BenchmarkPitch;
  scenario: BenchmarkScenario;
  isLastPitch: boolean;
  isLastScenario: boolean;
  onNext: () => void;
}) {
  const verdict = answer.detailHit
    ? "상세 적중 · +3점"
    : answer.familyHit
      ? "계열 적중 · +1점"
      : "다른 계열 · +0점";
  const selectedLabel = optionByKey[answer.selected].label;
  return (
    <section className="hb-reveal" aria-live="polite">
      <p className="hb-reveal-guide">
        실제 투구 결과입니다. 영상을 확인한 뒤 다음으로 이동하세요.
      </p>
      <div
        className={
          answer.detailHit
            ? "hb-verdict correct"
            : answer.familyHit
              ? "hb-verdict family"
              : "hb-verdict"
        }
      >
        <small>{verdict}</small>
        <strong>
          {optionByKey[pitch.actual].label}
          {pitch.rawLabel !== optionByKey[pitch.actual].label
            ? <em>실제 {pitch.rawLabel}</em>
            : null}
        </strong>
        <span>
          내 선택 · {familyByKey[answer.selectedFamily].label} &gt; {selectedLabel}
          <b>{pitch.result}</b>
        </span>
      </div>
      <div className="hb-video">
        <iframe
          allow="autoplay; fullscreen; picture-in-picture; encrypted-media"
          src={officialVideoUrl(pitch.playId)}
          title={`MLB 공식 ${scenario.pitcher} ${pitch.number}구 영상`}
        />
      </div>
      <div className="hb-video-actions">
        <a
          href={officialVideoUrl(pitch.playId)}
          rel="noreferrer"
          target="_blank"
        >
          영상이 안 보이면 MLB에서 열기 ↗
        </a>
        <button onClick={onNext} type="button">
          {isLastPitch
            ? isLastScenario
              ? "최종 리더보드 보기 →"
              : "다음 타석으로 →"
            : "다음 투구로 →"}
        </button>
      </div>
    </section>
  );
}

function PitchHistory({
  scenario,
  answers,
  activeNumber,
}: {
  scenario: BenchmarkScenario;
  answers: Answer[];
  activeNumber: number;
}) {
  return (
    <ol className="hb-history" aria-label="현재 타석 투구 기록">
      {scenario.pitches.slice(0, activeNumber).map((pitch) => {
        const answer = answers.find(
          (item) =>
            item.scenarioId === scenario.id &&
            item.pitchNumber === pitch.number,
        );
        return (
          <li
            className={
              pitch.number === activeNumber
                ? "active"
                : answer
                  ? answer.detailHit
                    ? "done correct"
                    : answer.familyHit
                      ? "done family"
                      : "done"
                  : ""
            }
            key={pitch.number}
          >
            <span>{pitch.number}</span>
            <div>
              <small>{pitch.count}</small>
              <strong>{answer ? pitch.rawLabel : "—"}</strong>
            </div>
            <i>{answer ? `+${answer.points}` : ""}</i>
          </li>
        );
      })}
    </ol>
  );
}

interface PredictionProps {
  scenario: BenchmarkScenario;
  scenarioNumber: number;
  scenarioCount: number;
  pitch: BenchmarkPitch;
  globalPitchNumber: number;
  totalPitchCount: number;
  versionLabel: string;
  isLastPitch: boolean;
  isLastScenario: boolean;
  selectedFamily?: PitchFamily;
  selected?: PitchGroup;
  answer?: Answer;
  answers: Answer[];
  showResult: boolean;
  onFamilySelect: (family: PitchFamily) => void;
  onSelect: (pitch: PitchGroup) => void;
  onRevealResult: () => void;
  onSubmit: () => void;
  onNext: () => void;
}

function PredictionFlow({
  scenario,
  scenarioNumber,
  scenarioCount,
  pitch,
  globalPitchNumber,
  totalPitchCount,
  versionLabel,
  isLastPitch,
  isLastScenario,
  selectedFamily,
  selected,
  answer,
  answers,
  showResult,
  onFamilySelect,
  onSelect,
  onRevealResult,
  onSubmit,
  onNext,
}: PredictionProps) {
  return (
    <div className="hb-layout-a">
      <header className="hb-topbar">
        <div className="hb-wordmark"><i /> PITCH / TEST</div>
        <span>{versionLabel} · {scenarioCount}개 시나리오</span>
      </header>
      <main className="hb-form-shell">
        <section className="hb-form-card">
          <div className="hb-progress-copy">
            <span>SCENARIO {scenarioNumber} · PITCH {pitch.number}</span>
            <strong>{globalPitchNumber} / {totalPitchCount}</strong>
          </div>
          <div className="hb-progress">
            <i style={{ width: `${(globalPitchNumber / totalPitchCount) * 100}%` }} />
          </div>
          <p className="hb-kicker">
            {scenario.pitcher.toUpperCase()} vs {scenario.batter.toUpperCase()}
          </p>
          <h1>다음 구종은<br />무엇일까요?</h1>
          <SituationStrip pitch={pitch} scenario={scenario} />
          {answer && showResult ? (
            <VideoReveal
              answer={answer}
              isLastPitch={isLastPitch}
              isLastScenario={isLastScenario}
              onNext={onNext}
              pitch={pitch}
              scenario={scenario}
            />
          ) : answer ? (
            <ModelPredictionReveal
              answer={answer}
              modelPrediction={pitch.modelPrediction}
              onRevealResult={onRevealResult}
            />
          ) : (
            <>
              <RepertoireStrip scenario={scenario} />
              <PitchChoices
                availableGroups={scenario.availableGroups}
                locked={false}
                onFamilySelect={onFamilySelect}
                onSelect={onSelect}
                selected={selected}
                selectedFamily={selectedFamily}
              />
              <button
                className="hb-submit"
                disabled={!selected}
                onClick={onSubmit}
                type="button"
              >
                답 확정하기
              </button>
            </>
          )}
        </section>
        <aside>
          <span>AT-BAT LOG · 진행된 투구</span>
          <PitchHistory
            activeNumber={pitch.number}
            answers={answers}
            scenario={scenario}
          />
        </aside>
      </main>
    </div>
  );
}

function Intro({
  benchmark,
  nickname,
  experience,
  onBack,
  onNickname,
  onExperience,
  onStart,
}: {
  benchmark: BenchmarkDefinition;
  nickname: string;
  experience: string;
  onBack: () => void;
  onNickname: (value: string) => void;
  onExperience: (value: string) => void;
  onStart: () => void;
}) {
  const levels = [
    ["new", "거의 안 봐요", "MLB 구종에 익숙하지 않음"],
    ["casual", "가끔 봐요", "선수와 주요 구종은 조금 앎"],
    ["fan", "자주 봐요", "경기와 구종을 자주 접함"],
  ];
  return (
    <main className="hb-intro">
      <div className="hb-intro-mark"><i /> PITCH / TEST</div>
      <p>{benchmark.label.toUpperCase()} · 약 4~5분</p>
      <h1>당신은 다음 공을<br /><em>읽을 수 있나요?</em></h1>
      <p className="hb-set-context">{benchmark.description}</p>
      <p className="hb-intro-copy">
        {new Set(benchmark.scenarios.map((scenario) => scenario.pitcher)).size}명의
        MLB 투수가 {benchmark.year}년에 던진 실제 {benchmark.scenarios.length}개
        타석, 총{" "}
        {benchmark.scenarios.reduce(
          (sum, scenario) => sum + scenario.pitches.length,
          0,
        )}구를 연속으로 예측합니다.
        매 투구 전에 3개 계열을 먼저 고르고 상세 구종까지 선택한 뒤 MLB
        공식 영상으로 결과를 확인합니다.
      </p>
      <div className="hb-scoring-rules">
        <div><strong>+3</strong><span>상세 구종 적중</span></div>
        <div><strong>+1</strong><span>계열만 적중</span></div>
        <div><strong>+0</strong><span>다른 계열</span></div>
      </div>
      <label className="hb-nickname" htmlFor="benchmark-nickname">
        <span>결과에 표시할 닉네임</span>
        <input
          autoComplete="nickname"
          id="benchmark-nickname"
          maxLength={20}
          onBlur={() => onNickname(normalizeNickname(nickname))}
          onChange={(event) => onNickname(event.target.value)}
          placeholder="예: 야구팬"
          type="text"
          value={nickname}
        />
        <small>1~20자 · 한글, 영문, 숫자, 공백, _, -</small>
      </label>
      <section>
        <span>평소 MLB를 얼마나 보나요?</span>
        <div>
          {levels.map(([value, label, detail]) => (
            <button
              aria-pressed={experience === value}
              className={experience === value ? "selected" : ""}
              key={value}
              onClick={() => onExperience(value)}
              type="button"
            >
              <strong>{label}</strong>
              <small>{detail}</small>
            </button>
          ))}
        </div>
      </section>
      <button
        className="hb-start"
        disabled={
          !experience || !nicknamePattern.test(normalizeNickname(nickname))
        }
        onClick={onStart}
        type="button"
      >
        {benchmark.scenarios.length}개 타석 ·{" "}
        {benchmark.scenarios.reduce(
          (sum, scenario) => sum + scenario.pitches.length,
          0,
        )}구 예측 시작하기 →
      </button>
      <button className="hb-version-back" onClick={onBack} type="button">
        ← 다른 버전 선택
      </button>
      <small className="hb-privacy">
        닉네임과 테스트 응답만 수집하며 이메일 등 연락처는 받지 않습니다.
      </small>
    </main>
  );
}

function Summary({
  answers,
  experience,
  scenarios,
  totalPitchCount,
  submissionStatus,
  onRetry,
  onRestart,
}: {
  answers: Answer[];
  experience: string;
  scenarios: BenchmarkScenario[];
  totalPitchCount: number;
  submissionStatus?: "saving" | "saved" | "error";
  onRetry: () => void;
  onRestart: () => void;
}) {
  const detailHits = answers.filter((answer) => answer.detailHit).length;
  const familyHits = answers.filter((answer) => answer.familyHit).length;
  const totalPoints = answers.reduce((sum, answer) => sum + answer.points, 0);
  const maxPoints = totalPitchCount * 3;
  const modelResults = scenarios.flatMap((scenario) =>
    scenario.pitches.flatMap((pitch) => {
      const selected = pitch.modelPrediction?.selected;
      if (!selected) return [];
      const detailHit = selected === pitch.actual;
      const familyHit = groupFamily[selected] === groupFamily[pitch.actual];
      return [{
        scenarioId: scenario.id,
        detailHit,
        familyHit,
        points: detailHit ? 3 : familyHit ? 1 : 0,
      }];
    }),
  );
  const hasModelResults = modelResults.length === totalPitchCount;
  const modelDetailHits = modelResults.filter((result) => result.detailHit).length;
  const modelFamilyHits = modelResults.filter((result) => result.familyHit).length;
  const modelPoints = modelResults.reduce((sum, result) => sum + result.points, 0);
  const averageMs = Math.round(
    answers.reduce((sum, answer) => sum + answer.responseMs, 0) / answers.length,
  );
  return (
    <main className="hb-summary">
      <p>ALL SCENARIOS COMPLETE</p>
      <div className="hb-summary-score">
        <span>총점</span>
        <strong>{totalPoints}<i>/{maxPoints}</i></strong>
        <small>{Math.round((totalPoints / maxPoints) * 100)}%</small>
      </div>
      <h1>당신의 최종 점수는<br />{totalPoints}점입니다.</h1>
      <div className="hb-summary-grid">
        <div><span>상세 적중 · 3점</span><strong>{detailHits} / {totalPitchCount}</strong></div>
        <div><span>계열 적중 포함</span><strong>{familyHits} / {totalPitchCount}</strong></div>
        <div><span>평균 응답 시간</span><strong>{(averageMs / 1000).toFixed(1)}초</strong></div>
      </div>
      {hasModelResults ? (
        <section className="hb-model-summary" aria-labelledby="model-summary-title">
          <header>
            <div>
              <span>HUMAN vs V8.4</span>
              <h2 id="model-summary-title">모델과 종합 결과 비교</h2>
            </div>
            <p>
              {totalPoints === modelPoints
                ? "동점"
                : totalPoints > modelPoints
                  ? `내가 ${totalPoints - modelPoints}점 앞섰습니다`
                  : `모델이 ${modelPoints - totalPoints}점 앞섰습니다`}
            </p>
          </header>
          <div className="hb-model-summary-grid">
            <article>
              <small>나</small>
              <strong>{totalPoints}<i>/{maxPoints}</i></strong>
              <span>상세 {detailHits} · 계열 포함 {familyHits}</span>
            </article>
            <b>VS</b>
            <article className="model">
              <small>OyarZabal V8.4</small>
              <strong>{modelPoints}<i>/{maxPoints}</i></strong>
              <span>상세 {modelDetailHits} · 계열 포함 {modelFamilyHits}</span>
            </article>
          </div>
        </section>
      ) : null}
      <section className="hb-scenario-results" aria-label="시나리오별 결과">
        {scenarios.map((scenario, scenarioIndex) => {
          const scenarioAnswers = answers.filter(
            (answer) => answer.scenarioId === scenario.id,
          );
          const scenarioPoints = scenarioAnswers.reduce(
            (sum, answer) => sum + answer.points,
            0,
          );
          const scenarioModelPoints = modelResults
            .filter((result) => result.scenarioId === scenario.id)
            .reduce((sum, result) => sum + result.points, 0);
          return (
            <article key={scenario.id}>
              <header>
                <div>
                  <span>SCENARIO {scenarioIndex + 1}</span>
                  <strong>{scenario.pitcher} vs {scenario.batter}</strong>
                  <small>{scenario.outcome}</small>
                </div>
                <div className="hb-scenario-score">
                  <b>나 {scenarioPoints}/{scenario.pitches.length * 3}</b>
                  {hasModelResults ? (
                    <small>MODEL {scenarioModelPoints}/{scenario.pitches.length * 3}</small>
                  ) : null}
                </div>
              </header>
              <div className="hb-answer-sheet">
                {scenario.pitches.map((pitch) => {
                  const pitchAnswer = scenarioAnswers.find(
                    (answer) => answer.pitchNumber === pitch.number,
                  )!;
                  return (
                    <div key={`${scenario.id}-${pitch.number}`}>
                      <span>{pitch.number}구</span>
                      <strong>{pitch.rawLabel}</strong>
                      <i
                        className={
                          pitchAnswer.detailHit
                            ? "correct"
                            : pitchAnswer.familyHit
                              ? "family"
                              : ""
                        }
                      >
                        +{pitchAnswer.points}
                      </i>
                    </div>
                  );
                })}
              </div>
              <a href={scenario.gameUrl} rel="noreferrer" target="_blank">
                MLB 원본 경기 ↗
              </a>
            </article>
          );
        })}
      </section>
      <p className="hb-summary-meta">
        MLB 시청 경험 · {experience === "fan" ? "자주 봄" : experience === "casual" ? "가끔 봄" : "거의 안 봄"}
        <span>완료 · {scenarios.length}개 시나리오 · {totalPitchCount}구</span>
      </p>
      {submissionStatus ? (
        <div className={`hb-save-status ${submissionStatus}`}>
          {submissionStatus === "saving"
            ? "결과를 저장하고 있습니다…"
            : submissionStatus === "saved"
              ? "결과가 저장되었습니다."
              : (
                <>
                  결과 저장에 실패했습니다.
                  <button onClick={onRetry} type="button">다시 저장</button>
                </>
              )}
        </div>
      ) : null}
      <div className="hb-summary-actions">
        <button onClick={onRestart} type="button">다시 해보기</button>
      </div>
      <small>PITCH / TEST · HUMAN BENCHMARK</small>
    </main>
  );
}

function getDailyParticipantId() {
  const storageKey = "pitch-test-daily-participant";
  try {
    const existing = window.localStorage.getItem(storageKey);
    if (existing) return existing;
    const created = window.crypto.randomUUID();
    window.localStorage.setItem(storageKey, created);
    return created;
  } catch {
    return window.crypto.randomUUID();
  }
}

async function jsonResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      typeof payload.error === "string"
        ? payload.error
        : "요청을 처리하지 못했습니다.",
    );
  }
  return payload as T;
}

function DailyHistory({
  history,
  scenarioId,
}: {
  history: DailyReveal[];
  scenarioId: string;
}) {
  const scenarioHistory = history.filter(
    (answer) => answer.scenarioId === scenarioId,
  );
  return (
    <ol className="hb-history" aria-label="현재 타석 투구 기록">
      {scenarioHistory.map((answer) => (
        <li
          className={
            answer.detailHit
              ? "done correct"
              : answer.familyHit
                ? "done family"
                : "done"
          }
          key={answer.pitchNumber}
        >
          <span>{answer.pitchNumber}</span>
          <div>
            <small>{answer.count}</small>
            <strong>{answer.rawLabel}</strong>
          </div>
          <i>+{answer.points}</i>
        </li>
      ))}
    </ol>
  );
}

function DailyIntro({
  challenge,
  nickname,
  error,
  loading,
  onBack,
  onNickname,
  onStart,
}: {
  challenge: DailyChallenge;
  nickname: string;
  error: string;
  loading: boolean;
  onBack: () => void;
  onNickname: (value: string) => void;
  onStart: () => void;
}) {
  return (
    <main className="hb-intro hb-daily-intro">
      <div className="hb-intro-mark"><i /> PITCH / TEST</div>
      <p>DAILY #{challenge.number} · 약 3~4분</p>
      <h1>최근 MLB {challenge.scenarioCount}타석을<br /><em>읽어보세요.</em></h1>
      <p className="hb-intro-copy">
        {challenge.sourceDate} MLB 경기에서 고른 {challenge.scenarioCount}타석입니다. 모든 사용자가
        같은 투구를 하루 한 번 예측하고, 완료하면 오늘의 순위와 연속 참여일을
        확인할 수 있습니다.
      </p>
      <div className="hb-scoring-rules">
        <div><strong>+3</strong><span>상세 구종 적중</span></div>
        <div><strong>+1</strong><span>계열만 적중</span></div>
        <div><strong>{challenge.scenarioCount}</strong><span>오늘의 타석</span></div>
      </div>
      <label className="hb-nickname" htmlFor="daily-nickname">
        <span>리더보드에 표시할 닉네임</span>
        <input
          autoComplete="nickname"
          id="daily-nickname"
          maxLength={20}
          onBlur={() => onNickname(normalizeNickname(nickname))}
          onChange={(event) => onNickname(event.target.value)}
          placeholder="예: 야구팬"
          type="text"
          value={nickname}
        />
        <small>첫 완료 기록만 오늘의 공식 순위에 반영됩니다.</small>
      </label>
      {error ? <p className="hb-daily-error" role="alert">{error}</p> : null}
      <button
        className="hb-start"
        disabled={loading || !nicknamePattern.test(normalizeNickname(nickname))}
        onClick={onStart}
        type="button"
      >
        {loading
          ? "오늘의 게임을 불러오는 중…"
          : `오늘의 ${challenge.scenarioCount}타석 시작하기 →`}
      </button>
      <button className="hb-version-back" onClick={onBack} type="button">
        ← 다른 모드 선택
      </button>
      <small className="hb-privacy">
        닉네임과 투구 응답만 저장하며, 같은 브라우저의 첫 기록만 반영합니다.
      </small>
    </main>
  );
}

function DailyResultView({
  session,
  resumeNotice,
  onBack,
}: {
  session: DailySession;
  resumeNotice?: string;
  onBack: () => void;
}) {
  const [shareStatus, setShareStatus] = useState("");
  const result = session.result!;
  const rowsByScenario = Array.from(
    { length: session.challenge.scenarioCount },
    (_, index) =>
      session.history.filter(
        (answer) => answer.scenarioNumber === index + 1,
      ),
  );
  const shareText = [
    `PITCH TEST DAILY #${session.challenge.number}`,
    "",
    ...rowsByScenario.map(
      (answers, index) =>
        `PA ${index + 1}  ${answers
          .map((answer) =>
            answer.detailHit ? "🟩" : answer.familyHit ? "🟨" : "⬛",
          )
          .join("")}`,
    ),
    "",
    `${result.points} / ${result.maxPoints}점 · ${result.rank}위`,
    `🔥 ${result.streak}일 연속`,
  ].join("\n");

  const share = async () => {
    try {
      if (navigator.share) {
        await navigator.share({ text: shareText, title: "Pitch Test Daily" });
      } else {
        await navigator.clipboard.writeText(shareText);
      }
      setShareStatus("공유 결과를 복사했습니다.");
    } catch {
      setShareStatus("공유를 취소했거나 복사하지 못했습니다.");
    }
  };

  return (
    <main className="hb-summary hb-daily-summary">
      <p>DAILY #{session.challenge.number} COMPLETE</p>
      {resumeNotice ? (
        <p className="hb-daily-notice" role="status">{resumeNotice}</p>
      ) : null}
      <div className="hb-summary-score">
        <span>오늘 순위</span>
        <strong>{result.rank}<i>위</i></strong>
        <small>{result.leaderboard.participantCount}명 참여</small>
      </div>
      <h1>{session.nickname}님의 점수는<br />{result.points}점입니다.</h1>
      <div className="hb-summary-grid">
        <div><span>상세 적중 · 3점</span><strong>{result.detailHits} / {result.pitchCount}</strong></div>
        <div><span>계열 적중 포함</span><strong>{result.familyHits} / {result.pitchCount}</strong></div>
        <div><span>연속 참여</span><strong>🔥 {result.streak}일</strong></div>
      </div>
      <section className="hb-daily-share" aria-label="공유 결과">
        <pre>{shareText}</pre>
        <button onClick={() => void share()} type="button">결과 공유하기</button>
        {shareStatus ? <small aria-live="polite">{shareStatus}</small> : null}
      </section>
      <section className="hb-leaderboard" aria-labelledby="daily-leaderboard-title">
        <div>
          <span>TODAY&apos;S LEADERBOARD</span>
          <h2 id="daily-leaderboard-title">오늘의 리더보드</h2>
        </div>
        <p>{result.leaderboard.participantCount}명의 첫 완료 기록</p>
        <div className="hb-leaderboard-table">
          <table>
            <thead>
              <tr>
                <th>순위</th>
                <th>닉네임</th>
                <th>점수</th>
                <th>상세</th>
                <th>계열</th>
              </tr>
            </thead>
            <tbody>
              {result.leaderboard.rows.map((row, index) => (
                <tr
                  className={[
                    row.isCurrent ? "current" : "",
                    row.isModel ? "model" : "",
                  ].filter(Boolean).join(" ")}
                  key={`${row.rank}-${row.nickname}-${index}`}
                >
                  <td>{row.rank}</td>
                  <th scope="row">{row.isModel ? `🤖 ${row.nickname}` : row.nickname}</th>
                  <td>{row.points}/{result.maxPoints}</td>
                  <td>{row.detailHits}</td>
                  <td>{row.familyHits}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <div className="hb-summary-actions">
        <button onClick={onBack} type="button">다른 모드 보기</button>
      </div>
      <small>PITCH / TEST · DAILY MLB CHALLENGE</small>
    </main>
  );
}

function DailyPitchTest({
  challenge,
  onBack,
}: {
  challenge: DailyChallenge;
  onBack: () => void;
}) {
  const [nickname, setNickname] = useState(() => {
    try {
      return window.localStorage.getItem("pitch-test-nickname") ?? "";
    } catch {
      return "";
    }
  });
  const [session, setSession] = useState<DailySession>();
  const [nextSession, setNextSession] = useState<DailySession>();
  const [selectedFamily, setSelectedFamily] = useState<PitchFamily>();
  const [selected, setSelected] = useState<PitchGroup>();
  const [showResult, setShowResult] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resumeNotice, setResumeNotice] = useState("");
  const questionStartedAt = useRef(Date.now());

  const start = async () => {
    const normalizedNickname = normalizeNickname(nickname);
    if (!nicknamePattern.test(normalizedNickname)) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/daily/attempts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          nickname: normalizedNickname,
          participantId: getDailyParticipantId(),
        }),
      });
      const payload = await jsonResponse<DailySession>(response);
      setResumeNotice(
        response.status === 200
          ? payload.status === "completed"
            ? "오늘 이미 완료한 기록을 불러왔습니다."
            : `저장된 기록을 불러왔습니다. ${payload.history.length}구 다음부터 이어집니다.`
          : "",
      );
      setNickname(payload.nickname);
      try {
        window.localStorage.setItem("pitch-test-nickname", payload.nickname);
      } catch {
        // Storage can be unavailable in private browsing; the game still works.
      }
      setSession(payload);
      questionStartedAt.current = Date.now();
      window.scrollTo({ top: 0 });
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "오늘의 게임을 시작하지 못했습니다.",
      );
    } finally {
      setLoading(false);
    }
  };

  const submit = async () => {
    if (
      !session?.current ||
      !selectedFamily ||
      !selected ||
      loading ||
      nextSession
    ) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(
        `/api/daily/attempts/${session.attemptId}/answers`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            selectedFamily,
            selected,
            responseMs: Math.max(0, Date.now() - questionStartedAt.current),
          }),
        },
      );
      const payload = await jsonResponse<DailySession>(response);
      setNextSession(payload);
      setShowResult(false);
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "답안을 저장하지 못했습니다.",
      );
    } finally {
      setLoading(false);
    }
  };

  const next = () => {
    if (!nextSession) return;
    setSession(nextSession);
    setNextSession(undefined);
    setSelectedFamily(undefined);
    setSelected(undefined);
    setShowResult(false);
    questionStartedAt.current = Date.now();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  if (!session) {
    return (
      <DailyIntro
        challenge={challenge}
        error={error}
        loading={loading}
        nickname={nickname}
        onBack={onBack}
        onNickname={setNickname}
        onStart={() => void start()}
      />
    );
  }
  if (session.status === "completed") {
    return (
      <DailyResultView
        onBack={onBack}
        resumeNotice={resumeNotice}
        session={session}
      />
    );
  }

  const current = session.current!;
  const reveal = nextSession?.reveal;
  const pitch: BenchmarkPitch = {
    number: current.pitch.number,
    count: current.pitch.count,
    actual: reveal?.actual ?? "FOUR_SEAM",
    rawLabel: reveal?.rawLabel ?? "",
    result: reveal?.result ?? "",
    playId: reveal?.playId ?? "",
  };
  const scenario: BenchmarkScenario = {
    ...current.scenario,
    pitches: [pitch],
  };
  const movedToNextScenario =
    nextSession?.current?.scenarioNumber !== undefined &&
    nextSession.current.scenarioNumber !== current.scenarioNumber;
  const history = nextSession?.history ?? session.history;

  return (
    <div className="hb-layout-a">
      <header className="hb-topbar">
        <div className="hb-wordmark"><i /> PITCH / TEST</div>
        <span>
          DAILY #{session.challenge.number} · {session.challenge.scenarioCount}개 타석
        </span>
      </header>
      <main className="hb-form-shell">
        <section className="hb-form-card">
          {resumeNotice ? (
            <p className="hb-daily-notice" role="status">{resumeNotice}</p>
          ) : null}
          <div className="hb-progress-copy">
            <span>AT BAT {current.scenarioNumber} · PITCH {pitch.number}</span>
            <strong>QUESTION {current.questionNumber}</strong>
          </div>
          <div className="hb-progress">
            <i
              style={{
                width: `${
                  (current.scenarioNumber / current.scenarioCount) * 100
                }%`,
              }}
            />
          </div>
          <p className="hb-kicker">
            {scenario.pitcher.toUpperCase()} vs {scenario.batter.toUpperCase()}
          </p>
          <h1>다음 구종은<br />무엇일까요?</h1>
          <SituationStrip pitch={pitch} scenario={scenario} />
          {reveal ? (
            showResult ? (
              <VideoReveal
                answer={reveal}
                isLastPitch={Boolean(nextSession?.result) || movedToNextScenario}
                isLastScenario={Boolean(nextSession?.result)}
                onNext={next}
                pitch={pitch}
                scenario={scenario}
              />
            ) : (
              <ModelPredictionReveal
                answer={reveal}
                modelPrediction={reveal.modelPrediction}
                onRevealResult={() => setShowResult(true)}
              />
            )
          ) : (
            <>
              <RepertoireStrip scenario={scenario} />
              <PitchChoices
                availableGroups={scenario.availableGroups}
                locked={loading}
                onFamilySelect={(family) => {
                  setSelectedFamily(family);
                  setSelected(undefined);
                }}
                onSelect={setSelected}
                selected={selected}
                selectedFamily={selectedFamily}
              />
              {error ? <p className="hb-daily-error" role="alert">{error}</p> : null}
              <button
                className="hb-submit"
                disabled={!selected || loading}
                onClick={() => void submit()}
                type="button"
              >
                {loading ? "답안을 저장하는 중…" : "답 확정하기"}
              </button>
            </>
          )}
        </section>
        <aside>
          <span>AT-BAT LOG · 진행된 투구만 표시</span>
          <DailyHistory history={history} scenarioId={scenario.id} />
        </aside>
      </main>
    </div>
  );
}

function MemorableLeaderboardView({
  chapter,
  onBack,
}: {
  chapter: MemorableChapter;
  onBack: () => void;
}) {
  const [leaderboard, setLeaderboard] = useState<MemorableLeaderboard>();
  const [error, setError] = useState("");
  const modelPitches = chapter.scenarios.flatMap((scenario) =>
    scenario.pitches.filter((pitch) => pitch.modelPrediction),
  );
  const modelDetailHits = modelPitches.filter(
    (pitch) => pitch.modelPrediction?.selected === pitch.actual,
  ).length;
  const modelFamilyHits = modelPitches.filter(
    (pitch) =>
      groupFamily[pitch.modelPrediction!.selected] ===
      groupFamily[pitch.actual],
  ).length;
  const modelVersion =
    modelPitches[0]?.modelPrediction?.modelVersion ?? "V8.4";
  const leaderboardRows = leaderboard
    ? [
        ...leaderboard.rows,
        {
          rank: 0,
          nickname: `OyarZabal ${modelVersion}`,
          points: modelDetailHits * 3 + modelFamilyHits - modelDetailHits,
          detailHits: modelDetailHits,
          familyHits: modelFamilyHits,
          averageResponseMs: null,
          isModel: true,
        },
      ]
        .sort(
          (left, right) =>
            right.points - left.points ||
            right.detailHits - left.detailHits ||
            Number(Boolean(left.isModel)) - Number(Boolean(right.isModel)) ||
            (left.averageResponseMs ?? Number.POSITIVE_INFINITY) -
              (right.averageResponseMs ?? Number.POSITIVE_INFINITY),
        )
        .reduce<MemorableLeaderboard["rows"]>((ranked, row) => {
          const previous = ranked[ranked.length - 1];
          ranked.push({
            ...row,
            rank:
              previous &&
              row.points === previous.points &&
              row.detailHits === previous.detailHits
                ? previous.rank
                : (previous?.rank ?? 0) + 1,
          });
          return ranked;
        }, [])
    : [];

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const response = await fetch(
          `/api/benchmark/memorable-leaderboard?version=${encodeURIComponent(
            `memorable:${chapter.id}`,
          )}`,
        );
        const payload = await jsonResponse<MemorableLeaderboard>(response);
        if (active) setLeaderboard(payload);
      } catch {
        if (active) setError("리더보드를 불러오지 못했습니다.");
      }
    };
    void load();
    return () => {
      active = false;
    };
  }, [chapter.id]);

  return (
    <main className="hb-summary hb-memorable-leaderboard-page">
      <button className="hb-version-back" onClick={onBack} type="button">
        ← 명승부 Set 목록
      </button>
      <p>MEMORABLE SET LEADERBOARD</p>
      <h1>{chapter.title}<br /><em>리더보드</em></h1>
      <p className="hb-memorable-leaderboard-context">{chapter.context}</p>
      {error ? <p className="hb-daily-error" role="alert">{error}</p> : null}
      {!leaderboard && !error ? (
        <p className="hb-memorable-leaderboard-loading">순위를 불러오는 중…</p>
      ) : leaderboard ? (
        <section
          className="hb-leaderboard"
          aria-labelledby="memorable-leaderboard-title"
        >
          <div>
            <span>{leaderboard.pitchCount} PITCHES · {leaderboard.maxPoints} POINTS</span>
            <h2 id="memorable-leaderboard-title">Set 순위</h2>
          </div>
          <p>{leaderboard.participantCount}개의 완료 기록 · 상위 50위</p>
          <div className="hb-leaderboard-table">
            <table>
              <thead>
                <tr>
                  <th>순위</th>
                  <th>닉네임</th>
                  <th>점수</th>
                  <th>상세</th>
                  <th>계열</th>
                  <th>평균 응답</th>
                </tr>
              </thead>
              <tbody>
                {leaderboardRows.length ? (
                  leaderboardRows.map((row, index) => (
                    <tr
                      className={row.isModel ? "model" : ""}
                      key={`${row.rank}-${row.nickname}-${index}`}
                    >
                      <td>{row.rank}</td>
                      <th scope="row">
                        {row.isModel ? `🤖 ${row.nickname}` : row.nickname}
                      </th>
                      <td>{row.points}/{leaderboard.maxPoints}</td>
                      <td>{row.detailHits}/{leaderboard.pitchCount}</td>
                      <td>{row.familyHits}/{leaderboard.pitchCount}</td>
                      <td>
                        {row.averageResponseMs === null
                          ? "—"
                          : `${(row.averageResponseMs / 1000).toFixed(1)}초`}
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={6}>아직 완료된 기록이 없습니다.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
      <small>PITCH / TEST · MEMORABLE SET</small>
    </main>
  );
}

function MemorableSetPicker({
  onBack,
  onSelect,
}: {
  onBack: () => void;
  onSelect: (chapter: MemorableChapter) => void;
}) {
  const [leaderboardChapter, setLeaderboardChapter] =
    useState<MemorableChapter>();

  if (leaderboardChapter) {
    return (
      <MemorableLeaderboardView
        chapter={leaderboardChapter}
        onBack={() => setLeaderboardChapter(undefined)}
      />
    );
  }

  return (
    <main className="hb-version-picker hb-memorable-picker">
      <nav className="hb-memorable-nav" aria-label="명승부 Set 탐색">
        <div className="hb-intro-mark"><i /> PITCH / TEST</div>
        <button onClick={onBack} type="button">← 메인으로</button>
      </nav>
      <div className="hb-memorable-hero">
        <div>
          <p className="hb-page-eyebrow">MEMORABLE SET · 4 GAMES</p>
          <h1>그 순간의 다음 공을<br /><em>직접 읽어보세요.</em></h1>
        </div>
        <p className="hb-memorable-copy">
          결과는 시작 전까지 공개하지 않습니다. 경기·이닝·점수만 보고
          같은 이닝에서 이어진 실제 타석의 다음 구종을 선택하세요.
        </p>
      </div>
      <div className="hb-memorable-stories">
        {memorableStories.map((story, storyIndex) => (
          <article key={story.id}>
            <span className="hb-story-index">
              {String(storyIndex + 1).padStart(2, "0")}
            </span>
            <header>
              <small>{story.stage}</small>
              <h2>{story.title}</h2>
              <p>{story.date} · {story.matchup}</p>
              <div className="hb-story-leaderboards">
                {story.chapters.map((chapter, chapterIndex) => (
                  <button
                    key={chapter.id}
                    onClick={() => setLeaderboardChapter(chapter)}
                    type="button"
                  >
                    SET {String(chapterIndex + 1).padStart(2, "0")} 리더보드
                    <i>→</i>
                  </button>
                ))}
              </div>
            </header>
            <div className="hb-story-sets">
              {story.chapters.map((chapter, chapterIndex) => {
                const pitchCount = chapter.scenarios.reduce(
                  (sum, scenario) => sum + scenario.pitches.length,
                  0,
                );
                return (
                  <button
                    key={chapter.id}
                    onClick={() => onSelect(chapter)}
                    type="button"
                  >
                    <span className="hb-set-meta">
                      <b>SET {String(chapterIndex + 1).padStart(2, "0")}</b>
                      <i>
                        {chapter.scenarios.length}타석 · {pitchCount}구
                      </i>
                    </span>
                    <strong>{chapter.title}</strong>
                    <small>{chapter.context}</small>
                    <em>이 세트 시작하기 <i>→</i></em>
                  </button>
                );
              })}
            </div>
          </article>
        ))}
      </div>
    </main>
  );
}

function VersionPicker({
  dailyChallenge,
  dailyLoading,
  onSelect,
}: {
  dailyChallenge?: DailyChallenge;
  dailyLoading: boolean;
  onSelect: (version: GameMode) => void;
}) {
  return (
    <main className="hb-version-picker">
      <div className="hb-intro-mark"><i /> PITCH / TEST</div>
      <p className="hb-page-eyebrow">HUMAN BENCHMARK</p>
      <h1>오늘의 투구를<br /><em>읽어보세요.</em></h1>
      <div className="hb-version-options">
        <button
          className="hb-mode-card"
          disabled={!dailyChallenge}
          onClick={() => onSelect("daily")}
          type="button"
        >
          <span className="hb-mode-card-head">
            <b>01</b>
            <i>DAILY · {dailyChallenge?.date ?? "TODAY"}</i>
          </span>
          <strong>오늘의 Pitch Test</strong>
          <small>
            최근 MLB 경기에서 고른 {dailyChallenge?.scenarioCount ?? 3}타석 · 하루 한 번
          </small>
          <span className="hb-mode-card-foot">
            <b>
              {dailyLoading
                ? "오늘의 문제를 불러오는 중…"
                : dailyChallenge
                  ? `${dailyChallenge.participantCount}명 완료 · 순위 도전`
                  : "오늘의 문제가 아직 준비되지 않았습니다."}
            </b>
            <i>→</i>
          </span>
        </button>
        <button
          className="hb-mode-card"
          onClick={() => onSelect("memorable")}
          type="button"
        >
          <span className="hb-mode-card-head">
            <b>02</b>
            <i>ARCHIVE · 4 GAMES / 5 SETS</i>
          </span>
          <strong>명승부 Set</strong>
          <small>
            결정적 순간 직전부터 이어지는 실제 MLB 연속 타석
          </small>
          <span className="hb-mode-card-foot">
            <b>결과를 모른 채 다음 공 예측하기</b>
            <i>→</i>
          </span>
        </button>
      </div>
      <a className="hb-admin-link" href="/admin/human-benchmark">
        관리자 대시보드 →
      </a>
    </main>
  );
}

export default function HumanBenchmarkPrototype() {
  const [version, setVersion] = useState<GameMode>();
  const [dailyChallenge, setDailyChallenge] = useState<DailyChallenge>();
  const [dailyLoading, setDailyLoading] = useState(true);
  const [memorableChapter, setMemorableChapter] =
    useState<MemorableChapter>();
  const [nickname, setNickname] = useState("");
  const [experience, setExperience] = useState("");
  const [started, setStarted] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const [pitchIndex, setPitchIndex] = useState(0);
  const [selectedFamily, setSelectedFamily] = useState<PitchFamily>();
  const [selected, setSelected] = useState<PitchGroup>();
  const [answers, setAnswers] = useState<Answer[]>([]);
  const [showResult, setShowResult] = useState(false);
  const [submissionStatus, setSubmissionStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const questionStartedAt = useRef(Date.now());
  const benchmarkVersion: BenchmarkVersion = version === "v2" ? "v2" : "v1";
  const submissionVersion =
    version === "memorable" && memorableChapter
      ? `memorable:${memorableChapter.id}`
      : benchmarkVersion;
  const benchmark: BenchmarkDefinition = memorableChapter
    ? {
        label: `명승부 Set · ${memorableChapter.title}`,
        year: memorableChapter.year,
        description: memorableChapter.context,
        scenarios: memorableChapter.scenarios,
      }
    : benchmarkVersions[benchmarkVersion];
  const scenarios = benchmark.scenarios;
  const totalPitchCount = scenarios.reduce(
    (sum, item) => sum + item.pitches.length,
    0,
  );

  useEffect(() => {
    const previousTitle = document.title;
    document.title = "Pitch Test — Human Benchmark";
    return () => {
      document.title = previousTitle;
    };
  }, []);

  useEffect(() => {
    let active = true;
    const loadDailyChallenge = async () => {
      try {
        const response = await fetch("/api/daily/challenge");
        const payload = await jsonResponse<DailyChallenge>(response);
        if (active) setDailyChallenge(payload);
      } catch {
        if (active) setDailyChallenge(undefined);
      } finally {
        if (active) setDailyLoading(false);
      }
    };
    void loadDailyChallenge();
    return () => {
      active = false;
    };
  }, []);

  const scenario = scenarios[scenarioIndex];
  const pitch = scenario.pitches[pitchIndex];
  const globalPitchNumber =
    scenarios
      .slice(0, scenarioIndex)
      .reduce((sum, item) => sum + item.pitches.length, 0) +
    pitchIndex +
    1;
  const isLastPitch = pitchIndex === scenario.pitches.length - 1;
  const isLastScenario = scenarioIndex === scenarios.length - 1;
  const answer = answers.find(
    (item) =>
      item.scenarioId === scenario.id &&
      item.pitchNumber === pitch.number,
  );
  const start = () => {
    const normalizedNickname = normalizeNickname(nickname);
    if (
      !nicknamePattern.test(normalizedNickname) ||
      !["new", "casual", "fan"].includes(experience)
    ) return;
    setNickname(normalizedNickname);
    setStarted(true);
    questionStartedAt.current = Date.now();
    window.scrollTo({ top: 0 });
  };

  const submit = () => {
    if (!selectedFamily || !selected || answer) return;
    setShowResult(false);
    const actualFamily = groupFamily[pitch.actual];
    const detailHit = selected === pitch.actual;
    const familyHit = selectedFamily === actualFamily;
    setAnswers((current) => [
      ...current,
      {
        scenarioId: scenario.id,
        pitchNumber: pitch.number,
        selectedFamily,
        selected,
        actualFamily,
        actual: pitch.actual,
        familyHit,
        detailHit,
        points: detailHit ? 3 : familyHit ? 1 : 0,
        responseMs: Math.max(0, Date.now() - questionStartedAt.current),
      },
    ]);
  };

  const saveSubmission = async () => {
    setSubmissionStatus("saving");
    try {
      const response = await fetch("/api/benchmark/submissions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          version: submissionVersion,
          nickname,
          experience,
          answers: answers.map((item) => ({
            scenarioId: item.scenarioId,
            pitchNumber: item.pitchNumber,
            selectedFamily: item.selectedFamily,
            selected: item.selected,
            responseMs: item.responseMs,
          })),
        }),
      });
      if (!response.ok) throw new Error("submission failed");
      setSubmissionStatus("saved");
    } catch {
      setSubmissionStatus("error");
    }
  };

  const next = () => {
    if (!answer) return;
    if (isLastPitch && isLastScenario) {
      setCompleted(true);
      void saveSubmission();
      return;
    }
    if (isLastPitch) {
      setScenarioIndex((current) => current + 1);
      setPitchIndex(0);
    } else {
      setPitchIndex((current) => current + 1);
    }
    setSelectedFamily(undefined);
    setSelected(undefined);
    setShowResult(false);
    questionStartedAt.current = Date.now();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const restart = () => {
    setVersion(undefined);
    setMemorableChapter(undefined);
    setStarted(false);
    setCompleted(false);
    setScenarioIndex(0);
    setPitchIndex(0);
    setSelectedFamily(undefined);
    setSelected(undefined);
    setShowResult(false);
    setAnswers([]);
    setSubmissionStatus("idle");
    window.scrollTo({ top: 0 });
  };

  const chooseVersion = (selectedVersion?: GameMode) => {
    setVersion(selectedVersion);
    if (selectedVersion !== "memorable") setMemorableChapter(undefined);
    window.scrollTo({ top: 0 });
  };

  const predictionProps: PredictionProps = {
    scenario,
    scenarioNumber: scenarioIndex + 1,
    scenarioCount: scenarios.length,
    pitch,
    globalPitchNumber,
    totalPitchCount,
    versionLabel: benchmark.label,
    isLastPitch,
    isLastScenario,
    selectedFamily,
    selected,
    answer,
    answers,
    showResult,
    onFamilySelect: (family) => {
      setSelectedFamily(family);
      setSelected(undefined);
    },
    onSelect: setSelected,
    onRevealResult: () => setShowResult(true),
    onSubmit: submit,
    onNext: next,
  };

  return (
    <div
      className={`benchmark-prototype${
        !version || (version === "memorable" && !memorableChapter)
          ? " hb-version-screen"
          : ""
      }`}
    >
      {!version ? (
        <VersionPicker
          dailyChallenge={dailyChallenge}
          dailyLoading={dailyLoading}
          onSelect={chooseVersion}
        />
      ) : version === "daily" ? (
        <DailyPitchTest
          challenge={dailyChallenge!}
          onBack={() => chooseVersion()}
        />
      ) : version === "memorable" && !memorableChapter ? (
        <MemorableSetPicker
          onBack={() => chooseVersion()}
          onSelect={(chapter) => {
            setMemorableChapter(chapter);
            window.scrollTo({ top: 0 });
          }}
        />
      ) : !started ? (
        <Intro
          benchmark={benchmark}
          experience={experience}
          nickname={nickname}
          onBack={() => {
            if (version === "memorable") {
              setMemorableChapter(undefined);
              window.scrollTo({ top: 0 });
            } else {
              chooseVersion();
            }
          }}
          onExperience={setExperience}
          onNickname={setNickname}
          onStart={start}
        />
      ) : completed ? (
        <Summary
          answers={answers}
          experience={experience}
          onRestart={restart}
          onRetry={() => void saveSubmission()}
          scenarios={scenarios}
          submissionStatus={
            submissionStatus === "idle" ? "saving" : submissionStatus
          }
          totalPitchCount={totalPitchCount}
        />
      ) : (
        <PredictionFlow {...predictionProps} />
      )}
    </div>
  );
}
