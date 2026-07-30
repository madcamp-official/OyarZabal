import type { BenchmarkScenario } from "./HumanBenchmarkPrototype";
import { memorableModelPredictions } from "./memorableModelPredictions";

export interface MemorableChapter {
  id: string;
  title: string;
  context: string;
  year: string;
  scenarios: BenchmarkScenario[];
}

export interface MemorableStory {
  id: string;
  title: string;
  date: string;
  matchup: string;
  stage: string;
  chapters: MemorableChapter[];
}

export const memorableStories: MemorableStory[] = [
  {
    "id": "ws-2025-game-7",
    "title": "World Series Game 7",
    "date": "2025.11.01",
    "matchup": "LAD @ TOR",
    "stage": "WORLD SERIES · GAME 7",
    "chapters": [
      {
        "id": "ws7-top-9",
        "title": "9회초 · LAD 공격",
        "context": "LAD 3 — 4 TOR · 9회초 · 무사 주자 없음",
        "year": "2025",
        "scenarios": [
          {
            "id": "memorable-813024-72",
            "pitcher": "Jeff Hoffman",
            "batter": "Enrique Hernández",
            "date": "2025-11-01",
            "inning": "9회 초",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 3,
            "homeScore": 4,
            "outs": 0,
            "bases": [
              false,
              false,
              false
            ],
            "outcome": "삼진",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "42c8f1bb-43a0-34d9-b206-130667685428"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "헛스윙",
                "playId": "31175e15-5c97-3f9d-9ec9-2bb3c0c46274"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "파울",
                "playId": "35781700-4a0e-3c29-9df4-10d9de870e38"
              },
              {
                "number": 4,
                "count": "1–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "피치아웃",
                "playId": "2c98750e-2d6d-3c45-b949-caf8dd687e4a"
              }
            ]
          },
          {
            "id": "memorable-813024-73",
            "pitcher": "Jeff Hoffman",
            "batter": "Miguel Rojas",
            "date": "2025-11-01",
            "inning": "9회 초",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 3,
            "homeScore": 4,
            "outs": 1,
            "bases": [
              false,
              false,
              false
            ],
            "outcome": "홈런",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "헛스윙",
                "playId": "75c043cb-e55c-338a-8ea1-1ec03e9d9c47"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "342320bb-a876-3e01-b2be-62424c308edf"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "671e2309-576f-38fc-a8b9-0f75833c9bf1"
              },
              {
                "number": 4,
                "count": "2–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "파울",
                "playId": "bb262c25-de36-3a8b-aa42-863234456e12"
              },
              {
                "number": 5,
                "count": "2–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "파울",
                "playId": "34d0f488-4741-31be-9d0d-8b860292f7a2"
              },
              {
                "number": 6,
                "count": "2–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "ba010303-e924-3a23-982d-9a8ac0bc27b5"
              },
              {
                "number": 7,
                "count": "3–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "인플레이",
                "playId": "7df05ee7-24b5-3c34-84aa-89316ec41110"
              }
            ]
          },
          {
            "id": "memorable-813024-74",
            "pitcher": "Jeff Hoffman",
            "batter": "Shohei Ohtani",
            "date": "2025-11-01",
            "inning": "9회 초",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 4,
            "homeScore": 4,
            "outs": 1,
            "bases": [
              false,
              false,
              false
            ],
            "outcome": "범타",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SPLITTER_FORK",
                "rawLabel": "스플리터",
                "result": "인플레이",
                "playId": "c2059204-82be-35f8-ae6b-f3b2e39313f5"
              }
            ]
          },
          {
            "id": "memorable-813024-75",
            "pitcher": "Jeff Hoffman",
            "batter": "Will Smith",
            "date": "2025-11-01",
            "inning": "9회 초",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 4,
            "homeScore": 4,
            "outs": 2,
            "bases": [
              false,
              false,
              false
            ],
            "outcome": "삼진",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "루킹 스트라이크",
                "playId": "058021f9-8a3f-3821-acb0-c8ca350237d8"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "루킹 스트라이크",
                "playId": "30723ee1-7ae2-3093-8c39-789a58d1e941"
              },
              {
                "number": 3,
                "count": "0–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "ae2c6fb1-ff67-3d74-b541-252d4d345a4d"
              },
              {
                "number": 4,
                "count": "1–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "ec1b4bde-78b1-3b47-8707-d74a20c5c46d"
              },
              {
                "number": 5,
                "count": "2–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "11c568f5-3cfa-3797-8526-0a100d705954"
              },
              {
                "number": 6,
                "count": "3–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "루킹 스트라이크",
                "playId": "75e804ec-3252-324d-b87b-083dce989986"
              }
            ]
          }
        ]
      },
      {
        "id": "ws7-bottom-9",
        "title": "9회말 · TOR 공격",
        "context": "LAD 4 — 4 TOR · 9회말 · 1사 1루",
        "year": "2025",
        "scenarios": [
          {
            "id": "memorable-813024-78",
            "pitcher": "Blake Snell",
            "batter": "Addison Barger",
            "date": "2025-11-01",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 4,
            "homeScore": 4,
            "outs": 1,
            "bases": [
              true,
              false,
              false
            ],
            "outcome": "볼넷",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더",
                "커브"
              ],
              "OFFSPEED": [
                "체인지업"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "CURVE",
              "CHANGEUP"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "3d7bc074-5c5b-34e7-883e-f163c1e5a916"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "5f12d688-7888-3f8d-94e4-4dd19a0d253e"
              },
              {
                "number": 3,
                "count": "2–0",
                "actual": "CURVE",
                "rawLabel": "커브",
                "result": "루킹 스트라이크",
                "playId": "53632952-f6f6-3cc9-80e0-98b819ee737e"
              },
              {
                "number": 4,
                "count": "2–1",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "루킹 스트라이크",
                "playId": "cfa9a030-38b0-3cd7-b85b-a67e7e4325a6"
              },
              {
                "number": 5,
                "count": "2–2",
                "actual": "CHANGEUP",
                "rawLabel": "체인지업",
                "result": "파울",
                "playId": "a185f06d-65ab-31c4-bc8f-40849c79c627"
              },
              {
                "number": 6,
                "count": "2–2",
                "actual": "CURVE",
                "rawLabel": "커브",
                "result": "파울",
                "playId": "78f3d6e2-5762-3655-b531-4e4bd982058f"
              },
              {
                "number": 7,
                "count": "2–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "파울",
                "playId": "1a629230-5340-30b5-978d-691a6f3945c1"
              },
              {
                "number": 8,
                "count": "2–2",
                "actual": "CURVE",
                "rawLabel": "커브",
                "result": "볼",
                "playId": "3e2298a9-5f97-3763-9243-3bd8deb8eb72"
              },
              {
                "number": 9,
                "count": "3–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "e1711380-6776-310c-babf-751241a8a02f"
              }
            ]
          },
          {
            "id": "memorable-813024-79",
            "pitcher": "Yoshinobu Yamamoto",
            "batter": "Alejandro Kirk",
            "date": "2025-11-01",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 4,
            "homeScore": 4,
            "outs": 1,
            "bases": [
              true,
              true,
              false
            ],
            "outcome": "몸에 맞는 공",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "싱커",
                "포심",
                "커터"
              ],
              "BREAKING": [
                "커브"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "CURVE",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SPLITTER_FORK",
                "rawLabel": "스플리터",
                "result": "루킹 스트라이크",
                "playId": "9016c1b5-8177-3efd-8624-bf6898cbbd53"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "몸에 맞는 공",
                "playId": "fab3fd46-af69-35bf-adb5-84077afcacdc"
              }
            ]
          },
          {
            "id": "memorable-813024-80",
            "pitcher": "Yoshinobu Yamamoto",
            "batter": "Daulton Varsho",
            "date": "2025-11-01",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 4,
            "homeScore": 4,
            "outs": 1,
            "bases": [
              true,
              true,
              true
            ],
            "outcome": "포스 아웃",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "싱커",
                "포심",
                "커터"
              ],
              "BREAKING": [
                "커브"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "CURVE",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SPLITTER_FORK",
                "rawLabel": "스플리터",
                "result": "볼",
                "playId": "a91643b7-6cd9-3504-8b48-04cb4aec5f20"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "SPLITTER_FORK",
                "rawLabel": "스플리터",
                "result": "파울",
                "playId": "07032c2b-4d24-3a65-ad00-f3c31c47311d"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "루킹 스트라이크",
                "playId": "d538f82d-1e8d-3f33-a8a9-5deec16d7e67"
              },
              {
                "number": 4,
                "count": "1–2",
                "actual": "SPLITTER_FORK",
                "rawLabel": "스플리터",
                "result": "인플레이",
                "playId": "787b6bc0-3420-343c-9c79-b3dc06b22586"
              }
            ]
          },
          {
            "id": "memorable-813024-81",
            "pitcher": "Yoshinobu Yamamoto",
            "batter": "Ernie Clement",
            "date": "2025-11-01",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "TOR",
            "awayScore": 4,
            "homeScore": 4,
            "outs": 2,
            "bases": [
              true,
              true,
              true
            ],
            "outcome": "범타",
            "gameUrl": "https://www.mlb.com/gameday/813024",
            "repertoire": {
              "FASTBALL": [
                "싱커",
                "포심",
                "커터"
              ],
              "BREAKING": [
                "커브"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "CURVE",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "CURVE",
                "rawLabel": "커브",
                "result": "인플레이",
                "playId": "dd474c08-1865-3406-a209-7e3bbd673cfa"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    "id": "sf-2026-june-10",
    "title": "Oracle Park · 9회말",
    "date": "2026.06.10",
    "matchup": "WSH @ SF",
    "stage": "REGULAR SEASON · 9TH INNING",
    "chapters": [
      {
        "id": "sf-bottom-9",
        "title": "9회말 · SF 공격",
        "context": "WSH 10 — 6 SF · 9회말 · 무사 2루",
        "year": "2026",
        "scenarios": [
          {
            "id": "memorable-823215-80",
            "pitcher": "Gus Varland",
            "batter": "Matt Chapman",
            "date": "2026-06-10",
            "inning": "9회 말",
            "awayTeam": "WSH",
            "homeTeam": "SF",
            "awayScore": 10,
            "homeScore": 6,
            "outs": 0,
            "bases": [
              false,
              true,
              false
            ],
            "outcome": "2루타",
            "gameUrl": "https://www.mlb.com/gameday/823215",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더"
              ],
              "OFFSPEED": [
                "체인지업"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "CHANGEUP"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "파울",
                "playId": "f26998ae-ea1e-3124-9bee-cb655bc5e2d5"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "헛스윙",
                "playId": "4e0802d6-0d09-3f28-8fcd-2481363166cd"
              },
              {
                "number": 3,
                "count": "0–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "인플레이",
                "playId": "47d6bf2f-badd-3e0f-9712-70d60f37af38"
              }
            ]
          },
          {
            "id": "memorable-823215-81",
            "pitcher": "Gus Varland",
            "batter": "Rafael Devers",
            "date": "2026-06-10",
            "inning": "9회 말",
            "awayTeam": "WSH",
            "homeTeam": "SF",
            "awayScore": 10,
            "homeScore": 7,
            "outs": 0,
            "bases": [
              false,
              true,
              false
            ],
            "outcome": "볼넷",
            "gameUrl": "https://www.mlb.com/gameday/823215",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더"
              ],
              "OFFSPEED": [
                "체인지업"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "CHANGEUP"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "루킹 스트라이크",
                "playId": "bfa42e22-a3c6-3c9f-9607-b09bbad715a6"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "파울",
                "playId": "6852291d-65be-3a4a-80ac-0d33e90481f2"
              },
              {
                "number": 3,
                "count": "0–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "e3b1af7d-9064-30c8-9d8f-ddac21524c95"
              },
              {
                "number": 4,
                "count": "1–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "5b75b42b-52c8-36f6-9299-c1789c4ee22f"
              },
              {
                "number": 5,
                "count": "2–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "4564ff3e-e608-3e84-b373-b97260c46945"
              },
              {
                "number": 6,
                "count": "3–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "f99ba1ca-da1f-3d54-92e7-3171bfa659e9"
              }
            ]
          },
          {
            "id": "memorable-823215-82",
            "pitcher": "Mitchell Parker",
            "batter": "Jung Hoo Lee",
            "date": "2026-06-10",
            "inning": "9회 말",
            "awayTeam": "WSH",
            "homeTeam": "SF",
            "awayScore": 10,
            "homeScore": 7,
            "outs": 0,
            "bases": [
              true,
              true,
              false
            ],
            "outcome": "안타",
            "gameUrl": "https://www.mlb.com/gameday/823215",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더",
                "커브"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "CURVE"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "루킹 스트라이크",
                "playId": "666d3e92-8396-3b86-b3cb-b9b2a25d5143"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "파울",
                "playId": "75dc9fc2-eb06-33dd-8fd5-e4c62939320b"
              },
              {
                "number": 3,
                "count": "0–2",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "05b08440-e53d-34c4-aefc-71462b0a84a2"
              },
              {
                "number": 4,
                "count": "1–2",
                "actual": "CURVE",
                "rawLabel": "커브",
                "result": "파울",
                "playId": "9ea56ce6-a1be-30b7-9206-4fb12f503187"
              },
              {
                "number": 5,
                "count": "1–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "인플레이",
                "playId": "3c06ade8-472e-304a-9897-50f6ec2c177e"
              }
            ]
          },
          {
            "id": "memorable-823215-83",
            "pitcher": "Mitchell Parker",
            "batter": "Bryce Eldridge",
            "date": "2026-06-10",
            "inning": "9회 말",
            "awayTeam": "WSH",
            "homeTeam": "SF",
            "awayScore": 10,
            "homeScore": 7,
            "outs": 0,
            "bases": [
              true,
              true,
              true
            ],
            "outcome": "홈런",
            "gameUrl": "https://www.mlb.com/gameday/823215",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [
                "슬라이더",
                "커브"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "SLIDER",
              "CURVE"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "eb0839db-2c5a-3317-a3d1-3ab11c9bba5b"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "SLIDER",
                "rawLabel": "슬라이더",
                "result": "볼",
                "playId": "516e0582-7639-3979-acd4-5e5bd4f06b11"
              },
              {
                "number": 3,
                "count": "2–0",
                "actual": "CURVE",
                "rawLabel": "커브",
                "result": "인플레이",
                "playId": "dadbb09f-3c03-3ec2-b314-2a6d03e6300f"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    "id": "bal-2025-september-6",
    "title": "Camden Yards · 9회말",
    "date": "2025.09.06",
    "matchup": "LAD @ BAL",
    "stage": "REGULAR SEASON · 9TH INNING",
    "chapters": [
      {
        "id": "bal-bottom-9",
        "title": "9회말 · 2아웃 이후",
        "context": "LAD 3 — 0 BAL · 9회말 · 2사 주자 없음",
        "year": "2025",
        "scenarios": [
          {
            "id": "memorable-776443-65",
            "pitcher": "Yoshinobu Yamamoto",
            "batter": "Jackson Holliday",
            "date": "2025-09-06",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "BAL",
            "awayScore": 3,
            "homeScore": 0,
            "outs": 2,
            "bases": [
              false,
              false,
              false
            ],
            "outcome": "홈런",
            "gameUrl": "https://www.mlb.com/gameday/776443",
            "repertoire": {
              "FASTBALL": [
                "포심",
                "커터",
                "싱커"
              ],
              "BREAKING": [
                "커브",
                "슬라이더"
              ],
              "OFFSPEED": [
                "스플리터"
              ]
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER",
              "CURVE",
              "SPLITTER_FORK"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "파울",
                "playId": "df259841-e8b9-3778-bdfd-09e0ac0eae2d"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "SPLITTER_FORK",
                "rawLabel": "스플리터",
                "result": "볼",
                "playId": "18e8fd9b-5789-3853-a49a-158df6790bab"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "0084d853-a42b-3f9e-a8e5-9f06d11bedd4"
              },
              {
                "number": 4,
                "count": "2–1",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "커터",
                "result": "인플레이",
                "playId": "3d3a8c2d-26bb-367a-9a79-6b199560d51d"
              }
            ]
          },
          {
            "id": "memorable-776443-66",
            "pitcher": "Blake Treinen",
            "batter": "Jeremiah Jackson",
            "date": "2025-09-06",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "BAL",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 2,
            "bases": [
              false,
              false,
              false
            ],
            "outcome": "2루타",
            "gameUrl": "https://www.mlb.com/gameday/776443",
            "repertoire": {
              "FASTBALL": [
                "싱커",
                "커터",
                "포심"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "루킹 스트라이크",
                "playId": "a26852b4-a539-3093-a0c4-a70c32fcd0b2"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "헛스윙",
                "playId": "151ef9d8-cae2-3d83-a85d-06d71eb881ac"
              },
              {
                "number": 3,
                "count": "0–2",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "볼",
                "playId": "2cef920f-e5a3-3fd4-9711-b09e529ec9f2"
              },
              {
                "number": 4,
                "count": "1–2",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "볼",
                "playId": "410e6c80-b843-3a19-adfe-008e199ab008"
              },
              {
                "number": 5,
                "count": "2–2",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "파울",
                "playId": "662ee4a6-1152-318d-96ae-367966974049"
              },
              {
                "number": 6,
                "count": "2–2",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "볼",
                "playId": "c5a2b66c-6a6a-30d3-9d2f-e178fbeaa458"
              },
              {
                "number": 7,
                "count": "3–2",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "인플레이",
                "playId": "cdea0c07-e417-37dd-8636-91b12a988cfa"
              }
            ]
          },
          {
            "id": "memorable-776443-67",
            "pitcher": "Blake Treinen",
            "batter": "Gunnar Henderson",
            "date": "2025-09-06",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "BAL",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 2,
            "bases": [
              false,
              true,
              false
            ],
            "outcome": "몸에 맞는 공",
            "gameUrl": "https://www.mlb.com/gameday/776443",
            "repertoire": {
              "FASTBALL": [
                "싱커",
                "커터",
                "포심"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "볼",
                "playId": "b5e73873-7fe6-3fe9-b47d-05f8a48f0e0f"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "루킹 스트라이크",
                "playId": "2d478570-3d92-3dd0-8615-94dd58df39c0"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "커터",
                "result": "파울",
                "playId": "12f8ecb6-7806-3c9a-ba9d-7c1f4d584e1f"
              },
              {
                "number": 4,
                "count": "1–2",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "파울",
                "playId": "7d27ddd2-5350-32f7-b6fb-61bab231f120"
              },
              {
                "number": 5,
                "count": "1–2",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "몸에 맞는 공",
                "playId": "e1c8df63-b81a-3c83-89e4-66ded67ea7d8"
              }
            ]
          },
          {
            "id": "memorable-776443-68",
            "pitcher": "Blake Treinen",
            "batter": "Ryan Mountcastle",
            "date": "2025-09-06",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "BAL",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 2,
            "bases": [
              true,
              true,
              false
            ],
            "outcome": "볼넷",
            "gameUrl": "https://www.mlb.com/gameday/776443",
            "repertoire": {
              "FASTBALL": [
                "싱커",
                "커터",
                "포심"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "피치아웃",
                "playId": "b86980e0-714b-3ebd-9b1f-47dbbe3db35e"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "8fbd9d45-15cf-36a1-9a9e-754b6eec09ea"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "볼",
                "playId": "3f67f7c8-5905-3c82-a31e-30a8ac72aea1"
              },
              {
                "number": 4,
                "count": "2–1",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "볼",
                "playId": "61dd3e38-62cb-337e-8a52-1d3df3dc275f"
              },
              {
                "number": 5,
                "count": "3–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "f6789948-f423-3b1b-a399-e1ee6bad7774"
              }
            ]
          },
          {
            "id": "memorable-776443-69",
            "pitcher": "Blake Treinen",
            "batter": "Colton Cowser",
            "date": "2025-09-06",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "BAL",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 2,
            "bases": [
              true,
              true,
              true
            ],
            "outcome": "볼넷",
            "gameUrl": "https://www.mlb.com/gameday/776443",
            "repertoire": {
              "FASTBALL": [
                "싱커",
                "커터",
                "포심"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "볼",
                "playId": "f72b07f7-3e10-3244-8ba4-ec4ac29a7177"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "볼",
                "playId": "33b27ad5-95c3-3ada-ab96-6bce96cf6aec"
              },
              {
                "number": 3,
                "count": "2–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "5c1501ea-ae54-3ad6-966b-43a31b0812fd"
              },
              {
                "number": 4,
                "count": "3–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "커터",
                "result": "루킹 스트라이크",
                "playId": "a6b462c6-d5f1-3ab1-ab0d-08aa30025fb8"
              },
              {
                "number": 5,
                "count": "3–1",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "볼",
                "playId": "33302097-8aff-3e96-8374-7deba894d8fc"
              }
            ]
          },
          {
            "id": "memorable-776443-70",
            "pitcher": "Tanner Scott",
            "batter": "Emmanuel Rivera",
            "date": "2025-09-06",
            "inning": "9회 말",
            "awayTeam": "LAD",
            "homeTeam": "BAL",
            "awayScore": 3,
            "homeScore": 2,
            "outs": 2,
            "bases": [
              true,
              true,
              true
            ],
            "outcome": "안타",
            "gameUrl": "https://www.mlb.com/gameday/776443",
            "repertoire": {
              "FASTBALL": [
                "포심"
              ],
              "BREAKING": [],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "4f17c537-e9b7-3e35-87f4-ac802a100e9c"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "루킹 스트라이크",
                "playId": "dade655a-d118-3129-8717-e9641e58a115"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "인플레이",
                "playId": "a31830e1-b33d-3af2-bf00-eb98a2ff4ee6"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    "id": "alcs-2025-game-7",
    "title": "ALCS Game 7",
    "date": "2025.10.20",
    "matchup": "SEA @ TOR",
    "stage": "ALCS · GAME 7",
    "chapters": [
      {
        "id": "alcs7-bottom-7",
        "title": "7회말 · TOR 공격",
        "context": "SEA 3 — 1 TOR · 7회말 · 무사 주자 없음",
        "year": "2025",
        "scenarios": [
          {
            "id": "memorable-813037-55",
            "pitcher": "Bryan Woo",
            "batter": "Addison Barger",
            "date": "2025-10-20",
            "inning": "7회 말",
            "awayTeam": "SEA",
            "homeTeam": "TOR",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 0,
            "bases": [
              false,
              false,
              false
            ],
            "outcome": "볼넷",
            "gameUrl": "https://www.mlb.com/gameday/813037",
            "repertoire": {
              "FASTBALL": [
                "포심",
                "싱커"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "볼",
                "playId": "d4dc358a-120c-3640-a1f5-7fcd48621c9a"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "파울",
                "playId": "72b07142-b85e-3355-bc6c-5bacb3d7e2b3"
              },
              {
                "number": 3,
                "count": "1–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "aeec2aa6-00e0-3de6-8753-cab844cf0919"
              },
              {
                "number": 4,
                "count": "2–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "a78647cd-813a-3cc8-8d07-bbeb4589bd77"
              },
              {
                "number": 5,
                "count": "3–1",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "bc7fe9b2-d557-3a30-9d0d-88fc83d6841d"
              }
            ]
          },
          {
            "id": "memorable-813037-56",
            "pitcher": "Bryan Woo",
            "batter": "Isiah Kiner-Falefa",
            "date": "2025-10-20",
            "inning": "7회 말",
            "awayTeam": "SEA",
            "homeTeam": "TOR",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 0,
            "bases": [
              true,
              false,
              false
            ],
            "outcome": "안타",
            "gameUrl": "https://www.mlb.com/gameday/813037",
            "repertoire": {
              "FASTBALL": [
                "포심",
                "싱커"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "루킹 스트라이크",
                "playId": "8bdf37ae-aa97-33b1-a221-bb8bcde8cbfa"
              },
              {
                "number": 2,
                "count": "0–1",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "파울",
                "playId": "2c9ecad7-f3aa-34e8-af85-d7818eef60ca"
              },
              {
                "number": 3,
                "count": "0–2",
                "actual": "SLIDER",
                "rawLabel": "스위퍼",
                "result": "인플레이",
                "playId": "74e44092-a550-33c0-aba7-1d773dfd311d"
              }
            ]
          },
          {
            "id": "memorable-813037-57",
            "pitcher": "Bryan Woo",
            "batter": "Andrés Giménez",
            "date": "2025-10-20",
            "inning": "7회 말",
            "awayTeam": "SEA",
            "homeTeam": "TOR",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 0,
            "bases": [
              true,
              true,
              false
            ],
            "outcome": "희생 번트",
            "gameUrl": "https://www.mlb.com/gameday/813037",
            "repertoire": {
              "FASTBALL": [
                "포심",
                "싱커"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "FOUR_SEAM",
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "볼",
                "playId": "508c1058-2172-312a-909c-f18d7c44875a"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "FOUR_SEAM",
                "rawLabel": "포심",
                "result": "인플레이",
                "playId": "cb50aa0a-cde7-3f34-8345-5efdd79870f3"
              }
            ]
          },
          {
            "id": "memorable-813037-58",
            "pitcher": "Eduard Bazardo",
            "batter": "George Springer",
            "date": "2025-10-20",
            "inning": "7회 말",
            "awayTeam": "SEA",
            "homeTeam": "TOR",
            "awayScore": 3,
            "homeScore": 1,
            "outs": 1,
            "bases": [
              false,
              true,
              true
            ],
            "outcome": "홈런",
            "gameUrl": "https://www.mlb.com/gameday/813037",
            "repertoire": {
              "FASTBALL": [
                "싱커"
              ],
              "BREAKING": [
                "스위퍼"
              ],
              "OFFSPEED": []
            },
            "availableGroups": [
              "MOVING_FASTBALL",
              "SLIDER"
            ],
            "pitches": [
              {
                "number": 1,
                "count": "0–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "볼",
                "playId": "c5084cd1-87ff-3ddb-9f64-8c01da829a7e"
              },
              {
                "number": 2,
                "count": "1–0",
                "actual": "MOVING_FASTBALL",
                "rawLabel": "싱커",
                "result": "인플레이",
                "playId": "43ff90e6-f8cd-3a3f-b93b-de5f2237a323"
              }
            ]
          }
        ]
      }
    ]
  }
];

memorableStories.forEach((story) =>
  story.chapters.forEach((chapter) =>
    chapter.scenarios.forEach((scenario) =>
      scenario.pitches.forEach((pitch) => {
        pitch.modelPrediction = memorableModelPredictions[pitch.playId];
      }),
    ),
  ),
);
