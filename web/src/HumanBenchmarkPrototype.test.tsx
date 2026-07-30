import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import HumanBenchmarkPrototype from "./HumanBenchmarkPrototype";

afterEach(() => {
  cleanup();
  window.localStorage.clear();
  vi.unstubAllGlobals();
});

test("opens a spoiler-free memorable set with verified consecutive at-bats", async () => {
  window.scrollTo = vi.fn();
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      id: "daily-1",
      number: 1,
      date: "2026-07-29",
      sourceDate: "2026-07-28",
      scenarioCount: 3,
      participantCount: 0,
    }),
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<HumanBenchmarkPrototype />);

  const dailyButton = await screen.findByRole("button", {
    name: /오늘의 Pitch Test/,
  });
  const memorableButton = screen.getByRole("button", { name: /명승부 Set/ });
  expect(dailyButton).toHaveClass("hb-mode-card");
  expect(memorableButton).toHaveClass("hb-mode-card");
  fireEvent.click(memorableButton);
  expect(
    screen.getByRole("heading", { name: "World Series Game 7" }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: "Oracle Park · 9회말" }),
  ).toBeInTheDocument();
  expect(screen.queryByText("홈런")).not.toBeInTheDocument();
  expect(screen.queryByText("끝내기")).not.toBeInTheDocument();

  fireEvent.click(screen.getByText("9회말 · SF 공격").closest("button")!);
  expect(screen.getByText("WSH 10 — 6 SF · 9회말 · 무사 2루")).toBeInTheDocument();
  expect(screen.getByText(/실제 4개 타석, 총 17구/)).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText(/결과에 표시할 닉네임/), {
    target: { value: "명승부팬" },
  });
  fireEvent.click(screen.getByRole("button", { name: /자주 봐요/ }));
  fireEvent.click(
    screen.getByRole("button", { name: /4개 타석 · 17구 예측 시작하기/ }),
  );

  expect(screen.getByText("GUS VARLAND vs MATT CHAPMAN")).toBeInTheDocument();
  expect(screen.queryByText("2루타")).not.toBeInTheDocument();
  expect(screen.queryByTitle(/MLB 공식 Gus Varland .*구 영상/)).not.toBeInTheDocument();

  fireEvent.click(
    within(screen.getByRole("radiogroup", { name: "예상 구종 계열" }))
      .getByRole("radio", { name: /패스트볼 계열/ }),
  );
  fireEvent.click(
    within(screen.getByRole("radiogroup", { name: "예상 상세 구종" }))
      .getByRole("radio", { name: /포심 Four-seam/ }),
  );
  fireEvent.click(screen.getByRole("button", { name: "답 확정하기" }));
  expect(screen.getByText("V8.4 MODEL PREDICTION")).toBeInTheDocument();
  expect(
    screen.queryByTitle(/MLB 공식 Gus Varland .*구 영상/),
  ).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "결과 확인하기 →" }));
  expect(
    screen.getByTitle(/MLB 공식 Gus Varland .*구 영상/),
  ).toHaveAttribute(
    "src",
    expect.stringContaining("f26998ae-ea1e-3124-9bee-cb655bc5e2d5"),
  );

  for (let pitchIndex = 1; pitchIndex < 17; pitchIndex += 1) {
    fireEvent.click(
      screen.getByRole("button", { name: /다음 (투구로|타석으로)/ }),
    );
    const familyChoice = within(
      screen.getByRole("radiogroup", { name: "예상 구종 계열" }),
    )
      .getAllByRole("radio")
      .find((choice) => !choice.hasAttribute("disabled"))!;
    fireEvent.click(familyChoice);
    fireEvent.click(
      within(screen.getByRole("radiogroup", { name: "예상 상세 구종" }))
        .getAllByRole("radio")[0],
    );
    fireEvent.click(screen.getByRole("button", { name: "답 확정하기" }));
    expect(screen.getByText("V8.4 MODEL PREDICTION")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "결과 확인하기 →" }));
  }
  fireEvent.click(
    screen.getByRole("button", { name: "최종 리더보드 보기 →" }),
  );

  const modelSummary = screen.getByRole("region", {
    name: "모델과 종합 결과 비교",
  });
  expect(within(modelSummary).getByText("OyarZabal V8.4")).toBeInTheDocument();
  expect(
    within(modelSummary).getAllByText(/상세 \d+ · 계열 포함 \d+/),
  ).toHaveLength(2);
  expect(document.querySelectorAll(".hb-scenario-score small")).toHaveLength(4);
  await waitFor(() => {
    const submission = fetchMock.mock.calls.find(
      ([url]) => url === "/api/benchmark/submissions",
    );
    expect(submission).toBeDefined();
    const payload = JSON.parse(String(submission![1]?.body));
    expect(payload).toMatchObject({
      version: "memorable:sf-bottom-9",
      nickname: "명승부팬",
      experience: "fan",
    });
    expect(payload.answers).toHaveLength(17);
  });
});

test("renders a first-base runner on the right side of the diamond", async () => {
  window.scrollTo = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: "daily-1",
        number: 1,
        date: "2026-07-30",
        sourceDate: "2026-07-28",
        scenarioCount: 3,
        participantCount: 0,
      }),
    }),
  );
  render(<HumanBenchmarkPrototype />);

  fireEvent.click(
    await screen.findByRole("button", { name: /명승부 Set/ }),
  );
  fireEvent.click(
    screen.getByRole("button", { name: /9회말 · TOR 공격/ }),
  );
  fireEvent.change(screen.getByLabelText(/결과에 표시할 닉네임/), {
    target: { value: "루상검증" },
  });
  fireEvent.click(screen.getByRole("button", { name: /자주 봐요/ }));
  fireEvent.click(screen.getByRole("button", { name: /예측 시작하기/ }));

  expect(screen.getByLabelText("1루 주자 있음")).toBeInTheDocument();
  expect(
    Array.from(document.querySelectorAll(".hb-base i")).map((base) =>
      base.classList.contains("occupied"),
    ),
  ).toEqual([false, false, true]);
});

test("opens each memorable set leaderboard from below the game date", async () => {
  window.scrollTo = vi.fn();
  const fetchMock = vi.fn(async (url: string) => {
    if (url === "/api/daily/challenge") {
      return {
        ok: true,
        json: async () => ({
          id: "daily-1",
          number: 1,
          date: "2026-07-30",
          sourceDate: "2026-07-28",
          scenarioCount: 3,
          participantCount: 0,
        }),
      };
    }
    if (
      url ===
      "/api/benchmark/memorable-leaderboard?version=memorable%3Aws7-top-9"
    ) {
      return {
        ok: true,
        json: async () => ({
          version: "memorable:ws7-top-9",
          label: "World Series Game 7 · 9회초 · LAD 공격",
          participantCount: 1,
          pitchCount: 18,
          maxPoints: 54,
          rows: [{
            rank: 1,
            nickname: "명승부선두",
            points: 24,
            detailHits: 6,
            familyHits: 12,
            averageResponseMs: 1800,
          }],
        }),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<HumanBenchmarkPrototype />);

  fireEvent.click(
    await screen.findByRole("button", { name: /명승부 Set/ }),
  );
  const gameHeader = screen
    .getByRole("heading", { name: "World Series Game 7" })
    .closest("header")!;
  expect(within(gameHeader).getByText("2025.11.01 · LAD @ TOR")).toBeInTheDocument();
  const leaderboardButton = within(gameHeader).getByRole("button", {
    name: /SET 01 리더보드/,
  });
  fireEvent.click(leaderboardButton);

  expect(
    await screen.findByRole("heading", { name: "Set 순위" }),
  ).toBeInTheDocument();
  expect(screen.getByText("명승부선두")).toBeInTheDocument();
  expect(screen.getByText("🤖 OyarZabal V8.4")).toBeInTheDocument();
  expect(document.querySelector(".hb-leaderboard tr.model")).toBeInTheDocument();
  expect(screen.getByText("24/54")).toBeInTheDocument();
  const leaderboardRows = document.querySelectorAll(
    ".hb-leaderboard tbody tr",
  );
  expect(leaderboardRows[0]).toHaveTextContent("OyarZabal V8.4");
  expect(leaderboardRows[0]).toHaveTextContent("42/54");
  expect(leaderboardRows[0].querySelector("td")).toHaveTextContent("1");
  expect(leaderboardRows[1]).toHaveTextContent("명승부선두");
  expect(leaderboardRows[1].querySelector("td")).toHaveTextContent("2");
  expect(screen.getByText("1개의 완료 기록 · 상위 50위")).toBeInTheDocument();
});

test.skip("legacy Version 1 flow", async () => {
  window.scrollTo = vi.fn();
  const fetchMock = vi.fn().mockResolvedValue({ ok: true });
  vi.stubGlobal("fetch", fetchMock);
  render(<HumanBenchmarkPrototype />);

  expect(
    screen.getByRole("button", { name: /Pitch Test Version 1/ }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: /Pitch Test Version 2/ }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("link", { name: "관리자 대시보드 →" }),
  ).toHaveAttribute("href", "/admin/human-benchmark");
  fireEvent.click(
    screen.getByRole("button", { name: /Pitch Test Version 1/ }),
  );

  expect(screen.queryByText("중계 화면")).not.toBeInTheDocument();
  expect(screen.queryByText("기록지")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Merrill Kelly 구종")).not.toBeInTheDocument();
  expect(screen.getByText(/실제 5개 타석, 총 25구/)).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText(/결과에 표시할 닉네임/), {
    target: { value: "<script>" },
  });
  fireEvent.click(screen.getByRole("button", { name: /가끔 봐요/ }));
  expect(
    screen.getByRole("button", { name: /25구 예측 시작하기/ }),
  ).toBeDisabled();
  fireEvent.change(screen.getByLabelText(/결과에 표시할 닉네임/), {
    target: { value: "야구팬" },
  });
  fireEvent.click(screen.getByRole("button", { name: /25구 예측 시작하기/ }));
  expect(screen.getByLabelText("Hunter Brown 구종")).toHaveTextContent(
    "포심 · 커터 · 싱커",
  );
  expect(screen.queryByText("4구 삼진")).not.toBeInTheDocument();

  const scenarioStarts = new Set([0, 3, 8, 12, 15]);
  const scenarioEnds = new Set([2, 7, 11, 14]);
  const snellStartsAt = 8;
  const kellyStartsAt = 15;

  Array.from({ length: 25 }).forEach((_, index) => {
    expect(screen.queryByTitle(/MLB 공식 .*구 영상/)).not.toBeInTheDocument();
    if (scenarioStarts.has(index)) {
      expect(
        within(screen.getByRole("list", { name: "현재 타석 투구 기록" }))
          .getAllByRole("listitem"),
      ).toHaveLength(1);
    }
    if (index === snellStartsAt) {
      expect(screen.getByLabelText("Blake Snell 구종")).toHaveTextContent(
        /포심.*슬라이더 · 커브.*체인지업/,
      );
    }
    if (index === kellyStartsAt) {
      expect(screen.getByLabelText("Merrill Kelly 구종")).toBeInTheDocument();
      expect(screen.getByText("SCENARIO 5 · PITCH 1")).toBeInTheDocument();
    }
    fireEvent.click(
      within(screen.getByRole("radiogroup", { name: "예상 구종 계열" }))
        .getByRole("radio", { name: /패스트볼 계열/ }),
    );
    const detailGroup = screen.getByRole("radiogroup", { name: "예상 상세 구종" });
    if (index === snellStartsAt) {
      expect(within(detailGroup).getAllByRole("radio")).toHaveLength(1);
    }
    fireEvent.click(
      within(detailGroup).getByRole("radio", { name: /포심 Four-seam/ }),
    );
    fireEvent.click(screen.getByRole("button", { name: "답 확정하기" }));
    expect(screen.getByTitle(/MLB 공식 .*구 영상/)).toHaveAttribute(
      "src",
      expect.stringContaining("baseballsavant.mlb.com/sporty-videos"),
    );
    fireEvent.click(
      screen.getByRole("button", {
        name: index === 24
          ? "최종 결과 보기"
          : scenarioEnds.has(index)
            ? "영상 확인 · 다음 시나리오"
            : "영상 확인 · 다음 투구",
      }),
    );
  });

  expect(
    screen.getByRole("heading", { name: "당신의 최종 점수는 28점입니다." }),
  ).toBeInTheDocument();
  expect(screen.getAllByText(/SCENARIO \d/)).toHaveLength(5);
  expect(await screen.findByText("결과가 저장되었습니다.")).toBeInTheDocument();
  const submissionCall = fetchMock.mock.calls.find(
    ([url]) => url === "/api/benchmark/submissions",
  )!;
  expect(JSON.parse(submissionCall[1].body)).toMatchObject({
    version: "v1",
    nickname: "야구팬",
    experience: "casual",
  });
});

test.skip("legacy Version 2 flow", () => {
  window.scrollTo = vi.fn();
  render(<HumanBenchmarkPrototype />);

  fireEvent.click(
    screen.getByRole("button", { name: /Pitch Test Version 2/ }),
  );
  expect(screen.getByText(/2026년에 던진 실제 5개/)).toBeInTheDocument();
  expect(screen.getByText(/5개 시나리오 · 25구 예측 시작하기/)).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText(/결과에 표시할 닉네임/), {
    target: { value: "버전2" },
  });
  fireEvent.click(screen.getByRole("button", { name: /가끔 봐요/ }));
  fireEvent.click(screen.getByRole("button", { name: /25구 예측 시작하기/ }));

  const scenarioEnds = new Set([4, 10, 14]);
  Array.from({ length: 15 }).forEach((_, index) => {
    fireEvent.click(
      within(screen.getByRole("radiogroup", { name: "예상 구종 계열" }))
        .getByRole("radio", { name: /패스트볼 계열/ }),
    );
    fireEvent.click(
      within(screen.getByRole("radiogroup", { name: "예상 상세 구종" }))
        .getByRole("radio", { name: /포심 Four-seam/ }),
    );
    fireEvent.click(screen.getByRole("button", { name: "답 확정하기" }));
    fireEvent.click(
      screen.getByRole("button", {
        name: scenarioEnds.has(index)
          ? "영상 확인 · 다음 시나리오"
          : "영상 확인 · 다음 투구",
      }),
    );
  });

  expect(screen.getByLabelText("Justin Verlander 구종")).toHaveTextContent(
    /포심.*슬라이더 · 스위퍼 · 커브.*체인지업/,
  );
  expect(screen.queryByLabelText("Reid Detmers 구종")).not.toBeInTheDocument();
  expect(screen.getByText("SCENARIO 4 · PITCH 1")).toBeInTheDocument();
  fireEvent.click(
    within(screen.getByRole("radiogroup", { name: "예상 구종 계열" }))
      .getByRole("radio", { name: /패스트볼 계열/ }),
  );
  expect(
    within(screen.getByRole("radiogroup", { name: "예상 상세 구종" }))
      .getAllByRole("radio"),
  ).toHaveLength(1);
});

test("plays the daily three-at-bat game and reveals the leaderboard at the end", async () => {
  window.scrollTo = vi.fn();
  window.localStorage.setItem(
    "pitch-test-daily-participant",
    "11111111-1111-4111-8111-111111111111",
  );
  const challenge = {
    id: "challenge-1",
    number: 1,
    date: "2026-07-29",
    sourceDate: "2026-07-28",
    scenarioCount: 3,
    participantCount: 7,
  };
  const repertoire = {
    FASTBALL: ["포심", "싱커"],
    BREAKING: ["슬라이더", "커브"],
    OFFSPEED: ["체인지업"],
  };
  const availableGroups = [
    "FOUR_SEAM",
    "MOVING_FASTBALL",
    "SLIDER",
    "CURVE",
    "CHANGEUP",
  ];
  const scenario = (number: number) => ({
    id: `daily-scenario-${number}`,
    pitcher: `Daily Pitcher ${number}`,
    batter: `Daily Batter ${number}`,
    date: "2026-07-28",
    inning: number === 1 ? "3회 초" : "7회 말",
    awayTeam: "NYY",
    homeTeam: "CWS",
    awayScore: number,
    homeScore: 0,
    outs: number - 1,
    bases: [false, false, false],
    outcome: number === 1 ? "삼진" : "안타",
    gameUrl: `https://www.mlb.com/gameday/${number}`,
    repertoire,
    availableGroups,
  });
  const current = (number: number) => ({
    questionNumber: number,
    scenarioNumber: number,
    scenarioCount: 3,
    scenario: scenario(number),
    pitch: { number: 1, count: "0–0" },
  });
  const reveal = (number: number) => ({
    scenarioId: `daily-scenario-${number}`,
    scenarioNumber: number,
    pitchNumber: 1,
    count: "0–0",
    selectedFamily: "FASTBALL",
    selected: "FOUR_SEAM",
    actualFamily: number === 1 ? "FASTBALL" : "BREAKING",
    actual: number === 1 ? "FOUR_SEAM" : "SLIDER",
    rawLabel: number === 1 ? "포심" : "슬라이더",
    result: number === 1 ? "루킹 스트라이크" : "인플레이",
    playId: `daily-play-${number}`,
    familyHit: number === 1,
    detailHit: number === 1,
    points: number === 1 ? 3 : 0,
    responseMs: 1000,
    modelPrediction: {
      selected: "CURVE",
      confidence: 0.42,
      probabilities: {
        CURVE: 0.42,
        FOUR_SEAM: 0.33,
        SLIDER: 0.25,
      },
      modelVersion: "V8.4",
    },
  });
  const firstReveal = reveal(1);
  const secondReveal = reveal(2);
  const thirdReveal = reveal(3);
  const reveals = [firstReveal, secondReveal, thirdReveal];
  const fetchMock = vi.fn(async (url: string, options?: RequestInit) => {
    if (url === "/api/daily/challenge") {
      return { ok: true, json: async () => challenge };
    }
    if (url === "/api/daily/attempts") {
      return {
        ok: true,
        json: async () => ({
          attemptId: "attempt-1",
          nickname: "데일리팬",
          challenge,
          status: "playing",
          current: current(1),
          history: [],
        }),
      };
    }
    if (url === "/api/daily/attempts/attempt-1/answers") {
      const body = JSON.parse(String(options?.body));
      const answerCount = fetchMock.mock.calls.filter(
        ([calledUrl]) =>
          calledUrl === "/api/daily/attempts/attempt-1/answers",
      ).length;
      expect(body).toMatchObject({
        selectedFamily: "FASTBALL",
        selected: "FOUR_SEAM",
      });
      if (answerCount < 3) {
        return {
          ok: true,
          json: async () => ({
            attemptId: "attempt-1",
            nickname: "데일리팬",
            challenge,
            status: "playing",
            current: current(answerCount + 1),
            history: reveals.slice(0, answerCount),
            reveal: reveals[answerCount - 1],
          }),
        };
      }
      return {
        ok: true,
        json: async () => ({
          attemptId: "attempt-1",
          nickname: "데일리팬",
          challenge,
          status: "completed",
          history: reveals,
          reveal: thirdReveal,
          result: {
            points: 3,
            maxPoints: 9,
            pitchCount: 3,
            detailHits: 1,
            familyHits: 1,
            rank: 4,
            streak: 2,
            leaderboard: {
              participantCount: 8,
              currentRank: 4,
              rows: [
                {
                  rank: 1,
                  nickname: "선두",
                  points: 6,
                  detailHits: 2,
                  familyHits: 2,
                  isCurrent: false,
                  isModel: false,
                },
                {
                  rank: 2,
                  nickname: "OyarZabal V8.4",
                  points: 5,
                  detailHits: 1,
                  familyHits: 2,
                  isCurrent: false,
                  isModel: true,
                },
                {
                  rank: 4,
                  nickname: "데일리팬",
                  points: 3,
                  detailHits: 1,
                  familyHits: 1,
                  isCurrent: true,
                  isModel: false,
                },
              ],
            },
          },
        }),
      };
    }
    throw new Error(`unexpected fetch: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<HumanBenchmarkPrototype />);

  expect(screen.queryByText(/Pitch Test Version 1/)).not.toBeInTheDocument();
  expect(screen.queryByText(/Pitch Test Version 2/)).not.toBeInTheDocument();
  fireEvent.click(
    await screen.findByRole("button", { name: /오늘의 Pitch Test/ }),
  );
  fireEvent.change(screen.getByLabelText(/리더보드에 표시할 닉네임/), {
    target: { value: "데일리팬" },
  });
  fireEvent.click(
    screen.getByRole("button", { name: "오늘의 3타석 시작하기 →" }),
  );
  expect(
    await screen.findByText("DAILY PITCHER 1 vs DAILY BATTER 1"),
  ).toBeInTheDocument();
  expect(screen.getByText("DAILY #1 · 3개 타석")).toBeInTheDocument();
  expect(document.querySelector(".hb-progress i")).toHaveStyle({ width: "33.33333333333333%" });

  for (let index = 0; index < 3; index += 1) {
    fireEvent.click(
      within(screen.getByRole("radiogroup", { name: "예상 구종 계열" }))
        .getByRole("radio", { name: /패스트볼 계열/ }),
    );
    fireEvent.click(
      within(screen.getByRole("radiogroup", { name: "예상 상세 구종" }))
        .getByRole("radio", { name: /포심 Four-seam/ }),
    );
    fireEvent.click(screen.getByRole("button", { name: "답 확정하기" }));
    expect(await screen.findByText("V8.4 MODEL PREDICTION")).toBeInTheDocument();
    expect(
      screen.queryByTitle(/MLB 공식 Daily Pitcher .*구 영상/),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "결과 확인하기 →" }),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "결과 확인하기 →" }),
    );
    expect(
      await screen.findByTitle(/MLB 공식 Daily Pitcher .*구 영상/),
    ).toHaveAttribute("src", expect.stringContaining(`daily-play-${index + 1}`));
    expect(screen.queryByText("V8.4 MODEL PREDICTION")).not.toBeInTheDocument();
    expect(screen.queryByText("모델의 선택")).not.toBeInTheDocument();
    expect(
      screen.getByText(`DAILY PITCHER ${index + 1} vs DAILY BATTER ${index + 1}`),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(
        `DAILY PITCHER ${index + 2} vs DAILY BATTER ${index + 2}`,
      ),
    ).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", {
        name: index < 2
          ? "다음 타석으로 →"
          : "최종 리더보드 보기 →",
      }),
    );
  }

  expect(
    screen.getByRole("heading", { name: "데일리팬님의 점수는 3점입니다." }),
  ).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "오늘의 리더보드" })).toBeInTheDocument();
  expect(screen.getByText("🔥 2일")).toBeInTheDocument();
  expect(screen.getByText("🤖 OyarZabal V8.4")).toBeInTheDocument();
  expect(screen.getByLabelText("공유 결과")).toHaveTextContent("PA 3");
  expect(screen.getByRole("row", { name: /4 데일리팬 3\/9 1 1/ })).toHaveClass(
    "current",
  );
});
