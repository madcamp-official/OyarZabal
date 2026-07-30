import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";

import HumanBenchmarkAdmin from "./HumanBenchmarkAdmin";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("requires a token and renders aggregate benchmark results", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      totals: {
        participants: 3,
        attempts: 75,
        detailRate: 0.2,
        familyRate: 0.35,
        averageScoreRate: 0.42,
        averageResponseMs: 2100,
      },
      scenarios: [{
        id: "brown-rodriguez",
        pitcher: "Hunter Brown",
        batter: "Julio Rodríguez",
        attempts: 12,
        detailRate: 0.25,
        familyRate: 0.5,
        averagePoints: 6,
      }],
      pitches: [{
        scenarioId: "brown-rodriguez",
        pitchNumber: 1,
        attempts: 3,
        detailRate: 0.33,
        familyRate: 0.66,
      }],
      scoreDistribution: [{ score: 41, participants: 2 }],
      submissions: [{
        id: "id-1",
        participantLabel: "야구팬",
        experience: "casual",
        submittedAt: "2026-07-27T00:00:00+00:00",
        points: 41,
        maxPoints: 75,
        detailHits: 7,
        familyHits: 12,
        averageResponseMs: 2100,
      }],
      memorable: {
        totals: {
          completedAttempts: 1,
          answers: 12,
          detailRate: 0.25,
          familyRate: 0.5,
        },
        chapters: [{
          version: "memorable:alcs7-bottom-7",
          label: "ALCS Game 7 · 7회말 · TOR 공격",
          participants: 1,
          pitchCount: 12,
          answers: 12,
          detailRate: 0.25,
          familyRate: 0.5,
          averageScoreRate: 0.33,
        }],
        submissions: [{
          id: "memorable-id-1",
          participantLabel: "명승부팬",
          version: "memorable:alcs7-bottom-7",
          versionLabel: "ALCS Game 7 · 7회말 · TOR 공격",
          experience: "fan",
          submittedAt: "2026-07-29T01:00:00+00:00",
          points: 12,
          maxPoints: 36,
          detailHits: 3,
          familyHits: 6,
          averageResponseMs: 1500,
        }],
      },
      daily: {
        totals: {
          challenges: 1,
          completedAttempts: 2,
          answers: 30,
          humanDetailRate: 0.23,
          humanFamilyRate: 0.4,
          modelDetailRate: 0.33,
          modelFamilyRate: 0.53,
        },
        challenges: [{
          date: "2026-07-29",
          sourceDate: "2026-07-27",
          participants: 2,
          pitchCount: 15,
          humanDetailRate: 0.23,
          humanFamilyRate: 0.4,
          modelDetailRate: 0.33,
          modelFamilyRate: 0.53,
          modelPoints: 19,
          modelMaxPoints: 45,
        }],
        attempts: [{
          id: "daily-id-1",
          nickname: "데일리팬",
          date: "2026-07-29",
          status: "completed",
          points: 14,
          maxPoints: 45,
          detailHits: 3,
          familyHits: 5,
          answeredPitches: 15,
          pitchCount: 15,
          startedAt: "2026-07-29T00:00:00+00:00",
          completedAt: "2026-07-29T00:05:00+00:00",
        }],
      },
    }),
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<HumanBenchmarkAdmin />);

  fireEvent.change(screen.getByLabelText("관리자 토큰"), {
    target: { value: "secret-token" },
  });
  fireEvent.click(screen.getByRole("button", { name: "대시보드 열기" }));

  expect(await screen.findByText("야구팬")).toBeInTheDocument();
  expect(screen.getByText("데일리팬")).toBeInTheDocument();
  expect(screen.getByText("명승부팬")).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "명승부 Set 결과" })).toBeInTheDocument();
  expect(screen.getAllByText("ALCS Game 7 · 7회말 · TOR 공격")).toHaveLength(2);
  expect(screen.getByText("V8.4 고정 예측")).toBeInTheDocument();
  expect(screen.getByText("35%")).toBeInTheDocument();
  await waitFor(() => {
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/admin/benchmark/summary",
      { headers: { Authorization: "Bearer secret-token" } },
    );
  });
});
