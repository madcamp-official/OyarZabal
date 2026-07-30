import { useEffect, useState, type CSSProperties, type FormEvent } from "react";

import "./human-benchmark-admin.css";

interface AdminSummary {
  totals: {
    participants: number;
    attempts: number;
    detailRate: number;
    familyRate: number;
    averageScoreRate: number;
    averageResponseMs: number;
  };
  scenarios: Array<{
    id: string;
    version?: string;
    pitcher: string;
    batter: string;
    attempts: number;
    detailRate: number;
    familyRate: number;
    averagePoints: number;
  }>;
  pitches: Array<{
    scenarioId: string;
    pitchNumber: number;
    attempts: number;
    detailRate: number;
    familyRate: number;
  }>;
  scoreDistribution: Array<{ score: number; participants: number }>;
  submissions: Array<{
    id: string;
    participantLabel: string;
    version?: string;
    versionLabel?: string;
    experience: "new" | "casual" | "fan";
    submittedAt: string;
    points: number;
    maxPoints: number;
    detailHits: number;
    familyHits: number;
    averageResponseMs: number;
  }>;
  daily: {
    totals: {
      challenges: number;
      completedAttempts: number;
      answers: number;
      humanDetailRate: number;
      humanFamilyRate: number;
      modelDetailRate: number;
      modelFamilyRate: number;
    };
    challenges: Array<{
      date: string;
      sourceDate: string;
      participants: number;
      pitchCount: number;
      humanDetailRate: number;
      humanFamilyRate: number;
      modelDetailRate: number;
      modelFamilyRate: number;
      modelPoints: number;
      modelMaxPoints: number;
    }>;
    attempts: Array<{
      id: string;
      nickname: string;
      date: string;
      status: "playing" | "completed";
      points: number;
      maxPoints: number;
      detailHits: number;
      familyHits: number;
      answeredPitches: number;
      pitchCount: number;
      startedAt: string;
      completedAt: string | null;
    }>;
  };
  memorable: {
    totals: {
      completedAttempts: number;
      answers: number;
      detailRate: number;
      familyRate: number;
    };
    chapters: Array<{
      version: string;
      label: string;
      participants: number;
      pitchCount: number;
      answers: number;
      detailRate: number;
      familyRate: number;
      averageScoreRate: number;
    }>;
    submissions: Array<{
      id: string;
      participantLabel: string;
      version: string;
      versionLabel: string;
      experience: "new" | "casual" | "fan";
      submittedAt: string;
      points: number;
      maxPoints: number;
      detailHits: number;
      familyHits: number;
      averageResponseMs: number;
    }>;
  };
}

const experienceLabel = {
  new: "거의 안 봄",
  casual: "가끔 봄",
  fan: "자주 봄",
};

const percent = (rate: number) => `${Math.round(rate * 100)}%`;
const versionLabel = (version = "v1") =>
  version.startsWith("v") ? `V${version.slice(1)}` : version;

export default function HumanBenchmarkAdmin() {
  const [token, setToken] = useState("");
  const [summary, setSummary] = useState<AdminSummary>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const previousTitle = document.title;
    document.title = "Pitch Test — Admin";
    return () => {
      document.title = previousTitle;
    };
  }, []);

  const loadSummary = async (event?: FormEvent) => {
    event?.preventDefault();
    if (!token) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/admin/benchmark/summary", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error("unauthorized");
      setSummary(await response.json());
    } catch {
      setSummary(undefined);
      setError("관리자 토큰을 확인해 주세요.");
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = async () => {
    setError("");
    try {
      const response = await fetch("/api/admin/benchmark/submissions.csv", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error("download failed");
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "human-benchmark-submissions.csv";
      link.click();
      URL.revokeObjectURL(url);
    } catch {
      setError("CSV 다운로드에 실패했습니다.");
    }
  };

  if (!summary) {
    return (
      <main className="hba-login">
        <span>PITCH / TEST · ADMIN</span>
        <h1>Human benchmark<br />dashboard</h1>
        <p>VM의 BENCHMARK_ADMIN_TOKEN 값을 입력하세요.</p>
        <form onSubmit={loadSummary}>
          <label htmlFor="admin-token">관리자 토큰</label>
          <input
            autoComplete="current-password"
            id="admin-token"
            onChange={(event) => setToken(event.target.value)}
            type="password"
            value={token}
          />
          <button disabled={!token || loading} type="submit">
            {loading ? "확인 중…" : "대시보드 열기"}
          </button>
        </form>
        {error ? <strong role="alert">{error}</strong> : null}
      </main>
    );
  }

  const maxDistribution = Math.max(
    1,
    ...summary.scoreDistribution.map((item) => item.participants),
  );

  return (
    <div className="hba-shell">
      <header className="hba-header">
        <div>
          <span>PITCH / TEST · ADMIN</span>
          <h1>Pitch Test 결과 대시보드</h1>
        </div>
        <div>
          <button onClick={() => void loadSummary()} type="button">새로고침</button>
          <button onClick={() => void downloadCsv()} type="button">CSV 받기</button>
        </div>
      </header>

      {error ? <p className="hba-error" role="alert">{error}</p> : null}

      <main className="hba-main">
        <section className="hba-metrics" aria-label="Daily 전체 결과">
          <article><span>Daily 완료</span><strong>{summary.daily.totals.completedAttempts}</strong><small>명</small></article>
          <article><span>사람 상세 적중</span><strong>{percent(summary.daily.totals.humanDetailRate)}</strong><small>{summary.daily.totals.answers}개 답안</small></article>
          <article><span>모델 상세 적중</span><strong>{percent(summary.daily.totals.modelDetailRate)}</strong><small>V8.4 고정 예측</small></article>
          <article><span>모델 계열 적중</span><strong>{percent(summary.daily.totals.modelFamilyRate)}</strong><small>{summary.daily.totals.challenges}일 비교</small></article>
        </section>

        <section className="hba-panel">
          <header><span>DAILY HUMAN VS MODEL</span><h2>날짜별 적중률</h2></header>
          <div className="hba-table-wrap">
            <table>
              <thead>
                <tr><th>문제 날짜</th><th>경기 날짜</th><th>완료</th><th>투구</th><th>사람 계열</th><th>사람 상세</th><th>모델 계열</th><th>모델 상세</th><th>모델 점수</th></tr>
              </thead>
              <tbody>
                {summary.daily.challenges.map((challenge) => (
                  <tr key={challenge.date}>
                    <td><strong>{challenge.date}</strong></td>
                    <td>{challenge.sourceDate}</td>
                    <td>{challenge.participants}명</td>
                    <td>{challenge.pitchCount}구</td>
                    <td>{percent(challenge.humanFamilyRate)}</td>
                    <td>{percent(challenge.humanDetailRate)}</td>
                    <td>{percent(challenge.modelFamilyRate)}</td>
                    <td>{percent(challenge.modelDetailRate)}</td>
                    <td>{challenge.modelPoints}/{challenge.modelMaxPoints}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="hba-panel">
          <header><span>DAILY ATTEMPTS</span><h2>참가자별 진행 기록</h2></header>
          <div className="hba-table-wrap">
            <table>
              <thead>
                <tr><th>날짜</th><th>닉네임</th><th>상태</th><th>점수</th><th>계열</th><th>상세</th><th>진행</th><th>시작 시각</th></tr>
              </thead>
              <tbody>
                {summary.daily.attempts.map((attempt) => (
                  <tr key={attempt.id}>
                    <td>{attempt.date}</td>
                    <td><strong>{attempt.nickname}</strong></td>
                    <td>{attempt.status === "completed" ? "완료" : "진행 중"}</td>
                    <td>{attempt.points}/{attempt.maxPoints}</td>
                    <td>{attempt.familyHits}/{attempt.answeredPitches}</td>
                    <td>{attempt.detailHits}/{attempt.answeredPitches}</td>
                    <td>{attempt.answeredPitches}/{attempt.pitchCount}구</td>
                    <td>{new Date(attempt.startedAt).toLocaleString("ko-KR")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="hba-panel hba-memorable">
          <header><span>MEMORABLE SET</span><h2>명승부 Set 결과</h2></header>
          <section className="hba-metrics" aria-label="명승부 Set 전체 결과">
            <article><span>완료 기록</span><strong>{summary.memorable.totals.completedAttempts}</strong><small>회</small></article>
            <article><span>수집 답안</span><strong>{summary.memorable.totals.answers}</strong><small>개 투구</small></article>
            <article><span>계열 적중</span><strong>{percent(summary.memorable.totals.familyRate)}</strong><small>상세 적중 포함</small></article>
            <article><span>상세 적중</span><strong>{percent(summary.memorable.totals.detailRate)}</strong><small>정확한 구종</small></article>
          </section>
          <h3>Set별 집계</h3>
          <div className="hba-table-wrap">
            <table>
              <thead>
                <tr><th>명승부 Set</th><th>완료</th><th>투구</th><th>계열 적중</th><th>상세 적중</th><th>평균 점수율</th></tr>
              </thead>
              <tbody>
                {summary.memorable.chapters.map((chapter) => (
                  <tr key={chapter.version}>
                    <td><strong>{chapter.label}</strong></td>
                    <td>{chapter.participants}회</td>
                    <td>{chapter.pitchCount}구</td>
                    <td>{percent(chapter.familyRate)}</td>
                    <td>{percent(chapter.detailRate)}</td>
                    <td>{percent(chapter.averageScoreRate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <h3>최근 완료 기록</h3>
          <div className="hba-table-wrap">
            <table>
              <thead>
                <tr><th>참가자</th><th>명승부 Set</th><th>경험</th><th>점수</th><th>계열</th><th>상세</th><th>평균 응답</th><th>완료 시각</th></tr>
              </thead>
              <tbody>
                {summary.memorable.submissions.length ? (
                  summary.memorable.submissions.map((submission) => (
                    <tr key={submission.id}>
                      <td><strong>{submission.participantLabel}</strong></td>
                      <td>{submission.versionLabel}</td>
                      <td>{experienceLabel[submission.experience]}</td>
                      <td>{submission.points}/{submission.maxPoints}</td>
                      <td>{submission.familyHits}/{submission.maxPoints / 3}</td>
                      <td>{submission.detailHits}/{submission.maxPoints / 3}</td>
                      <td>{(submission.averageResponseMs / 1000).toFixed(1)}초</td>
                      <td>{new Date(submission.submittedAt).toLocaleString("ko-KR")}</td>
                    </tr>
                  ))
                ) : (
                  <tr><td colSpan={8}>아직 완료된 명승부 Set 기록이 없습니다.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="hba-panel hba-legacy">
          <header><span>ARCHIVED BENCHMARK</span><h2>기존 공개 실험 기록</h2></header>
        </section>

        <section className="hba-metrics" aria-label="전체 결과">
          <article><span>참가자</span><strong>{summary.totals.participants}</strong><small>명</small></article>
          <article><span>계열 적중</span><strong>{percent(summary.totals.familyRate)}</strong><small>상세 적중 포함</small></article>
          <article><span>상세 적중</span><strong>{percent(summary.totals.detailRate)}</strong><small>정확한 구종</small></article>
          <article><span>평균 점수</span><strong>{percent(summary.totals.averageScoreRate)}</strong><small>문항 수 정규화</small></article>
        </section>

        <section className="hba-panel">
          <header><span>SCENARIO BREAKDOWN</span><h2>타석별 적중률</h2></header>
          <div className="hba-table-wrap">
            <table>
              <thead>
                <tr><th>시나리오</th><th>투수 vs 타자</th><th>계열</th><th>상세</th><th>평균 점수</th></tr>
              </thead>
              <tbody>
                {summary.scenarios.map((scenario, index) => (
                  <tr key={scenario.id}>
                    <td>{versionLabel(scenario.version)} · {index + 1}</td>
                    <td><strong>{scenario.pitcher}</strong><small>vs {scenario.batter}</small></td>
                    <td>{percent(scenario.familyRate)}</td>
                    <td>{percent(scenario.detailRate)}</td>
                    <td>{scenario.averagePoints}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <div className="hba-two-column">
          <section className="hba-panel">
            <header><span>PITCH ACCURACY</span><h2>투구별 상세 적중률</h2></header>
            <div className="hba-pitch-grid">
              {summary.scenarios.map((scenario) => (
                <div key={scenario.id}>
                  <strong>{scenario.pitcher}</strong>
                  <span>vs {scenario.batter}</span>
                  <ol>
                    {summary.pitches
                      .filter((pitch) => pitch.scenarioId === scenario.id)
                      .map((pitch) => (
                        <li
                          key={`${scenario.id}-${pitch.pitchNumber}`}
                          style={{ "--hit-rate": pitch.detailRate } as CSSProperties}
                          title={`계열 ${percent(pitch.familyRate)}`}
                        >
                          <small>{pitch.pitchNumber}구</small>
                          <b>{percent(pitch.detailRate)}</b>
                        </li>
                      ))}
                  </ol>
                </div>
              ))}
            </div>
          </section>

          <section className="hba-panel">
            <header><span>SCORE DISTRIBUTION</span><h2>정규화 점수 분포</h2></header>
            <div className="hba-distribution">
              {summary.scoreDistribution.length ? (
                summary.scoreDistribution.map((item) => (
                  <div key={item.score}>
                    <span>{item.score}%</span>
                    <i style={{ width: `${(item.participants / maxDistribution) * 100}%` }} />
                    <b>{item.participants}명</b>
                  </div>
                ))
              ) : <p>아직 제출된 결과가 없습니다.</p>}
            </div>
          </section>
        </div>

        <section className="hba-panel">
          <header><span>RECENT SUBMISSIONS</span><h2>참가자별 결과</h2></header>
          <div className="hba-table-wrap">
            <table>
              <thead>
                <tr><th>참가자</th><th>버전</th><th>경험</th><th>점수</th><th>계열 적중</th><th>상세 적중</th><th>평균 응답</th><th>제출 시각</th></tr>
              </thead>
              <tbody>
                {summary.submissions.map((submission) => (
                  <tr key={submission.id}>
                    <td><strong>{submission.participantLabel}</strong></td>
                    <td>{versionLabel(submission.version)}</td>
                    <td>{experienceLabel[submission.experience]}</td>
                    <td>{submission.points}/{submission.maxPoints}</td>
                    <td>{submission.familyHits}/{submission.maxPoints / 3}</td>
                    <td>{submission.detailHits}/{submission.maxPoints / 3}</td>
                    <td>{(submission.averageResponseMs / 1000).toFixed(1)}초</td>
                    <td>{new Date(submission.submittedAt).toLocaleString("ko-KR")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
