import { useEffect, useMemo, useState } from "react";

import type { Game, Manifest, ModelKey, Pitch } from "./types";

const modelOrder: ModelKey[] = ["final", "similarity", "baseline"];

function useReplayData() {
  const [data, setData] = useState<{ manifest: Manifest; game: Game }>();
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/data/manifest.json")
      .then((response) => {
        if (!response.ok) throw new Error("매니페스트를 불러오지 못했습니다.");
        return response.json() as Promise<Manifest>;
      })
      .then(async (manifest) => {
        if (manifest.schemaVersion !== 6) {
          throw new Error("지원하지 않는 경기 데이터 버전입니다.");
        }
        const response = await fetch(manifest.games[0].path);
        if (!response.ok) throw new Error("경기 데이터를 불러오지 못했습니다.");
        setData({ manifest, game: (await response.json()) as Game });
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  return { data, error };
}

function Diamond({ bases }: { bases: boolean[] }) {
  return (
    <div className="diamond" aria-label={`주자 ${bases.filter(Boolean).length}명`}>
      <i className={bases[1] ? "on second" : "second"} />
      <i className={bases[2] ? "on third" : "third"} />
      <i className={bases[0] ? "on first" : "first"} />
    </div>
  );
}

function Situation({ game, pitch }: { game: Game; pitch: Pitch }) {
  return (
    <section className="situation panel" aria-label="경기 상황">
      <div className="inning">
        <span>{pitch.half === "Top" ? "▲" : "▼"}</span>
        <strong>{pitch.inning}회</strong>
      </div>
      <div className="scoreboard">
        <div><span>{game.awayTeam}</span><strong>{pitch.score.away}</strong></div>
        <div><span>{game.homeTeam}</span><strong>{pitch.score.home}</strong></div>
      </div>
      <Diamond bases={pitch.bases} />
      <div className="count">
        <span><b>{pitch.count.balls}</b> B</span>
        <span><b>{pitch.count.strikes}</b> S</span>
        <span><b>{pitch.outs}</b> O</span>
      </div>
    </section>
  );
}

function ProbabilityBars({
  pitch,
  model,
  labels,
}: {
  pitch: Pitch;
  model: ModelKey;
  labels: Record<string, string>;
}) {
  const prediction = pitch.predictions[model];
  const sorted = Object.entries(prediction.probabilities).sort(
    ([, left], [, right]) => right - left,
  );
  return (
    <div className="bars">
      {sorted.map(([group, probability], index) => (
        <div className={index === 0 ? "bar-row leading" : "bar-row"} key={group}>
          <span>{labels[group]}</span>
          <div className="track">
            <i style={{ width: `${probability * 100}%` }} />
          </div>
          <b>{Math.round(probability * 100)}%</b>
        </div>
      ))}
    </div>
  );
}

function ModelTabs({
  active,
  onChange,
  names,
}: {
  active: ModelKey;
  onChange: (model: ModelKey) => void;
  names: Record<ModelKey, string>;
}) {
  return (
    <div className="tabs" role="tablist" aria-label="예측 모델">
      {modelOrder.map((model) => (
        <button
          aria-selected={active === model}
          className={active === model ? "active" : ""}
          key={model}
          onClick={() => onChange(model)}
          role="tab"
        >
          {names[model]}
        </button>
      ))}
    </div>
  );
}

function MetricsTable({
  game,
  manifest,
}: {
  game: Game;
  manifest: Manifest;
}) {
  const diagnostics = game.metrics.final;
  return (
    <section className="metrics panel">
      <div className="section-title">
        <span>MODEL REPORT</span>
        <h2>한 경기에서, 누가 더 잘 읽었나</h2>
      </div>
      <div className="metric-grid">
        {modelOrder.map((model) => (
          <article key={model}>
            <span>{manifest.models[model]}</span>
            <strong>{(game.metrics[model].accuracy * 100).toFixed(1)}%</strong>
            <small>정확도</small>
            <dl>
              <div><dt>Top 3</dt><dd>{(game.metrics[model].top3Accuracy * 100).toFixed(1)}%</dd></div>
              <div><dt>Macro F1</dt><dd>{game.metrics[model].macroF1.toFixed(3)}</dd></div>
              <div><dt>Log loss</dt><dd>{game.metrics[model].logLoss.toFixed(3)}</dd></div>
            </dl>
          </article>
        ))}
      </div>
      <div className="class-diagnostics">
        <div>
          <span>FINAL MODEL / SIX PITCH GROUPS</span>
          <h3>구종별 진단</h3>
        </div>
        <div className="diagnostic-head">
          <span>구종</span><span>실제 / 예측</span><span>Recall</span>
        </div>
        {Object.entries(manifest.pitchGroups).map(([group, label]) => {
          const actual = diagnostics.actualDistribution[group] ?? 0;
          const predicted = diagnostics.predictedDistribution[group] ?? 0;
          const recall = diagnostics.perClass[group]?.recall ?? 0;
          return (
            <div className="diagnostic-row" key={group}>
              <strong>{label}</strong>
              <div>
                <span style={{ width: `${actual * 100}%` }} />
                <i style={{ width: `${predicted * 100}%` }} />
                <small>
                  {(actual * 100).toFixed(1)}% / {(predicted * 100).toFixed(1)}%
                </small>
              </div>
              <b>{(recall * 100).toFixed(1)}%</b>
            </div>
          );
        })}
      </div>
      <p className="caveat">{game.caveat}</p>
    </section>
  );
}

export default function App() {
  const { data, error } = useReplayData();
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [model, setModel] = useState<ModelKey>("final");

  const runningAccuracy = useMemo(() => {
    if (!data || index < 0) return 0;
    const seen = data.game.pitches.slice(0, index + (revealed ? 1 : 0));
    if (!seen.length) return 0;
    return seen.filter(
      (pitch) => pitch.predictions.final.topPitch === pitch.actual.pitchGroup,
    ).length / seen.length;
  }, [data, index, revealed]);

  if (error) return <main className="state">{error}</main>;
  if (!data) return <main className="state">경기 기록을 정리하는 중…</main>;

  const { game, manifest } = data;
  const pitch = game.pitches[index];
  const next = () => {
    setIndex((current) => Math.min(current + 1, game.pitches.length - 1));
    setRevealed(false);
  };

  return (
    <main>
      <header>
        <a className="brand" href="#" aria-label="OyarZabal 처음으로">
          OYAR<span>ZABAL</span>
        </a>
        <div className="game-label">
          <span>{game.title}</span>
          <strong>{game.awayTeam} <i>at</i> {game.homeTeam}</strong>
        </div>
        <div className="live-badge"><i /> HISTORICAL REPLAY</div>
      </header>

      <div className="progress">
        <i style={{ width: `${((index + 1) / game.pitchCount) * 100}%` }} />
      </div>

      <section className="hero">
        <div>
          <p className="eyebrow">PITCH INTELLIGENCE / {game.date}</p>
          <h1>다음 공을<br /><em>먼저 읽는다.</em></h1>
          <p className="intro">
            결과를 열기 전에, 모델이 실제 경기 상황만으로 선택한 다음 구종을
            확인하세요.
          </p>
        </div>
        <div className="pitch-counter">
          <span>PITCH</span>
          <strong>{String(index + 1).padStart(3, "0")}</strong>
          <small>/ {game.pitchCount}</small>
        </div>
      </section>

      <Situation game={game} pitch={pitch} />

      <section className="matchup">
        <article>
          <span>PITCHING</span>
          <h2>{pitch.pitcher.name}</h2>
          <p>{pitch.pitcher.throws === "R" ? "우투" : "좌투"} · #{pitch.pitcher.id}</p>
        </article>
        <div className="versus">VS</div>
        <article className="batter">
          <span>AT BAT</span>
          <h2>{pitch.batter.name}</h2>
          <p>{pitch.batter.stand === "R" ? "우타" : "좌타"} · #{pitch.batter.id}</p>
        </article>
      </section>

      <section className="prediction panel">
        <div className="section-title">
          <span>NEXT PITCH PROBABILITY</span>
          <h2>모델의 선택</h2>
        </div>
        <ModelTabs active={model} onChange={setModel} names={manifest.models} />
        {model === "final" ? (
          <div className={`model-source ${pitch.modelSource.type}`}>
            <span>MODEL SOURCE</span>
            <strong>{pitch.modelSource.label}</strong>
            <small>
              {pitch.modelSource.pitcherReliability !== undefined
                ? [
                    `투수 신뢰도 ${Math.round(pitch.modelSource.pitcherReliability * 100)}%`,
                    `상황 Gate ${Math.round((pitch.modelSource.contextGate ?? 0) * 100)}%`,
                    `적용 ${Math.round((pitch.modelSource.effectiveScale ?? 0) * 100)}%`,
                    pitch.modelSource.capReason
                      ? `안전 제한: ${pitch.modelSource.capReason}`
                      : null,
                    pitch.modelSource.hardGateReason
                      ? `Global 전환: ${pitch.modelSource.hardGateReason}`
                      : null,
                  ].filter(Boolean).join(" · ")
                : pitch.modelSource.type !== "global"
                  ? `Residual scale ${Math.round((pitch.modelSource.residualScale ?? pitch.modelSource.specialistWeight) * 100)}%`
                  : "Global 100%"}
            </small>
          </div>
        ) : null}
        <div className="recent">
          <span>THIS AT-BAT</span>
          {pitch.recentPitches.length ? (
            pitch.recentPitches.map((code, recentIndex) => (
              <i key={`${code}-${recentIndex}`}>{code}</i>
            ))
          ) : (
            <small>첫 투구</small>
          )}
        </div>
        <ProbabilityBars pitch={pitch} model={model} labels={manifest.pitchGroups} />
        <div className={revealed ? "reveal open" : "reveal"}>
          {revealed ? (
            <>
              <div>
                <span>ACTUAL PITCH</span>
                <strong>{manifest.pitchGroups[pitch.actual.pitchGroup]}</strong>
                <small>
                  {pitch.actual.speed ? `${pitch.actual.speed} mph · ` : ""}
                  {pitch.actual.description.replaceAll("_", " ")}
                </small>
              </div>
              <div className={
                pitch.predictions[model].topPitch === pitch.actual.pitchGroup
                  ? "verdict correct"
                  : "verdict"
              }>
                {pitch.predictions[model].topPitch === pitch.actual.pitchGroup
                  ? "✓ 정확한 예측"
                  : "× 다른 선택"}
              </div>
            </>
          ) : (
            <button onClick={() => setRevealed(true)}>실제 투구 공개</button>
          )}
        </div>
        <ul className="explain">
          {pitch.explanations.map((line) => <li key={line}>{line}</li>)}
        </ul>
      </section>

      <nav className="controls" aria-label="리플레이 이동">
        <button
          disabled={index === 0}
          onClick={() => {
            setIndex((current) => Math.max(0, current - 1));
            setRevealed(false);
          }}
        >
          ← 이전
        </button>
        <div>
          <span>최종 후보 누적 정확도</span>
          <strong>{(runningAccuracy * 100).toFixed(1)}%</strong>
        </div>
        <button disabled={!revealed || index === game.pitchCount - 1} onClick={next}>
          다음 투구 →
        </button>
      </nav>

      {index === game.pitchCount - 1 && revealed ? (
        <MetricsTable game={game} manifest={manifest} />
      ) : null}

      <footer>
        <span>OyarZabal / Madcamp</span>
        <p>Prediction is not certainty. It is a better question.</p>
      </footer>
    </main>
  );
}
