# OyarZabal

완료된 MLB 경기를 한 구씩 재생하며 다음 구종 예측을 검증하는 웹 데모입니다.

현재 MVP는 2024 월드시리즈 1차전의 실제 투구를 사용해 다음 모델을 같은
조건에서 비교합니다.

- 투수·카운트별 과거 구종 분포
- PitchPredict Similarity
- 2022–2025 MLB 전체 Global XGBoost
- 검증을 통과한 유명 투수의 shrinkage personalizer와 Global 결합

실제 구종은 사용자가 `실제 투구 공개`를 누르기 전까지 숨겨지며, 경기 종료 후
Accuracy, Top-3 Accuracy, Macro F1, Log Loss를 비교합니다.

## 구조

```text
ml/       데이터·피처·모델·정적 산출물 파이프라인
web/      Vite + React + TypeScript 리플레이 앱
docs/     기획, 결정, 실험 기록
artifacts 실행별 설정·지표·자원 기록
```

## Python 파이프라인

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync --extra dev
uv run oyarzabal-fetch-statcast
uv run oyarzabal-train-hybrid
uv run oyarzabal-build-demo \
  --history data/raw/statcast \
  --game /root/workspace/pitchpredict-smoke-test/data/cache/games/775300.parquet \
  --predictions /root/workspace/pitchpredict-smoke-test/outputs/predictions.csv
uv run pytest -q
```

수집은 `data/raw/statcast`의 월별 Parquet로 재개할 수 있습니다. 학습 모델은
`models/hybrid`, 정적 JSON은 `web/public/data`, 실행 기록은
`artifacts/runs/<run-id>`에 저장합니다.

## 웹앱

```bash
cd web
npm install
npm run dev
```

검증:

```bash
npm test
npm run build
```

상세 기획은 [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)를 참고하세요.
