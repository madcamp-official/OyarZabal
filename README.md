# OyarZabal

완료된 MLB 경기를 한 구씩 재생하며 다음 구종 예측을 검증하는 웹 데모입니다.

현재 MVP는 2024 월드시리즈 1차전의 실제 투구를 사용해 다음 모델을 같은
조건에서 비교합니다.

- 투수·카운트별 과거 구종 분포
- PitchPredict Similarity
- 2022–2025 MLB 전체 Global XGBoost
- 검증을 통과한 투수의 pooled contextual residual과 Global 결합

실제 구종은 사용자가 `실제 투구 공개`를 누르기 전까지 숨겨지며, 경기 종료 후
Accuracy, Top-3 Accuracy, Macro F1, Log Loss를 비교합니다.

## 구조

```text
ml/       데이터·피처·모델·정적 산출물 파이프라인
web/      Vite + React + TypeScript 리플레이 앱
docs/     기획, 결정, 실험 기록
reports/  주요 모델 버전별 성능 리포트와 현재 모델 진단
artifacts 실행별 설정·지표·자원 기록
```

## Python 파이프라인

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync --extra dev
uv run oyarzabal-fetch-statcast
uv run oyarzabal-train-hybrid
uv run oyarzabal-fetch-statcast \
  --start 2026-03-25 --end 2026-07-26 \
  --output data/holdout/statcast-2026
uv run oyarzabal-evaluate-holdout --models models/v6
uv run oyarzabal-build-demo \
  --history data/raw/statcast \
  --game /path/to/game.parquet \
  --predictions /path/to/predictions.csv
uv run pytest -q
```

수집은 `data/raw/statcast`의 월별 Parquet로 재개할 수 있습니다. 학습 모델은
`models/v6`, 정적 JSON은 `web/public/data`, 실행 기록은
`artifacts/runs/<run-id>`에 저장합니다.
2026 홀드아웃 평가는 2025년 말까지 학습한 모델만 허용하며, 학습 폴더나
registry에서 2026 데이터가 감지되면 중단합니다.

## Codex Worktree

Codex 앱에서 이 저장소 루트를 프로젝트로 등록한 뒤 Local Environment의 setup
script를 다음과 같이 설정합니다.

```bash
bash .codex/setup.sh
```

이 script는 Python 및 web dependency만 설치합니다. Git에서 제외된 원본 데이터와
학습 모델은 복사하지 않습니다.

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

상세 기획은 [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md), 모델 버전별 성능과
현재 진단은 [reports/README.md](reports/README.md)를 참고하세요.
