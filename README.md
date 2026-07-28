# OyarZabal

완료된 MLB 경기를 한 구씩 재생하며 다음 구종 예측을 검증하는 웹 데모입니다.

현재 MVP는 2024 월드시리즈 1차전의 실제 투구를 사용해 다음 모델을 같은
조건에서 비교합니다.

- 투수·카운트별 과거 구종 분포
- PitchPredict Similarity
- 2022–2025 MLB 전체 Global XGBoost
- 검증을 통과한 투수의 pooled contextual residual과 Global 결합

실제 구종은 사용자가 `실제 투구 공개`를 누르기 전까지 숨겨지며, 경기 종료 후
Exact·Family·Hierarchical Accuracy, Top-3 Accuracy, Macro F1, Log Loss를
비교합니다. 확률은 3개 계열 아래 기존 6개 세부 구종으로 표시합니다.
공식 예측은 세부 확률을 계열별로 합산해 가장 높은 계열을 고른 뒤, 그 계열
안에서 확률이 가장 높은 구종을 선택합니다.

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
uv run oyarzabal-evaluate-holdout \
  --models models/v7 \
  --reference-models models/hybrid
# prospective 승격 평가는 공개된 회귀 진단 구간을 제외한다.
uv run oyarzabal-evaluate-holdout \
  --models models/v7 \
  --reference-models models/hybrid \
  --prospective-manifest config/v7-prospective.json \
  --output artifacts/prospective/v7-first-look.json
uv run oyarzabal-build-demo \
  --history data/raw/statcast \
  --game /path/to/game.parquet \
  --predictions /path/to/predictions.csv
uv run pytest -q
```

수집은 `data/raw/statcast`의 월별 Parquet로 재개할 수 있습니다. 학습 모델은
`models/v7`, 정적 JSON은 `web/public/data`, 실행 기록은
`artifacts/runs/<run-id>`에 저장합니다.
2026 홀드아웃 평가는 2025년 말까지 학습한 모델만 허용하며, 학습 폴더나
registry에서 2026 데이터가 감지되면 중단합니다.
버전 간 개인화 비교는 V5 활성 선수 30명을 동결한
`v5-enabled-pitchers-v1` cohort를 자동으로 사용합니다. 같은 benchmark의
정확한 투구가 달라지면 row fingerprint 검사에서 평가가 중단됩니다.
V7 Registry는 선수를 `full / limited / shadow`로 나누며, prospective 승격
전까지 제품 배포 상태는 shadow입니다.
독립 검증에서는 현재 배율 `1.0`과 제한 선수 배율만 `1.5`로 높인 후보를
동시에 동결합니다. 2026-07-26 이후 최소 30일·100,000구·후보별 개입
15,000구가 모이기 전에는 성능 지표를 열지 않습니다.

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
