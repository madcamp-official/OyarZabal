# OyarZabal V6 계획

## 목표

완료된 MLB 경기를 한 구씩 재생하며 결과 공개 전에 다음 구종 확률을 확인하고,
경기 종료 후 Accuracy·Top-3 Accuracy·Macro F1·Log Loss와 구종 분포를
검증한다. V6는 미래 prospective 승격 전까지 shadow로 운영한다.

## 모델 구조

```text
모든 투구 ──> Calibrated Global XGBoost ──> 6종 logits ────────────┐
                                                                   ├─> 최종 확률
등록 투수 ──> pooled residual × reliability × context Gate ───────┘
미등록·탈락·안전 게이트 실패 ───────────────────────────> Global 100%
```

- Global은 `pitcher`, `batter` ID를 입력으로 사용하지 않는다.
- taxonomy는 포심, 무빙 패스트볼, 슬라이더 계열, 커브 계열, 체인지업,
  스플리터·포크의 6종이다.
- Global temperature 1.0465 뒤에 검증된 class-wise logit bias를 적용한다.
- Residual은 투수 ID, count bucket, 타자 손잡이와 같은 타석 직전 구종을
  사용해 Global logits의 보정량만 학습한다.
- Context Gate는 residual이 정답 log probability를 높일 가능성을 예측하는
  binary XGBoost다. Gate에는 투수·타자 ID를 넣지 않는다.

투수 \(i\), 투구 \(t\)의 scale은 다음으로 고정한다.

```text
reliability =
  0.5 × n/(n+1000) × min(전체 bootstrap 개선 확률, 최근 90일 개선 확률)

effective scale =
  hard safety pass × reliability × context Gate
```

- 경기 단위 paired bootstrap은 500회 수행한다.
- scale의 수식상 최대치는 0.5다.
- 최종 분포는 Global과 JS divergence 0.05 이하, 클래스별 확률 변화 20%p
  이하가 되도록 필요할 때 scale을 축소한다.
- 별도 drift 모델, residual magnitude damping과 scale grid search는 두지 않는다.

## 시간 검증과 Registry

- 2023 OOF로 residual을 최초 학습한다.
- 2024 앞 70%로 Gate를 학습하고 다음 10%에서 early stopping한다.
- 2024 마지막 20%에서 V6와 선수별 게이트를 평가한다.
- 2025는 후향 검증으로 사용한다.
- 2026-03-25~07-25는 이미 공개된 benchmark이므로 회귀 진단 외 선택에 쓰지
  않는다.
- 2026-07-26 이후 최소 30일·전체 100,000구·V6 개입 15,000구가 쌓인 첫
  시점에 V5 대비 paired game bootstrap으로 승격을 판단한다.

Active 선수는 2024·2025에서 다음을 모두 통과해야 한다.

- Global보다 Log Loss 개선
- Accuracy·Macro F1 하락 각각 0.5%p 이내
- 주요 구종 zero recall 없음
- `maxClassShareError ≤ 20%p`
- `TVD ≤ 20%p`
- `maxClassCalibrationError ≤ 10%p`

Active 목표나 상한은 없다. 통과하지 못한 선수는 shadow로 보존하고 Global만
사용한다. 최신 시즌 표본이 없는 provisional 선수는 reliability 최대 0.15와
365일 half-life를 적용한다.

## 피처와 진단

- 모든 누적·rolling·support 피처는 현재 투구를 제외한다.
- 커리어·시즌·최근 100구·카운트·손잡이·전환확률과 현재 경기 레퍼토리를
  유지한다.
- 같은 타석의 직전 1·2·3구, 첫 구종, 구종별 사용 횟수·비율과 연속 횟수를
  유지한다.
- Gate 전용으로 career/count/stand/transition support와 Global entropy,
  top1 margin, Global-reference disagreement와 JS를 사용한다.
- `majorityPredictionGap`은 호환 리포트에만 남긴다.
- 선택과 리포트에는 class별 signed share error, TVD, class recall,
  `classCalibrationError`, 선수별·월별 game bootstrap CI를 기록한다.

## 산출물과 운영

- Registry schema v5: calibration, reliability 구성요소, Gate metadata,
  active/shadow/provisional 상태와 실패 사유
- Holdout schema v2: 전체·선수별·월별 지표와 paired bootstrap CI
- Replay schema v6: reliability, context Gate, effective scale과 cap/fallback 사유
- 쇼케이스는 경기 시작 전 데이터만으로 별도 학습하며 `V6 Shadow`로 표시한다.
- 대형 작업은 순차 실행하고 RAM 9GiB 미만, 디스크 15GiB 미만 또는 GPU
  사용량 20GiB 초과 시 다음 단계를 시작하지 않는다.
- 실행별 자원 스냅샷, 설정, 지표와 실패 원인을 `artifacts/runs`에 남긴다.

Similarity stacker, xLSTM, 라이브 경기, 사용자 대결, 계정과 영상 분석은
V6 범위에서 제외한다.
