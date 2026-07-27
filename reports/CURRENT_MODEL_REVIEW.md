# 현재 모델 문제점과 개선안

## 결론

현재 모델은 V3 Global보다 개인화 성능과 시간 재현성이 좋아졌고, pooled
residual은 독립 2026의 활성 선수 표본에서도 Accuracy를 1.83%p, Log Loss를
0.0239 개선했다. 구조를 폐기할 이유는 없다.

다만 “쏠림 해결”이라고 부르기에는 이르다. 전역 포심 과대예측은 사라졌지만,
선수별 residual이 특정 구종을 과도하게 선택하는 국소 쏠림이 남아 있다.
현재 active 25명은 품질 목표가 아니라 노출 상한이며, 선수별 분포와 Macro F1
게이트가 충분하지 않다.

## “active 25명 → 15명”의 정확한 의미

이전 판단의 15명은 현재 active 25명에게 **2025
`majorityPredictionGap ≤ 20%p`만 사후 적용**한 결과다. 전체 후보를 다시
선발한 결과는 다르다.

| 조건 | 현재 registry에서 남는 후보 | active 상한 적용 후 |
|---|---:|---:|
| 기존 게이트 | 52 | 25 |
| 2025 majority gap ≤20%p | 36 | 25 |
| 2024·2025 majority gap 모두 ≤20%p | 28 | 25 |
| 2024·2025 구종별 최대 분포 오차 모두 ≤20%p | 23 | 23 |
| 위 조건 + Global보다 분포 오차 악화 ≤5%p | 17 | 17 |
| 위 조건 + Macro F1 하락 ≤1%p | 12 | 12 |

따라서 “게이트 하나를 넣으면 active가 15명이 된다”는 표현은 정확하지 않다.
현재 25명만 제거하고 exposure-cap 후보로 다시 채우지 않을 때만 15명이다.
품질을 우선한다면 목표 인원수를 먼저 정하지 말고 게이트 통과자 수를 결과로
받아들여야 한다.

52명은 2026 동결 평가 직전에 다시 생성한 최신 registry 기준이다. 이전 실험
로그의 53명은 그보다 앞선 run snapshot이며, 현재 모델 진단에는 최신 registry를
사용했다.

또한 `majorityPredictionGap`만으로는 부족하다. 이 지표는 “예측 최다 클래스
비율이 실제 최다 클래스 비율보다 얼마나 더 큰가”만 본다. 특정 클래스를 크게
과소예측하거나 최다 클래스가 서로 바뀐 경우에는 0이 될 수 있다. 예를 들어
Taijuan Walker의 2025 candidate는 majority gap이 0이지만 구종별 최대 분포
오차는 23.56%p다.

## 문제 1 — 선수별 국소 구종 쏠림

현재 active 25명의 2025 결과:

- 평균 majority gap: Global 12.40%p → residual 15.78%p
- 중앙값: Global 9.55%p → residual 11.97%p
- 개선 9명, 악화 16명
- residual majority gap 20%p 초과: 10명
- 주요 구종 zero recall은 없지만, 0보다 조금 큰 recall도 통과할 수 있다.

심한 사례:

| 투수 | Global gap | Residual gap | 실제 최다 비율 | 예측 최다 비율 |
|---|---:|---:|---:|---:|
| Nathan Eovaldi | 55.50%p | 46.22%p | 31.42% | 77.64% |
| Chris Bassitt | 25.19%p | 36.13%p | 58.18% | 94.31% |
| Luis Castillo | 17.89%p | 34.49%p | 45.48% | 79.98% |
| Aaron Civale | 25.42%p | 31.58%p | 51.11% | 82.69% |
| Tarik Skubal | 29.41%p | 27.11%p | 30.65% | 57.76% |

Eovaldi처럼 residual이 Global보다 낫지만 여전히 절대 오차가 큰 경우와,
Bassitt·Castillo처럼 residual이 오히려 쏠림을 키우는 경우를 구분해야 한다.

### 원인

선수별 통과 함수는 현재 다음만 확인한다.

- 평가 표본 300구 이상
- Log Loss 개선
- Accuracy 하락 0.5%p 이내
- 실제 비율 5% 이상 클래스의 zero recall 없음

선수별 Macro F1, 분포 오차, Global 대비 분포 악화는 확인하지 않는다. 모든
active 선수에게 공통 scale 0.5를 쓰는 점도 과보정을 만든다.

### 개선

1. `majorityPredictionGap` 대신 모든 클래스에 대해
   `max(abs(predicted_share - actual_share))`를 기록한다.
2. 2024와 2025에서 모두 최대 분포 오차 20%p 이하를 요구한다.
3. candidate 분포 오차가 같은 행의 Global보다 5%p 이상 악화되면 탈락시킨다.
4. Macro F1은 두 해 모두 Global 대비 1%p 이상 하락하지 않도록 한다.
5. 실제 비율 5% 이상 클래스는 zero recall뿐 아니라 recall과 support를
   진단표에 남긴다. hard floor는 bootstrap 검증 후 정한다.

현재 결과에 1–4를 적용하면 high-trust active 후보는 12명이다. 인원수를
늘리기 위해 기준을 낮추지 않고 나머지는 Global로 라우팅하는 것이 안전하다.

## 문제 2 — 공통 residual scale과 불안정한 순위

scale 0.5는 pool aggregate에서 선택됐고 모든 active 선수에게 동일하게
적용된다. 투수마다 데이터 양과 residual 반응이 달라 일부는 0.25가 더 안전할
수 있다. 또한 현재 순위는 2024·2025 Log Loss 개선량의 합을 사용하고 파일럿
선수를 먼저 배치한다. 한 해의 큰 이득이 다른 해의 불안정을 가릴 수 있다.

### 개선

- scale 후보는 `0, 0.25, 0.5`로 제한한다.
- 2024에서 투수별 scale을 선택하고 2025에서 그대로 검증한다.
- 선택 표본이 작으면 0.25로 shrink하거나 Global을 유지한다.
- 순위는 두 해 개선량의 합보다 `min(2024 개선, 2025 개선)`을 우선한다.
- 유명 선수 우선순위는 품질 게이트를 모두 통과한 뒤 표시 순서에만 사용한다.

## 문제 3 — Accuracy 개선에 비해 Macro F1 개선이 작음

2026 active/provisional 28,734구에서 Accuracy는 44.15%에서 45.97%로
1.83%p 올랐지만 Macro F1은 41.90%에서 41.94%로 0.04%p만 올랐다. 즉
개인화 이득이 주력 구종과 흔한 상황에 집중됐을 가능성이 높다.

### 개선

- per-player 게이트에 Macro F1 비열화 조건을 추가한다.
- 실제 비율 5% 이상인 클래스별 recall을 선수별 리포트에 항상 기록한다.
- Accuracy와 Log Loss만 좋아지고 Macro F1이 나빠지는 후보를 별도
  `probability-only` 상태로 남기되 Top-1 라우팅에는 사용하지 않는다.

## 문제 4 — 전역 prior/calibration 오차

2026 전체에서 포심은 실제 30.60%인데 최종 예측은 21.31%다. 반대로
체인지업은 11.14%→16.88%, 스플리터/포크는 3.31%→7.69%다.
`majorityPredictionGap`은 0이지만 클래스별 분포는 맞지 않는다.

클래스 가중치가 rare class recall을 높이는 과정에서 예측 prior를 이동시켰을
가능성이 있다. 하나의 temperature는 confidence만 조절할 뿐 클래스별 bias는
바로잡지 못한다.

### 개선

- 2023·2024·2025 OOF만 사용해 class-wise intercept calibration 후보를
  평가한다.
- 채택 조건은 Log Loss 개선, Accuracy 하락 0.5%p 이내, Macro F1 비열화
  없음, 최대 클래스 분포 오차 감소로 고정한다.
- 2026은 calibration 선택에 사용하지 않고 보고만 한다.
- 인위적으로 실제 분포에 예측 비율을 맞추는 post-hoc 강제 보정은 하지 않는다.

## 문제 5 — residual 적용 범위가 좁음

2026 전체 459,530구 중 residual 적용은 28,734구, 6.25%다. 활성 선수
범위에서는 효과가 크지만 전체 Accuracy 개선은 0.11%p에 그친 이유다.

### 개선

- 먼저 active 품질을 강화하고, 통과자 수를 억지로 25명에 맞추지 않는다.
- 이후 2023–2025 support가 충분한 exposure-cap 후보를 같은 strict gate로
  재평가한다.
- 신규·복귀 투수는 최근 데이터가 쌓이기 전까지 Global을 사용한다.
- coverage 자체보다 `개선된 투구 수 × 신뢰 가능한 효과`를 KPI로 둔다.

## 문제 6 — 불확실성과 drift 리포트 부족

현재 평가는 point estimate만 저장한다. 300구는 몇 경기의 결과에 좌우될 수
있고, 2026 holdout도 시즌 전반부뿐이다. 선수별 2026 지표와 월별 drift도
artifact에 없다.

### 개선

- OOF 예측을 투구 단위로 보존하고 `game_pk` 단위 bootstrap 95% CI를 만든다.
- 선수별·월별 Accuracy, Macro F1, Log Loss, 최대 분포 오차를 기록한다.
- 활성 조건은 가능하면 Log Loss 개선 CI가 0 아래인지 확인한다.
- 2026-03-25~07-25는 이미 공개된 benchmark이므로 후속 선택에는 쓰지 않는다.
  다음 독립 평가는 이후에 잠근 시간 구간이나 2027로 남긴다.

## 권장 구현 순서

1. **평가 지표 보강**: 최대 클래스 분포 오차, total variation distance,
   선수별 Macro F1과 주요 클래스 recall을 artifact에 추가한다.
2. **Registry 재선발**: 2024·2025 strict gate를 적용하고 active 수를 결과로
   결정한다. 현재 데이터에서는 약 12명이 high-trust 후보가 된다.
3. **Scale 축소 실험**: 선수별 `0/0.25/0.5`를 2024에서 선택하고 2025에서
   검증한다.
4. **불확실성 추가**: 경기 단위 bootstrap과 월별 drift 리포트를 만든다.
5. **Global calibration 실험**: pre-2026 OOF에서만 class-wise intercept를
   검증한다.
6. **새 독립 평가 예약**: 현재 공개한 2026 구간은 재선택에 사용하지 않는다.

## 완료 조건

- 선수별 gate가 2024와 2025 모두에서 통과해야 한다.
- 모든 active의 최대 클래스 분포 오차가 20%p 이하이다.
- candidate 분포 오차가 Global보다 5%p 넘게 악화되지 않는다.
- Macro F1 하락은 선수별 1%p 이내, aggregate 0.5%p 이내이다.
- 실제 비율 5% 이상 클래스에 zero recall이 없다.
- active 수는 목표 숫자로 채우지 않고 통과 결과로 결정한다.
- 2026 공개 benchmark를 모델·scale·registry 선택에 사용하지 않는다.

## 근거

- 현재 registry: 로컬 `models/hybrid/registry.json`
- 2026 평가:
  `artifacts/runs/20260727T032547104596Z/holdout-2026.json`
- 선수별 게이트 구현:
  [`ml/oyarzabal/residual.py`](../ml/oyarzabal/residual.py)
- 노출 순위 구현:
  [`ml/oyarzabal/training.py`](../ml/oyarzabal/training.py)
- [V4 pooled residual 리포트](2026-07-27-pooled-residual.md)
- [V5 2026 holdout 리포트](2026-07-27-frozen-holdout.md)
