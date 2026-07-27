# V7 — 계층 decoder와 증분 Personalizer 재평가

> [!IMPORTANT]
> **결론:** 공식 예측을 `계열 합산 Top-1 → 계열 내부 Top-1`로 통일했다.
> 307구 쇼케이스는 Family 67.10%, Exact 56.35%로 개선됐다. 그러나 공개
> 2026 고정 cohort에서는 V5보다 낮아 V7은 계속 `shadow`다.

## 공식 선택 규칙

```text
각 계열의 두 세부 확률 합산
→ 합산 확률이 가장 높은 계열 선택
→ 선택한 계열 안에서 확률이 높은 구종 선택
```

- Exact Accuracy: 최종 세부 구종 적중률
- Family Accuracy: 합산 확률 Top-1 계열 적중률
- Hierarchical Accuracy: 세부 적중 1, 계열만 적중 0.5, 다른 계열 0의 평균
- `Hierarchical = (Exact + Family) / 2`

확률과 Log Loss 계산은 바뀌지 않는다. Top-1에 의존하는 Exact, Macro F1,
분포 진단, Gate disagreement와 선수 Registry는 모두 새 규칙으로 다시
계산했다.

## Registry 재판정

```text
full      4명  ██
limited  43명  ██████████████████████
shadow   51명  ██████████████████████████
```

| 이전 V7 → 새 V7 | 인원 |
|---|---:|
| full → full | 1 |
| full → limited | 7 |
| full → shadow | 2 |
| limited → full | 2 |
| limited → limited | 31 |
| limited → shadow | 7 |
| shadow → full | 1 |
| shadow → limited | 5 |
| shadow → shadow | 42 |

이전 `10 / 40 / 48`에서 새 `4 / 43 / 51`로 바뀌었다. 확률 모델만 유지한 채
표시를 바꾼 것이 아니라, 새 Exact·Macro F1·Hierarchical·구종 분포로
2024·2025 안전 배율을 전부 다시 선택한 결과다.

## 2024·2025 후향 검증

| 동일 98명 pool | Global Exact | V7 Exact | Global Family | V7 Family | Global Hier. | V7 Hier. | Global LL | V7 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 29,085구 | 45.41% | **45.68%** | 57.72% | **58.06%** | 51.57% | **51.87%** | 1.2122 | **1.2076** |
| 2025 189,721구 | 45.40% | **45.69%** | 56.31% | **56.67%** | 50.86% | **51.18%** | 1.2150 | **1.2104** |

두 해 모두 residual 라우팅이 Global보다 Exact, Family, Hierarchical과
Log Loss를 개선했다. Macro F1은 2024 `41.02% → 40.86%`, 2025
`42.65% → 42.55%`로 각각 0.5%p 안전 범위 안에서 소폭 하락했다.

## 공개 2026 공통 표본 재평가

기간은 2026-03-25~07-25다. V5·V6·V7 모두 같은
`family-sum-then-child` decoder로 다시 평가했다. 이미 공개된 구간이므로
승격 선택에는 사용하지 않는다.

### MLB 전체 459,530구

| 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|---:|
| V5 Final | 47.58% | 58.16% | 52.87% | **45.86%** | 1.1437 |
| V6 Final | **48.28%** | **59.27%** | **53.78%** | 45.49% | **1.1290** |
| V7 Final | 47.94% | 58.71% | 53.33% | 45.76% | 1.1359 |

```text
Family
V5 58.16%  █████████████████████████████
V6 59.27%  █████████████████████████████▋
V7 58.71%  █████████████████████████████▎
```

V7 자체 Global→Final 변화는 Exact `+0.03%p`, Family `+0.04%p`,
Hierarchical `+0.03%p`, Log Loss `−0.0006`이다. 개인화는 도움을 줬지만
V7 재학습에서 Macro F1 안전 조건 때문에 Global calibration weight가
기존 0.5에서 0.25로 낮아져 V6 전체 성능을 넘지 못했다.

### 고정 개인화 cohort 30명·28,734구

표본 fingerprint는 기존과 같은
`a2d7de0347b98e9cac05fe1ee22eedc1eab33aa5430e770089755f8945198232`다.

| 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|---:|
| V5 Final | **45.14%** | **58.61%** | **51.88%** | 39.23% | **1.2422** |
| V6 Final | 44.99% | 58.40% | 51.69% | 39.31% | 1.2475 |
| V7 Final | 44.53% | 57.83% | 51.18% | **39.77%** | 1.2515 |

V7은 V5보다 Exact `−0.62%p`, Family `−0.78%p`, Hierarchical
`−0.70%p`, Log Loss `+0.0094`다. paired game bootstrap 95% CI도
V5 대비 `−0.01076 ~ −0.00820`으로 음수라 승격할 수 없다.

## Tier별 2026 진단

| tier | 선수 | 행 | 개입 행 | Exact 변화 | Family 변화 | Hier. 변화 | Log Loss 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 4 | 4,124 | 4,109 | **+0.27%p** | −0.05%p | **+0.11%p** | **−0.0043** |
| limited | 43 | 40,746 | 40,720 | **+0.30%p** | **+0.45%p** | **+0.38%p** | **−0.0058** |
| shadow | 51 | 45,275 | 0 | 0 | 0 | 0 | 0 |

full의 Family는 0.05%p 낮아졌지만 Exact와 Hierarchical, Log Loss,
Macro F1 및 분포 진단이 개선돼 기존 안전 조건을 통과했다. limited는
Family를 포함한 세 정확도와 Log Loss를 모두 개선했다.

## 307구 쇼케이스

| 모델 | Exact | Family | Hierarchical | Top-3 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| Global | 56.03% | 66.78% | 61.40% | 95.44% | 47.71% | 0.9914 |
| V7 Final | **56.35%** | **67.10%** | **61.73%** | 95.44% | **47.88%** | **0.9895** |
| Similarity | 53.42% | 63.84% | 58.63% | **95.77%** | 42.55% | 1.0346 |
| Baseline | 51.47% | 57.65% | 54.56% | 91.21% | 33.63% | 1.1919 |

V7 residual은 188구에 limited로 적용됐고 119구는 shadow Global이었다.
이전 평면 Top-1 판정의 V7 쇼케이스와 비교하면 Exact `+3.26%p`, Family
`+5.21%p`, Hierarchical `+4.24%p`다. 한 경기 결과이므로 배포 성능
근거로 사용하지 않는다.

## 판정과 재현 근거

- 현재 `deploymentStatus`: `shadow`
- 첫 승격 평가: 2026-07-26 이후 최소 30일, 전체 100,000구,
  V7 개입 15,000구를 충족한 첫 prospective look
- 학습 run: `artifacts/runs/20260727T125835404946Z/result.json`
- V7 holdout:
  `artifacts/runs/20260727T125835404946Z/holdout-opened-2026-v7-hierarchical-decoder.json`
- V6 동등 비교:
  `artifacts/runs/20260727T125835404946Z/holdout-opened-2026-v6-hierarchical-decoder.json`
- 쇼케이스 run: `artifacts/runs/20260727T131317739205Z/result.json`
- Registry schema v7, Holdout schema v5, Replay schema v8
