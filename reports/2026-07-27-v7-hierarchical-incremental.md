# V7 — 계층형 구종 평가와 증분 Personalizer

> [!IMPORTANT]
> **결론:** V7은 Global 대비 full·limited 개인화를 안전하게 개선했지만,
> 공개된 2026 고정 cohort에서는 V5보다 Log Loss가 0.0019 나빴다.
> 따라서 Registry 라우팅은 구현됐어도 제품 상태는 prospective 평가 전까지
> `shadow`다.

## 한눈에 보기

```text
Registry
full     10명  █████
limited  40명  ████████████████████
shadow   48명  ████████████████████████
```

V6 후향 분석의 `10 / 39 / 49`에서 2025 기록이 비어 있으나 과거 안전 배율을
가진 선수 #676710 한 명이 stale limited로 이동했다. 이 선수는 365일
half-life 감쇠를 추가로 받는다.

| V6 후향 tier → V7 실제 tier | 인원 |
|---|---:|
| full → full | 10 |
| limited → limited | 39 |
| shadow → limited(stale) | 1 |
| shadow → shadow | 48 |

## 무엇이 바뀌었나

- 6종 출력은 유지하고 패스트볼·브레이킹볼·오프스피드 3계열을 확률 합으로
  표시한다.
- `Family Accuracy`와 `Hierarchical Accuracy`를 모든 평가 범위에 추가했다.
- 선수 단위 절대 게이트가 실패해도 같은 행의 Global보다 안전하게 나아지는
  최대 배율을 찾으면 `limited`로 실제 추론한다.
- `full`은 배율 1, `limited`는 선수별 배율, `shadow`는 배율 0이다.
- Context Gate, reliability, JS 0.05와 클래스별 20%p cap은 유지했다.

```text
Hierarchical Accuracy
정확한 6종 적중    1.0
같은 계열 오답     0.5
다른 계열          0.0

= (Exact Accuracy + Family Accuracy) / 2
```

## 2024·2025 후향 검증

| 동일 98명 pool | Global Exact | V7 Exact | Global Family | V7 Family | Global Hier. | V7 Hier. | Global LL | V7 LL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 29,085구 | 46.29% | **46.52%** | 56.06% | **56.36%** | 51.18% | **51.44%** | 1.2067 | **1.2037** |
| 2025 189,721구 | 46.39% | **46.56%** | 54.74% | **55.00%** | 50.56% | **50.78%** | 1.2077 | **1.2049** |

두 해 모두 residual 라우팅이 Global보다 Log Loss·Exact·Family·Hierarchical을
개선했다. 2025 Macro F1은 43.503%에서 43.478%로 0.025%p 하락했지만
0.5%p 안전 한도 안이다.

## 공개 2026 회귀 진단

기간은 2026-03-25~07-25다. 이 결과는 이미 모델 설계 전에 공개됐으므로
승격 선택에는 사용하지 않는다.

### MLB 전체 459,530구

| 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|---:|
| V5 Final | 47.73% | 55.88% | 51.81% | 46.50% | 1.1437 |
| V7 Global | 48.77% | 57.32% | 53.05% | **46.80%** | 1.1291 |
| V7 Final | **48.80%** | **57.37%** | **53.08%** | 46.80% | **1.1287** |

```text
Exact         V5 47.73%  ████████████████████████
              V7 48.80%  ████████████████████████▌
Hierarchical  V5 51.81%  ██████████████████████████
              V7 53.08%  ██████████████████████████▌
```

V7 전체 개선에는 새 Global 재학습 효과가 크게 포함된다. 개인화 자체의 증분은
V7 Global→Final 기준 Exact +0.03%p, Hierarchical +0.04%p, Log Loss
−0.0004다.

### 고정 개인화 cohort 30명·28,734구

표본 fingerprint는 기존과 같은
`a2d7de0347b98e9cac05fe1ee22eedc1eab33aa5430e770089755f8945198232`다.

| 모델 | Exact | Family | Hierarchical | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|---:|
| V5 Final | **45.96%** | **56.80%** | **51.38%** | **42.01%** | **1.2422** |
| V7 Global | 45.50% | 56.11% | 50.81% | 41.67% | 1.2478 |
| V7 Final | 45.80% | 56.68% | 51.24% | 41.69% | 1.2441 |

V7 개인화는 자체 Global보다 좋아졌지만 V5 Final보다 Exact −0.16%p,
Hierarchical −0.14%p, Log Loss +0.0019다. 고정 cohort paired bootstrap의
95% CI도 V5 대비 `−0.00279 ~ −0.00110`으로 음수여서 승격할 수 없다.

## Tier별 진단

| tier | 선수 | 2026 행 | 개입 행 | Exact 변화 | Family 변화 | Hier. 변화 | Log Loss 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 10 | 10,344 | 10,344 | **+0.28%p** | **+0.31%p** | **+0.29%p** | **−0.0030** |
| limited | 40 | 39,559 | 39,518 | **+0.29%p** | **+0.46%p** | **+0.37%p** | **−0.0040** |
| shadow | 48 | 40,242 | 0 | 0 | 0 | 0 | 0 |

limited는 이번 공개 진단에서 full보다 위험하지 않았고 오히려 증분 폭이
조금 컸다. 하지만 배율이 2024·2025에서 선택됐으므로 이 표만으로 배포를
승격하지 않는다.

## 307구 쇼케이스

| 모델 | Exact | Family | Hierarchical | Top-3 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| Global | 53.09% | 61.89% | 57.49% | 95.44% | 45.73% | 0.9914 |
| V7 Final | 53.09% | 61.89% | 57.49% | 95.44% | 45.73% | **0.9896** |

100구에 residual이 적용됐다(`full 90`, `limited 10`). Top-1은 바꾸지 않고
정답 확률만 개선했다. 한 경기 쇼케이스이므로 배포 성능 근거로 쓰지 않는다.

## 판정과 다음 평가

- 현재 `deploymentStatus`: `shadow`
- 공개 2026 데이터: 회귀 진단만
- 첫 승격 평가: 2026-07-26 이후만 잘라 최소 30일, 전체 100,000구,
  V7 개입 15,000구를 모두 충족한 시점
- 비교 기준: MLB 전체와 고정 30명 cohort에서 V5 대비 paired game bootstrap

## 재현 근거

- 학습 run: `artifacts/runs/20260727T113141910703Z/result.json`
- 공개 holdout:
  `artifacts/runs/20260727T113141910703Z/holdout-opened-2026-v7.json`
- 쇼케이스 run: `artifacts/runs/20260727T115105174501Z/result.json`
- Registry: `models/v7/registry.json`
- Replay: `web/public/data/games/775300.json`
