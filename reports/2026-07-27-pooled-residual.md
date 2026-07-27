# V4 — pooled contextual residual

> [!NOTE]
> **한 줄 결론:** 독립 개인화 대신 표본을 공유하는 residual 구조로 바꾸자
> 2025 pool에서 Accuracy와 Log Loss가 함께 개선됐고 실제 배포 후보가 생겼다.
> 아래 46.48%는 전체 MLB 성능이 아니며, 제품 단위 외부 비교에서는
> Global 47.62%를 Final routed 47.73%로 개선했다.

## 한눈에 보기

- 날짜: 2026-07-27
- 학습 데이터: 2022–2025 지원 구종 2,990,491구
- taxonomy: 현재 6종
- 구조: frozen Global logits + pooled multiclass XGBoost correction
- residual: 599 trees, 공통 scale 0.5, 학습 pool 98명
- registry: active 25명, provisional 5명

## 무엇이 바뀌었나

투수별 상수 bias를 카운트 bucket, 타자 손잡이, 직전 구종과 투수 ID를 입력으로
받는 얕은 residual로 교체했다. 2023 OOF로 residual을 만들고 2024에서 scale을
선택한 뒤 2025에서 최종 게이트를 확인했다. Gerrit Cole처럼 2025 표본이 없는
투수는 작은 보정을 시간에 따라 감쇠하는 provisional로 분리했다.

## 평가 설계

- residual 최초 학습: 2023 OOF
- scale 선택: 2024
- 최종 게이트: 2025
- 아래 pool 지표는 98명 모두에게 residual을 적용한 후보 평가다.
- 실제 제품은 그중 registry에서 활성화된 선수에게만 residual을 적용한다.
- 자격 미달·비활성 선수도 Global 학습에서는 제외되지 않는다. 전체
  2,990,491구가 Global에 사용되고, 실제 추론에서는 residual 미적용 선수의
  Global 확률을 그대로 유지한다.

## 성능 결과

### 내부 residual pool 평가

다음 45.08%와 46.48%는 2025 MLB 전체가 아니라 residual 학습 자격을
충족한 98명 189,721구의 동일 표본 비교다.

### 2024 scale 선택 — 234,444구

| 모델 | Accuracy | Top-3 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|
| Global | 45.57% | 90.27% | 43.24% | 1.2087 |
| Residual | 46.46% | 90.61% | 43.07% | 1.1977 |

### 2025 최종 게이트 — 189,721구

| 모델 | Accuracy | Top-3 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|
| Global | 45.08% | 89.99% | 43.46% | 1.2237 |
| Residual | 46.48% | 90.37% | 43.57% | 1.2054 |

#### Global 대비 변화

| 평가 구간 | Accuracy | Top-3 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|
| 2024 | **+0.89%p** | +0.34%p | −0.17%p | **−0.0110** |
| 2025 | **+1.40%p** | +0.38%p | +0.11%p | **−0.0183** |

2025에서 Accuracy는 1.40%p, Log Loss는 0.0182 개선됐다. 2024 Macro F1은
0.17%p 하락했지만 당시 허용 범위 0.5%p 안이었다.

### 제품 단위 동일 표본 비교

V4는 Global을 교체하지 않고 활성 선수에게만 residual을 라우팅한다. 따라서
전체 MLB 성능은 `Global 단독`과 `Final routed`를 같은 투구에 적용해
비교해야 한다. 2025는 선수 활성화에도 사용됐으므로, 독립적인 2026
459,530구 평가를 최종 근거로 사용한다.

| 모델 구조 | Accuracy | Top-3 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|
| Global 단독 | 47.62% | 91.90% | 46.47% | 1.1452 |
| Final routed | **47.73%** | **91.92%** | **46.50%** | **1.1437** |
| Final − Global | **+0.11%p** | +0.02%p | +0.03%p | **−0.0015** |

active/provisional 28,734구에서는 Accuracy가 44.15%에서 45.97%로
1.83%p 개선됐다. residual 적용 비율이 전체의 6.25%였기 때문에 MLB 전체
개선은 약 `6.25% × 1.83%p = 0.11%p`로 희석됐다.

### 307구 쇼케이스

| 모델 | Accuracy | Top-3 | Macro F1 | Log Loss |
|---|---:|---:|---:|---:|
| Final residual | 55.37% | 94.79% | 46.72% | 0.9910 |
| Global | 53.75% | 95.44% | 46.10% | 1.0019 |

현재 추적된 최종 JSON의 수치다. 초기 schema v5 생성 직후 실험 로그에 남은
54.07%는 최종 선수 게이트를 반영하기 전 값이므로 이 리포트에서는 Git에 남은
최종 artifact를 기준으로 삼았다.

## 판단

pool 전체와 2025 게이트에서는 residual의 Accuracy와 Log Loss가 모두
개선됐다. 선수별 독립 모델보다 표본 공유가 잘 작동했고, 2026 외부 평가를
진행할 근거를 확보했다. 이후 같은 2026 전체 표본에서도 Final routed가
Global 단독보다 Accuracy, Macro F1, Log Loss를 모두 개선해 제품 구조의
유효성을 확인했다.

다만 aggregate 통과가 각 투수의 구종 분포가 안전하다는 뜻은 아니다. 선수별
게이트에는 Log Loss, Accuracy 비열화, 주요 구종 zero recall만 있고 분포 오차와
Macro F1 비열화 조건은 없다.

## 비교 시 주의사항

- V3의 48.87%와 이 리포트의 46.48%는 각각 2025 MLB 전체와 98명 pool에서
  측정돼 직접 비교할 수 없다. 이 차이를 V4의 성능 하락으로 해석하지 않는다.
- V3 Global 구조와 V4 Final routed 구조의 직접 비교는 같은 2026 전체
  표본에서 측정한 `47.62% → 47.73%`다.
- 2025는 선수 활성화에도 사용됐으므로 이 버전의 완전한 외부 평가는 아니다.
- 모든 active 선수에게 같은 residual scale 0.5를 사용한다.
- active 상한 25명과 파일럿 우선순위가 품질 순위와 노출 순위를 섞는다.
- 307구 쇼케이스는 외부 성능 근거가 아니다.

## 재현 근거

- 로컬 run artifact:
  `artifacts/runs/20260727T032547104596Z/result.json`
- Git artifact: revision `861ff26`,
  `web/public/data/games/775300.json`
- [실험 로그](../docs/EXPERIMENT_LOG.md)
