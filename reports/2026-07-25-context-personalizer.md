# V2 — PA/Game 문맥 + shrinkage personalizer

> [!NOTE]
> **한 줄 결론:** 문맥 피처로 Global 지표는 소폭 개선됐지만, 상수 logit
> bias 방식의 개인화는 2025 외부 게이트에서 모두 탈락했다.

## 한눈에 보기

- 날짜: 2026-07-25
- 학습 데이터: 2022–2025 지원 구종 2,916,065구
- taxonomy: 포심, 싱커, 커터, 슬라이더, 커브, 체인지업
- 구조: Global XGBoost + 투수별 regularized logit bias
- Global: `global-sqrt-d6-m8`, 1,779 trees, temperature 1.0306

## 무엇이 바뀌었나

타석 내 구종별 사용 횟수, 첫 구종, 구종 다양성, 계열 비율, 연속 횟수와 현재
경기 레퍼토리 변화를 point-in-time 피처로 추가했다. 독립 Specialist는
Global 확률에 작은 투수별 logit bias만 더하는 shrinkage personalizer로
교체했다. Early stopping 상한은 2,000 trees로 늘렸다.

## 평가 설계

- 후보 선택: 2024
- 최종 게이트: 선택에 사용하지 않은 2025
- 2025 평가 표본: 725,576구
- 보조 평가: 같은 307구 역사적 쇼케이스

## 성능 결과

### 2025 MLB-wide

| Accuracy | Top-3 Accuracy | Macro F1 | Log Loss |
|---:|---:|---:|---:|
| 47.78% | 91.64% | 46.20% | 1.1413 |

두 번의 전체 재학습에서 같은 후보와 지표가 재현됐다. 파일럿 5명 중 최종
활성 personalizer는 0명이었다.

Nestor Cortes는 2024 내부 검증을 통과했지만 2025에서 다음과 같이 실패했다.

| 모델 | Accuracy | Macro F1 | Log Loss |
|---|---:|---:|---:|
| Global | 44.81% | 23.76% | 1.1839 |
| Personalizer | 43.32% | 23.68% | 1.1698 |

| 변화 | Accuracy | Macro F1 | Log Loss |
|---|---:|---:|---:|
| Personalizer − Global | **−1.49%p** | **−0.08%p** | **−0.0141** |

> [!WARNING]
> Log Loss는 개선됐지만 Top-1 Accuracy와 클래스 균형은 악화됐다. 확률
> 품질 하나만 좋아졌다는 이유로 개인화를 활성화하지 않았다.

### 307구 쇼케이스

| Accuracy | Top-3 | Macro F1 | Final Log Loss | Global Log Loss |
|---:|---:|---:|---:|---:|
| 50.16% | 93.81% | 46.47% | 1.0667 | 1.0678 |

## 판단

PA/Game 문맥과 늘어난 tree 수로 V1보다 시간 재현성과 지표가 소폭 개선됐다.
그러나 파일럿 personalizer는 MLB-wide 게이트에서 모두 탈락했다. 상수
logit bias만으로는 카운트·타자 손잡이·직전 구종에 따라 달라지는 투수 선택을
표현하기 어렵다는 결론으로 이어졌다.

## 비교 시 주의사항

- V1과 같은 taxonomy지만 주 평가 표본이 V1의 2개년 OOF와 다르다.
- 307구 쇼케이스 수치는 별도 13명 캐시 pregame 실행이라 MLB-wide 배포
  결과와 분리해야 한다.
- Log Loss 개선만으로 개인화 모델을 활성화할 수 없음을 보여준다.

## 재현 근거

- 로컬 run artifact:
  `artifacts/runs/20260725T020300325197Z/result.json`
- [실험 로그](../docs/EXPERIMENT_LOG.md)
