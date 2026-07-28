# V7 prospective 독립 검증 동결

> 이 동결안은 prospective 표본이 0구인 상태에서
> [V7.2 동결안](2026-07-28-v7.2-residual-tuning.md)으로 대체됐다.
> 아래 내용은 V7.0/V7.1 사전 등록의 역사 기록이다.

## 현재 결론

V7은 아직 승격할 수 없다. 로컬 holdout은 2026-07-25까지이며, 독립 검증은
2026-07-26 이후 투구만 허용한다. 따라서 현재 prospective 표본은 0구이고
`V7.0`, `V7.1` 모두 `awaiting_data`, 제품 상태는 `shadow`다.

현재 limited roster 43명은 2024·2025 결과로 후향 선택됐다. 이번 검증에서는
roster를 바꾸지 않고 고정된 미래 표본에서만 효과를 확인한다.

## 동결 후보

| 후보 | limited boost | 우선순위 | 현재 상태 |
|---|---:|---:|---|
| V7.1 | 1.5 | 1 | awaiting_data |
| V7.0 | 1.0 | 2 | awaiting_data |

boost는 limited 선수에게만 적용한다. full·shadow 라우팅과 Global·residual·
Context Gate 모델은 동일하다. effective scale은 기존 상한 0.5, hard safety,
JS divergence 0.05, 클래스별 확률 변화 20%p cap을 계속 적용한다.

## 첫 단일 look

다음 조건을 모두 채우기 전에는 후보 성능을 열지 않는다.

- 2026-07-26 이후 최소 30일
- 전체 100,000구
- 후보별 residual 개입 15,000구

조건 충족 후 V5와 경기 단위 paired bootstrap 1,000회로 비교한다.

| 조건 | 통과 기준 |
|---|---|
| Log Loss | V5 대비 개선량의 95% CI 하한 > 0 |
| Exact / Family / Hierarchical / Macro F1 | V5 대비 하락 ≤ 0.5%p |
| 주요 구종 | 새 zero recall 없음 |
| class share / TVD / calibration | 절대 한도 통과, V5 대비 악화 ≤ 0.5%p |
| 선수별 안전성 | 표본 300구 이상 선수에서 같은 분포 조건 통과 |

고정 우선순위에 따라 V7.1을 먼저 심사하고, 실패하면 V7.0을 심사한다. 이
look을 연 뒤에는 후보, 임계값, 우선순위를 바꾸지 않는다.

## 실행

```bash
uv run oyarzabal-evaluate-holdout \
  --models models/v7 \
  --reference-models models/hybrid \
  --prospective-manifest config/v7-prospective.json \
  --output artifacts/prospective/v7-first-look.json
```

manifest는 후보와 V5 모델 파일 hash를 검증한다. 첫 look이 소비된 결과 파일은
같은 명령으로 덮어쓸 수 없다.
