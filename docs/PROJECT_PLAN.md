# OyarZabal V7.2 계획

## 목표

완료된 MLB 경기를 한 구씩 재생하며 다음 구종의 6개 확률과 3개 계열 확률을
함께 보여준다. 개인화는 Global보다 안전하게 나아지는 선수에게만 적용하고,
V7.2를 현재 taxonomy·제품 계약의 기본 세대로 active 배포한다. 2026-07-26
이후 prospective 평가는 별도의 사후 성능 인증으로 유지한다.

## 계층형 구종

```text
패스트볼 계열: 포심, 무빙 패스트볼
브레이킹볼 계열: 슬라이더 계열, 커브 계열
오프스피드 계열: 체인지업, 스플리터·포크
```

6종 모델과 label 순서는 유지한다. 계열 확률은 두 자식 확률의 합이며 별도
모델을 학습하지 않는다. 공식 예측은 다음 계층 decoder로 결정한다.

```text
계열별 두 자식 확률 합산
→ 합산 확률 Top-1 계열 선택
→ 선택한 계열 안에서 확률 Top-1 구종 선택
```

- Exact Accuracy: 계층 decoder가 선택한 세부 구종 적중률
- Family Accuracy: 합산 확률 Top-1 계열 적중률
- Hierarchical Accuracy: 세부 적중 1점, 계열만 적중 0.5점, 다른 계열 0점의
  평균. `(Exact Accuracy + Family Accuracy) / 2`와 같다.

모델 선택의 주 지표는 Log Loss다. Hierarchical Accuracy는 후보가 Global보다
0.5%p 넘게 나빠지는 것을 막는 안전 조건으로 사용한다.

## 모델 구조

```text
모든 투구 ──> Calibrated Global XGBoost ──> 6종 logits ────────────┐
                                                                   ├─> 최종 확률
등록 투수 ──> pooled residual × reliability × context Gate         │
             × Registry multiplier ────────────────────────────────┘
shadow·hard safety 실패 ────────────────────────────────> Global 100%
```

```text
reliability =
  0.5 × n/(n+1000) × min(전체 bootstrap 개선 확률, 최근 90일 개선 확률)

effective scale =
  hard safety
  × min(0.5, 1.5 × reliability)
  × sqrt(context Gate)
  × Registry multiplier
```

- Context Gate는 residual이 Global보다 정답 log probability를 높일 가능성을
  예측한다. 투수·타자 ID는 사용하지 않는다.
- JS divergence 0.05와 클래스별 확률 변화 20%p cap을 유지한다.
- Global·residual 학습 구조와 기존 point-in-time 피처는 바꾸지 않는다.

## 3단계 Registry

2024·2025의 같은 시간순 OOF 행에서 선수 residual을 Global과 비교한다.

- `full`: 두 해의 기존 절대 게이트와 계층 안전 조건 통과. multiplier 1.
- `limited`: 절대 게이트는 실패했지만 두 해 모두 Global 대비 증분 게이트
  통과. 0.05~1.00 중 두 해의 최대 안전 배율 중 작은 값을 사용한다.
- `shadow`: 표본 부족 또는 0.05에서도 실패. multiplier 0으로 Global만 사용.

증분 게이트는 Log Loss 개선, Exact·Macro F1·Hierarchical 하락 0.5%p 이내,
새 주요 구종 zero recall 없음과 분포·calibration 오류의 추가 악화 방지를
요구한다. Global 자체의 기존 오류를 residual의 실패로 중복 처벌하지 않는다.
최신 시즌이 비어 있는 과거 통과 선수는 limited로 두고 기존 365일 half-life
감쇠를 적용한다.

## 평가와 승격

- 학습과 선수 선택은 2022–2025만 사용한다.
- 공개된 2026-03-25~07-25는 회귀 진단에만 사용한다.
- 버전 비교는 MLB 전체와 고정된 `v5-enabled-pitchers-v1` 30명 cohort만
  사용하며 exact row fingerprint를 검사한다.
- tier별 표본은 운영 진단일 뿐 버전 우위 판단 표본이 아니다.
- 2026-07-26 이후 데이터만 지정해 최소 30일·전체 100,000구·V7 개입
  15,000구를 채운 첫 look에서 paired game bootstrap으로 사후 성능을
  인증한다. `deploymentStatus=active`와 별개로 인증 전 상태는
  `performanceCertification=prospective-pending`이다.
- 2024 OOF 선택과 2025 OOF 확인으로 reliability boost 1.5와 Context Gate
  power 0.5를 선택했다. limited 추가 boost는 선수별 안전 조건 때문에
  1.0으로 유지한다.
- Registry는 각 연도 최대 배율의 최솟값이 아니라 2024·2025에서 동시에
  안전한 5% 단위 배율의 교집합을 사용한다.
- 최종 prospective 후보는 `V7.2 reliability=1.5, gatePower=0.5,
  limitedBoost=1.0` 하나로 동결한다. prospective 결과를 보고 설정이나
  조건을 다시 조정하지 않는다.
- 첫 look은 V5 대비 Log Loss 개선의 경기 단위 bootstrap 95% CI 하한이
  0보다 커야 한다. Exact·Family·Hierarchical·Macro F1 하락은 각각
  0.5%p 이내여야 하며, 주요 구종 zero recall이 없어야 한다.
- 분포 안전 조건은 `maxClassShareError ≤ 20%p`, `TVD ≤ 20%p`,
  `maxClassCalibrationError ≤ 10%p`와 V5 대비 악화 허용치 0.5%p로
  사전 고정한다. 선수별 평가는 최소 300구인 선수를 대상으로 같은 조건을
  확인한다.
- 현재 limited roster는 2024·2025 후향 데이터로 선택된 cohort다. 따라서
  roster 자체의 품질 주장은 prospective 첫 look이 통과해야만 가능하다.

## 산출물

- Registry schema v7: 판정 규칙, tier, multiplier, 연도별 안전 배율, reliability,
  증분 지표와 실패 사유
- Holdout schema v6: 동결 manifest와 model hash, 단일 look 진행 상태,
  Exact·Family·Hierarchical, 전체·고정 cohort·tier·선수·월별 지표와
  bootstrap CI
- Replay schema v8: 계열 매핑, 세 지표, tier, multiplier, effective scale과
  cap/fallback 사유
- UI: 3개 부모 계열 아래 6개 확률 막대, 세부 Top-1 배지, 공개 후
  `정확한 구종 / 계열 적중 / 다른 계열` 판정

사용자 직접 예측 대결, 계정, 라이브 API, 영상 분석과 98명 pool 확장은
V7 범위에서 제외한다.
