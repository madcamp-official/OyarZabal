# 설계 결정

## 2026-07-24 — 핵심 기능 1을 MVP로 고정

완료 경기 리플레이와 모델 검증을 먼저 만든다. 라이브 연동과 사용자 예측
대결은 후속 기능으로 남긴다.

## 2026-07-24 — 6개 구종군

포심, 싱커/투심, 커터, 슬라이더/스위퍼, 커브, 체인지업을 target으로 사용한다.
희귀 유효 구종은 시퀀스 문맥에만 보존한다.

## 2026-07-25 — 6개 구종군 재정의

싱커·투심·커터를 무빙 패스트볼로 합치고 스플리터·포크를 정식 target으로
추가한다. 최종 target은 포심, 무빙 패스트볼, 슬라이더 계열, 커브 계열,
체인지업, 스플리터·포크다. 기존 모델의 클래스 의미가 달라지므로 재학습 없이
재사용하지 않는다.

## 2026-07-24 — Global + 검증형 Specialist

13명 데이터로 만든 모델을 Global이라고 부르지 않는다. 2022–2025 MLB 전체
데이터와 선수 ID 비의존 피처로 Global 모델을 만들고, 유명 투수는 데이터
요건과 시간순 검증을 통과할 때만 Specialist를 혼합한다. 하드 스위치 대신
투수별 검증 weight를 사용한다.

## 2026-07-24 — 정적 리플레이

완료 경기 예측을 JSON으로 미리 계산하고 Vite 앱에서 재생한다. 핵심 기능 1에는
API, 데이터베이스, 계정 시스템을 추가하지 않는다.

## 2026-07-24 — 역사적 쇼케이스 표기

첫 2024 경기는 제품 흐름을 증명하는 쇼케이스다. 경기 시작 전 데이터만
사용하되 동결된 2026 홀드아웃 성능으로 표현하지 않는다.

## 2026-07-25 — 독립 Specialist를 shrinkage personalizer로 교체

투수별 독립 XGBoost는 검증 Accuracy가 올라도 주력 구종으로 붕괴해 Macro F1이
하락했다. Global 확률에 투수별 regularized logit bias만 더하고, 2024 내부
시간 분할에서 통과한 투수만 활성화한다. 모델과 shrinkage 선택에 쓰지 않은
2025 시즌을 최종 평가로 사용한다.

## 2026-07-25 — 타석과 현재 경기 문맥 추가

현재 투구를 제외한 타석 내 구종 사용, 첫 구종, 구종 다양성, 계열 비율과
현재 경기 레퍼토리 변화를 Global 피처에 포함한다. 원시 비율 대신 smoothing된
확률을 사용하고 타석·경기 경계에서 상태를 초기화한다.

## 2026-07-27 — fixed bias를 pooled contextual residual로 교체

선수별 상수 bias는 투수의 전체 레퍼토리 비율만 보정하고 카운트·타자 손잡이·
직전 구종에 따라 달라지는 선택을 학습하지 못했다. 자격을 충족한 투수의 표본을
하나로 모아 Global logits를 `base_margin`으로 사용하는 얕은 XGBoost residual을
학습한다. 2024와 2025를 모두 통과한 선수만 active로 노출하며, 2025 기록이 없는
선수는 감쇠된 provisional 보정만 허용한다.

## 2026-07-27 — 2026 시즌을 동결 홀드아웃으로 격리

모델, temperature, residual scale, 선수 registry는 2025년 말까지의 데이터로
확정한다. 2026 투구는 이 값들의 학습·선택·활성화에 사용하지 않고 최종 성능
평가에만 사용한다. 단, 실제 서비스 상황과 같이 이미 공개된 2026 이전 투구는
뒤따르는 투구의 point-in-time 문맥 피처를 갱신할 수 있다.

## 2026-07-27 — 고정 scale을 reliability-gated residual로 교체

공통 residual scale 0.5와 active 25명 상한을 제거한다. 투수별 OOF 표본,
전체 기간과 최근 90일 경기 bootstrap 개선 확률, 투수 ID가 없는 binary
상황 Gate를 곱해 투구별 scale을 계산한다. 별도 drift·magnitude 모델은
추가하지 않고 JS 0.05와 클래스별 20%p 변화 cap만 유지한다.

선수 활성화에는 Log Loss·Accuracy·Macro F1뿐 아니라 signed class share,
TVD, 주요 구종 recall과 class probability calibration을 사용한다. 2024·2025
게이트를 통과한 수만 active로 두며 V6는 2026-07-26 이후 prospective 조건을
통과할 때까지 shadow로 유지한다.

## 2026-07-27 — V5 개인화 평가 cohort를 버전 공통 기준으로 동결

모델마다 active 선수가 달라 V5의 28,734구와 V6의 10,344구를 직접 비교할 수
없었다. V5에서 활성화된 30명을 `v5-enabled-pitchers-v1`로 동결하고, 이후
모든 모델의 개인화 성능을 이 선수들의 동일 기간·동일 투구에서 비교한다.
모델별 active·registry pool은 운영 진단으로만 남긴다. 같은 benchmark를
재평가할 때는 exact row fingerprint까지 일치해야 한다.

## 2026-07-27 — 계층형 평가와 증분 Registry 도입

6종 출력은 유지하고 포심·무빙 패스트볼, 슬라이더·커브, 체인지업·
스플리터/포크를 각각 패스트볼·브레이킹볼·오프스피드 계열로 합산해 표시한다.
공식 예측은 6종 Top-1을 유지하며 Exact 1점, 같은 계열 오답 0.5점, 다른 계열
0점의 Hierarchical Accuracy를 보조 안전 지표로 추가한다.

선수 개인화는 절대 기준만으로 켜고 끄지 않는다. 두 해의 엄격 조건을 통과하면
full, 엄격 조건은 실패해도 같은 행의 Global보다 안전하게 개선되면 limited,
최소 배율에서도 증분 조건을 실패하면 shadow로 둔다. 이 변경은 98명 pool의
활용도를 높이되 Global이 이미 가진 오류를 residual에 중복 처벌하지 않기
위한 것이다. V7은 공개된 2026 구간으로 승격하지 않고 prospective 평가 전까지
shadow를 유지한다.

## 2026-07-27 — 공식 예측을 계층 decoder로 변경

6종의 평면 argmax가 속한 계열을 Family 예측으로 사용하던 결정을 폐기한다.
각 계열의 두 확률을 합산해 부모 계열을 먼저 고르고, 선택한 부모 안에서 확률이
높은 자식을 공식 세부 구종으로 고른다. 이 규칙은 계열과 세부 예측의 모순을
없애고 Family Accuracy를 실제 3계열 판정으로 만든다.

확률과 Log Loss는 바뀌지 않지만 Exact·Macro F1·구종 분포와 선수별 안전
게이트는 달라진다. 따라서 기존 Registry를 재사용하지 않고 2024·2025 OOF로
full/limited/shadow를 다시 판정하며, 공개 2026과 쇼케이스 리포트도 전부
새 decoder로 재계산한다.

## 2026-07-28 — V7 독립 검증 후보와 단일 look을 동결

V7 구조와 Registry는 바꾸지 않는다. 현재 limited 배율 1.0을 `V7.0`, limited
배율만 1.5로 높인 모델을 `V7.1`로 고정하고, 같은 2026-07-26 이후 투구에서
동시에 shadow 평가한다. full과 shadow 선수의 배율은 두 후보에서 같다.

최초 평가는 최소 30일·100,000구·후보별 residual 개입 15,000구가 모두
충족될 때 한 번만 연다. 이미 공개된 2026-03-25~07-25는 후보 선택과 승격에
사용하지 않는다. V7.1을 먼저 심사하고 실패하면 V7.0을 심사하며, prospective
결과를 본 뒤 후보·임계값·순서를 바꾸지 않는다.

승격에는 V5 대비 경기 단위 paired bootstrap Log Loss 개선 CI 하한 양수,
Exact·Family·Hierarchical·Macro F1 하락 0.5%p 이내, 주요 구종 zero recall
없음을 요구한다. TVD·class share·calibration과 선수별 쏠림도 사전에 동결한
절대 한도 및 V5 대비 0.5%p 악화 허용치로 검사한다. 현재 limited roster는
2024·2025 후향 데이터로 선택됐음을 모든 독립 검증 리포트에 명시한다.

## 2026-07-28 — V7.2는 reliability와 Gate만 확대

prospective 표본이 아직 0구인 동안 2022–2025 시간순 OOF로 Residual 적용
강도를 다시 선택했다. 사전 고정한 9개 동적 후보 중 reliability 배율 1.5와
Context Gate power 0.5를 채택했다. 이는 Gate 확률에 제곱근을 적용해 중간
확률에서도 Residual을 더 신뢰한다.

limited 추가 배율 1.5와 2.0은 전체 Log Loss를 더 낮췄지만 2024·2025에서
각각 다수 선수의 Accuracy·Macro F1·zero recall·분포 조건을 악화시켜
기각했다. 따라서 limited boost는 1.0으로 유지한다.

기존의 `min(maxSafe2024, maxSafe2025)`는 분포 지표가 scale에 대해
단조적이라고 잘못 가정했다. V7.2부터는 두 연도에서 동시에 안전한 5% 단위
배율을 직접 찾는다. 최종 Registry는 `full 3 / limited 42 / shadow 53`이며
OOF 선수별 실패는 두 연도 모두 0명이다.

공개된 2026 구간은 후보 동결 뒤 회귀 진단에만 사용했다. V7.2는 V7.0보다
개선됐지만 V5 고정 cohort의 Log Loss를 넘지 못했으므로 shadow를 유지한다.
기존 V7.0/V7.1 prospective 후보는 표본이 0구인 상태에서 V7.2 단일 후보로
대체한다.

## 2026-07-28 — V7.2를 현재 taxonomy 기본 세대로 승격

V5 대비 통계적 성능 승격이 아니라 taxonomy·계층 decoder·Registry·UI 계약의
세대교체로 V7.2를 `active`로 지정한다. V5는 삭제하지 않고 legacy 비교
기준으로 보존한다.

선수별 Registry의 `shadow`는 계속 Global fallback을 뜻하며 모델 전체
`deploymentStatus=active`와 구분한다. prospective 단일 look도 폐기하지
않고 `performanceCertification=prospective-pending` 상태의 사후 성능
인증으로 유지한다. 향후 V8의 주 기준선은 같은 taxonomy의 Global-only와
V7.2로 정한다.

## 2026-07-28 — V8.3은 2024 클래스 균형 게이트에서 기각

Global-conditioned Sequence Residual은 Log Loss와 계층 Accuracy를
개선했지만 Macro F1과 TVD를 크게 악화시켰다. 사전 정의한 2024 정상·물리결측
게이트를 통과한 후보가 없으므로 2025 정답을 열어 추가 튜닝하지 않는다.

V7.2를 active로 유지한다. V8.3 코드와 실패 artifact는 결측 마스크,
확장 데이터 계약, point-in-time physical/catcher 연구 기반으로 보존하되
제품 추론에는 연결하지 않는다.
