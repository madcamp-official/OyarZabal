# 실험 로그

성공뿐 아니라 실패와 중요한 수정도 append-only로 기록한다.

## 기록 형식

```text
실행:
시각:
Git revision:
데이터 manifest:
가설:
명령:
설정:
결과:
실패/한계:
수정:
검증:
다음:
```

## 2026-07-24 — 최초 MVP

- 13명 투수의 캐시로 2024 월드시리즈 1차전 307구 리플레이를 만들었다.
- 최초 7종 분류와 기본 XGBoost는 포심으로 예측이 쏠렸다.
- 한계: 13명 데이터는 MLB 전체 Global 성능을 나타내지 않는다.

## 2026-07-24 — 누수 방지 6종 피처 전환

- 현재 투구를 제외한 커리어·시즌·최근 100구·카운트·타자 손잡이·전환확률과
  직전 1·2·3구 시퀀스를 구현했다.
- `FS/FO/SC/KN/EP`는 문맥에 남기고 target에서는 제외했다.
- 기존 16개 후보 OOF 학습은 완료 직전 Global + Specialist 방향으로 전환되어
  의도적으로 중단했다. 중단 시 정적 산출물은 덮어쓰지 않았다.

## 2026-07-24 — Global + Specialist 구현

- 2022–2025 월별 재개형 수집기, 선수 ID 비의존 Global 피처, 2024·2025
  validation fold, 5명 Specialist 자격 검사와 blend registry를 구현했다.
- 각 투구의 모델 출처를 표시하는 schema v3 UI를 추가했다.
- 실제 전체 데이터 수집과 학습 결과는 해당 run artifact가 생성된 뒤 이어서
  이 로그에 기록한다.

## 2026-07-24 — MLB 전체 학습과 schema v3 쇼케이스

- 수집: 2022–2025 월별 48개 shard, 원시 약 319만 행, 지원 6종 2,916,065구.
- Global 2024·2025 OOF: Accuracy 0.4739, Macro F1 0.4594,
  Log Loss 1.1436, temperature 0.9865.
- 전체 OOF에서는 5명 중 2명이 데이터 자격 미달, 3명은 4개 specialist 후보가
  Global의 Log Loss와 Macro F1을 동시에 개선하지 못해 비활성화됐다.
- 경기 전 MLB-wide 쇼케이스 검증에서는 Gerrit Cole의
  `global-sqrt-d4-m8` specialist가 weight 0.25로 통과했다. 307구 중 88구는
  Hybrid, 219구는 Global로 라우팅됐다.
- 쇼케이스 최종: Accuracy 0.5147, Macro F1 0.4711, Log Loss 1.0841.
- 중요 수정: 모델용 문자열 ID와 registry 정수 ID 비교 때문에 모든 specialist
  데이터가 0구로 보이던 문제를 별도 `pitcher_id`로 수정했다.
- 중요 수정: 경기 파일에만 존재하는 `bat_score_diff` 때문에 과거 행의
  `score_diff`가 전부 결측이 되던 문제를 행별 점수 차 fallback으로 수정했다.

## 2026-07-25 — PA/Game 문맥과 personalizer 구현

- 타석 내 구종별 사용 횟수, 구종 다양성, 첫 구종, 타석 내 연속 횟수와
  3개 구종 계열 비율을 point-in-time 피처로 추가했다.
- 현재 경기 구종 사용률을 시즌 사용률로 smoothing하고 시즌 대비 변화량을
  추가했다.
- Global 후보를 `none/light/sqrt × depth 4/6`으로 제한하고 early stopping
  상한을 2,000 trees, patience를 75 rounds로 조정했다.
- weighted 학습 후보도 early stopping은 실제 평가 분포의 unweighted
  `mlogloss`를 사용하도록 수정했다.
- 독립 Specialist XGBoost를 투수별 shrunk logit-bias personalizer로
  교체했다. 2024에서 선택하고 2025를 선택에 사용하지 않는 평가로 남겼다.
- 작은 4개년 합성 데이터로 후보 선택, 2025 평가, 최종 모델과 schema v2
  registry 저장 경로를 smoke 검증했다.
- 전체 MLB 재학습 결과와 새 쇼케이스 지표는 실제 장시간 실행 완료 후 별도
  항목으로 기록한다.

## 2026-07-25 — MLB 전체 재학습 및 2025 최종 게이트

- 2,916,065구로 두 차례 재학습했으며 선택 결과와 지표가 동일하게 재현됐다.
- 2024 후보 선택에서는 `global-sqrt-d6-m8`을 Log Loss와 Macro F1 모두에서
  이긴 후보가 없어 기존 구조를 유지했다. 최종 tree 수는 650에서 1,779로
  늘어 early-stopping 상한 문제를 해소했다.
- 선택에 사용하지 않은 2025 725,576구에서 Accuracy 0.4778, Macro F1
  0.4620, Log Loss 1.1413, Top-3 Accuracy 0.9164를 기록했다.
- Nestor Cortes personalizer는 2024 내부 검증을 통과했지만 2025에서 Global
  대비 Accuracy가 0.4481에서 0.4332로, Macro F1이 0.2376에서 0.2368로
  하락했다. Log Loss는 1.1839에서 1.1698로 개선됐지만 전체 배포 게이트를
  만족하지 못해 `test_validation_failed`로 비활성화했다.
- 파일럿 5명 중 최종 활성 personalizer는 0명이다. 실패 결과는 삭제하지 않고
  registry validation과 post-evaluation run artifact에 보존했다.

## 2026-07-25 — 307구 쇼케이스 재생성

- Final/Global: Accuracy 0.5016, Top-3 Accuracy 0.9381, Macro F1 0.4647.
- Final Log Loss 1.0667, Global Log Loss 1.0678로 personalizer가 확률 품질만
  소폭 개선했고 argmax 예측은 바꾸지 않았다.
- 13명 캐시 기반 쇼케이스에서는 Tommy Kahnle personalizer만 선택됐지만,
  이는 MLB-wide 2025 holdout 배포 판단과 분리된 데모 결과다.

## 2026-07-25 — 구종군 재정의 및 전체 재학습

- target을 포심(`FF`), 무빙 패스트볼(`SI/FT/FC`), 슬라이더 계열
  (`SL/ST/SV`), 커브 계열(`CU/KC/CS`), 체인지업(`CH`), 스플리터·포크
  (`FS/FO`)로 재정의했다. `KN/SC`와 결측·미분류는 target에서 제외한다.
- 기존 클래스 의미와 순서가 달라져 MLB-wide 모델을 재사용하지 않고
  2,990,491구로 다시 학습했다. 선택 모델은 `global-sqrt-d6-m8`, 최종 tree
  수는 1,898, temperature는 1.0465다.
- 2025 750,581구에서 Accuracy 0.4887, Top-3 Accuracy 0.9308, Macro F1
  0.4724, Log Loss 1.1103을 기록했고 6개 클래스 모두 recall이 0보다 컸다.
- 파일럿 5명 중 활성 personalizer는 0명이다. Cole은 2025 검증 행 부재,
  Flaherty와 Cortes는 내부 검증 실패, Treinen과 Kahnle은 데이터 자격
  미달이었다.
- schema v4 쇼케이스는 Accuracy 0.5375, Top-3 Accuracy 0.9544, Macro F1
  0.4610, Log Loss 1.0012를 기록했다. 해당 경기에는 스플리터·포크가 0구여서
  클래스별 성능 판단에 사용할 수 없고, 이전 taxonomy 지표와 직접 비교하지
  않는다.

## 2026-07-27 — pooled contextual residual 전체 학습

- 2,990,491구에서 자격을 충족한 98명을 pooled residual 학습 풀로 사용했다.
  Global은 `global-sqrt-d6-m8`, temperature `1.0465`로 고정했다.
- 2023 OOF로 residual을 학습하고 2024에서 scale `0.5`를 선택했다. 2024
  후보군 Log Loss는 1.2087에서 1.1977로, Accuracy는 0.4557에서 0.4646으로
  개선됐다. Macro F1은 0.4324에서 0.4307로 0.0017 하락해 허용 범위
  0.005 안에 머물렀다.
- 선택에 사용하지 않은 2025 후보군 189,721구에서 Log Loss는 1.2237에서
  1.2054로, Accuracy는 0.4508에서 0.4648로, Macro F1은 0.4346에서
  0.4357로 개선돼 최종 전체 게이트를 통과했다.
- 선수별 2024·2025 게이트 통과자는 53명이었으며 제품 상한을 적용해 active
  25명을 노출한다. 2025 기록이 정확히 0구이고 2024 게이트를 통과한
  provisional 6명 중 Gerrit Cole을 포함한 5명을 노출한다. 나머지는
  `exposure_cap`으로 registry에 보존한다.
- 첫 실행은 2024 표본이 0인 후보의 지표 계산을 시도해 실패했다. 빈 시즌을
  `insufficient_2024_support`로 처리하도록 수정했다. 다음 실행 전에는 선수
  ID를 Python `int`로 정규화해 registry JSON 직렬화 실패 가능성도 제거했다.
  두 중단 모두 각 run의 `error.json`에 보존했다.
- schema v5의 307구 역사적 쇼케이스는 Accuracy 0.5407, Top-3 Accuracy
  0.9544, Macro F1 0.4639, Log Loss 0.9965를 기록했다. 이 값은 13명 캐시의
  pregame 검증 결과이며 MLB-wide 2025 게이트와 분리해 해석한다.

## 2026-07-27 — 첫 동결 2026 홀드아웃 평가

- 2022–2025 지원 구종 2,990,491구만 사용해 모델과 registry를 재생성했다.
  학습 cutoff는 2025-11-01이며 2026 데이터는 학습·선택·활성화에서 제외했다.
- 2026-03-25부터 데이터 소스에서 완결된 2026-07-25까지 지원 구종
  459,530구를 평가했다. 전체 라우팅 모델은 Accuracy 0.4773, Top-3 Accuracy
  0.9192, Macro F1 0.4650, Log Loss 1.1437을 기록했다.
- 같은 행의 Global 단독은 Accuracy 0.4762, Macro F1 0.4647, Log Loss
  1.1452였다. residual 라우팅은 전체 Accuracy를 0.11%p, registry pool
  90,145구에서는 0.58%p 개선했다.
- 실제 활성·provisional 선수 28,734구에서는 Accuracy가 0.4415에서
  0.4597로 1.83%p, Log Loss가 1.2658에서 1.2418로 개선됐다.
- 이 시점부터 해당 2026-03-25~07-25 구간은 공개된 benchmark다. 이후 모델
  변경이나 게이트 선택에 이 결과를 사용하면 같은 구간을 독립 홀드아웃으로
  다시 주장하지 않는다.

## 2026-07-27 — V6 reliability-gated residual

- 2,990,491구로 Global calibration, pooled residual, binary context Gate를
  재학습했다. 클래스별 logit calibration effective weight는 0.5가 채택됐다.
- 2025 MLB 전체에서 calibrated Global은 Accuracy 0.4992, Macro F1 0.4711,
  Log Loss 1.0963을 기록했다.
- 2025 residual pool 189,721구에서 V6는 calibrated Global 대비 Accuracy를
  0.4639에서 0.4656으로, Log Loss를 1.2077에서 1.2049로 개선했다.
  Macro F1은 0.4350에서 0.4348로 0.00025 하락해 허용 범위 안이었다.
- 엄격한 선수별 분포·calibration 게이트를 통과한 active는 10명이며 25명
  목표나 상한을 적용하지 않았다.
- 첫 전체 실행 후 최근 90일 기준을 리그 마지막 날짜로 계산한 오류를 발견했다.
  각 투수의 마지막 경기 기준으로 수정하고 회귀 테스트를 추가한 뒤 전체
  재학습했다. 최종 active 수는 10명으로 같았지만 공백 선수의 reliability가
  정의대로 복원됐다.
- 이미 공개된 2026-03-25~07-25는 회귀 진단에만 사용했다. V6 Final은 동일
  표본에서 Accuracy 0.4878, Log Loss 1.1290이었으나 V6 개입 표본이
  10,344구로 prospective 최소 15,000구에 미달해 승격하지 않았다.
- schema v6 쇼케이스는 307구 중 90구에 V6를 적용해 Accuracy 0.5407,
  Macro F1 0.4615, Log Loss 0.9996을 기록했다.

## 2026-07-27 — V5·V6 공통 evaluation cohort 재평가

- 기존 V5 활성·provisional 30명을 `v5-enabled-pitchers-v1`로 동결했다.
  2026-03-25~07-25의 정확한 표본은 28,734구이며 fingerprint는
  `a2d7de0347b98e9cac05fe1ee22eedc1eab33aa5430e770089755f8945198232`다.
- 같은 28,734구에서 V5 Final은 Accuracy 0.4596, Macro F1 0.4201,
  Log Loss 1.2422를 기록했다. V6 Final은 각각 0.4555, 0.4168, 1.2475로
  V5보다 Accuracy 0.41%p, Macro F1 0.34%p 낮고 Log Loss가 0.0053
  나빴다.
- V6 residual 자체는 V6 Global 대비 Accuracy를 0.05%p, Log Loss를
  0.00026 개선했지만 V5 Final을 넘지 못했다. 따라서 V6는 shadow를
  유지하며, prospective 승격에는 MLB 전체와 고정 개인화 cohort를 모두
  통과하도록 게이트를 강화했다.

## 2026-07-27 — V6 선수별 안전 scale 분석

- 동일 V6 시간순 OOF에서 기존 투구별 effective scale에 0.05–1.00의
  선수별 배율을 적용하고, 2024·2025 각각 Global 대비 새 악화를 만들지
  않는 최대값을 계산했다. 운영 Registry는 변경하지 않았다.
- 98명은 `full 10명 / limited 39명 / shadow 49명`으로 재분류됐다. 기존
  2024 선수 게이트 실패 61명 중 31명, 2025 실패 10명 중 8명이 limited
  후보가 됐다.
- 2024 표본 부족 17명과 최소 배율에서도 안전 조건을 실패한 32명은
  shadow를 유지한다. 따라서 Context Gate만 남겨 전원에게 residual을
  적용하는 방식은 채택하지 않았다.
- 이 배율은 2024·2025 후향 결과로 선택했으므로 배포 근거가 아니다.
  prospective 구간에서 고정 배율을 검증한 뒤에만 Registry 라우팅 승격을
  검토한다.

## 2026-07-27 — V7 계층형 평가와 증분 Registry

- 기존 6종 확률을 패스트볼·브레이킹볼·오프스피드 3계열로 합산하고 Exact,
  Family, Hierarchical Accuracy를 학습·holdout·쇼케이스 전 구간에 추가했다.
- 2022–2025 2,990,491구를 재학습한 실제 Registry는
  `full 10 / limited 40 / shadow 48`이다. 기존 후향 분석보다 limited가 한 명
  늘어난 것은 최신 시즌 표본이 없는 과거 안전 선수를 stale limited로
  보존한 결과다.
- 2025 98명 pool에서 V7은 Global 대비 Exact 46.39%→46.56%,
  Family 54.74%→55.00%, Hierarchical 50.56%→50.78%, Log Loss
  1.2077→1.2049로 개선했다.
- 공개된 2026 MLB 전체 459,530구에서 V7 Final은 Exact 48.80%,
  Family 57.37%, Hierarchical 53.08%, Log Loss 1.1287이었다.
- 같은 고정 30명·28,734구에서 V7 Final은 V5 Final보다 Exact 0.16%p,
  Hierarchical 0.14%p 낮고 Log Loss가 0.0019 나빴다. V7은 shadow를
  유지한다.
- full 10명과 limited 40명은 공개 2026 tier 진단에서 모두 자체 Global보다
  Log Loss와 Hierarchical Accuracy를 개선했다. shadow 48명은 최종 확률이
  Global과 완전히 같았다.
- 307구 쇼케이스는 Exact 53.09%, Family 61.89%, Hierarchical 57.49%,
  Log Loss 0.9896을 기록했다. residual은 full 90구, limited 10구에 적용됐다.
- 첫 holdout 실행에서 평가 행이 0개인 stale 선수가 발생해 sklearn 진단이
  중단됐다. 공용 지표 함수가 빈 평가 slice를 0 support로 반환하도록 수정하고
  회귀 테스트를 추가했다.

## 2026-07-27 — V7 계층 decoder 전면 재평가

- 공식 선택을 평면 6종 argmax에서 `계열별 확률 합산 Top-1 → 선택 계열 내부
  Top-1`로 변경했다. 공용 metrics, Gate disagreement, Replay `topPitch`,
  UI 판정이 같은 decoder를 사용한다.
- 새 Exact·Macro F1·Hierarchical·분포 조건으로 2024·2025 Registry를 다시
  계산했다. 상태는 `full 10 / limited 40 / shadow 48`에서
  `full 4 / limited 43 / shadow 51`로 바뀌었다.
- 2025 98명 pool에서 V7은 Global 대비 Exact 45.40%→45.69%,
  Family 56.31%→56.67%, Hierarchical 50.86%→51.18%, Log Loss
  1.2150→1.2104로 개선했다.
- 공개 2026 MLB 전체에서 V7 Final은 Exact 47.94%, Family 58.71%,
  Hierarchical 53.33%, Log Loss 1.1359였다. V6를 같은 decoder로 다시
  평가한 48.28%, 59.27%, 53.78%, 1.1290보다는 낮았다.
- 고정 30명·28,734구에서 V5는 Exact 45.14%, Family 58.61%,
  Hierarchical 51.88%, Log Loss 1.2422였고 V7은 44.53%, 57.83%,
  51.18%, 1.2515였다. V7은 shadow를 유지한다.
- 307구 쇼케이스 V7은 Exact 56.35%, Family 67.10%, Hierarchical
  61.73%, Log Loss 0.9895를 기록했다. 이전 평면 판정 대비 세 정확도는 각각
  3.26%p, 5.21%p, 4.24%p 높다.
- 산출물은 Registry schema v7, Holdout schema v5, Replay schema v8로
  올리고 `decisionRule=family-sum-then-child`를 기록했다.

## 2026-07-28 — V7 prospective 독립 검증 동결

- V7 구조를 유지한 채 `V7.0 limitedBoost=1.0`과
  `V7.1 limitedBoost=1.5`를 동결했다. boost는 limited 선수에만 적용되고
  기존 reliability 상한 0.5, Context Gate, hard safety, JS·확률 변화 cap은
  그대로 유지된다.
- 후보와 V5 reference의 model hash, 후보 순서, 표본 조건, metric gate를
  `config/v7-prospective.json`에 기록했다.
- 첫 look 전에는 성능 지표를 계산하지 않는다. 최소 30일·100,000구·후보별
  개입 15,000구를 모두 채운 뒤 단 한 번 V5와 비교하며, 결과를 본 뒤
  재튜닝하지 않는다.
- 현재 로컬 2026 holdout은 2026-07-25까지라 prospective 표본은 0구다.
  두 후보 모두 `awaiting_data`, V7 전체는 `shadow`를 유지한다.
- limited roster는 2024·2025에서 후향 선택됐으며, 독립 검증은 이 cohort를
  고정한 채 앞으로 들어오는 투구만 평가한다.

## 2026-07-28 — V7.2 Residual 강도 확대

- `config/v7-residual-tuning.json`에 reliability boost 3개, Gate power
  3개, limited boost 3개와 안전 조건을 결과 확인 전에 고정했다.
- 2024 선택·2025 확인에서 `reliabilityScaleBoost=1.5`,
  `contextGatePower=0.5`가 두 해 모두 V7.0보다 Log Loss와 세 정확도를
  개선해 선택됐다.
- limited boost 1.5와 2.0은 전체 Log Loss를 추가 개선했지만 각각
  15–16명, 19–21명의 선수별 안전 조건을 실패해 기각했다.
- 기존 Registry가 연도별 최대 안전 배율의 최솟값을 사용해 더 작은 배율에서
  오히려 분포 조건을 실패할 수 있음을 발견했다. 두 해의 안전 배율 교집합을
  직접 선택하도록 수정한 뒤 limited boost 1.0의 OOF 실패 선수는
  2024·2025 모두 0명이 됐다.
- 최종 Registry는 `full 3 / limited 42 / shadow 53`이다.
- 후보 선택 이후 공개 2026-03-25~07-25를 한 번 진단했다. V7.2는 V7.0
  대비 MLB 전체 Log Loss 1.13593→1.13588, 고정 30명 Log Loss
  1.25154→1.25080으로 개선했다.
- 공개 2026의 non-zero effective scale 평균은 0.125→0.157, p90은
  0.249→0.387로 증가했다. 개입 투구는 44,829→42,518로 줄었다.
- V5 고정 30명 Log Loss 1.24217은 넘지 못했으므로 V7.2는 shadow를
  유지하고 2026-07-26 이후 첫 prospective look만 승격 근거로 사용한다.

## 2026-07-28 — Prospective 수집 시작

- 부분 월 Parquet가 존재하면 날짜 범위를 보지 않고 skip하던 수집기 결함을
  수정했다. 기존 월의 마지막 날짜를 한 번 겹쳐 다시 받아 식별 키로 병합해
  Statcast 지연 갱신과 신규 날짜를 함께 반영한다.
- 2026-07-26~28을 요청해 4,548개 원시 행을 7월 shard에 병합했다.
- 현재 가용 범위는 7월 26일까지이며 prospective 유효 표본은 4,514구,
  Residual 개입은 600구다.
- 최소 임계값 전이라 성능은 공개하지 않았고 `firstLookConsumed=false`다.

## 2026-07-28 — V7.2 taxonomy 세대교체

- 사용자 결정에 따라 V7.2를 현재 taxonomy·제품 구조의 기본 모델로
  `active` 지정했다.
- 승격 근거는 V5 대비 통계적 우위가 아니라 6종→3계열 계층 decoder,
  `full / limited / shadow` Registry, 현재 UI·산출물 계약의 일관성이다.
- V5 모델과 과거 지표는 legacy benchmark로 보존한다.
- prospective 평가는 계속 blinded 상태로 수집하며 배포 게이트가 아닌
  사후 성능 인증으로 해석한다.

## 2026-07-28 — V7.3 tier scale stress test

- V7.2 모델과 Registry를 유지하고 `fullTierBoost=3`,
  `limitedTierBoost=4`만 적용했다. reliability boost 1.5, Gate power 0.5,
  scale 상한 0.5, JS 0.05와 클래스 확률 변화 20%p cap은 유지했다.
- 공개 2026 고정 459,530구에서 V7.3은 Global 대비 Exact +0.071%p,
  Family +0.099%p, Hierarchical +0.085%p, Log Loss −0.00119였다.
  V7.2 대비로도 세 정확도와 Log Loss가 모두 개선됐다.
- 고정 30명에서는 Global 대비 Exact +0.68%p, Family +0.89%p,
  Hierarchical +0.78%p, Log Loss −0.00999였지만 Macro F1은 −0.35%p,
  TVD는 +3.29%p였다. V5 Log Loss보다도 0.00381 나빴다.
- non-zero scale 평균은 0.157→0.332, 중앙값은 0.099→0.397, p90은
  0.387→0.5로 증가했다. 개입 투구는 같은 42,518구였다.
- full tier는 Log Loss와 Macro F1·TVD가 함께 개선됐지만 limited tier는
  Log Loss 개선과 함께 TVD가 1.44%p 악화됐다. V7.3은 공개 표본 진단
  실험으로만 기록하고 V7.2의 active 상태는 바꾸지 않는다.

## 2026-07-28 — V7.4 equal tier scale stress test

- V7.3에서 full 배율만 3→4로 높였다. limited는 4, shadow는 0이며
  reliability boost 1.5, Gate power 0.5, scale·JS·확률 변화 cap은 유지했다.
- 공개 2026 전체에서 V7.4는 V7.3 대비 Exact +0.001%p, Family +0.001%p,
  Hierarchical +0.001%p, Log Loss −0.000004로 아주 조금 개선됐다.
- full 3명·3,841구에서는 Exact +0.10%p, Family +0.16%p, Macro F1
  +0.14%p, Log Loss −0.00048, TVD −0.31%p로 모든 주요 방향이 개선됐다.
- 전체 non-zero scale 평균은 0.332→0.335였고 개입 투구는 같은
  42,518구였다. 고정 30명 cohort에는 full tier 투구가 없어 V7.3과 결과가
  동일했다.
- V7.4는 공개 표본 stress test로만 보존하고 V7.2의 active 상태는 유지한다.
