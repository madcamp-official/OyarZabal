# OyarZabal MVP 계획

## 목표

완료된 MLB 경기를 한 구씩 재생하며 결과 공개 전에 다음 구종 확률을 확인하고,
경기 종료 후 모델의 정확도·Top-3 정확도·Macro F1·Log Loss를 검증한다.
첫 공개 경기는 제품 흐름을 보여주는 역사적 쇼케이스이며 동결된 2026 홀드아웃이
아니다.

## 모델 구조

모든 투수에게 적용되는 MLB 전체 Global XGBoost와 검증을 통과한 투수의
pooled contextual residual을 결합한다.

```text
모든 투구 ──> Global XGBoost ──> 6종 logits ──────────────┐
                                                          ├─> 최종 6종 확률
등록 투수 문맥 ──> pooled residual correction × scale ───┘
미등록/탈락 투수 ──────────────────────────────> Global 100%
```

- Global 모델은 `pitcher`, `batter` ID를 입력으로 사용하지 않는다.
- Residual 후보는 5,000구 이상, 최근 시즌 500구 이상, 사용률 5% 이상인
  구종군 3개 이상이어야 한다.
- Global은 앞선 실험에서 채택한 `global-sqrt-d6-m8`과 temperature `1.0465`를
  고정한다.
- 2023 OOF로 residual을 학습하고 2024에서 scale을 선택한다. 2023+2024로
  재학습한 뒤 2025를 선택에 사용하지 않은 최종 게이트로 남긴다.
- residual 입력은 투수 ID, 5개 count bucket, 타자 손잡이, 같은 타석의 직전
  구종군이다. XGBoost `base_margin`에 Global logits를 넣어 보정량만 학습한다.
- Global보다 Log Loss가 개선되고 Accuracy·Macro F1 하락이 각각 0.5%p
  이내이며 주요 구종 recall이 0보다 클 때만 활성화한다.
- 2023·2024·2025 각각 300구 이상인 선수는 active 후보, 2025가 정확히
  0구인 선수는 provisional 후보다. 노출은 active 25명과 provisional 5명으로
  제한한다.
- provisional scale은 최대 0.25이며 마지막 경기 이후 365일 half-life로
  감쇠한다.

학습 target은 포심(`FF`), 무빙 패스트볼(`SI/FT/FC`), 슬라이더 계열
(`SL/ST/SV`), 커브 계열(`CU/KC/CS`), 체인지업(`CH`), 스플리터·포크
(`FS/FO`)의 6종이다. `KN/SC`와 결측·미분류는 target에서 제외한다.

## 구현 순서

1. 2022–2025 Statcast를 월별, 재개 가능한 Parquet shard로 수집한다.
2. 선수 ID 비의존 Global을 2023·2024·2025 시간순 OOF로 추론한다.
3. 자격을 충족한 투수 전체를 공유하는 pooled residual을 학습한다.
4. 2024 scale 선택과 2025 최종 게이트를 통과한 선수만 registry에 활성화한다.
5. 쇼케이스는 경기 시작 전 데이터만으로 별도 학습한다.
6. schema v5 정적 JSON에 각 투구의 Global/Residual 출처와 유효 scale을
   기록한다.

## 현재 피처 단계

- 기존 커리어·시즌·최근 100구·카운트·손잡이·전환확률을 유지한다.
- 같은 타석의 직전 1·2·3구와 경기 전체 동일 구종 연속 횟수를 유지한다.
- 타석 내 6종별 사용 횟수, 보여준 구종 수, 첫 구종, 타석 내 연속 횟수와
  패스트볼·브레이킹·오프스피드 비율을 추가한다.
- 현재 경기 구종별 smoothed 사용률, 시즌 대비 변화량과 보여준 구종 수를
  추가한다.

Similarity logistic stacker, xLSTM, 라이브 경기, 사용자 대결, 계정, 영상 분석은
이번 범위에서 제외한다.

## 운영 안전장치

- 모든 누적·rolling 피처는 현재 투구를 제외한다.
- 월별 수집 파일과 모델은 중간 저장하며 재실행 시 완료 항목을 건너뛴다.
- 대형 작업은 순차 실행하고 RAM 9 GiB 미만, 디스크 15 GiB 미만,
  GPU 사용량 20 GiB 초과 시 다음 작업을 시작하지 않는다.
- 각 실행은 자원 스냅샷, 설정, 지표, 실패 사유를 `artifacts/runs`에 남긴다.
