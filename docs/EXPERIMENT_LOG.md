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
