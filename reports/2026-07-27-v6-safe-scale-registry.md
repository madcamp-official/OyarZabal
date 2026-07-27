# V6 — 안전 scale과 3단계 Registry 분석

> [!IMPORTANT]
> **한 줄 결론:** Gate만 남겨 98명 모두에게 residual을 적용하는 것은 아직
> 위험하다. 기존 엄격 조건을 전부 통과한 `full` 10명 외에, 선수별 scale을
> 낮추면 Global보다 새 악화를 만들지 않는 `limited` 후보 39명을 찾았다.
> 나머지 49명은 `shadow`를 유지해야 한다.

## 한눈에 보기

동일한 V6 시간순 OOF 결과에서 기존 투구별 scale을 5% 단위로 낮추며 선수별
최대 안전 배율을 계산했다. 이 분석은 Registry 변경 가능성을 진단한 것이며
운영 라우팅은 바꾸지 않았다.

```text
full     10명  █████
limited  39명  ████████████████████
shadow   49명  █████████████████████████
```

| 현재 상태·사유 | 인원 | 새 full | 새 limited | 새 shadow |
|---|---:|---:|---:|---:|
| active | 10 | 10 | 0 | 0 |
| 2024 선수 게이트 실패 | 61 | 0 | **31** | 30 |
| 2025 선수 게이트 실패 | 10 | 0 | **8** | 2 |
| 2024 평가 표본 부족 | 17 | 0 | 0 | 17 |
| 합계 | **98** | **10** | **39** | **49** |

## 왜 Gate만 남기면 위험한가

Context Gate는 각 투구에서 “reference residual이 Global보다 정답에 더 높은
확률을 줬을 가능성”을 예측한다. 선수의 한 시즌 전체에서 Log Loss가
악화되는지, 특정 구종으로 쏠리는지, 주요 구종 recall이 새로 0이 되는지는
직접 보장하지 않는다.

실제 계산에서도 98명 중 49명은 안전한 제한 사용을 입증하지 못했다.

- 17명: 2024 평가 표본이 100구 미만이라 판단 불가
- 32명: 표본은 충분하지만 최소 배율 5%에서도 2024 또는 2025 상대 안전
  조건 실패
- 평가 가능 shadow의 실패 사유는 서로 겹치며, Log Loss 미개선 22건,
  calibration 악화 7건, argmax 구종 분포 악화 6건, TVD 악화 6건이다.

따라서 Gate는 남기되, 최소한 `평가 표본`, `선수별 새 악화 여부`,
`JS/클래스 확률 변화 cap`은 별도 안전장치로 유지하는 편이 맞다.

## 계산 방법

V6의 기존 투구별 scale은 다음과 같다.

```text
s_v6(i,t) = hardSafety × reliability(i) × contextGate(i,t)
```

이번 분석에서는 모델을 하나 더 만들지 않고 선수별 배율만 추가했다.

```text
s_limited(i,t) = alpha(i) × s_v6(i,t)
alpha 후보 = 0.05, 0.10, ..., 1.00
```

2024와 2025에서 각각 아래 상대 안전 조건을 만족하는 가장 큰 `alpha`를
찾고, 두 값 중 작은 값을 보수적 최종 배율로 사용했다.

- residual Log Loss가 같은 행의 Global보다 개선
- Accuracy와 Macro F1 하락이 각각 0.5%p 이내
- Global에는 없던 주요 구종 zero recall을 새로 만들지 않음
- `maxClassShareError`, TVD, `maxClassCalibrationError`가 절대 한도를
  넘는 경우에도 Global보다 더 악화시키지 않음

이 상대 기준은 Global 자체의 분포 문제까지 residual 탓으로 돌리던 기존
게이트의 맹점을 보완한다. `full`은 기존 절대 조건을 모두 통과한 선수만
유지하고, 상대 기준만 통과한 선수는 `limited`로 분리한다.

## Limited 후보의 배율

| 보수적 배율 | 인원 | 선수 |
|---:|---:|---|
| 1.00 | 24 | Yu Darvish, Tyler Anderson, Nathan Eovaldi, Kyle Hendricks, Michael Lorenzen, Marcus Stroman, Yusei Kikuchi, Kevin Gausman, Taijuan Walker, Trevor Williams, Frankie Montas, Aaron Nola, Nick Martinez, Seth Lugo, José Berríos, Pablo López, Michael King, Ryan Feltner, Jake Irvin, Dean Kremer, Cole Ragans, Corbin Burnes, Logan Gilbert, Brayan Bello |
| 0.75 | 1 | Nick Pivetta |
| 0.70 | 1 | Luis Severino |
| 0.65 | 1 | Freddy Peralta |
| 0.40 | 3 | Sonny Gray, Jameson Taillon, Kyle Freeland |
| 0.30 | 1 | George Kirby |
| 0.25 | 1 | Tanner Bibee |
| 0.20 | 1 | Bailey Falter |
| 0.15 | 1 | Zack Wheeler |
| 0.05 | 5 | Jose Quintana, Carlos Rodón, Aaron Civale, MacKenzie Gore, Tarik Skubal |

39명의 중앙 배율은 1.00, 평균은 0.729다. 배율 1.00인 선수가 24명인 것은
residual이 강하다는 뜻이 아니라, 기존 V6 scale을 더 줄일 필요가 없었다는
뜻이다. 일부 선수는 reliability가 매우 낮아 원래 effective scale 자체가
거의 0이다.

## 2024 통과 후 2025에서 탈락했던 10명

| 선수 | 2024 최대 | 2025 최대 | 보수적 배율 | 권장 상태 |
|---|---:|---:|---:|---|
| Jose Quintana | 1.00 | 0.05 | 0.05 | limited |
| Kyle Hendricks | 1.00 | 1.00 | 1.00 | limited |
| Erick Fedde | 1.00 | 0.00 | 0.00 | shadow |
| Nick Martinez | 1.00 | 1.00 | 1.00 | limited |
| Max Fried | 1.00 | 0.00 | 0.00 | shadow |
| José Berríos | 1.00 | 1.00 | 1.00 | limited |
| Luis Severino | 1.00 | 0.70 | 0.70 | limited |
| Pablo López | 1.00 | 1.00 | 1.00 | limited |
| Dean Kremer | 1.00 | 1.00 | 1.00 | limited |
| Cole Ragans | 1.00 | 1.00 | 1.00 | limited |

8명은 기존의 단일 실패 조건 때문에 residual을 완전히 끄기보다 제한적으로
사용할 여지가 있다. Erick Fedde와 Max Fried는 2025 최소 배율에서도
`maxClassShareError`와 TVD를 Global보다 악화시켜 shadow가 맞다.

## 3단계 Registry를 실제로 적용하는 방법

| 상태 | 추론 동작 | 승격 근거 |
|---|---|---|
| full | 기존 V6 scale 그대로 사용 | 2024·2025 엄격 절대 게이트 통과 |
| limited | 기존 V6 scale에 선수별 `alpha`를 곱함 | 두 해의 상대 안전 게이트 통과 |
| shadow | 최종 scale 0, Global만 제공 | 표본 부족 또는 최소 배율도 실패 |

Registry에는 `status`, `scaleMultiplier`, 2024·2025 최대 안전 배율, 표본 수,
실패 사유를 저장한다. 추론 시에는 기존 reliability와 Context Gate를 계산한
뒤 `scaleMultiplier`만 한 번 더 곱한다. JS 0.05와 클래스별 확률 변화 20%p
cap은 그대로 적용한다.

여기서 “inactive 선수 데이터를 제한적으로 사용한다”는 말은 맞지만 두 층을
구분해야 한다.

- 98명 pool의 shadow 선수 데이터는 이미 pooled residual 학습에 사용된다.
- 지금까지 막혀 있던 것은 그 선수에게 residual을 실제 추론에 적용하는
  단계다. `limited`는 이 적용을 작은 scale로 허용한다.
- 98명 pool 밖의 선수는 이번 변경만으로 residual 학습·추론 대상이 되지
  않는다.

## 판단과 배포 조건

권장 방향은 **Gate-only가 아니라 `full / limited / shadow + 최소 안전장치`**
다. 기존의 여러 절대 조건 중 Global에서 물려받은 문제는 limited의 상대
안전 조건으로 완화하되, 표본 부족과 실제 새 악화는 계속 차단한다.

다만 이번 배율은 2024·2025 결과를 보고 선택했으므로 후향 최적값이다.
즉시 active로 배포하면 안 된다. 다음 구현에서는 값을 Registry에
`shadow recommendation`으로만 기록하고, 2026-07-26 이후 고정된 prospective
구간에서 V5 대비 성능과 선수별 안전성을 통과한 뒤 실제 라우팅을 승격해야
한다.

## 재현 근거

- 학습 데이터: 2022–2025 지원 구종 2,990,491구
- 2024 선택 표본: 마지막 20%, 98명 29,085구
- 2025 후향 검증 표본: 같은 98명 pool 189,721구
- 분석 run:
  `artifacts/runs/20260727T104308093224Z/result.json`
- 운영 Registry 변경 여부: `false`

이 분석은 버전 간 성능 비교가 아니라 V6 내부 Registry 라우팅 진단이다.
V5·V6 공식 비교에는 계속 `v5-enabled-pitchers-v1` 고정 cohort와 동일 기간
전체 MLB 표본만 사용한다.
