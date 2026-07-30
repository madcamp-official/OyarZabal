# 2026 MLB 인간 투구 예측 벤치마크 후보

조사일: 2026-07-28

## 결론

현재 25구 형식을 그대로 유지한다면 아래 5타석 조합이 가장 균형이 좋다.

- 정확히 25구: `4 + 5 + 6 + 6 + 4`
- 타석 결과가 홈런, 삼진, 뜬공, 볼넷, 안타로 모두 다름
- 타자 중복 없음
- 모든 타석이 3~6구 범위
- 사용 구종은 현재 분류 체계로 모두 처리 가능
- 모델의 정답률이나 난이도는 선별 기준으로 사용하지 않음

| 권장 노출 순서 | 투수 | 타자 | 경기 상황 | 결과 | 구종 순서 | 투구 수 |
|---:|---|---|---|---|---|---:|
| 1 | Hunter Brown | Andrew Benintendi | 2026-07-25, 7회말 | 헛스윙 삼진 | CH · SI · SI · FF · KC | 5 |
| 2 | David Peterson | Pete Alonso | 2026-07-09, 4회말 | 볼넷 | CU · CH · FF · SL · FF · FF | 6 |
| 3 | Griffin Canning | Ozzie Albies | 2026-07-23, 4회말 | 홈런 | SL · CH · SL · SI | 4 |
| 4 | Justin Verlander | Nolan Arenado | 2026-03-30, 3회말 | 좌익수 뜬공 | SL · SL · SL · CU · CH · FF | 6 |
| 5 | Merrill Kelly | Jacob Wilson | 2026-07-22, 3회초 | 우전 안타 | SL · CH · SI · FC | 4 |
|  |  |  |  |  | **합계** | **25** |

권장 노출 순서는 타석 길이가 오름차순처럼 보이지 않도록 섞었고, 기존 요구대로 Merrill Kelly를 마지막에 두었다.

구종 코드:

- FF: 포심 패스트볼
- SI: 싱커
- SL: 슬라이더
- CU: 커브
- KC: 너클 커브
- CH: 체인지업

## 타석별 공식 소스와 투구 영상

### 1. Hunter Brown vs Andrew Benintendi

- 경기: Houston Astros @ Chicago White Sox, 2026-07-25
- gamePk: `824571`
- 공식 타석 번호: AB 50
- 결과: Andrew Benintendi 헛스윙 삼진
- [MLB Film Room 경기](https://www.mlb.com/video/game/824571)
- [MLB 공식 live feed](https://statsapi.mlb.com/api/v1.1/game/824571/feed/live)

| # | 투구 전 카운트 | 구종 | 결과 | Savant |
|---:|---|---|---|---|
| 1 | 0-0 | CH | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=c0248f99-7bfe-3381-abc2-66db0281b153) |
| 2 | 1-0 | SI | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=2b92e029-d9e6-3899-9c53-2862051a2233) |
| 3 | 2-0 | SI | 헛스윙 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=4ce56d94-f561-3da6-b8df-e6bb28c8839b) |
| 4 | 2-1 | FF | 파울 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=8d96c56f-4392-39d2-b564-84e6b9e83536) |
| 5 | 2-2 | KC | 헛스윙 삼진 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=68e5507b-af4e-307f-bee3-60fa6bc514c8) |

### 2. David Peterson vs Pete Alonso

- 경기: Chicago Cubs @ Baltimore Orioles, 2026-07-09
- gamePk: `824816`
- 공식 타석 번호: AB 28
- 결과: Pete Alonso 볼넷
- [MLB Film Room 경기](https://www.mlb.com/video/game/824816)
- [MLB 공식 live feed](https://statsapi.mlb.com/api/v1.1/game/824816/feed/live)

| # | 투구 전 카운트 | 구종 | 결과 | Savant |
|---:|---|---|---|---|
| 1 | 0-0 | CU | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=1448362a-255e-3dc8-833e-9148d0140251) |
| 2 | 1-0 | CH | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=33b527c2-de46-3d1d-a85b-a202998303b0) |
| 3 | 2-0 | FF | 파울 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=0b55b332-b8ed-3023-8e67-e60afdaa41b3) |
| 4 | 2-1 | SL | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=bea1051e-091c-32f0-a243-cbb3bc6eb9e2) |
| 5 | 3-1 | FF | 헛스윙 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=98214476-7eab-360f-be4a-4fa7a27cd534) |
| 6 | 3-2 | FF | 볼넷 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=d1a12e84-94b0-3d08-96c6-365a36d6e4b5) |

### 3. Griffin Canning vs Ozzie Albies

- 경기: San Diego Padres @ Atlanta Braves, 2026-07-23
- gamePk: `824893`
- 공식 타석 번호: AB 29
- 결과: Ozzie Albies 우중월 홈런
- [MLB Film Room 경기](https://www.mlb.com/video/game/824893)
- [MLB 공식 live feed](https://statsapi.mlb.com/api/v1.1/game/824893/feed/live)

| # | 투구 전 카운트 | 구종 | 결과 | Savant |
|---:|---|---|---|---|
| 1 | 0-0 | SL | 헛스윙 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=8d357201-6eb1-3411-a0c2-15104f722811) |
| 2 | 0-1 | CH | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=0aeb9e12-7966-32f1-9fc1-b1f978f593e7) |
| 3 | 1-1 | SL | 파울 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=849226e4-c200-3c1f-98cc-0fd152a9808d) |
| 4 | 1-2 | SI | 홈런 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=cefd2c3a-4f7e-33f8-8179-1727d5ac0cf4) |

### 4. Justin Verlander vs Nolan Arenado

- 경기: Detroit Tigers @ Arizona Diamondbacks, 2026-03-30
- gamePk: `825108`
- 공식 타석 번호: AB 28
- 결과: Nolan Arenado 좌익수 뜬공
- [MLB Film Room 경기](https://www.mlb.com/video/game/825108)
- [MLB 공식 live feed](https://statsapi.mlb.com/api/v1.1/game/825108/feed/live)

| # | 투구 전 카운트 | 구종 | 결과 | Savant |
|---:|---|---|---|---|
| 1 | 0-0 | SL | 콜드 스트라이크 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=4f53c70d-8181-3a01-a992-9b0c2e203d66) |
| 2 | 0-1 | SL | 헛스윙 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=1ae6818d-80f1-3289-b0ff-028a83f31071) |
| 3 | 0-2 | SL | 파울 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=02a55c7e-271c-3906-91c2-9ed0944854ca) |
| 4 | 0-2 | CU | 파울 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=98604fe0-8c56-3a4c-ba06-0134927d7c28) |
| 5 | 0-2 | CH | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=03d2bee0-58ab-3bc7-b4eb-25a204201b41) |
| 6 | 1-2 | FF | 뜬공 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=870732ef-c899-3692-9565-866a459ac142) |

### 5. Merrill Kelly vs Jacob Wilson

- 경기: Athletics @ Arizona Diamondbacks, 2026-07-22
- gamePk: `825055`
- 공식 타석 번호: AB 18
- 결과: Jacob Wilson 우전 안타
- [MLB Film Room 경기](https://www.mlb.com/video/game/825055)
- [MLB 공식 live feed](https://statsapi.mlb.com/api/v1.1/game/825055/feed/live)

| # | 투구 전 카운트 | 구종 | 결과 | Savant |
|---:|---|---|---|---|
| 1 | 0-0 | SL | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=0e8238d4-2cc2-3644-806f-327ebb73686e) |
| 2 | 1-0 | CH | 파울팁 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=ea160849-0035-36d4-a8c5-0bbc9566217d) |
| 3 | 1-1 | SI | 볼 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=53571ca2-3d6b-3571-a3f9-4296c9d62cae) |
| 4 | 2-1 | FC | 안타 | [영상](https://baseballsavant.mlb.com/sporty-videos?playId=cc862cad-9879-3c89-85c3-4027db1c5bc9) |

## 투수 선정 이유

- Hunter Brown은 기존 파일럿 선수라 2025판과의 체감 비교가 가능하다.
- David Peterson과 Griffin Canning은 2026 MLB 등판 표본이 충분하고 서로 다른 결과의 타석을 구성하기 좋다.
- Justin Verlander는 사용자 요청에 따라 Reid Detmers를 대신하며, 2026 Arizona전 6구 뜬공 타석을 사용한다.
- Merrill Kelly는 기존 요구대로 마지막 타석에 두면서 안타 결과를 채울 수 있다.
- Chris Flexen은 2026 MLB pitching game log가 없어 이번 2026판 주 후보에서는 제외한다.
- Blake Snell은 조사 시점의 2026 MLB pitching game log가 1경기뿐이라 주 후보에서는 제외하고 예비 후보로 둔다.

공식 game log:

- [Hunter Brown](https://statsapi.mlb.com/api/v1/people/686613/stats?stats=gameLog&group=pitching&season=2026)
- [David Peterson](https://statsapi.mlb.com/api/v1/people/656849/stats?stats=gameLog&group=pitching&season=2026)
- [Griffin Canning](https://statsapi.mlb.com/api/v1/people/656288/stats?stats=gameLog&group=pitching&season=2026)
- [Justin Verlander](https://statsapi.mlb.com/api/v1/people/434378/stats?stats=gameLog&group=pitching&season=2026)
- [Merrill Kelly](https://statsapi.mlb.com/api/v1/people/518876/stats?stats=gameLog&group=pitching&season=2026)
- [Chris Flexen](https://statsapi.mlb.com/api/v1/people/623167/stats?stats=gameLog&group=pitching&season=2026)
- [Blake Snell](https://statsapi.mlb.com/api/v1/people/605483/stats?stats=gameLog&group=pitching&season=2026)

## 예비 교체 후보

실제 영상을 이어 붙인 뒤 중계 자막, 노출된 구종 표기, 영상 끊김이 발견될 때만 아래 후보로 교체한다.

| 투수 | 타자 | 경기 상황 | 결과 | 구종 수 |
|---|---|---|---|---:|
| Griffin Canning | Dominic Smith | 2026-07-23, 4회말 | 홈런 | 3 |
| Hunter Brown | Munetaka Murakami | 2026-07-25, 1회말 | 삼진 | 4 |
| Merrill Kelly | Brandon Lowe | 2026-07-27, 6회말 | 1타점 안타 | 7 |

## 구현 전 확인 사항

1. 각 Savant 영상을 직접 재생해 중계 그래픽에 구종명이 먼저 노출되지 않는지 확인한다.
2. 투수별 안내 구종 목록은 이 타석에 나온 구종만이 아니라 2026 시즌 실제 repertoire 전체에서 만든다.
3. 타석 길이를 참가자가 미리 추측하지 못하도록 현재처럼 미래 투구 칸과 총 투구 수를 숨긴다.
4. 이 세트는 사람 대상 공개 실험용 후보이며, 모델 성능을 보고 고른 showcase나 동결된 2026 holdout으로 표현하지 않는다.
