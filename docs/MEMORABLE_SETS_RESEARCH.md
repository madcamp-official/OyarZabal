# 명승부 Set S 후보 검증

- 검증일: 2026-07-29
- 자료 범위: MLB 공식 뉴스, MLB Stats API `feed/live`, Baseball Savant `sporty-videos`
- 영상 판정: 각 `playId` 페이지가 HTTP 응답을 반환하고 HTML 안에 `id="sporty"`와 `var playId`를 모두 포함할 때 **영상 HTML 있음**으로 판정했다. 브라우저에서 미디어 스트림 끝까지 재생한 판정은 아니다.
- 상태 표기: `주자`는 `1루/2루/3루`, `-`는 주자 없음이다. 점수는 `원정-홈`이다.

## 결론

| 우선순위 | 경기 / 챕터 | 제안 범위 | 타석 | 투구 | 영상 HTML | 공개 판단 |
|---|---|---|---:|---:|---:|---|
| S1-A | 2025 WS G7 LAD@TOR, 9회초 Rojas 동점 과정 | AB 72–75 | 4 | 18 | 18/18 | **공개 가능** |
| S1-B | 2025 WS G7 LAD@TOR, 9회말 만루 위기 | AB 78–81 | 4 | 16 | 16/16 | **공개 가능** |
| S2 | 2026-06-10 WSH@SF, 9회말 Eldridge 끝내기 | AB 80–83 | 4 | 17 | 17/17 | **공개 가능** |
| S3-A | 2025-09-06 LAD@BAL, 9회말 노히트 저지 | AB 65–67 | 3 | 16 | 16/16 | **공개 가능** |
| S3-B | 2025-09-06 LAD@BAL, 9회말 만루·끝내기 | AB 68–70 | 3 | 13 | 13/13 | **공개 가능**. A부터 연속 재생하면 29구 |
| S4 | 2025 ALCS G7 SEA@TOR, 7회말 Springer 역전 | AB 55–58 | 4 | 12 | 12/12 | **공개 가능** |

전체 공개 후보는 **22타석, 92구이며 Savant 영상 HTML은 92/92**다.

## S1. 2025 World Series Game 7 LAD@TOR

MLB 공식 보도는 Miguel Rojas가 9회초 1아웃에서 동점 홈런을 쳤고, 9회말 Toronto가 만루를 만들었지만 Daulton Varsho의 홈 강제 아웃과 Ernie Clement의 중견수 뜬공으로 연장에 들어갔다고 기록한다. 이 두 장면은 결과를 미리 밝히지 않는 별도 챕터 두 개가 적합하다. [MLB 공식 보도](https://www.mlb.com/news/dodgers-win-2025-world-series) · [MLB Stats API feed](https://statsapi.mlb.com/api/v1.1/game/813024/feed/live)

- `gamePk`: `813024`
- 경기: Los Angeles Dodgers 5 @ Toronto Blue Jays 4, World Series Game 7
- 챕터 A 시작 노출 정보: `9회초 · LAD 3-4 TOR · 0아웃 · 주자 없음`
- 챕터 B 시작 노출 정보: `9회말 · LAD 4-4 TOR · 1아웃 · 1루 Isiah Kiner-Falefa`
- 챕터 B 범위는 이미 출루한 주자를 둔 AB 78부터다. 그 주자의 출루까지 포함하려면 AB 77 Bo Bichette의 안타를 추가해야 하지만, 제품의 12~18구 길이와 공식 기사의 “두 주자가 나간 뒤” 장면에 맞춰 16구인 AB 78–81을 선택했다.

### 챕터 A — 9회초 동점 과정

| AB | 타자 vs 투수 | 시작: count / out / 주자 / 점수 | 종료: count / out / 주자 / 점수 | 결과 | 구수 |
|---:|---|---|---|---|---:|
| 72 | Enrique Hernández vs Jeff Hoffman | 0-0 / 0 / - / 3-4 | 1-3 / 1 / - / 3-4 | Strikeout | 4 |
| 73 | Miguel Rojas vs Jeff Hoffman | 0-0 / 1 / - / 3-4 | 3-2 / 1 / - / 4-4 | Home Run | 7 |
| 74 | Shohei Ohtani vs Jeff Hoffman | 0-0 / 1 / - / 4-4 | 0-0 / 2 / - / 4-4 | Flyout | 1 |
| 75 | Will Smith vs Jeff Hoffman | 0-0 / 2 / - / 4-4 | 3-3 / 3 / - / 4-4 | Strikeout | 6 |

| AB | # | 투구 전 count | 구종 | 결과 | playId / Savant | HTML |
|---:|---:|---|---|---|---|---|
| 72 | 1 | 0-0 | SL | Ball | [`42c8f1bb-43a0-34d9-b206-130667685428`](https://baseballsavant.mlb.com/sporty-videos?playId=42c8f1bb-43a0-34d9-b206-130667685428) | ✓ |
| 72 | 2 | 1-0 | SL | Swinging Strike | [`31175e15-5c97-3f9d-9ec9-2bb3c0c46274`](https://baseballsavant.mlb.com/sporty-videos?playId=31175e15-5c97-3f9d-9ec9-2bb3c0c46274) | ✓ |
| 72 | 3 | 1-1 | SL | Foul | [`35781700-4a0e-3c29-9df4-10d9de870e38`](https://baseballsavant.mlb.com/sporty-videos?playId=35781700-4a0e-3c29-9df4-10d9de870e38) | ✓ |
| 72 | 4 | 1-2 | SL | Swinging Strike (Blocked) | [`2c98750e-2d6d-3c45-b949-caf8dd687e4a`](https://baseballsavant.mlb.com/sporty-videos?playId=2c98750e-2d6d-3c45-b949-caf8dd687e4a) | ✓ |
| 73 | 1 | 0-0 | SL | Swinging Strike | [`75c043cb-e55c-338a-8ea1-1ec03e9d9c47`](https://baseballsavant.mlb.com/sporty-videos?playId=75c043cb-e55c-338a-8ea1-1ec03e9d9c47) | ✓ |
| 73 | 2 | 0-1 | SL | Ball | [`342320bb-a876-3e01-b2be-62424c308edf`](https://baseballsavant.mlb.com/sporty-videos?playId=342320bb-a876-3e01-b2be-62424c308edf) | ✓ |
| 73 | 3 | 1-1 | FF | Ball | [`671e2309-576f-38fc-a8b9-0f75833c9bf1`](https://baseballsavant.mlb.com/sporty-videos?playId=671e2309-576f-38fc-a8b9-0f75833c9bf1) | ✓ |
| 73 | 4 | 2-1 | FF | Foul | [`bb262c25-de36-3a8b-aa42-863234456e12`](https://baseballsavant.mlb.com/sporty-videos?playId=bb262c25-de36-3a8b-aa42-863234456e12) | ✓ |
| 73 | 5 | 2-2 | FF | Foul | [`34d0f488-4741-31be-9d0d-8b860292f7a2`](https://baseballsavant.mlb.com/sporty-videos?playId=34d0f488-4741-31be-9d0d-8b860292f7a2) | ✓ |
| 73 | 6 | 2-2 | SL | Ball | [`ba010303-e924-3a23-982d-9a8ac0bc27b5`](https://baseballsavant.mlb.com/sporty-videos?playId=ba010303-e924-3a23-982d-9a8ac0bc27b5) | ✓ |
| 73 | 7 | 3-2 | SL | In play, run(s) | [`7df05ee7-24b5-3c34-84aa-89316ec41110`](https://baseballsavant.mlb.com/sporty-videos?playId=7df05ee7-24b5-3c34-84aa-89316ec41110) | ✓ |
| 74 | 1 | 0-0 | FS | In play, out(s) | [`c2059204-82be-35f8-ae6b-f3b2e39313f5`](https://baseballsavant.mlb.com/sporty-videos?playId=c2059204-82be-35f8-ae6b-f3b2e39313f5) | ✓ |
| 75 | 1 | 0-0 | SL | Called Strike | [`058021f9-8a3f-3821-acb0-c8ca350237d8`](https://baseballsavant.mlb.com/sporty-videos?playId=058021f9-8a3f-3821-acb0-c8ca350237d8) | ✓ |
| 75 | 2 | 0-1 | SL | Called Strike | [`30723ee1-7ae2-3093-8c39-789a58d1e941`](https://baseballsavant.mlb.com/sporty-videos?playId=30723ee1-7ae2-3093-8c39-789a58d1e941) | ✓ |
| 75 | 3 | 0-2 | SL | Ball | [`ae2c6fb1-ff67-3d74-b541-252d4d345a4d`](https://baseballsavant.mlb.com/sporty-videos?playId=ae2c6fb1-ff67-3d74-b541-252d4d345a4d) | ✓ |
| 75 | 4 | 1-2 | SL | Ball | [`ec1b4bde-78b1-3b47-8707-d74a20c5c46d`](https://baseballsavant.mlb.com/sporty-videos?playId=ec1b4bde-78b1-3b47-8707-d74a20c5c46d) | ✓ |
| 75 | 5 | 2-2 | SL | Ball | [`11c568f5-3cfa-3797-8526-0a100d705954`](https://baseballsavant.mlb.com/sporty-videos?playId=11c568f5-3cfa-3797-8526-0a100d705954) | ✓ |
| 75 | 6 | 3-2 | FF | Called Strike | [`75e804ec-3252-324d-b87b-083dce989986`](https://baseballsavant.mlb.com/sporty-videos?playId=75e804ec-3252-324d-b87b-083dce989986) | ✓ |

### 챕터 B — 9회말 만루 위기

| AB | 타자 vs 투수 | 시작: count / out / 주자 / 점수 | 종료: count / out / 주자 / 점수 | 결과 | 구수 |
|---:|---|---|---|---|---:|
| 78 | Addison Barger vs Blake Snell | 0-0 / 1 / 1루 Isiah Kiner-Falefa / 4-4 | 4-2 / 1 / 1루 Addison Barger, 2루 Isiah Kiner-Falefa / 4-4 | Walk | 9 |
| 79 | Alejandro Kirk vs Yoshinobu Yamamoto | 0-0 / 1 / 1·2루 / 4-4 | 1-1 / 1 / 만루 / 4-4 | Hit By Pitch | 2 |
| 80 | Daulton Varsho vs Yoshinobu Yamamoto | 0-0 / 1 / 만루 / 4-4 | 1-2 / 2 / 1루 Daulton Varsho, 2루 Alejandro Kirk, 3루 Addison Barger / 4-4 | Forceout at home | 4 |
| 81 | Ernie Clement vs Yoshinobu Yamamoto | 0-0 / 2 / 만루 / 4-4 | 0-0 / 3 / - / 4-4 | Flyout | 1 |

| AB | # | 투구 전 count | 구종 | 결과 | playId / Savant | HTML |
|---:|---:|---|---|---|---|---|
| 78 | 1 | 0-0 | SL | Ball | [`3d7bc074-5c5b-34e7-883e-f163c1e5a916`](https://baseballsavant.mlb.com/sporty-videos?playId=3d7bc074-5c5b-34e7-883e-f163c1e5a916) | ✓ |
| 78 | 2 | 1-0 | FF | Ball | [`5f12d688-7888-3f8d-94e4-4dd19a0d253e`](https://baseballsavant.mlb.com/sporty-videos?playId=5f12d688-7888-3f8d-94e4-4dd19a0d253e) | ✓ |
| 78 | 3 | 2-0 | CU | Called Strike | [`53632952-f6f6-3cc9-80e0-98b819ee737e`](https://baseballsavant.mlb.com/sporty-videos?playId=53632952-f6f6-3cc9-80e0-98b819ee737e) | ✓ |
| 78 | 4 | 2-1 | SL | Called Strike | [`cfa9a030-38b0-3cd7-b85b-a67e7e4325a6`](https://baseballsavant.mlb.com/sporty-videos?playId=cfa9a030-38b0-3cd7-b85b-a67e7e4325a6) | ✓ |
| 78 | 5 | 2-2 | CH | Foul | [`a185f06d-65ab-31c4-bc8f-40849c79c627`](https://baseballsavant.mlb.com/sporty-videos?playId=a185f06d-65ab-31c4-bc8f-40849c79c627) | ✓ |
| 78 | 6 | 2-2 | CU | Foul | [`78f3d6e2-5762-3655-b531-4e4bd982058f`](https://baseballsavant.mlb.com/sporty-videos?playId=78f3d6e2-5762-3655-b531-4e4bd982058f) | ✓ |
| 78 | 7 | 2-2 | SL | Foul | [`1a629230-5340-30b5-978d-691a6f3945c1`](https://baseballsavant.mlb.com/sporty-videos?playId=1a629230-5340-30b5-978d-691a6f3945c1) | ✓ |
| 78 | 8 | 2-2 | CU | Ball In Dirt | [`3e2298a9-5f97-3763-9243-3bd8deb8eb72`](https://baseballsavant.mlb.com/sporty-videos?playId=3e2298a9-5f97-3763-9243-3bd8deb8eb72) | ✓ |
| 78 | 9 | 3-2 | FF | Ball | [`e1711380-6776-310c-babf-751241a8a02f`](https://baseballsavant.mlb.com/sporty-videos?playId=e1711380-6776-310c-babf-751241a8a02f) | ✓ |
| 79 | 1 | 0-0 | FS | Called Strike | [`9016c1b5-8177-3efd-8624-bf6898cbbd53`](https://baseballsavant.mlb.com/sporty-videos?playId=9016c1b5-8177-3efd-8624-bf6898cbbd53) | ✓ |
| 79 | 2 | 0-1 | SI | Hit By Pitch | [`fab3fd46-af69-35bf-adb5-84077afcacdc`](https://baseballsavant.mlb.com/sporty-videos?playId=fab3fd46-af69-35bf-adb5-84077afcacdc) | ✓ |
| 80 | 1 | 0-0 | FS | Ball | [`a91643b7-6cd9-3504-8b48-04cb4aec5f20`](https://baseballsavant.mlb.com/sporty-videos?playId=a91643b7-6cd9-3504-8b48-04cb4aec5f20) | ✓ |
| 80 | 2 | 1-0 | FS | Foul | [`07032c2b-4d24-3a65-ad00-f3c31c47311d`](https://baseballsavant.mlb.com/sporty-videos?playId=07032c2b-4d24-3a65-ad00-f3c31c47311d) | ✓ |
| 80 | 3 | 1-1 | FF | Called Strike | [`d538f82d-1e8d-3f33-a8a9-5deec16d7e67`](https://baseballsavant.mlb.com/sporty-videos?playId=d538f82d-1e8d-3f33-a8a9-5deec16d7e67) | ✓ |
| 80 | 4 | 1-2 | FS | In play, out(s) | [`787b6bc0-3420-343c-9c79-b3dc06b22586`](https://baseballsavant.mlb.com/sporty-videos?playId=787b6bc0-3420-343c-9c79-b3dc06b22586) | ✓ |
| 81 | 1 | 0-0 | CU | In play, out(s) | [`dd474c08-1865-3406-a209-7e3bbd673cfa`](https://baseballsavant.mlb.com/sporty-videos?playId=dd474c08-1865-3406-a209-7e3bbd673cfa) | ✓ |

## S2. 2026-06-10 WSH@SF — Eldridge 끝내기 만루홈런

MLB Giants 공식 보도는 San Francisco가 9회말 6-10에서 시작해 Matt Chapman의 RBI 2루타, Rafael Devers의 볼넷, Jung Hoo Lee의 안타로 만루를 만든 뒤 Bryce Eldridge가 Mitchell Parker의 2-0 투구를 끝내기 만루홈런으로 연결했다고 기록한다. 기사는 이를 slider라고 서술하지만 Stats API의 해당 투구 분류는 `CU`이므로 구현 데이터는 feed 값을 따른다. 공식 feed도 같은 AB 80–83 연속 4타석을 기록한다. [MLB Giants 공식 보도](https://www.mlb.com/giants/news/giants-bryce-eldridge-hits-walk-off-grand-slam) · [MLB Stats API feed](https://statsapi.mlb.com/api/v1.1/game/823215/feed/live)

- `gamePk`: `823215`
- 경기: Washington Nationals 10 @ San Francisco Giants 11
- 세트 시작 노출 정보: `9회말 · WSH 10-6 SF · 0아웃 · 2루 Luis Arraez`
- 세트 결과: Chapman RBI 2루타 → Devers 볼넷 → Lee 안타 → Eldridge 끝내기 만루홈런

| AB | 타자 vs 투수 | 시작: count / out / 주자 / 점수 | 종료: count / out / 주자 / 점수 | 결과 | 구수 |
|---:|---|---|---|---|---:|
| 80 | Matt Chapman vs Gus Varland | 0-0 / 0 / 2루 Luis Arraez / 10-6 | 0-2 / 0 / 2루 Matt Chapman / 10-7 | Double, 1 RBI | 3 |
| 81 | Rafael Devers vs Gus Varland | 0-0 / 0 / 2루 Matt Chapman / 10-7 | 4-2 / 0 / 1루 Rafael Devers, 2루 Matt Chapman / 10-7 | Walk | 6 |
| 82 | Jung Hoo Lee vs Mitchell Parker | 0-0 / 0 / 1·2루 / 10-7 | 1-2 / 0 / 만루 / 10-7 | Single | 5 |
| 83 | Bryce Eldridge vs Mitchell Parker | 0-0 / 0 / 만루 / 10-7 | 2-0 / 0 / - / 10-11 | Walk-off Grand Slam, 4 RBI | 3 |

| AB | # | 투구 전 count | 구종 | 결과 | playId / Savant | HTML |
|---:|---:|---|---|---|---|---|
| 80 | 1 | 0-0 | FF | Foul | [`f26998ae-ea1e-3124-9bee-cb655bc5e2d5`](https://baseballsavant.mlb.com/sporty-videos?playId=f26998ae-ea1e-3124-9bee-cb655bc5e2d5) | ✓ |
| 80 | 2 | 0-1 | SL | Swinging Strike | [`4e0802d6-0d09-3f28-8fcd-2481363166cd`](https://baseballsavant.mlb.com/sporty-videos?playId=4e0802d6-0d09-3f28-8fcd-2481363166cd) | ✓ |
| 80 | 3 | 0-2 | FF | In play, run(s) | [`47d6bf2f-badd-3e0f-9712-70d60f37af38`](https://baseballsavant.mlb.com/sporty-videos?playId=47d6bf2f-badd-3e0f-9712-70d60f37af38) | ✓ |
| 81 | 1 | 0-0 | SL | Called Strike | [`bfa42e22-a3c6-3c9f-9607-b09bbad715a6`](https://baseballsavant.mlb.com/sporty-videos?playId=bfa42e22-a3c6-3c9f-9607-b09bbad715a6) | ✓ |
| 81 | 2 | 0-1 | FF | Foul | [`6852291d-65be-3a4a-80ac-0d33e90481f2`](https://baseballsavant.mlb.com/sporty-videos?playId=6852291d-65be-3a4a-80ac-0d33e90481f2) | ✓ |
| 81 | 3 | 0-2 | FF | Ball | [`e3b1af7d-9064-30c8-9d8f-ddac21524c95`](https://baseballsavant.mlb.com/sporty-videos?playId=e3b1af7d-9064-30c8-9d8f-ddac21524c95) | ✓ |
| 81 | 4 | 1-2 | FF | Ball | [`5b75b42b-52c8-36f6-9299-c1789c4ee22f`](https://baseballsavant.mlb.com/sporty-videos?playId=5b75b42b-52c8-36f6-9299-c1789c4ee22f) | ✓ |
| 81 | 5 | 2-2 | FF | Ball | [`4564ff3e-e608-3e84-b373-b97260c46945`](https://baseballsavant.mlb.com/sporty-videos?playId=4564ff3e-e608-3e84-b373-b97260c46945) | ✓ |
| 81 | 6 | 3-2 | FF | Ball | [`f99ba1ca-da1f-3d54-92e7-3171bfa659e9`](https://baseballsavant.mlb.com/sporty-videos?playId=f99ba1ca-da1f-3d54-92e7-3171bfa659e9) | ✓ |
| 82 | 1 | 0-0 | SL | Called Strike | [`666d3e92-8396-3b86-b3cb-b9b2a25d5143`](https://baseballsavant.mlb.com/sporty-videos?playId=666d3e92-8396-3b86-b3cb-b9b2a25d5143) | ✓ |
| 82 | 2 | 0-1 | SL | Foul | [`75dc9fc2-eb06-33dd-8fd5-e4c62939320b`](https://baseballsavant.mlb.com/sporty-videos?playId=75dc9fc2-eb06-33dd-8fd5-e4c62939320b) | ✓ |
| 82 | 3 | 0-2 | SL | Ball | [`05b08440-e53d-34c4-aefc-71462b0a84a2`](https://baseballsavant.mlb.com/sporty-videos?playId=05b08440-e53d-34c4-aefc-71462b0a84a2) | ✓ |
| 82 | 4 | 1-2 | CU | Foul | [`9ea56ce6-a1be-30b7-9206-4fb12f503187`](https://baseballsavant.mlb.com/sporty-videos?playId=9ea56ce6-a1be-30b7-9206-4fb12f503187) | ✓ |
| 82 | 5 | 1-2 | FF | In play, no out | [`3c06ade8-472e-304a-9897-50f6ec2c177e`](https://baseballsavant.mlb.com/sporty-videos?playId=3c06ade8-472e-304a-9897-50f6ec2c177e) | ✓ |
| 83 | 1 | 0-0 | SL | Ball | [`eb0839db-2c5a-3317-a3d1-3ab11c9bba5b`](https://baseballsavant.mlb.com/sporty-videos?playId=eb0839db-2c5a-3317-a3d1-3ab11c9bba5b) | ✓ |
| 83 | 2 | 1-0 | SL | Ball | [`516e0582-7639-3979-acd4-5e5bd4f06b11`](https://baseballsavant.mlb.com/sporty-videos?playId=516e0582-7639-3979-acd4-5e5bd4f06b11) | ✓ |
| 83 | 3 | 2-0 | CU | In play, run(s) | [`dadbb09f-3c03-3ec2-b314-2a6d03e6300f`](https://baseballsavant.mlb.com/sporty-videos?playId=dadbb09f-3c03-3ec2-b314-2a6d03e6300f) | ✓ |

## S3. 2025-09-06 LAD@BAL — 노히트 저지부터 끝내기

MLB는 이 경기를 2025년 최고의 경기 10위로 선정하면서, 9회 2아웃까지 노히트 중이던 Yoshinobu Yamamoto를 상대로 Jackson Holliday가 홈런을 쳤고, 이후 Blake Treinen이 만루와 밀어내기 득점을 허용한 뒤 Emmanuel Rivera가 Tanner Scott에게 끝내기 2타점 안타를 쳤다고 정리한다. 따라서 Holliday와 Rivera를 모두 담는 정확한 연속 구간은 3~4타석이 아니라 **AB 65–70의 6타석, 29구**다. [MLB 공식 후보 근거](https://www.mlb.com/news/mlb-top-games-of-2025#10-from-imminent-no-no-to-loss-in-minutes-dodgers-at-orioles-sept-6) · [MLB Stats API feed](https://statsapi.mlb.com/api/v1.1/game/776443/feed/live)

- `gamePk`: `776443`
- 경기: Los Angeles Dodgers 3 @ Baltimore Orioles 4
- 세트 시작 노출 정보: `9회말 · LAD 3-0 BAL · 2아웃 · 주자 없음`
- 세트 결과: Holliday HR → Jeremiah Jackson 2루타 → Gunnar Henderson HBP → Ryan Mountcastle 볼넷 → Colton Cowser 밀어내기 볼넷 → Rivera 끝내기 2타점 안타
- 구성 판단: 12~18구에 맞춰 중간 타석을 생략하면 실제로 주자와 점수를 만든 과정이 끊어진다. 이벤트 정확성 우선 원칙에 따라 29구 전체를 한 세트로 두거나, UI에서 `노히트 저지`와 `만루·끝내기` 두 챕터로 시각적으로만 나누는 것이 안전하다.

### 연속 타석 상태

| AB | 타자 vs 투수 | 시작: count / out / 주자 / 점수 | 종료: count / out / 주자 / 점수 | 결과 | 구수 |
|---:|---|---|---|---|---:|
| 65 | Jackson Holliday vs Yoshinobu Yamamoto | 0-0 / 2 / - / 3-0 | 2-1 / 2 / - / 3-1 | Home Run | 4 |
| 66 | Jeremiah Jackson vs Blake Treinen | 0-0 / 2 / - / 3-1 | 3-2 / 2 / 2루 Jeremiah Jackson / 3-1 | Double | 7 |
| 67 | Gunnar Henderson vs Blake Treinen | 0-0 / 2 / 2루 Daniel Johnson / 3-1 | 2-2 / 2 / 1루 Gunnar Henderson, 2루 Daniel Johnson / 3-1 | Hit By Pitch | 5 |
| 68 | Ryan Mountcastle vs Blake Treinen | 0-0 / 2 / 1루 Gunnar Henderson, 2루 Daniel Johnson / 3-1 | 4-1 / 2 / 1루 Ryan Mountcastle, 2루 Gunnar Henderson, 3루 Daniel Johnson / 3-1 | Walk | 5 |
| 69 | Colton Cowser vs Blake Treinen | 0-0 / 2 / 1루 Jorge Mateo, 2루 Gunnar Henderson, 3루 Daniel Johnson / 3-1 | 4-1 / 2 / 1루 Colton Cowser, 2루 Jorge Mateo, 3루 Gunnar Henderson / 3-2 | Walk, 1 RBI | 5 |
| 70 | Emmanuel Rivera vs Tanner Scott | 0-0 / 2 / 만루 / 3-2 | 1-1 / 2 / 1루 Emmanuel Rivera, 2루 Colton Cowser / 3-4 | Walk-off Single, 2 RBI | 3 |

AB 67 시작 전 Daniel Johnson이 Jeremiah Jackson의 대주자로, AB 69 시작 전 Jorge Mateo가 Ryan Mountcastle의 대주자로 들어간 사실과 AB 68의 폭투 주자 진루도 같은 [공식 feed](https://statsapi.mlb.com/api/v1.1/game/776443/feed/live)의 action/runners 기록을 반영했다.

### 투구와 Savant 영상 HTML

아래 29개 모두 2026-07-29 검사에서 영상 HTML 마커를 반환했다.

| AB | # | 투구 전 count | 구종 | 결과 | playId / Savant | HTML |
|---:|---:|---|---|---|---|---|
| 65 | 1 | 0-0 | FF | Foul | [`df259841-e8b9-3778-bdfd-09e0ac0eae2d`](https://baseballsavant.mlb.com/sporty-videos?playId=df259841-e8b9-3778-bdfd-09e0ac0eae2d) | ✓ |
| 65 | 2 | 0-1 | FS | Ball | [`18e8fd9b-5789-3853-a49a-158df6790bab`](https://baseballsavant.mlb.com/sporty-videos?playId=18e8fd9b-5789-3853-a49a-158df6790bab) | ✓ |
| 65 | 3 | 1-1 | FF | Ball | [`0084d853-a42b-3f9e-a8e5-9f06d11bedd4`](https://baseballsavant.mlb.com/sporty-videos?playId=0084d853-a42b-3f9e-a8e5-9f06d11bedd4) | ✓ |
| 65 | 4 | 2-1 | FC | In play, run(s) | [`3d3a8c2d-26bb-367a-9a79-6b199560d51d`](https://baseballsavant.mlb.com/sporty-videos?playId=3d3a8c2d-26bb-367a-9a79-6b199560d51d) | ✓ |
| 66 | 1 | 0-0 | SI | Called Strike | [`a26852b4-a539-3093-a0c4-a70c32fcd0b2`](https://baseballsavant.mlb.com/sporty-videos?playId=a26852b4-a539-3093-a0c4-a70c32fcd0b2) | ✓ |
| 66 | 2 | 0-1 | ST | Swinging Strike | [`151ef9d8-cae2-3d83-a85d-06d71eb881ac`](https://baseballsavant.mlb.com/sporty-videos?playId=151ef9d8-cae2-3d83-a85d-06d71eb881ac) | ✓ |
| 66 | 3 | 0-2 | ST | Ball | [`2cef920f-e5a3-3fd4-9711-b09e529ec9f2`](https://baseballsavant.mlb.com/sporty-videos?playId=2cef920f-e5a3-3fd4-9711-b09e529ec9f2) | ✓ |
| 66 | 4 | 1-2 | ST | Ball | [`410e6c80-b843-3a19-adfe-008e199ab008`](https://baseballsavant.mlb.com/sporty-videos?playId=410e6c80-b843-3a19-adfe-008e199ab008) | ✓ |
| 66 | 5 | 2-2 | SI | Foul | [`662ee4a6-1152-318d-96ae-367966974049`](https://baseballsavant.mlb.com/sporty-videos?playId=662ee4a6-1152-318d-96ae-367966974049) | ✓ |
| 66 | 6 | 2-2 | ST | Ball | [`c5a2b66c-6a6a-30d3-9d2f-e178fbeaa458`](https://baseballsavant.mlb.com/sporty-videos?playId=c5a2b66c-6a6a-30d3-9d2f-e178fbeaa458) | ✓ |
| 66 | 7 | 3-2 | SI | In play, no out | [`cdea0c07-e417-37dd-8636-91b12a988cfa`](https://baseballsavant.mlb.com/sporty-videos?playId=cdea0c07-e417-37dd-8636-91b12a988cfa) | ✓ |
| 67 | 1 | 0-0 | ST | Ball In Dirt | [`b5e73873-7fe6-3fe9-b47d-05f8a48f0e0f`](https://baseballsavant.mlb.com/sporty-videos?playId=b5e73873-7fe6-3fe9-b47d-05f8a48f0e0f) | ✓ |
| 67 | 2 | 1-0 | SI | Called Strike | [`2d478570-3d92-3dd0-8615-94dd58df39c0`](https://baseballsavant.mlb.com/sporty-videos?playId=2d478570-3d92-3dd0-8615-94dd58df39c0) | ✓ |
| 67 | 3 | 1-1 | FC | Foul | [`12f8ecb6-7806-3c9a-ba9d-7c1f4d584e1f`](https://baseballsavant.mlb.com/sporty-videos?playId=12f8ecb6-7806-3c9a-ba9d-7c1f4d584e1f) | ✓ |
| 67 | 4 | 1-2 | FF | Foul | [`7d27ddd2-5350-32f7-b6fb-61bab231f120`](https://baseballsavant.mlb.com/sporty-videos?playId=7d27ddd2-5350-32f7-b6fb-61bab231f120) | ✓ |
| 67 | 5 | 1-2 | ST | Hit By Pitch | [`e1c8df63-b81a-3c83-89e4-66ded67ea7d8`](https://baseballsavant.mlb.com/sporty-videos?playId=e1c8df63-b81a-3c83-89e4-66ded67ea7d8) | ✓ |
| 68 | 1 | 0-0 | ST | Swinging Strike (Blocked) | [`b86980e0-714b-3ebd-9b1f-47dbbe3db35e`](https://baseballsavant.mlb.com/sporty-videos?playId=b86980e0-714b-3ebd-9b1f-47dbbe3db35e) | ✓ |
| 68 | 2 | 0-1 | FF | Ball | [`8fbd9d45-15cf-36a1-9a9e-754b6eec09ea`](https://baseballsavant.mlb.com/sporty-videos?playId=8fbd9d45-15cf-36a1-9a9e-754b6eec09ea) | ✓ |
| 68 | 3 | 1-1 | ST | Ball | [`3f67f7c8-5905-3c82-a31e-30a8ac72aea1`](https://baseballsavant.mlb.com/sporty-videos?playId=3f67f7c8-5905-3c82-a31e-30a8ac72aea1) | ✓ |
| 68 | 4 | 2-1 | SI | Ball | [`61dd3e38-62cb-337e-8a52-1d3df3dc275f`](https://baseballsavant.mlb.com/sporty-videos?playId=61dd3e38-62cb-337e-8a52-1d3df3dc275f) | ✓ |
| 68 | 5 | 3-1 | FF | Ball | [`f6789948-f423-3b1b-a399-e1ee6bad7774`](https://baseballsavant.mlb.com/sporty-videos?playId=f6789948-f423-3b1b-a399-e1ee6bad7774) | ✓ |
| 69 | 1 | 0-0 | SI | Ball | [`f72b07f7-3e10-3244-8ba4-ec4ac29a7177`](https://baseballsavant.mlb.com/sporty-videos?playId=f72b07f7-3e10-3244-8ba4-ec4ac29a7177) | ✓ |
| 69 | 2 | 1-0 | SI | Ball | [`33b27ad5-95c3-3ada-ab96-6bce96cf6aec`](https://baseballsavant.mlb.com/sporty-videos?playId=33b27ad5-95c3-3ada-ab96-6bce96cf6aec) | ✓ |
| 69 | 3 | 2-0 | FF | Ball | [`5c1501ea-ae54-3ad6-966b-43a31b0812fd`](https://baseballsavant.mlb.com/sporty-videos?playId=5c1501ea-ae54-3ad6-966b-43a31b0812fd) | ✓ |
| 69 | 4 | 3-0 | FC | Called Strike | [`a6b462c6-d5f1-3ab1-ab0d-08aa30025fb8`](https://baseballsavant.mlb.com/sporty-videos?playId=a6b462c6-d5f1-3ab1-ab0d-08aa30025fb8) | ✓ |
| 69 | 5 | 3-1 | ST | Ball In Dirt | [`33302097-8aff-3e96-8374-7deba894d8fc`](https://baseballsavant.mlb.com/sporty-videos?playId=33302097-8aff-3e96-8374-7deba894d8fc) | ✓ |
| 70 | 1 | 0-0 | FF | Ball | [`4f17c537-e9b7-3e35-87f4-ac802a100e9c`](https://baseballsavant.mlb.com/sporty-videos?playId=4f17c537-e9b7-3e35-87f4-ac802a100e9c) | ✓ |
| 70 | 2 | 1-0 | FF | Called Strike | [`dade655a-d118-3129-8717-e9641e58a115`](https://baseballsavant.mlb.com/sporty-videos?playId=dade655a-d118-3129-8717-e9641e58a115) | ✓ |
| 70 | 3 | 1-1 | FF | In play, run(s) | [`a31830e1-b33d-3af2-bf00-eb98a2ff4ee6`](https://baseballsavant.mlb.com/sporty-videos?playId=a31830e1-b33d-3af2-bf00-eb98a2ff4ee6) | ✓ |

## S4. 2025 ALCS Game 7 SEA@TOR — Springer 역전 3점 홈런

MLB 공식 회고는 George Springer가 ALCS Game 7의 7회말, Toronto가 1-3으로 뒤진 1아웃 주자 2명 상황에서 역전 3점 홈런을 쳤다고 기록한다. 두 주자는 Addison Barger의 볼넷과 Isiah Kiner-Falefa의 안타로 나갔고, 둘을 진루시킨 Andrés Giménez의 희생번트가 사이에 있으므로 정확한 연속 구간은 요청된 3타석이 아니라 **AB 55–58의 4타석, 12구**다. [MLB 공식 회고](https://www.mlb.com/news/biggest-moments-of-2025-mlb-postseason#blue-jays-mariners-al-championship-series-by-george-he-s-done-it) · [MLB Stats API feed](https://statsapi.mlb.com/api/v1.1/game/813037/feed/live)

- `gamePk`: `813037`
- 경기: Seattle Mariners 3 @ Toronto Blue Jays 4
- 시리즈 상황: ALCS Game 7, 승자 World Series 진출
- 세트 시작 노출 정보: `7회말 · SEA 3-1 TOR · 0아웃 · 주자 없음`
- 세트 결과: Barger 볼넷 → Kiner-Falefa 안타 → Giménez 희생번트 → Springer 역전 3점 홈런

### 연속 타석 상태

| AB | 타자 vs 투수 | 시작: count / out / 주자 / 점수 | 종료: count / out / 주자 / 점수 | 결과 | 구수 |
|---:|---|---|---|---|---:|
| 55 | Addison Barger vs Bryan Woo | 0-0 / 0 / - / 3-1 | 4-1 / 0 / 1루 Addison Barger / 3-1 | Walk | 5 |
| 56 | Isiah Kiner-Falefa vs Bryan Woo | 0-0 / 0 / 1루 Addison Barger / 3-1 | 0-2 / 0 / 1루 Isiah Kiner-Falefa, 2루 Addison Barger / 3-1 | Single | 3 |
| 57 | Andrés Giménez vs Bryan Woo | 0-0 / 0 / 1루 Isiah Kiner-Falefa, 2루 Addison Barger / 3-1 | 1-0 / 1 / 2루 Isiah Kiner-Falefa, 3루 Addison Barger / 3-1 | Sac Bunt | 2 |
| 58 | George Springer vs Eduard Bazardo | 0-0 / 1 / 2루 Isiah Kiner-Falefa, 3루 Addison Barger / 3-1 | 1-0 / 1 / - / 3-4 | Home Run, 3 RBI | 2 |

### 투구와 Savant 영상 HTML

아래 12개 모두 2026-07-29 검사에서 영상 HTML 마커를 반환했다.

| AB | # | 투구 전 count | 구종 | 결과 | playId / Savant | HTML |
|---:|---:|---|---|---|---|---|
| 55 | 1 | 0-0 | SI | Ball | [`d4dc358a-120c-3640-a1f5-7fcd48621c9a`](https://baseballsavant.mlb.com/sporty-videos?playId=d4dc358a-120c-3640-a1f5-7fcd48621c9a) | ✓ |
| 55 | 2 | 1-0 | FF | Foul | [`72b07142-b85e-3355-bc6c-5bacb3d7e2b3`](https://baseballsavant.mlb.com/sporty-videos?playId=72b07142-b85e-3355-bc6c-5bacb3d7e2b3) | ✓ |
| 55 | 3 | 1-1 | FF | Ball | [`aeec2aa6-00e0-3de6-8753-cab844cf0919`](https://baseballsavant.mlb.com/sporty-videos?playId=aeec2aa6-00e0-3de6-8753-cab844cf0919) | ✓ |
| 55 | 4 | 2-1 | FF | Ball | [`a78647cd-813a-3cc8-8d07-bbeb4589bd77`](https://baseballsavant.mlb.com/sporty-videos?playId=a78647cd-813a-3cc8-8d07-bbeb4589bd77) | ✓ |
| 55 | 5 | 3-1 | FF | Ball | [`bc7fe9b2-d557-3a30-9d0d-88fc83d6841d`](https://baseballsavant.mlb.com/sporty-videos?playId=bc7fe9b2-d557-3a30-9d0d-88fc83d6841d) | ✓ |
| 56 | 1 | 0-0 | FF | Called Strike | [`8bdf37ae-aa97-33b1-a221-bb8bcde8cbfa`](https://baseballsavant.mlb.com/sporty-videos?playId=8bdf37ae-aa97-33b1-a221-bb8bcde8cbfa) | ✓ |
| 56 | 2 | 0-1 | SI | Foul | [`2c9ecad7-f3aa-34e8-af85-d7818eef60ca`](https://baseballsavant.mlb.com/sporty-videos?playId=2c9ecad7-f3aa-34e8-af85-d7818eef60ca) | ✓ |
| 56 | 3 | 0-2 | ST | In play, no out | [`74e44092-a550-33c0-aba7-1d773dfd311d`](https://baseballsavant.mlb.com/sporty-videos?playId=74e44092-a550-33c0-aba7-1d773dfd311d) | ✓ |
| 57 | 1 | 0-0 | FF | Ball | [`508c1058-2172-312a-909c-f18d7c44875a`](https://baseballsavant.mlb.com/sporty-videos?playId=508c1058-2172-312a-909c-f18d7c44875a) | ✓ |
| 57 | 2 | 1-0 | FF | In play, out(s) | [`cb50aa0a-cde7-3f34-8345-5efdd79870f3`](https://baseballsavant.mlb.com/sporty-videos?playId=cb50aa0a-cde7-3f34-8345-5efdd79870f3) | ✓ |
| 58 | 1 | 0-0 | SI | Ball | [`c5084cd1-87ff-3ddb-9f64-8c01da829a7e`](https://baseballsavant.mlb.com/sporty-videos?playId=c5084cd1-87ff-3ddb-9f64-8c01da829a7e) | ✓ |
| 58 | 2 | 1-0 | SI | In play, run(s) | [`43ff90e6-f8cd-3a3f-b93b-de5f2237a323`](https://baseballsavant.mlb.com/sporty-videos?playId=43ff90e6-f8cd-3a3f-b93b-de5f2237a323) | ✓ |

## 공개 전 재검사 기준

1. 공개 빌드 직전에 모든 `playId`에 대해 `sporty_video_available()`을 다시 실행한다.
2. `feed/live`의 `atBatIndex`, `playId`, 타자·투수·count·주자·점수를 저장 원본으로 사용하고, MLB 뉴스 문구를 데이터 대신 파싱하지 않는다.
3. 시작 화면에는 경기, 이닝, 현재 점수, 아웃, 주자, 시리즈 상황만 노출하고 이 문서의 `세트 결과`와 타석 결과는 숨긴다.
4. S3은 29구이므로 제품의 기본 길이보다 길다. 임의로 타석을 삭제하지 말고 챕터 UI 또는 별도 `longSet` 표시로 처리한다.
