# MLB Film Room 6개 타석 연속 투구 영상

- 검수일: 2026-07-27
- 범위: Hunter Brown 2타석, Chris Flexen 2타석, Blake Snell 2타석
- 합계: 6타석, 25구
- 목적: 한 타석의 모든 투구를 순서대로 보여 주는 사람 대상 구종 예측 파일럿

## 검수 방법

투구 순서는 공식 MLB live feed에서 `about.atBatIndex = AB - 1`인 타석을 찾은 뒤, `playEvents` 중 `isPitch=true`인 이벤트만 원래 순서대로 추출했다. Feed의 `count`는 해당 투구가 끝난 뒤 값이므로, `pre-count`는 첫 공을 0-0으로 두고 이후에는 직전 투구의 `count`를 사용했으며 로컬 Statcast의 `balls`·`strikes`와 교차 확인했다. 구종 코드는 투구 이벤트의 `details.type.code` 원문이다.

- [game 776262 공식 feed](https://statsapi.mlb.com/api/v1.1/game/776262/feed/live)
- [game 777153 공식 feed](https://statsapi.mlb.com/api/v1.1/game/777153/feed/live)
- [game 776717 공식 feed](https://statsapi.mlb.com/api/v1.1/game/776717/feed/live)
- [game 776534 공식 feed](https://statsapi.mlb.com/api/v1.1/game/776534/feed/live)
- [Statcast CSV 필드 문서](https://baseballsavant.mlb.com/csv-docs)

25개 `playId`를 각각 공식 Baseball Savant 개별 영상 페이지에서 열고 `<video>` 재생을 직접 시작했다. 모두 `readyState=4`, `currentTime>0`, `paused=false`를 충족했다. 따라서 아래의 `실제 재생 ✓`는 링크 존재 여부만 확인했다는 뜻이 아니다.

구종 코드: `FF` 포심 패스트볼, `SI` 싱커, `FC` 커터, `SL` 슬라이더, `KC` 너클 커브, `CU` 커브, `CH` 체인지업.

## Reel 생성 검증 결과

공식 [Film Room 경기 페이지](https://www.mlb.com/video/game/776262)의 `Pitch by Pitch`에서 첫 투구 카드의 메뉴를 열고 `Reels`를 눌렀다. 비로그인 상태에서는 클립 선택 화면이 열리지 않고 다음 로그인 URL로 즉시 이동했다.

`https://www.mlb.com/login?redirectUri=https%3A%2F%2Fwww.mlb.com%2Fvideo%2Fgame%2F776262`

따라서 비로그인 상태로는 첫 클립조차 Reel에 추가할 수 없으며, 여러 투구 결합·저장·공유도 진행할 수 없다. 외부 MLB 계정 로그인이나 계정 상태 변경은 하지 않았으므로 6개 Reel은 모두 미생성이다. 아래의 개별 Savant 링크를 순서대로 재생하는 구성은 검증됐지만, 공유 가능한 공식 Reel URL은 없다.

## 1. Hunter Brown vs. Randy Arozarena

- 경기: 2025-09-19, game 776262, AB 1, 1회초
- 타석 결과: 삼진(K)
- 총 투구: 4구
- Film Room: [경기 전체 Pitch by Pitch](https://www.mlb.com/video/game/776262)
- Reel 상태: **미생성** — 첫 클립의 `Reels` 선택 시 MLB 로그인이 요구됨

| 순서 | pre-count | raw 구종 | playId | 공식 Savant 영상 | 검증 |
|---:|---|---|---|---|---|
| 1 | 0-0 | FF | `9f341939-0461-3dde-aba0-53146c67b8ec` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=9f341939-0461-3dde-aba0-53146c67b8ec) | 실제 재생 ✓ |
| 2 | 0-1 | FF | `d171930f-1735-301e-98ca-2aedb6b58c93` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=d171930f-1735-301e-98ca-2aedb6b58c93) | 실제 재생 ✓ |
| 3 | 0-2 | SL | `7da5b796-7543-38ba-8b28-6ecd8134b9ad` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=7da5b796-7543-38ba-8b28-6ecd8134b9ad) | 실제 재생 ✓ |
| 4 | 1-2 | KC | `c065575f-17a7-3d62-9fde-0dde847b851c` | [4구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=c065575f-17a7-3d62-9fde-0dde847b851c) | 실제 재생 ✓ |

## 2. Hunter Brown vs. Julio Rodríguez

- 경기: 2025-09-19, game 776262, AB 3, 1회초
- 타석 결과: 홈런(HR)
- 총 투구: 3구
- Film Room: [경기 전체 Pitch by Pitch](https://www.mlb.com/video/game/776262)
- Reel 상태: **미생성** — 비로그인 Reel 추가가 차단됨

| 순서 | pre-count | raw 구종 | playId | 공식 Savant 영상 | 검증 |
|---:|---|---|---|---|---|
| 1 | 0-0 | KC | `52cf57c1-8e40-3d47-8a12-0deeac2d862a` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=52cf57c1-8e40-3d47-8a12-0deeac2d862a) | 실제 재생 ✓ |
| 2 | 1-0 | SL | `73c9f8b1-ef00-3797-afea-bab067cffeee` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=73c9f8b1-ef00-3797-afea-bab067cffeee) | 실제 재생 ✓ |
| 3 | 2-0 | SI | `cb946579-155c-3f5c-bc46-459d59cf2d09` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=cb946579-155c-3f5c-bc46-459d59cf2d09) | 실제 재생 ✓ |

## 3. Chris Flexen vs. Jasson Domínguez

- 경기: 2025-07-11, game 777153, AB 5, 1회말
- 타석 결과: 삼진(K)
- 총 투구: 6구
- Film Room: [경기 전체 Pitch by Pitch](https://www.mlb.com/video/game/777153)
- Reel 상태: **미생성** — 비로그인 Reel 추가가 차단됨

| 순서 | pre-count | raw 구종 | playId | 공식 Savant 영상 | 검증 |
|---:|---|---|---|---|---|
| 1 | 0-0 | FF | `eddc170c-f646-3146-b317-63d0cc36c2a8` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=eddc170c-f646-3146-b317-63d0cc36c2a8) | 실제 재생 ✓ |
| 2 | 0-1 | CH | `d749e5b1-32d7-3c1d-b58c-2bb03b7e5829` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=d749e5b1-32d7-3c1d-b58c-2bb03b7e5829) | 실제 재생 ✓ |
| 3 | 1-1 | FF | `1a43cec0-ccc9-39b0-a551-470437feb4f7` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=1a43cec0-ccc9-39b0-a551-470437feb4f7) | 실제 재생 ✓ |
| 4 | 2-1 | FF | `cb2bc7fb-19ac-324d-b8d4-c78d2cdbc305` | [4구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=cb2bc7fb-19ac-324d-b8d4-c78d2cdbc305) | 실제 재생 ✓ |
| 5 | 3-1 | FF | `f1bcad4b-35b2-3bcc-94ef-bfa7b318d29f` | [5구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=f1bcad4b-35b2-3bcc-94ef-bfa7b318d29f) | 실제 재생 ✓ |
| 6 | 3-2 | FF | `5a321cb1-7d37-3cdc-8c9d-e6eb8873d1e8` | [6구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=5a321cb1-7d37-3cdc-8c9d-e6eb8873d1e8) | 실제 재생 ✓ |

## 4. Chris Flexen vs. Aaron Judge

- 경기: 2025-07-11, game 777153, AB 6, 1회말
- 타석 결과: 볼넷(BB)
- 총 투구: 5구
- Film Room: [경기 전체 Pitch by Pitch](https://www.mlb.com/video/game/777153)
- Reel 상태: **미생성** — 비로그인 Reel 추가가 차단됨

| 순서 | pre-count | raw 구종 | playId | 공식 Savant 영상 | 검증 |
|---:|---|---|---|---|---|
| 1 | 0-0 | FC | `986e4c92-6723-37c3-a099-40d6b95b2a76` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=986e4c92-6723-37c3-a099-40d6b95b2a76) | 실제 재생 ✓ |
| 2 | 1-0 | FC | `9014ad03-dcb2-3e84-b16a-572274d70b32` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=9014ad03-dcb2-3e84-b16a-572274d70b32) | 실제 재생 ✓ |
| 3 | 2-0 | FF | `61f60747-5fab-3e01-9f40-94a3bef8019f` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=61f60747-5fab-3e01-9f40-94a3bef8019f) | 실제 재생 ✓ |
| 4 | 3-0 | FF | `aba18eca-7f64-35d0-b140-90ca100ccb5f` | [4구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=aba18eca-7f64-35d0-b140-90ca100ccb5f) | 실제 재생 ✓ |
| 5 | 3-1 | FF | `6c100aac-00d7-31a9-a55b-ba6cfe6e2238` | [5구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=6c100aac-00d7-31a9-a55b-ba6cfe6e2238) | 실제 재생 ✓ |

## 5. Blake Snell vs. Fernando Tatis Jr.

- 경기: 2025-08-16, game 776717, AB 25, 3회초
- 타석 결과: 삼진(K)
- 총 투구: 4구
- Film Room: [경기 전체 Pitch by Pitch](https://www.mlb.com/video/game/776717)
- Reel 상태: **미생성** — 비로그인 Reel 추가가 차단됨

| 순서 | pre-count | raw 구종 | playId | 공식 Savant 영상 | 검증 |
|---:|---|---|---|---|---|
| 1 | 0-0 | CH | `ba1da303-1cbf-3cc3-a8d4-cc8c8a9794f9` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=ba1da303-1cbf-3cc3-a8d4-cc8c8a9794f9) | 실제 재생 ✓ |
| 2 | 0-1 | FF | `dfcc2efe-ed10-3bfe-ad1f-c55442605802` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=dfcc2efe-ed10-3bfe-ad1f-c55442605802) | 실제 재생 ✓ |
| 3 | 1-1 | CH | `ca55dfcc-48c7-3d30-98be-b44169342c93` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=ca55dfcc-48c7-3d30-98be-b44169342c93) | 실제 재생 ✓ |
| 4 | 1-2 | CU | `ec1d712f-2ef5-3f68-b732-bec2fcbbd920` | [4구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=ec1d712f-2ef5-3f68-b732-bec2fcbbd920) | 실제 재생 ✓ |

## 6. Blake Snell vs. Corbin Carroll

- 경기: 2025-08-29, game 776534, AB 38, 6회초
- 타석 결과: 2루타(2B)
- 총 투구: 3구
- Film Room: [경기 전체 Pitch by Pitch](https://www.mlb.com/video/game/776534)
- Reel 상태: **미생성** — 비로그인 Reel 추가가 차단됨

| 순서 | pre-count | raw 구종 | playId | 공식 Savant 영상 | 검증 |
|---:|---|---|---|---|---|
| 1 | 0-0 | CU | `3b89133d-ebed-3f05-9c76-89ae50ded352` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=3b89133d-ebed-3f05-9c76-89ae50ded352) | 실제 재생 ✓ |
| 2 | 1-0 | SL | `6e427864-20b8-3db8-a2fb-bc96d1ea2b7c` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=6e427864-20b8-3db8-a2fb-bc96d1ea2b7c) | 실제 재생 ✓ |
| 3 | 2-0 | SL | `22fc72c4-17a0-35be-a4c1-6ecf265e9bf4` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=22fc72c4-17a0-35be-a4c1-6ecf265e9bf4) | 실제 재생 ✓ |

## 실사용 메모

- 순서는 각 표의 1구부터 마지막 구까지다.
- 공개 퀴즈에서 정답 노출을 피하려면 참가자에게 Savant URL, 페이지 제목, 브라우저 탭 제목을 직접 보여 주지 않아야 한다. 일부 페이지 제목은 타석 결과를 그대로 포함한다.
- 공식 Reel이 필요하면 사용자가 직접 MLB 계정으로 로그인한 세션에서 25개 클립을 추가하고, 저장·공유 URL 생성까지 별도로 검증해야 한다.
- 공개 직전에는 링크의 실제 재생 여부를 다시 확인한다. MLB가 영상 제공이나 접근 정책을 바꿀 수 있다.
