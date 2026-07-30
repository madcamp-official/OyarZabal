# MLB Film Room 첫 공개 파일럿 후보

- 검수일: 2026-07-27
- 범위: Hunter Brown(686613), Chris Flexen(623167), Blake Snell(605483), 각 10개
- 선정 규칙: 2025년의 3~6구 타석에서 마지막 투구를 `target pitch`로 사용
- 영상 원칙: 공식 Baseball Savant 개별 투구 페이지를 본 재생 링크로, MLB Film Room 경기 페이지를 보조 탐색 링크로 사용

## 검수 방법과 해석

후보 메타데이터는 [Baseball Savant Statcast Search](https://baseballsavant.mlb.com/statcast_search), [Statcast CSV 필드 문서](https://baseballsavant.mlb.com/csv-docs), 공식 MLB 경기 feed를 교차 확인했다. `game_pk`, `at_bat_number`, `pitch_number`의 의미는 CSV 문서를 따른다.

영상 링크는 추측한 Film Room slug가 아니다. 공식 MLB feed의 마지막 투구 `playId`를 공식 [Baseball Savant Videos](https://baseballsavant.mlb.com/sporty-videos) 페이지에서 직접 열었다. 본목록 30개 모두 다음 두 검사를 통과했다.

1. 페이지에 `HOME Broadcast Video` 또는 `AWAY Broadcast Video` 옵션이 표시됨.
2. 브라우저에서 실제 `<video>` 재생을 시작했을 때 `readyState=4`였고 `currentTime`이 0보다 커짐.

표의 `재생 확인`은 “페이지가 존재한다”는 뜻만이 아니라 위 두 조건을 모두 2026-07-27에 확인했다는 뜻이다. 다만 향후 MLB가 영상을 삭제하거나 지역·브라우저 정책을 변경할 수 있으므로 공개 직전 다시 확인해야 한다.

`target pitch`는 모두 타석의 마지막 투구다. `pre-count`는 그 투구가 들어오기 전 볼-스트라이크다. 그룹은 프로젝트의 기존 6분류에 맞춰 기록했다.

## Hunter Brown — 10개

구성: 삼진 3, 볼넷 2, 안타 3(1B·2B·HR), 타구 아웃 2.

| # | 날짜 | game_pk | 이닝 | 타자 | AB | PA 구수 | 결과 | target | pre-count | Statcast raw / 그룹 | 공식 재생 | Film Room | 재생 확인 |
|---|---|---:|---|---|---:|---:|---|---:|---|---|---|---|---|
| H01 | 2025-09-19 | 776262 | 1회초 | Randy Arozarena | 1 | 4 | K | 4 | 1-2 | KC / 커브 계열 | [재생][H01] | [경기][G776262] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H02 | 2025-09-19 | 776262 | 6회초 | Eugenio Suárez | 42 | 6 | K | 6 | 2-2 | SL / 슬라이더 계열 | [재생][H02] | [경기][G776262] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H03 | 2025-08-31 | 776510 | 1회초 | Yoán Moncada | 2 | 5 | K | 5 | 2-2 | KC / 커브 계열 | [재생][H03] | [경기][G776510] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H04 | 2025-09-24 | 776195 | 3회말 | Nick Kurtz | 20 | 4 | BB | 4 | 3-0 | SI / 무빙 패스트볼 | [재생][H04] | [경기][G776195] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H05 | 2025-08-31 | 776510 | 1회초 | Taylor Ward | 4 | 6 | BB | 6 | 3-2 | FF / 포심 | [재생][H05] | [경기][G776510] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H06 | 2025-08-31 | 776510 | 6회초 | Jo Adell | 41 | 3 | 1B | 3 | 2-0 | SI / 무빙 패스트볼 | [재생][H06] | [경기][G776510] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H07 | 2025-09-24 | 776195 | 5회말 | Shea Langeliers | 37 | 3 | 2B | 3 | 1-1 | SI / 무빙 패스트볼 | [재생][H07] | [경기][G776195] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H08 | 2025-09-19 | 776262 | 1회초 | Julio Rodríguez | 3 | 3 | HR | 3 | 2-0 | SI / 무빙 패스트볼 | [재생][H08] | [경기][G776262] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H09 | 2025-08-31 | 776510 | 3회초 | Mike Trout | 21 | 4 | GO | 4 | 2-1 | SI / 무빙 패스트볼 | [재생][H09] | [경기][G776510] | 방송 옵션 ✓ · 실제 재생 ✓ |
| H10 | 2025-09-19 | 776262 | 1회초 | Cal Raleigh | 2 | 5 | GO | 5 | 0-2 | CH / 체인지업 | [재생][H10] | [경기][G776262] | 방송 옵션 ✓ · 실제 재생 ✓ |

## Chris Flexen — 10개

구성: 삼진 3, 볼넷 2, 안타 3(1B 2개·HR), 타구 아웃 2(GO·SF).

| # | 날짜 | game_pk | 이닝 | 타자 | AB | PA 구수 | 결과 | target | pre-count | Statcast raw / 그룹 | 공식 재생 | Film Room | 재생 확인 |
|---|---|---:|---|---|---:|---:|---|---:|---|---|---|---|---|
| F01 | 2025-06-27 | 777336 | 7회말 | Christian Walker | 63 | 6 | K | 6 | 1-2 | CU / 커브 계열 | [재생][F01] | [경기][G777336] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F02 | 2025-07-05 | 777229 | 2회초 | Willson Contreras | 18 | 4 | K | 4 | 1-2 | SL / 슬라이더 계열 | [재생][F02] | [경기][G777229] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F03 | 2025-07-11 | 777153 | 1회말 | Jasson Domínguez | 5 | 6 | K | 6 | 3-2 | FF / 포심 | [재생][F03] | [경기][G777153] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F04 | 2025-06-27 | 777336 | 6회말 | Cam Smith | 54 | 6 | BB | 6 | 3-2 | FC / 무빙 패스트볼 | [재생][F04] | [경기][G777336] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F05 | 2025-07-11 | 777153 | 1회말 | Aaron Judge | 6 | 5 | BB | 5 | 3-1 | FF / 포심 | [재생][F05] | [경기][G777153] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F06 | 2025-06-27 | 777336 | 5회말 | Yainer Diaz | 45 | 5 | 1B | 5 | 2-2 | CU / 커브 계열 | [재생][F06] | [경기][G777336] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F07 | 2025-07-05 | 777229 | 4회초 | Nolan Gorman | 31 | 4 | HR | 4 | 1-2 | SL / 슬라이더 계열 | [재생][F07] | [경기][G777229] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F08 | 2025-07-21 | 777065 | 8회초 | John Rave | 63 | 4 | 1B | 4 | 1-2 | SV / 슬라이더 계열 | [재생][F08] | [경기][G777065] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F09 | 2025-07-05 | 777229 | 3회초 | Alec Burleson | 24 | 5 | GO | 5 | 1-2 | FC / 무빙 패스트볼 | [재생][F09] | [경기][G777229] | 방송 옵션 ✓ · 실제 재생 ✓ |
| F10 | 2025-07-11 | 777153 | 3회말 | Aaron Judge | 23 | 4 | SF | 4 | 2-1 | FC / 무빙 패스트볼 | [재생][F10] | [경기][G777153] | 방송 옵션 ✓ · 실제 재생 ✓ |

## Blake Snell — 10개

구성: 삼진 3, 볼넷 2, 안타 3(1B 2개·2B), 타구 아웃 2(GIDP·GO).

| # | 날짜 | game_pk | 이닝 | 타자 | AB | PA 구수 | 결과 | target | pre-count | Statcast raw / 그룹 | 공식 재생 | Film Room | 재생 확인 |
|---|---|---:|---|---|---:|---:|---|---:|---|---|---|---|---|
| S01 | 2025-08-16 | 776717 | 3회초 | Fernando Tatis Jr. | 25 | 4 | K | 4 | 1-2 | CU / 커브 계열 | [재생][S01] | [경기][G776717] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S02 | 2025-08-16 | 776717 | 5회초 | Jake Cronenworth | 37 | 6 | K | 6 | 2-2 | FF / 포심 | [재생][S02] | [경기][G776717] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S03 | 2025-09-17 | 776280 | 2회초 | Max Kepler | 9 | 6 | K | 6 | 2-2 | SL / 슬라이더 계열 | [재생][S03] | [경기][G776280] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S04 | 2025-08-16 | 776717 | 6회초 | Manny Machado | 46 | 4 | BB | 4 | 3-0 | FF / 포심 | [재생][S04] | [경기][G776717] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S05 | 2025-08-29 | 776534 | 1회초 | Geraldo Perdomo | 2 | 5 | BB | 5 | 3-1 | CH / 체인지업 | [재생][S05] | [경기][G776534] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S06 | 2025-08-22 | 776636 | 2회말 | Ryan O'Hearn | 10 | 5 | 1B | 5 | 2-2 | CU / 커브 계열 | [재생][S06] | [경기][G776636] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S07 | 2025-08-29 | 776534 | 6회초 | Corbin Carroll | 38 | 3 | 2B | 3 | 2-0 | SL / 슬라이더 계열 | [재생][S07] | [경기][G776534] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S08 | 2025-09-17 | 776280 | 3회초 | Bryson Stott | 19 | 5 | 1B | 5 | 2-2 | FF / 포심 | [재생][S08] | [경기][G776280] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S09 | 2025-08-22 | 776636 | 2회말 | Xander Bogaerts | 11 | 4 | GIDP | 4 | 2-1 | CH / 체인지업 | [재생][S09] | [경기][G776636] | 방송 옵션 ✓ · 실제 재생 ✓ |
| S10 | 2025-09-17 | 776280 | 4회초 | Bryce Harper | 25 | 4 | GO | 4 | 2-1 | CU / 커브 계열 | [재생][S10] | [경기][G776280] | 방송 옵션 ✓ · 실제 재생 ✓ |

## 본목록에서 제외한 후보

아래 두 페이지는 브라우저에서 영상 자체는 재생됐지만, 공식 Savant 페이지에서 `HOME Broadcast Video`/`AWAY Broadcast Video` 옵션이 확인되지 않았다. 링크 제공 방식의 일관성을 위해 본목록에서 제외했다.

| 투수 | 경기 / AB | 타자·결과 | 제외 링크 | 대체 |
|---|---|---|---|---|
| Hunter Brown | 776432 / 7 | Jake Burger K | [확인 페이지](https://baseballsavant.mlb.com/sporty-videos?playId=618a4358-a1d0-3fb0-a323-735ceec37eea) | H03 Yoán Moncada K |
| Hunter Brown | 776432 / 43 | Jake Burger GIDP | [확인 페이지](https://baseballsavant.mlb.com/sporty-videos?playId=15323ae8-ed7d-3c98-9ed9-3f445d7455f5) | H09 Mike Trout GO |

## 파일럿 사용 전 체크

- 공개 직전에 30개 링크의 방송 옵션과 실제 재생을 다시 확인한다.
- 페이지 제목이 타석 결과를 노출하므로, 퀴즈 UI에서는 참가자가 답을 제출하기 전 링크 제목·브라우저 탭 제목이 보이지 않게 해야 한다. 공식 플레이어를 우회하거나 영상을 재호스팅해서 해결하지 않는다.
- 본 링크는 target pitch 전체를 재생한다. 릴리스 전 자동 정지 제어는 공식 문서화된 기능으로 확인되지 않았으므로 별도 UX 파일럿이 필요하다.
- 2025년 후보만 사용한 이유는 최근 완료 시즌 중 공식 방송 옵션과 실제 재생을 모두 확인할 수 있는 균형 표본을 우선했기 때문이다.

## 공식 개별 투구 링크

[H01]: https://baseballsavant.mlb.com/sporty-videos?playId=c065575f-17a7-3d62-9fde-0dde847b851c
[H02]: https://baseballsavant.mlb.com/sporty-videos?playId=f19e7daa-c12e-3bf8-a12d-9eb4da1d7c30
[H03]: https://baseballsavant.mlb.com/sporty-videos?playId=6f5c28b6-67aa-3e58-9f27-603d3d93c062
[H04]: https://baseballsavant.mlb.com/sporty-videos?playId=3c2a982a-2aa0-340a-b9a8-019658160152
[H05]: https://baseballsavant.mlb.com/sporty-videos?playId=58b326f2-baf3-313f-929c-7483f4ad3cfd
[H06]: https://baseballsavant.mlb.com/sporty-videos?playId=b81d367e-f2e2-34f1-bea7-82fafc597906
[H07]: https://baseballsavant.mlb.com/sporty-videos?playId=48b1ce4c-dc26-3a2d-9785-c34e3893b9e0
[H08]: https://baseballsavant.mlb.com/sporty-videos?playId=cb946579-155c-3f5c-bc46-459d59cf2d09
[H09]: https://baseballsavant.mlb.com/sporty-videos?playId=d7956f8f-fd24-39c8-84cd-b0ad44a3e7c3
[H10]: https://baseballsavant.mlb.com/sporty-videos?playId=26bb62e0-10bb-3b0a-a1eb-699ed1b6c292

[F01]: https://baseballsavant.mlb.com/sporty-videos?playId=ac0ecc3a-7d5b-32e4-800b-9c129d15d912
[F02]: https://baseballsavant.mlb.com/sporty-videos?playId=64193ce0-3c08-3b30-88eb-048ba7eec334
[F03]: https://baseballsavant.mlb.com/sporty-videos?playId=5a321cb1-7d37-3cdc-8c9d-e6eb8873d1e8
[F04]: https://baseballsavant.mlb.com/sporty-videos?playId=f8b463c8-d6dc-397c-bc86-9605f960c3ea
[F05]: https://baseballsavant.mlb.com/sporty-videos?playId=6c100aac-00d7-31a9-a55b-ba6cfe6e2238
[F06]: https://baseballsavant.mlb.com/sporty-videos?playId=0680d114-b1f8-3cad-8b4e-f0294cb88699
[F07]: https://baseballsavant.mlb.com/sporty-videos?playId=a0a3f064-789f-32ac-aa87-7381672d9179
[F08]: https://baseballsavant.mlb.com/sporty-videos?playId=8dbb4aed-df2d-3874-a8b2-a15bf2e47475
[F09]: https://baseballsavant.mlb.com/sporty-videos?playId=4f5b8dcc-5433-3f2b-bb45-2b7516338d77
[F10]: https://baseballsavant.mlb.com/sporty-videos?playId=b61145a6-8010-332c-81b6-f6301ef863cc

[S01]: https://baseballsavant.mlb.com/sporty-videos?playId=ec1d712f-2ef5-3f68-b732-bec2fcbbd920
[S02]: https://baseballsavant.mlb.com/sporty-videos?playId=941afdd0-e59c-3407-9a3a-a15a0a48c69f
[S03]: https://baseballsavant.mlb.com/sporty-videos?playId=24848156-1872-3bdb-96fc-82a8d8fd0e85
[S04]: https://baseballsavant.mlb.com/sporty-videos?playId=09bc5279-5cd3-3ce1-82d0-7b1130a3c2c6
[S05]: https://baseballsavant.mlb.com/sporty-videos?playId=2eba0258-6e31-304c-9859-146fecf6fe56
[S06]: https://baseballsavant.mlb.com/sporty-videos?playId=a402e91d-a467-3acf-9063-884dde20829a
[S07]: https://baseballsavant.mlb.com/sporty-videos?playId=22fc72c4-17a0-35be-a4c1-6ecf265e9bf4
[S08]: https://baseballsavant.mlb.com/sporty-videos?playId=17b2a951-db52-315b-9ca4-70f773262e3a
[S09]: https://baseballsavant.mlb.com/sporty-videos?playId=0446b81b-6f5d-39f2-b1c8-7795fb565cb8
[S10]: https://baseballsavant.mlb.com/sporty-videos?playId=b36b14a3-9ffd-330d-a003-bd1321e85e04

## MLB Film Room 경기 링크

[G776195]: https://www.mlb.com/video/game/776195
[G776262]: https://www.mlb.com/video/game/776262
[G776280]: https://www.mlb.com/video/game/776280
[G776510]: https://www.mlb.com/video/game/776510
[G776534]: https://www.mlb.com/video/game/776534
[G776636]: https://www.mlb.com/video/game/776636
[G776717]: https://www.mlb.com/video/game/776717
[G777065]: https://www.mlb.com/video/game/777065
[G777153]: https://www.mlb.com/video/game/777153
[G777229]: https://www.mlb.com/video/game/777229
[G777336]: https://www.mlb.com/video/game/777336
