# Merrill Kelly 장기 타석 Film Room Reel 후보

- 검수일: 2026-07-27
- 투수: Merrill Kelly (`player_id=518876`)
- 범위: 2025 MLB 정규시즌
- 선정 기준: 8구 이상 타석 중 승부 흐름이 뚜렷하고 결과가 서로 다른 후보
- 최종 추천: 10구 홈런, 10구 삼진, 10구 볼넷 각 1개

## 결론

| 추천 순위 | 날짜 | game_pk | 이닝 | AB | 타자 | 결과 | 총 투구 | 추천 이유 |
|---:|---|---:|---|---:|---|---|---:|---|
| 1 | 2025-06-15 | 777491 | 4회초 | 24 | Jake Cronenworth | 2점 홈런 | 10 | 0-2에서 5차례 파울로 버틴 뒤 10구째 홈런 |
| 2 | 2025-07-13 | 777115 | 4회말 | 33 | Jo Adell | 삼진 | 10 | 풀카운트에서 4구 연속 파울 뒤 체인지업 파울팁 삼진 |
| 3 | 2025-05-24 | 777786 | 7회말 | 51 | Nolan Arenado | 볼넷 | 10 | 1-2에서 4차례 파울을 걷어 낸 뒤 변화구가 바운드되어 볼넷 |

세 후보 모두 다음 검수를 통과했다.

- 공식 MLB live feed에서 해당 타석과 10개 투구의 순서·구종·`playId`를 확인했다.
- 30개 Baseball Savant 개별 영상에서 실제 재생을 시작해 모두 `readyState=4`, `currentTime>0`, `paused=false`를 확인했다.
- 30개 페이지 모두 `HOME Broadcast Video`와 `AWAY Broadcast Video` 옵션을 확인했다.
- 세 MLB Film Room 경기 페이지 모두 `Pitch by Pitch` 섹션을 확인했다.
- Reel 추가에는 MLB 로그인이 필요하므로 Reel을 생성하거나 계정 상태를 변경하지 않았다.

`pre-count`는 투구가 들어오기 전의 볼-스트라이크다. raw 구종 코드는 공식 feed 원문이며, `FC`는 커터, `FF`는 포심, `SI`는 싱커, `SL`은 슬라이더, `CU`는 커브, `CH`는 체인지업이다.

## 1위 — Jake Cronenworth, 10구 2점 홈런

- 날짜: 2025-06-15
- game_pk: `777491`
- 이닝: 4회초
- AB: `24`
- 타자: Jake Cronenworth
- 결과: 2점 홈런 — Xander Bogaerts 득점
- 총 투구: 10구
- 공식 feed: [game 777491 live feed](https://statsapi.mlb.com/api/v1.1/game/777491/feed/live)
- Film Room: [Padres at D-backs — Pitch by Pitch](https://www.mlb.com/video/game/777491)
- Film Room 검수: `Pitch by Pitch` 있음
- Reel 상태: **미생성 — MLB 로그인 필요**

추천 이유: 초구 스트라이크 뒤 곧바로 0-2가 됐지만 3~6구를 네 차례 연속 파울로 버텼다. 7구째 볼 뒤 8구를 다시 파울로 걷어 내고, 9구째 볼을 골라 2-2를 만든 다음 10구째 체인지업을 2점 홈런으로 연결한다. 세 후보 중 서사가 가장 강하고 마지막 결과도 가장 선명하다.

| 구 | pre-count | raw | 투구 결과 | playId | 공식 Savant 영상 | 영상 검수 |
|---:|---|---|---|---|---|---|
| 1 | 0-0 | FC | Called Strike | `fd55d110-d28c-361a-be21-1e513b60b231` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=fd55d110-d28c-361a-be21-1e513b60b231) | 실제 재생 · HOME/AWAY ✓ |
| 2 | 0-1 | FC | Foul | `1aea2de1-4b90-3501-b2ce-1f56a78badf0` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=1aea2de1-4b90-3501-b2ce-1f56a78badf0) | 실제 재생 · HOME/AWAY ✓ |
| 3 | 0-2 | FF | Foul | `f6430d0e-b5b1-3281-97a3-013daee24b0f` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=f6430d0e-b5b1-3281-97a3-013daee24b0f) | 실제 재생 · HOME/AWAY ✓ |
| 4 | 0-2 | CH | Foul | `14dffd39-d218-3920-b33f-87aeda928ee6` | [4구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=14dffd39-d218-3920-b33f-87aeda928ee6) | 실제 재생 · HOME/AWAY ✓ |
| 5 | 0-2 | FF | Foul | `f02c927f-ed6f-3f94-825a-f692bf55d125` | [5구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=f02c927f-ed6f-3f94-825a-f692bf55d125) | 실제 재생 · HOME/AWAY ✓ |
| 6 | 0-2 | CU | Foul | `e3eddbec-e6f4-359a-921e-0ecd18d7b6c3` | [6구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=e3eddbec-e6f4-359a-921e-0ecd18d7b6c3) | 실제 재생 · HOME/AWAY ✓ |
| 7 | 0-2 | FC | Ball | `52edd36c-89fb-3af6-9fd0-185544ca9257` | [7구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=52edd36c-89fb-3af6-9fd0-185544ca9257) | 실제 재생 · HOME/AWAY ✓ |
| 8 | 1-2 | FC | Foul | `5c030906-2f05-370c-9a81-a0940529ae20` | [8구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=5c030906-2f05-370c-9a81-a0940529ae20) | 실제 재생 · HOME/AWAY ✓ |
| 9 | 1-2 | FF | Ball | `00f1d414-7529-3536-92c3-fe0e764f6639` | [9구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=00f1d414-7529-3536-92c3-fe0e764f6639) | 실제 재생 · HOME/AWAY ✓ |
| 10 | 2-2 | CH | In play, run(s) | `1a32b39b-51f1-371c-a4b9-31bf02eeecfe` | [10구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=1a32b39b-51f1-371c-a4b9-31bf02eeecfe) | 실제 재생 · HOME/AWAY ✓ |

## 2위 — Jo Adell, 10구 파울팁 삼진

- 날짜: 2025-07-13
- game_pk: `777115`
- 이닝: 4회말
- AB: `33`
- 타자: Jo Adell
- 결과: 파울팁 삼진
- 총 투구: 10구
- 공식 feed: [game 777115 live feed](https://statsapi.mlb.com/api/v1.1/game/777115/feed/live)
- Film Room: [D-backs at Angels — Pitch by Pitch](https://www.mlb.com/video/game/777115)
- Film Room 검수: `Pitch by Pitch` 있음
- Reel 상태: **미생성 — MLB 로그인 필요**

추천 이유: 5구 만에 3-2가 된 뒤 Adell이 6~9구를 네 차례 연속 파울로 버틴다. Kelly가 싱커·슬라이더·체인지업을 섞은 끝에 10구째 체인지업으로 파울팁 삼진을 잡는다. 투수가 긴 승부를 이기는 사례로 1위 후보와 대비하기 좋다.

| 구 | pre-count | raw | 투구 결과 | playId | 공식 Savant 영상 | 영상 검수 |
|---:|---|---|---|---|---|---|
| 1 | 0-0 | SI | Ball | `41bafbd6-162a-380e-b554-7795306a6890` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=41bafbd6-162a-380e-b554-7795306a6890) | 실제 재생 · HOME/AWAY ✓ |
| 2 | 1-0 | FC | Foul | `d4efce76-e5b2-3a83-b732-139d26ebd8dd` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=d4efce76-e5b2-3a83-b732-139d26ebd8dd) | 실제 재생 · HOME/AWAY ✓ |
| 3 | 1-1 | SL | Swinging Strike | `851819e5-3aa0-3eaf-9b79-43e1d57cbd51` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=851819e5-3aa0-3eaf-9b79-43e1d57cbd51) | 실제 재생 · HOME/AWAY ✓ |
| 4 | 1-2 | CH | Ball | `9176f0c6-3588-3246-877f-1557a784062e` | [4구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=9176f0c6-3588-3246-877f-1557a784062e) | 실제 재생 · HOME/AWAY ✓ |
| 5 | 2-2 | FF | Ball | `20209e24-273b-332d-9e12-f3630f7f42e0` | [5구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=20209e24-273b-332d-9e12-f3630f7f42e0) | 실제 재생 · HOME/AWAY ✓ |
| 6 | 3-2 | CH | Foul | `0b2f9818-de65-3047-9249-8d8feb22d4e0` | [6구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=0b2f9818-de65-3047-9249-8d8feb22d4e0) | 실제 재생 · HOME/AWAY ✓ |
| 7 | 3-2 | SI | Foul | `2e57da19-4d33-3cad-b204-c5acce3c1cc7` | [7구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=2e57da19-4d33-3cad-b204-c5acce3c1cc7) | 실제 재생 · HOME/AWAY ✓ |
| 8 | 3-2 | SL | Foul | `657ef2b0-0a67-3077-ade7-fd29cff5db2c` | [8구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=657ef2b0-0a67-3077-ade7-fd29cff5db2c) | 실제 재생 · HOME/AWAY ✓ |
| 9 | 3-2 | SL | Foul | `fe24a253-601e-352c-a32c-1b5c46ad20ee` | [9구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=fe24a253-601e-352c-a32c-1b5c46ad20ee) | 실제 재생 · HOME/AWAY ✓ |
| 10 | 3-2 | CH | Foul Tip | `a86da716-e4c1-36fb-aa15-bcb938259fe4` | [10구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=a86da716-e4c1-36fb-aa15-bcb938259fe4) | 실제 재생 · HOME/AWAY ✓ |

## 3위 — Nolan Arenado, 10구 볼넷

- 날짜: 2025-05-24
- game_pk: `777786`
- 이닝: 7회말
- AB: `51`
- 타자: Nolan Arenado
- 결과: 볼넷 — Alec Burleson 2루 진루
- 총 투구: 10구
- 공식 feed: [game 777786 live feed](https://statsapi.mlb.com/api/v1.1/game/777786/feed/live)
- Film Room: [D-backs at Cardinals — Pitch by Pitch](https://www.mlb.com/video/game/777786)
- Film Room 검수: `Pitch by Pitch` 있음
- Reel 상태: **미생성 — MLB 로그인 필요**

추천 이유: Arenado가 3구째부터 1-2에 몰린 뒤 4~6구를 연속 파울로 걷어 낸다. 8구째 바운드된 슬라이더로 풀카운트를 만들고, 9구째 체인지업도 파울로 버틴 뒤 10구째 바운드된 커브를 골라 볼넷을 얻는다. 긴 승부가 안타나 삼진이 아닌 출루로 끝나는 대조군이다.

| 구 | pre-count | raw | 투구 결과 | playId | 공식 Savant 영상 | 영상 검수 |
|---:|---|---|---|---|---|---|
| 1 | 0-0 | FC | Called Strike | `e58b1ced-f7b4-3b88-9f47-c3ba3aae26d7` | [1구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=e58b1ced-f7b4-3b88-9f47-c3ba3aae26d7) | 실제 재생 · HOME/AWAY ✓ |
| 2 | 0-1 | CH | Ball | `8dfe3c2f-fea6-3e31-bf38-ebfd58db1d5c` | [2구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=8dfe3c2f-fea6-3e31-bf38-ebfd58db1d5c) | 실제 재생 · HOME/AWAY ✓ |
| 3 | 1-1 | SL | Foul Tip | `a455a30e-c8f4-3f4c-b7e5-5d098e12dc93` | [3구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=a455a30e-c8f4-3f4c-b7e5-5d098e12dc93) | 실제 재생 · HOME/AWAY ✓ |
| 4 | 1-2 | SL | Foul | `cb9fe4c3-060b-3d81-a563-fea18b832d21` | [4구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=cb9fe4c3-060b-3d81-a563-fea18b832d21) | 실제 재생 · HOME/AWAY ✓ |
| 5 | 1-2 | CH | Foul | `41b410a8-941d-363e-8feb-846a97640488` | [5구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=41b410a8-941d-363e-8feb-846a97640488) | 실제 재생 · HOME/AWAY ✓ |
| 6 | 1-2 | CH | Foul | `ca06f5a1-24cf-3679-b1ff-e4182aba8250` | [6구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=ca06f5a1-24cf-3679-b1ff-e4182aba8250) | 실제 재생 · HOME/AWAY ✓ |
| 7 | 1-2 | FF | Ball | `2d1493f0-6f6e-375c-8a11-ac60bd5ee8c6` | [7구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=2d1493f0-6f6e-375c-8a11-ac60bd5ee8c6) | 실제 재생 · HOME/AWAY ✓ |
| 8 | 2-2 | SL | Ball In Dirt | `51b87595-3d83-3517-8d2e-2e532c61fcdb` | [8구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=51b87595-3d83-3517-8d2e-2e532c61fcdb) | 실제 재생 · HOME/AWAY ✓ |
| 9 | 3-2 | CH | Foul | `2390f8c5-d15e-34e8-9ab7-88cf7edae92a` | [9구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=2390f8c5-d15e-34e8-9ab7-88cf7edae92a) | 실제 재생 · HOME/AWAY ✓ |
| 10 | 3-2 | CU | Ball In Dirt | `20f08df3-96c3-344b-92d8-869a5c9bd1a3` | [10구 재생](https://baseballsavant.mlb.com/sporty-videos?playId=20f08df3-96c3-344b-92d8-869a5c9bd1a3) | 실제 재생 · HOME/AWAY ✓ |

## 조사·선정 방법

로컬 `data/raw/statcast/2025-*.parquet`에서 `pitcher=518876`을 필터링하고 `game_pk + at_bat_number`로 묶어 `max(pitch_number) >= 8`인 타석을 찾았다. 2025 정규시즌 후보 중 구수가 가장 길고 결과가 겹치지 않는 10구 타석 세 개를 우선했다.

후보의 타자·결과·투구 순서·구종·`playId`는 각 경기의 공식 MLB live feed로 다시 확인했다. feed의 투구 이벤트에 기록된 count는 해당 투구 뒤의 count이므로, 표의 `pre-count`는 직전 투구 이후 count를 이어 받아 계산했으며 초구는 `0-0`으로 두었다. 구종 필드 해석은 [Statcast CSV 문서](https://baseballsavant.mlb.com/csv-docs)를 참고했다.

## 사용 시 주의

- 개별 Savant 페이지 제목은 마지막 투구의 타석 결과를 노출할 수 있다. 예측 응답 전에는 URL·페이지 제목·브라우저 탭 제목이 참가자에게 보이지 않게 해야 한다.
- 공식 영상을 재호스팅하거나 다운로드하는 대신 MLB가 제공하는 페이지와 Film Room 흐름을 사용한다.
- 공개 직전에 30개 링크의 실제 재생과 방송 옵션을 다시 확인한다.
- Reel 생성은 MLB 계정 로그인 후의 별도 작업이다. 이 문서는 후보와 순서만 확정하며 공유 가능한 Reel URL을 제공하지 않는다.
