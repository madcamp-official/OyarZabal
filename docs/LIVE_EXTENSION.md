# Gameday 라이브 확장 프로그램

MLB Gameday 옆 Chrome Side Panel에서 사용자와 V7.2 모델이 다음 구종을
예측하는 로컬 프로토타입이다.

경기 상황 패널은 완료 경기 리플레이에서만 표시한다. 실제 라이브 경기에서는
Gameday 자체 상황판과 중복되므로 MLB feed 기준 이닝·볼카운트·타자·이번 타석의
다음 투구 번호만 작게 표시한다.

## 실행

저장소 루트에서 API를 실행한다.

```bash
uv sync
uv run python -m oyarzabal.live_api
```

기본 설정은 다음 파일을 사용한다.

- V7.2 Global·보정 bias·pooled residual·context gate: `models/v7.2`
- 과거 기록: `data/raw/statcast-v8`, `data/holdout/statcast-v8-2026`
- 답안 DB: `data/live/live.sqlite3`
- API: `https://api.pitchtest.madcamp-kaist.org`

경로는 환경 변수로 바꿀 수 있다.

```bash
OYARZABAL_LIVE_MODEL_DIR=models/v7.2
OYARZABAL_HISTORY_DIRS=data/raw/statcast-v8:data/holdout/statcast-v8-2026
OYARZABAL_LIVE_DB_PATH=data/live/live.sqlite3
```

라이브 예측 제한시간은 서버 기준 8초로 고정한다.

## Chrome에 설치

1. Chrome에서 `chrome://extensions`를 연다.
2. 오른쪽 위 `개발자 모드`를 켠다.
3. `압축해제된 확장 프로그램을 로드합니다`를 누른다.
4. 저장소의 `extension` 디렉터리를 선택한다.
5. `mlb.com/gameday/.../live` 경기 페이지를 연다.
6. OyarZabal 확장 프로그램 아이콘을 눌러 Side Panel을 연다.

Side Panel은 열린 탭의 URL에서 `gamePk`만 읽는다. 진행 중인 API 요청이 끝나면
최대 0.25초 안에 다음 조회를 시작한다. API는 매 요청에서 MLB Stats API live
feed의 `currentPlay`를 읽어 이닝·카운트와 다음 투구 문맥을 만든다. 따라서
영상이나 Gameday DOM과 프레임 단위로 맞추는 방식이 아니며, MLB feed 반영
시각에 polling·네트워크 시간이 더해질 수 있다.
새 투구 문맥의 모델 계산이 끝나면 feed를 다시 확인하고, 계산 도중 다음 투구로
진행됐다면 낡은 예측을 응답하지 않고 최신 상태를 다시 계산한다.

## 대결 규칙

직전 투구가 feed에 반영되고 다음 투수·타자·카운트가 확정되면 서버가 모델
예측을 생성하고 8초 대결을 연다.

V7.2는 과거 Statcast 기록과 현재 Gameday 문맥으로 Global 확률을 만든 뒤
보정 bias, pooled residual, context gate, Registry tier·scale과 안전 cap을
순서대로 적용한다. Registry 비활성 투수나 안전 gate를 통과하지 못한 문맥은
정식 V7.2 라우팅 규칙에 따라 Global 결과를 유지한다.

- 사용자는 3개 구종 계열을 먼저 고른 뒤, 해당 계열의 상세 구종 2개 중
  하나를 선택한다. 상세 구종 선택은 즉시 서버에 임시 저장되며 8초 안에는
  자유롭게 바꿀 수 있다.
- 서버 deadline 또는 실제 투구 수신 전에는 답안 저장 여부와 관계없이 모델
  예측을 숨긴다.
- 8초가 지나거나 실제 투구가 먼저 들어오면 마지막 저장 답안을 잠그고,
  모델의 3개 계열 확률과 계열별 2개 상세 구종 확률을 공개한다.
- 사용자와 모델은 상세 구종 적중 3점, 같은 계열 적중 1점으로 경쟁하며
  누적 상세·계열 정확도를 함께 표시한다.
- 답을 고르지 않은 투구도 사용자 0점·오답으로 기록해 Accuracy의 분모에
  포함한다.
- 리플레이 진행 모드와 점수는 Game PK별로 구분한다. 다른 탭이나 경기를
  다녀와도 해당 경기의 진행 위치와 점수를 다시 불러오며, 새 리플레이를
  시작하거나 `처음부터`를 누르면 그 경기 점수만 0으로 초기화한다.
- `통계 보기` 팝업에서 모델 입력에 쓰인 시즌·최근 100구·현재 경기
  구사율, 투수의 실제 구종 목록과 투구 운용 상태를 확인한다.
- 다음 실제 pitch 이벤트가 들어오면 사용자와 모델을 함께 채점한다.
- 직전 투구 결과에는 해당 타석의 몇 구째였는지도 함께 표시한다.
- 투수·타자·카운트·주자 상태가 투구 없이 바뀌면 기존 대결을 취소한다.
- 실제 결과가 먼저 들어온 대결에는 답을 제출할 수 없다.

자동 볼·스트라이크는 카운트만, 견제는 주자 상태만 갱신한다. 구종이 없는
pitch 이벤트는 채점에서 제외한다.

## 완료 경기 테스트 모드

완료된 Gameday 페이지에서도 Side Panel의 `테스트 시작`을 누르면 서버가
완료 feed를 첫 투구 직전 상태부터 잘라서 재생한다.

1. 계열과 상세 구종을 선택한다. 상세 구종을 고르면 즉시 잠긴다.
2. 공개된 V7.2 예측과 비교한다.
3. `실제 투구 공개`를 눌러 채점하고 다음 투구로 이동한다.
4. `처음부터`는 replay cursor와 로컬 점수를 초기화한다.

Replay cursor와 답안은 테스트용으로 서버 메모리에만 저장된다. 서버를
재시작하면 진행 중인 replay session도 사라진다.
현재 프로토타입은 inning·count·score와 투구 직전 주자 이동을 복원한다.
실제 live feed의 수신 지연은 이 모드에서 측정할 수 없다.

## 확인

```bash
uv run pytest -q ml/tests/test_live.py
uv run ruff check ml/oyarzabal/live.py ml/oyarzabal/live_api.py \
  ml/tests/test_live.py
node --check extension/service-worker.js
node --check extension/sidepanel.js
```

완료 경기 Gameday에서는 `종료된 경기입니다`가 표시된다. 실제 갱신 시차는
라이브 경기 Shadow 운영으로 별도 측정해야 한다.
