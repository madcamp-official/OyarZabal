# Human benchmark API 배포

현재 프런트엔드와 같은 KCloud VM에서 API를 `127.0.0.1:8000`으로 실행하고,
Nginx가 `/api/` 요청만 프록시하는 구성이다. 참가자가 입력한 닉네임과 응답은
SQLite에 저장된다.

## 1. 환경 준비

```bash
cd /opt/oyarzabal
uv sync --extra benchmark
sudo install -d -o oyarzabal -g oyarzabal /var/lib/oyarzabal
sudo openssl rand -hex 32 -out /etc/oyarzabal-benchmark-token
sudo chown oyarzabal:oyarzabal /etc/oyarzabal-benchmark-token
sudo chmod 600 /etc/oyarzabal-benchmark-token
```

`/opt/oyarzabal/.env`를 만든다.

```dotenv
BENCHMARK_ADMIN_TOKEN_FILE=/etc/oyarzabal-benchmark-token
BENCHMARK_DB_PATH=/var/lib/oyarzabal/benchmark.sqlite3
BENCHMARK_ALLOWED_ORIGIN=https://www.pitchtest.madcamp-kaist.org
BENCHMARK_TRUST_PROXY=1
BENCHMARK_RATE_LIMIT=30
BENCHMARK_DAILY_RATE_LIMIT=120
```

`.env`는 Git에 커밋하지 않는다.

```bash
sudo chown oyarzabal:oyarzabal /opt/oyarzabal/.env
sudo chmod 600 /opt/oyarzabal/.env
```

## 2. 서비스와 Nginx 연결

`deploy/oyarzabal-benchmark.service.example`의 사용자와 경로가 VM과 일치하는지
확인한 뒤 systemd에 설치한다.

```bash
sudo cp deploy/oyarzabal-benchmark.service.example /etc/systemd/system/oyarzabal-benchmark.service
sudo systemctl daemon-reload
sudo systemctl enable --now oyarzabal-benchmark
sudo systemctl status oyarzabal-benchmark
```

기존 HTTPS `server` 블록에 `deploy/nginx-benchmark.conf.example`의 두
`location` 블록을 반영한다. 기존 `location /`이 있으면 새로 추가하지 말고
`try_files $uri $uri/ /index.html;`만 유지한다.

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 3. 확인

```bash
curl https://www.pitchtest.madcamp-kaist.org/api/health
```

`{"ok":true}`가 나오면 참가자 페이지는 완료 시 자동으로 결과를 저장한다.
관리자 화면은 다음 주소에서 환경변수의 관리자 토큰으로 연다.

`https://www.pitchtest.madcamp-kaist.org/admin/human-benchmark`

SQLite 백업 대상은 `/var/lib/oyarzabal/benchmark.sqlite3`와 같은 디렉터리의
`-wal`, `-shm` 파일이다. 실행 중 복사하는 대신 `sqlite3`의 `.backup` 명령이나
서비스를 잠시 정지한 상태에서 백업한다.

## 4. Daily Pitch Test 자동 생성

Daily 모드는 매일 15:00 KST에 전날 MLB 공식 경기를 먼저 확인하고, Savant
플레이어가 모든 투구에 실제 생성된 3~6구 타석 세 개를 골라 공개한다. 전날
영상이 아직 처리 중이면 최대 3일 안의 가장 최근 영상 준비 완료 경기일을 쓴다.
각 투구 직전 상태로 V8.4 예측을 미리 계산해 DB에 저장하고, 사용자가 답을
확정한 뒤에만 모델 예측과 실제 결과를 함께 공개한다.

```bash
sudo cp deploy/oyarzabal-daily.service.example \
  /etc/systemd/system/oyarzabal-daily.service
sudo cp deploy/oyarzabal-daily.timer.example \
  /etc/systemd/system/oyarzabal-daily.timer
sudo systemctl daemon-reload
sudo systemctl enable --now oyarzabal-daily.timer
sudo systemctl start oyarzabal-daily.service
sudo systemctl status oyarzabal-daily.service
```

수동 생성은 같은 환경에서 다음 명령으로 실행한다.

```bash
uv run oyarzabal-generate-daily
```
