# Ubuntu CPU 서버 · HTTPS · 배포 설정

## 아직 필요한 정보와 작업

이 문서는 집 Ubuntu 컴퓨터 + Docker Compose + **호스트에 설치하는 Caddy** 기준입니다. 실제 서버 IP/SSH 계정/키/구매한 도메인이 제공되지 않아 서버 접속·설치·DNS·인증서 발급·실제 배포는 수행하지 않았습니다. 아래 예시 주소는 반드시 본인 값으로 바꿉니다.

## 1. 서버 준비

서버에 Git, curl, Docker Engine, Docker Compose v2, Caddy를 설치하고 확인합니다. Compose의 `up --wait`와 복수 `--env-file` 옵션을 지원하는 버전을 사용합니다.

```bash
git --version
docker version
docker compose version
caddy version
```

설치 안내: https://docs.docker.com/engine/install/ubuntu/ 와 https://caddyserver.com/docs/install#debian-ubuntu-raspbian

전용 SSH 배포 계정을 사용하세요. 그 계정이 Docker를 사용할 수 있어야 합니다. Docker 그룹 권한은 사실상 root 수준의 호스트 제어 권한이므로 일반 사용자 모두에게 부여하지 마세요. Docker 데몬 포트를 외부에 공개하지 않습니다.

PR을 main에 병합한 다음 배포 계정의 홈 폴더에서:

```bash
cd ~
git clone https://github.com/16minsun-afk/-.git yolo-api
cd ~/yolo-api
mkdir -p models/v1
cp deploy/.env.example .env
chmod 600 .env
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
```

생성한 값을 **서버의 `.env` 파일 `API_TOKEN=` 뒤에만** 입력합니다. 채팅, GitHub 코드, 로그 또는 Android 소스에 복사하지 마세요. `MODEL_MIN_MAP=0.5`는 예시 합격 기준이며 실제 모델 선정 기준과 맞춥니다. 모델 크기와 서버 메모리에 따라 compose.yaml의 4 GiB 메모리/CPU 2개 제한을 조정하세요.

## 2. 검증된 모델 두 파일 전송

Windows에서 `validate.py`가 성공해 `artifacts/v1`이 생긴 뒤, 실제 서버 계정/주소로 바꾸어 실행합니다. 비표준 SSH 포트는 scp의 `-P 포트`를 추가합니다.

```powershell
scp artifacts/v1/best.pt artifacts/v1/manifest.json 사용자@서버주소:~/yolo-api/models/v1/
```

서버에서 컨테이너의 UID 10001이 읽을 수 있도록 합니다. 파일 쓰기 권한은 배포 계정만 유지합니다.

```bash
cd ~/yolo-api
chmod 755 models models/v1
chmod 644 models/v1/best.pt models/v1/manifest.json
```

첫 모델을 실수로 덮어쓰지 마세요. 다음 배포에서는 새 `models/v2` 폴더를 사용합니다. 모델과 manifest가 다르면 배포 검사가 실패합니다. 합성 CI 전용 모델은 운영 설정에서 거부됩니다.

## 3. GHCR 읽기 권한

main CI의 publish가 완료되어 이미지가 있어야 합니다. 새 패키지는 비공개일 수 있습니다. 비공개 패키지라면 배포 계정으로 GHCR에 로그인해야 합니다. 패키지 읽기 권한이 있는 전용 토큰을 준비하고 서버에서만 사용합니다.

```bash
read -rsp 'GHCR read token: ' GHCR_TOKEN; echo
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u 16minsun-afk --password-stdin
unset GHCR_TOKEN
```

이 인증은 **워크플로의 GITHUB_TOKEN과 별개**입니다. Docker의 인증 파일 접근 권한도 보호하세요. 체크포인트와 개인 데이터는 GHCR 이미지에 들어가지 않습니다.

## 4. GitHub production 환경과 Secrets

저장소 → Settings → Environments → New environment → `production`을 만듭니다. 허용 배포 브랜치를 main으로 제한하고 필요한 승인자를 지정합니다. 워크플로에 environment 이름을 적는 것만으로 승인자·보호 규칙이 자동 설정되는 것은 아닙니다.

이 환경의 Secrets에 다음을 직접 등록합니다. 실제 키를 채팅으로 보내지 마세요.

| 이름 | 값 |
|---|---|
| DEPLOY_HOST | GitHub-hosted runner에서 도달 가능한 서버 DNS 이름 또는 IPv4. 프로토콜/경로 없이 입력 |
| DEPLOY_USER | Ubuntu 전용 배포 계정 |
| DEPLOY_PORT | SSH 외부 접속 포트. 기본이면 22 |
| DEPLOY_SSH_KEY | 해당 계정에만 쓰는 전용 비암호화 SSH 개인키. 공개키는 서버 authorized_keys에 등록 |
| DEPLOY_KNOWN_HOSTS | 신뢰할 수 있는 경로로 확인한 서버 공개 호스트 키의 known_hosts 형식 |

호스트 키는 서버 콘솔에서 `/etc/ssh/ssh_host_ed25519_key.pub`의 **공개키**와 fingerprint를 확인하세요. 개인 호스트 키 파일을 복사하면 안 됩니다. known_hosts는 기본 포트에서 `서버주소 ssh-ed25519 AAAA...`, 다른 포트에서는 `[서버주소]:포트 ssh-ed25519 AAAA...` 형식입니다. 단순 네트워크 스캔 결과를 확인 없이 신뢰하거나 `StrictHostKeyChecking=no`를 쓰지 마세요.

Settings의 main 보호 규칙에는 PR 검토와 CI의 `test` check 성공을 필수로 지정하는 것을 권장합니다. CI가 한 번 실행된 뒤 실제 표시되는 check 이름을 선택하세요. 이 저장소 작업은 보호 규칙·Secrets·서버 계정 자체를 생성하지 않습니다.

## 5. 첫 배포와 롤백

Actions → 성공한 main CI → Summary에서 이미지 전체 주소를 확인합니다. 예시 형식은 `ghcr.io/16minsun-afk/yolo-api@sha256:<64자리>`입니다.

Actions → Deploy → Run workflow:

- branch: `main`
- image_digest: `sha256:`를 제외한 64자리 hex
- model_version: `v1`

필요한 승인을 마치면 SSH로 접속해 새 이미지 pull, 모델 SHA/품질 검사, CPU 워밍업을 실행합니다. 사전 검사가 성공하기 전까지 이전 컨테이너를 교체하지 않습니다. 이후 readiness가 실패하면 이전 이미지·모델 선택으로 복구를 시도합니다. 첫 배포에는 이전 버전이 없으므로 실패한 서비스를 정지하고 오류를 보고합니다.

서버에서 동일 절차를 직접 실행하는 방법:

```bash
cd ~/yolo-api
bash scripts/deploy.sh 'ghcr.io/16minsun-afk/yolo-api@sha256:실제64자리digest' v1
```

상태와 로그:

```bash
cd ~/yolo-api
docker compose --env-file .env --env-file deployment.env ps
docker compose --env-file .env --env-file deployment.env logs --tail=100 api
curl --fail http://127.0.0.1:8000/readyz
```

이전 버전으로 되돌릴 때도 이전 digest와 model_version을 Deploy에 넣어 같은 검사를 실행합니다. 롤백은 이미지/모델 선택에 대한 것이며, OS·Caddy·`.env`·DB 같은 외부 상태 전체를 복구하지 않습니다. Compose 교체 과정에 짧은 중단이 생길 수 있고 무중단 배포는 아닙니다. 워크플로는 현재 서버의 compose.yaml을 사용하므로 이 파일의 변경은 내용을 검토한 뒤 서버에서 `git pull --ff-only`로 반영하세요.

## 6. Caddy HTTPS와 외부 네트워크

구매한 도메인의 DNS를 서버의 공인 주소에 맞추고, Caddy가 외부 80/443 요청을 받을 수 있도록 라우터/NAT와 방화벽을 구성합니다. CGNAT나 이중 NAT가 있으면 단순 포트포워딩만으로 접근되지 않을 수 있으므로 회선을 먼저 확인하세요. SSH도 GitHub runner에서 도달 가능해야 하며, 인증키·접근 제한 또는 별도의 관리망을 사용하세요. 이 프로젝트는 SSH 접속을 위해 사설망을 자동 구성하지 않습니다.

서버의 `deploy/Caddyfile`에서 `api.example.com`을 실제 도메인으로 바꾼 다음 Caddy 설정으로 적용합니다. 기존 `/etc/caddy/Caddyfile`이 있다면 먼저 백업하고 기존 사이트를 보존하면서 통합하세요.

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
sudo systemctl status caddy --no-pager
curl --fail https://본인도메인/readyz
```

Caddy 자동 HTTPS는 DNS/외부 도달성 조건이 맞아야 동작합니다. 이 저장소를 추가하는 것만으로 도메인이나 인증서가 생성되지는 않습니다. FastAPI는 `127.0.0.1:8000`에만 공개하고 **공유기에 8000번 포트포워딩을 만들지 않습니다**. 외부에서 노출되는 웹 포트는 Caddy의 80/443입니다.

Android 휴대전화의 Wi-Fi를 끄고 LTE/5G에서 `https://본인도메인/readyz`가 열리는지 먼저 확인한 뒤, 앱은 multipart의 필드 이름 `file`로 `/predict`에 이미지를 보냅니다. `/readyz` 성공은 HTTPS/API 준비 확인이지 인증된 이미지 업로드 전체 검증은 아니므로 실제 이미지 요청도 별도로 테스트하세요.

HTTPS는 전송을 보호할 뿐 사용자 인증을 대체하지 않습니다. 공통 API 키는 비공개 개발 테스트에만 사용하고, 플레이스토어 공개 앱용 사용자 인증·사용량 제한은 별도로 구현해야 합니다.

참고: https://caddyserver.com/docs/automatic-https 와 https://docs.docker.com/engine/network/port-publishing/
