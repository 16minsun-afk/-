# YOLO 학습 · 검증 · CI/CD

Windows RTX 5070 Ti Laptop에서 학습하고, Ubuntu CPU 서버에서 Docker + **호스트 Caddy**로 추론하는 기본 프로젝트입니다. Python 3.12, YOLO26m을 기본값으로 사용합니다. 기존 저장소에는 학습 코드가 없어 새로 추가한 출발점이며, 실제 데이터와 학습된 모델은 포함하지 않습니다.

```text
Windows/PyCharm + CUDA GPU
  train.py → best.pt → validate.py → artifacts/v1/{best.pt, manifest.json}
                                  ↓ 검증된 두 파일만 서버에 전송
GitHub PR → lint / 42개 테스트 / 합성 데이터 1 epoch / Docker 실행 검사
main CI 통과 → ghcr.io/16minsun-afk/yolo-api:sha-<commit> 이미지 발행
Actions의 Deploy 수동 실행 + production 환경 승인
  → SSH → 모델 해시·성능 기준·워밍업 → Compose 교체 → readiness 검사
  → 실패 시 이전 이미지 + 모델 선택으로 복구 시도
Android → HTTPS → 호스트 Caddy → 127.0.0.1:8000 → CPU FastAPI
```

## 1. Windows 학습 환경

기존 `C:\Users\dimpk\PycharmProjects\PythonProject`를 덮어쓰지 않도록 처음에는 옆 폴더에 받습니다. PR 병합 전 검토하려면 clone 후 `git switch feat/yolo-training-cicd`를 실행합니다.

```powershell
cd C:\Users\dimpk\PycharmProjects
git clone https://github.com/16minsun-afk/-.git yolo-cicd
cd yolo-cicd
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU unavailable')"
```

PyCharm 인터프리터를 이 폴더의 `.venv\Scripts\python.exe`로 지정합니다. CUDA가 `False`이면 실제 학습을 시작하지 말고 NVIDIA 드라이버와 가상환경을 확인합니다. 학습은 CUDA 12.8 wheel, CI/서버는 CPU wheel을 **먼저** 설치합니다. 주요 직접 의존성을 고정했으며, 전체 전이 의존성의 해시 잠금까지 구현한 것은 아닙니다.

## 2. 데이터 준비 → 학습 → 검증

```text
dataset/
├── images/train/  a.jpg
├── images/val/    b.jpg
├── labels/train/  a.txt
└── labels/val/    b.txt
```

`data.example.yaml`을 `data.local.yaml`로 복사하고 **본인 데이터의 클래스명과 경로**로 바꾸세요. plastic/can/paper는 예시일 뿐입니다. train/val은 필수이며 test를 별도로 추가할 수 있습니다. 이 검사기는 이미지 디렉터리 방식과 YOLO 객체 탐지의 `class x y width height` 라벨만 지원합니다. segmentation/pose 라벨이나 이미지 목록 TXT는 변환 없이 지원하지 않습니다.

```powershell
Copy-Item data.example.yaml data.local.yaml
# 실제 이미지와 라벨을 배치한 뒤 실행합니다.
.\.venv\Scripts\python.exe train.py --data data.local.yaml --check-only
.\scripts\train.bat --data data.local.yaml --model yolo26m.pt --epochs 200 --imgsz 640 --batch 16 --device 0 --workers 4 --name yolo26m_custom
```

첫 실행은 공식 사전학습 가중치를 내려받을 수 있습니다. GPU 메모리가 부족하면 batch를 8 또는 4로 낮춥니다. `patience=30`이므로 개선이 없으면 200 epochs 전에 끝날 수 있습니다. 같은 name을 재사용하면 결과 폴더에 번호가 붙을 수 있으므로 **콘솔에 출력된 실제 best.pt 경로**를 사용하세요.

```powershell
.\.venv\Scripts\python.exe validate.py --model runs/train/yolo26m_custom/weights/best.pt --data data.local.yaml --split val --device 0 --min-map 0.5 --out artifacts/v1
```

`0.5`는 mAP50-95의 **초기 예시 합격 기준**입니다. 달성 보장이나 프로젝트의 확정 목표가 아닙니다. 용도·클래스별 성능·오탐/미탐을 보고 기준을 정하고 서버의 `MODEL_MIN_MAP`에도 반영하세요. 최종 성능 보고에는 별도로 보관한 test 데이터와 `--split test`를 사용하고, test로 반복 튜닝하지 마세요.

검증 결과는 `reports/validation.json`에 기록됩니다. 합격한 경우에만 새 버전 폴더에 `best.pt`와 `manifest.json`이 생성됩니다. 버전 폴더를 덮어쓰지 않으므로 다음 모델에는 `artifacts/v2`처럼 새 이름을 씁니다. 이미지/라벨 SHA-256으로 데이터 버전을 기록하고, 모델 해시로 평가한 파일과 배포 파일을 연결합니다. 이 manifest는 전자서명이 아닙니다. 본인이 만든 신뢰할 수 있는 `.pt`와 보고서만 사용하세요.

검사 항목: 누락 라벨, 클래스 ID, 좌표 범위, 이미지 손상, train/val/test 간 **동일 파일 내용** 중복. 비슷한 사진, 연속 영상 프레임, 같은 사람/장소의 누수까지 자동 검출하는 것은 아니므로 분할을 직접 검토해야 합니다.

## 3. CI와 CD의 역할

- **CI:** PR, main push, 수동 실행에서 Ruff·pytest·Bash 문법을 검사합니다. 작은 합성 데이터로 YOLO26n 1 epoch 학습·검증·저장·재로딩을 실행하고, CPU Docker 컨테이너의 `/readyz`와 실제 모델 `/predict`를 검사합니다. 합성 데이터 결과는 실제 모델 정확도 검증이 아닙니다.
- **이미지 발행:** main push의 CI가 모두 성공한 뒤에만 GHCR에 코드를 발행합니다. 데이터·모델·`.env`는 Git과 Docker 이미지에 포함하지 않습니다. Actions 요약에서 `...@sha256:<digest>`를 확인합니다.
- **CD:** Actions → Deploy → Run workflow에서 main, image_digest, model_version을 선택합니다. 서버 준비와 production 환경 설정은 [서버 설정 문서](docs/SETUP_KO.md)를 먼저 따라야 합니다. PR만 열었다고 실제 서버가 배포되지는 않습니다.

실제 4,000장/200 epochs 학습은 로컬 GPU에서 명시적으로 실행합니다. 공개 저장소에 개인 노트북/운영 서버를 self-hosted runner로 등록하지 않았습니다. Actions가 노트북 GPU를 자동으로 빌려 쓰는 구조가 아닙니다.

## 4. 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

배포 스크립트의 Docker/SSH 서버 동작은 단위 테스트에서 모의 처리합니다. Windows에서 bash/flock이 없으면 해당 Linux 테스트 4개는 건너뛰고 Ubuntu CI에서 실행합니다. 실제 모델·컨테이너 통합 검사는 CI의 별도 단계입니다.

## 5. 범위와 보안

API는 `/healthz`, `/readyz`, 인증이 필요한 `POST /predict`를 제공합니다. 10 MiB 파일 제한, 요청 본문 제한, 이미지 형식·픽셀 수 검사, CPU 동시 추론 제한을 포함합니다. 모델 로딩 또는 품질 검사가 실패하면 정상 준비 상태로 표시하지 않습니다.

현재 `X-API-Key`는 **개발·비공개 파일럿용 공통 키**입니다. 이 키를 Android APK에 넣어 일반 사용자에게 배포하면 안 됩니다. 공개 앱에는 별도 로그인 서버의 짧은 수명 토큰/JWT, 사용자별 권한·할당량, 남용 방지, 개인정보·보관 정책이 필요합니다. 이 저장소만으로 공개 서비스 운영 준비가 모두 끝나는 것은 아닙니다.

Dependencies/Actions/Docker 업데이트 PR은 Dependabot이 제안합니다. 자동 병합은 하지 않습니다. main 보호 규칙과 production 승인 정책은 GitHub Settings에서 별도로 설정해야 합니다.

### 공식 참고

- YOLO 학습: https://docs.ultralytics.com/modes/train/
- YOLO 검증: https://docs.ultralytics.com/modes/val/
- PyTorch wheel 조합: https://pytorch.org/get-started/previous-versions/
- GitHub runner 보안: https://docs.github.com/en/actions/reference/security/secure-use
- 배포 전 Ultralytics 및 데이터셋의 라이선스도 확인하세요: https://www.ultralytics.com/license
