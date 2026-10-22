# RTX 5090 로컬 파인튜닝

기존 R06의 **weak-only head warm-up 1 epoch → Swin 마지막 stage와 norm 1 epoch 학습 → 특징 재추출 → Gold58 3-fold × 3-seed 평가**를 Windows 로컬 CUDA GPU에서 실행한다. 전체 Swin을 학습하는 U04/U05는 이 실행기의 대상이 아니다. 실제 학습 결과가 나오기 전에는 성능 개선을 주장하지 않는다.

학습 계산은 기존 `experiments/spatial_train.py`를 재사용한다. 로컬 실행기는 Kaggle 경로를 찾는 두 함수만 해당 프로세스 안에서 연결한다. 실행 중인 Kaggle 캠페인의 소스 hash, 대기 순서와 상태 파일은 변경하지 않는다. 로컬 실험 ID는 `r06_local_warmup`이며 Kaggle R06과 별도로 기록한다.

## 1. GPU 환경 준비

프로젝트 루트 `D:\RSNA_Abnormality_Detection`에서 실행한다. 현재 PC는 **기존 Python 3.11.9 + PyTorch 2.7.1+cu128 + torchvision 0.22.1+cu128**을 재사용한다. 해당 Python으로 `.venv-5090`을 `--system-site-packages` 방식으로 만들어 기존 GPU PyTorch를 공유하고, pandas 3.0.6·pydicom 3.0.2·python-gdcm 3.2.6만 추가했다. `.venv`의 CPU PyTorch와 구분한다. 이 환경은 기반 Python 3.11.9의 설치 패키지에 의존한다.

2026-10-22 검증: RTX 5090에서 GPU forward/backward 통과. S11 v1의 bundle·feature cache SHA256 검증, TorchScript 읽기, eager Swin 가중치 strict 복원도 통과했다. GPU 무작위 입력 2건에서 TorchScript와 eager 출력의 최대 절대 오차는 **0.0**이었다. 이는 아래 3건 smoke 및 전체 학습 검증과 별개다. [환경 호환성 기록](../outputs/local/r06-python311-compatibility.json).

현재 환경은 준비됐으므로 아래 설치 스크립트를 다시 실행할 필요가 없다. **새 GPU 환경을 처음 구성할 경우**에는 Python 3.13 64bit와 다음 스크립트를 사용할 수 있다. 이 스크립트는 PyTorch/torchvision을 업그레이드하므로 현재 검증 버전을 유지할 때는 실행하지 않는다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local_gpu.ps1
# Python launcher를 쓰지 않는 경우
# powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local_gpu.ps1 -PythonExe C:\Python313\python.exe
```

공식 CUDA 13.0 PyTorch/torchvision과 프로젝트 의존성을 설치하고, GPU 행렬 연산의 forward/backward를 확인한다. CUDA Toolkit 전체를 별도로 설치하는 절차는 포함하지 않는다. [PyTorch 지원표](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix)

2026-10-22 확인: RTX 5090 32GB, RAM 64GB, D드라이브 여유 약 1TB. 전체 학습 영상의 준비 여부는 별도로 확인해야 한다. **이 코드를 저장한 것만으로 전체 학습이 시작되지는 않는다.**

## 2. 입력 준비

| 경로 | 필요한 내용 |
|---|---|
| `data/raw/` | `train.csv`, `train_series.csv`, `train_series/<StudyUID>/<SeriesUID>/*.dcm` |
| `data/weak_labels/report_labels_v2.csv` | 기존 Pilkwang 약라벨 v1의 파일. S11 manifest와 SHA256이 같아야 함 |
| `data/local-assets/s11/run/` | 기존 S11 **train v1**의 전체 `run` 출력: `tokens.pt`, `bundle/assets.json`, `bundle/encoder.ts`, `bundle/model.pt`, `bundle/wheels/` 등 |

기존 로컬 `outputs/automation/spatial/s11_cross/train/run/`에는 지표와 manifest만 있다. 2026-10-22 S11 train의 현재 버전이 1임을 확인하고, 전체 산출물을 `data/local-assets/s11/run/`에 내려받아 기존 manifest와 파일 해시가 일치함을 확인했다. Kaggle 개인 notebook 접근 권한 및 API 인증은 기존 사용자 환경에서 제공한다. 토큰은 스크립트·Git에 넣지 않는다.

현재 Kaggle S11 출력 버전이 1인지 확인한 다음 다운로드한다. CLI output은 최신 버전을 받으므로, 이후 notebook을 다시 실행했다면 먼저 Kaggle에서 버전과 입력을 확인한다.

```powershell
.\.venv\Scripts\kaggle.exe kernels status choiyoungwhan/rsna-knee-s11-cross-train
.\.venv\Scripts\kaggle.exe kernels output choiyoungwhan/rsna-knee-s11-cross-train -p data/local-assets/s11
```

대회 영상은 Kaggle의 대회 다운로드 기능으로 받아 `data/raw/train_series/`에 풀어둔다. 전체 DICOM 목록은 약 570GB이며, 압축파일과 추출본·특징 캐시가 동시에 차지하는 공간을 고려한다. **실행기가 자동으로 대용량 데이터를 다운로드하거나 기존 파일을 삭제하지 않는다.** 원본 bundle의 Linux용 DICOM wheel은 Kaggle 제출용이므로 보존하고, Windows 로컬에서는 환경 설치 단계에서 받은 Windows용 decoder를 사용한다.

## 3. 준비 확인 → 소량 학습 → 전체 학습

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_training.ps1 -Mode check
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_training.ps1 -Mode smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_training.ps1 -Mode train
```

- `check`: CUDA 연산, parent bundle/약라벨/cache hash, 4,350건·Gold58, cache ID와 전처리, 선택되는 MRI series의 파일 존재를 검증한다. 학습하거나 출력 폴더를 만들지 않는다. DICOM 전체의 decode 무결성을 보장하는 검사는 아니다.
- `smoke`: 원래 report hash 분할의 **weak train 2건 + weak validation 1건**으로 warm-up과 실제 backward·optimizer update·validation을 수행한다. Gold는 제외한다. 전체 CSV와 parent feature cache는 필요하지만 이 단계의 영상은 선택된 3건만 있어도 된다. 성공 여부·시간·최대 CUDA 할당량을 기록하며 OOF 점수나 제출 모델로 취급하지 않는다.
- `train`: 원래 4,350건을 사용한다. Backbone 학습은 기존 분할의 weak 3,835건, weak validation 457건이며 Gold58은 제외한다. Gold로 학습한 S11 최종 head를 파인튜닝 초기값으로 쓰지 않는다. Backbone 적응 후 기존 방식으로 Gold58 OOF와 전체 head 재학습을 수행한다.

다른 위치에 데이터를 두었다면 경로를 명시한다. 공백이 있는 경로는 따옴표로 감싼다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_training.ps1 `
    -Mode train -DataDir D:\MRI\rsna-knee `
    -Labels D:\MRI\report_labels_v2.csv `
    -ParentRun D:\MRI\s11\run `
    -Output D:\RSNA_Abnormality_Detection\outputs\local\r06-second-attempt
```

Python 진입점은 `python -m scripts.train_local --help`로 확인한다. 기본 모드는 `check`다. 학습률과 입력 구조를 바꾸는 옵션은 제공하지 않으며 R06 비교 조건을 유지한다. 현재 학습 코어의 FT/특징 추출 단계별 7시간 제한도 유지한다. 실제 5090 소요 시간은 smoke 후 전체 학습에서 측정해야 한다.

## 2026-10-22: 기존 Python 3.11 GPU 환경의 smoke 결과

`scripts/run_local_training.ps1 -Mode smoke`가 exit 0으로 완료됐다. 기존 Python 3.11.9의 GPU PyTorch를 공유하는 `.venv-5090`에서 실행했으며 학습 코드와 Kaggle 큐는 변경하지 않았다.

| 항목 | 확인 결과 |
|---|---|
| 입력 | 기존 분할의 weak train 2건 + weak validation 1건, Gold 0건 |
| 내려받은 영상 | 선택된 14개 시리즈, DICOM 424개, 199,526,624 bytes; 파일 크기·DICOM 헤더 확인 및 SHA256 기록 |
| 부모 입력 | S11 train v1 전체 산출물; 저장된 manifest·bundle·tokens 해시 일치 |
| 학습 단계 | 새 head warm-up 1 epoch → Swin 마지막 stage/norm 1 epoch → 별도 weak validation |
| 측정 시간 | **3.188초** — warm-up·미세조정·검증·checkpoint 저장 포함, 다운로드·입력 사전검사·초기 모델 로딩 제외 |
| 최대 PyTorch CUDA 할당량 | **698.284 MiB** — 전체 Windows GPU 사용량과는 다른 값 |
| warm-up train loss | 0.691976; optimizer step 1회 |
| 미세조정 train loss / validation BCE | 0.643539 / 0.683164; 성능 비교용 OOF가 아님 |
| 실제 가중치 갱신 | 허용된 28개 parameter tensor 모두 변경, 고정된 tensor는 그대로 유지, checkpoint 값 모두 유한 |
| 관련 자동 테스트 | `tests.test_local_training` 6개 통과 |

기록: [실행 정보](../outputs/local/r06-rtx5090-smoke/local_run.json), [시간·메모리](../outputs/local/r06-rtx5090-smoke/smoke/smoke_metrics.json), [학습 분할 감사](../outputs/local/r06-rtx5090-smoke/smoke/backbone_audit.json), [가중치 갱신 검증](../outputs/local/r06-rtx5090-smoke/smoke/checkpoint_verification.json), [영상 다운로드 기록](../outputs/local/r06-smoke-download-receipt.json). 이 링크들은 Git에서 제외한 로컬 산출물이다.

smoke 완료 시점에는 영상 3건만 준비됐다. **당시 4,350건 전체를 확인하는 `-Mode check`는 누락 영상 오류를 반환했다.** 이후 전체 입력 준비·학습 실행은 아래 무인 실행 기록과 구분한다. 3건 측정값을 전체 학습 시간이나 성능으로 환산하지 않는다. 기본 smoke 출력 폴더는 이미 존재하므로 재실행할 때는 새 `-Output` 경로를 사용한다.

## 2026-10-22: 무인 전체 학습 시작

사용자의 전체 파인튜닝 실행 지시에 따라 **13:48 KST** 숨김 백그라운드 실행기를 시작했다. launcher PID 44392, 실제 worker PID 49764. 최초 확인 단계는 `downloading`이며 GPU 전체 학습 완료를 의미하지 않는다. 이후 agent는 모니터링하지 않고 사용자의 요청 때만 상태를 확인한다.

실행 순서: **대회 ZIP 다운로드 → 선택된 21,045개 시리즈 추출 → 전체 입력 검사 → R06 warm-up/부분 미세조정 → 4,350건 특징 재추출 → Gold58 3-fold × 3-seed 평가 → 최종 head 및 bundle 저장**.

- 다운로드 ZIP: 265,018,939,620 bytes(약 265GB). 입력 CSV의 SHA256이 기존 파일과 같은지 검사하며, 전체 원본 중 이번 R06에 필요한 시리즈만 추출한다. ZIP은 재개·재검증을 위해 보존한다.
- 실행기: [run_local_r06.py](../scripts/run_local_r06.py). 기존 `.venv`는 다운로드/준비에, 검증된 `.venv-5090`은 실제 학습에 사용한다.
- 현재 상태: [status.json](../outputs/local/r06-unattended-20261022/status.json).
- 진행 로그: [runner.log](../outputs/local/r06-unattended-20261022/runner.log), [stderr](../outputs/local/r06-unattended-20261022/runner.stderr.log).
- 설정·입력·소스 고정 기록: [inputs.json](../outputs/local/r06-unattended-20261022/inputs.json).
- 학습 산출물: `outputs/local/r06-rtx5090-train/`. 완료 여부는 해당 `local_run.json`과 실행기의 `status.json`으로 확인한다.

실행기와 기존 코드의 전체 테스트 **114개 통과** 후 시작했다. 기존 MRI와 충돌하는 파일은 덮어쓰지 않고, 누락 시리즈·파일 손상·공간 부족·입력/코드 변경은 학습 전에 중단한다. 중복 실행은 파일 잠금으로 차단한다. 일시적인 다운로드 연결 실패는 기존 Kaggle SDK의 유한 재시도/재개 기능을 사용하며, 학습 실패를 자동으로 다시 학습하지 않는다.

LLM 호출, 예약 agent, Kaggle 공식 제출은 없다. 실행 중 프로세스가 Windows 자동 절전을 억제하고 종료 시 해제한다. 수동 절전·종료·재부팅은 작업을 중단하므로 PC와 인터넷 연결을 유지한다. 다운로드 시간은 학습 시간 예상에 포함되지 않는다.

중단 후 재개 명령은 다음과 같다. 기존 다운로드·추출 파일을 검증해 재사용하지만, 학습 출력 폴더가 이미 있으면 덮어쓰거나 epoch 중간부터 재학습하지 않고 중단한다.

```powershell
.\.venv\Scripts\python.exe -u -X utf8 -m scripts.run_local_r06
```

## 출력과 재실행

기본 경로는 `outputs/local/r06-rtx5090-smoke/`, `outputs/local/r06-rtx5090-train/`이다.

- `local_run.json`: 실행 상태, 시작·종료 시간, 환경, 설정, 코드·입력 hash.
- `smoke/smoke_metrics.json`: 소량 실행의 시간·VRAM. 같은 폴더에 backbone audit와 checkpoint.
- `run/metrics.json`, `oof_predictions.csv`, `folds.csv`, `history.csv`, `warmup_history.csv`, `backbone_audit.json`, `timings.json`: 전체 실험 결과.
- `run/bundle/`: encoder와 최종 head, manifest, Kaggle용 offline decoder wheel.

출력 폴더는 새로 만들어야 하며 이미 존재하면 덮어쓰지 않는다. 중단된 출력은 보존하고 `local_run.json`에 실패를 기록한다(강제 프로세스 종료/전원 차단은 `running`으로 남을 수 있음). **epoch 중간 재개는 지원하지 않는다.** 재시도는 새로운 `-Output`으로 수행한다.

로컬 모델을 Kaggle에서 사용하려면 `run/bundle`을 private dataset으로 전달하고, 저장된 `assets.json`의 **로컬 실험 config와 동일한** spatial submit notebook을 생성해야 한다. 기존 S11 submit notebook의 config를 그대로 사용하면 검증이 실패한다. 현재 추론 코드는 **Python 주·부 버전과 PyTorch 전체 버전 문자열까지 동일해야** 모델을 읽는다. `+cu130`과 `+cu128` 차이도 거부하므로 config만 맞춰서는 충분하지 않다. Kaggle에 동일한 실행 환경을 준비하거나 별도로 검토한 모델 이식 절차가 필요하며, encoder load와 Internet OFF 추론을 검증한 뒤 제출한다. 이 실행기는 모델을 업로드하거나 공식 제출하지 않는다.

## 코드 검증 범위

`tests/test_local_training.py`는 feature 재사용 조건, Gold 제외 smoke 분할, 경로 연결 복구, 누락 MRI·잘못된 hash 거부, 출력 덮어쓰기 방지와 실패 기록을 검증한다. CPU 환경에서 실행할 수 있다. 실제 GPU 학습·전체 DICOM decode·Kaggle 제출 호환성은 입력과 CUDA 환경이 준비된 뒤의 검증 항목이다.
