# RTX 5090 로컬 파인튜닝

기존 R06의 **weak-only head warm-up 1 epoch → Swin 마지막 stage와 norm 1 epoch 학습 → 특징 재추출 → Gold58 3-fold × 3-seed 평가**를 Windows 로컬 CUDA GPU에서 실행한다. 전체 Swin을 학습하는 U04/U05는 이 실행기의 대상이 아니다. 실제 학습 결과가 나오기 전에는 성능 개선을 주장하지 않는다.

학습 계산은 기존 `experiments/spatial_train.py`를 재사용한다. 로컬 실행기는 Kaggle 경로를 찾는 두 함수만 해당 프로세스 안에서 연결한다. 실행 중인 Kaggle 캠페인의 소스 hash, 대기 순서와 상태 파일은 변경하지 않는다. 로컬 실험 ID는 `r06_local_warmup`이며 Kaggle R06과 별도로 기록한다.

## 1. GPU 환경 준비

프로젝트 루트 `D:\RSNA_Abnormality_Detection`에서 실행한다. **Python 3.13 64bit**를 먼저 설치한다. 기존 `.venv`와 별도인 `.venv-5090`을 사용한다. 기존 encoder export가 TorchScript이므로 Python 3.14보다 3.13을 권장한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local_gpu.ps1
# Python launcher를 쓰지 않는 경우
# powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_local_gpu.ps1 -PythonExe C:\Python313\python.exe
```

공식 CUDA 13.0 PyTorch/torchvision과 프로젝트 의존성을 설치하고, GPU 행렬 연산의 forward/backward를 확인한다. CUDA Toolkit 전체를 별도로 설치하는 절차는 포함하지 않는다. [PyTorch 지원표](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix)

2026-10-22 확인: RTX 5090 32GB, RAM 64GB, D드라이브 여유 약 1TB. 기존 `.venv`는 CPU 전용 PyTorch이고 DICOM 영상은 없다. **이 코드를 저장한 것만으로 환경 설치나 학습이 시작되지는 않는다.**

## 2. 입력 준비

| 경로 | 필요한 내용 |
|---|---|
| `data/raw/` | `train.csv`, `train_series.csv`, `train_series/<StudyUID>/<SeriesUID>/*.dcm` |
| `data/weak_labels/report_labels_v2.csv` | 기존 Pilkwang 약라벨 v1의 파일. S11 manifest와 SHA256이 같아야 함 |
| `data/local-assets/s11/run/` | 기존 S11 **train v1**의 전체 `run` 출력: `tokens.pt`, `bundle/assets.json`, `bundle/encoder.ts`, `bundle/model.pt`, `bundle/wheels/` 등 |

기존 로컬 `outputs/automation/spatial/s11_cross/train/run/`에는 지표와 manifest만 있고 모델과 feature cache는 없다. 이를 완전한 학습 입력으로 오인하지 않는다. Kaggle 개인 notebook 접근 권한 및 API 인증은 기존 사용자 환경에서 제공한다. 토큰은 스크립트·Git에 넣지 않는다.

현재 Kaggle S11 출력 버전이 1인지 확인한 다음 다운로드한다. CLI output은 최신 버전을 받으므로, 이후 notebook을 다시 실행했다면 먼저 Kaggle에서 버전과 입력을 확인한다.

```powershell
.\.venv-5090\Scripts\kaggle.exe kernels status choiyoungwhan/rsna-knee-s11-cross-train
.\.venv-5090\Scripts\kaggle.exe kernels output choiyoungwhan/rsna-knee-s11-cross-train -p data/local-assets/s11
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
