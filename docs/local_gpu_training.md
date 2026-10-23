# RTX 5090 로컬 파인튜닝

대회 종료 시점의 완료 결과와 진행 중인 작업은 [2026-10-23 정리](competition_wrapup_20261023.md)를 참고한다. 아래 실행·검증 기록의 시간은 각 작업 당시 기준이다.

기본 모드는 기존 R06의 **weak-only head warm-up 1 epoch → Swin 마지막 stage와 norm 1 epoch 학습 → 특징 재추출 → Gold58 3-fold × 3-seed 평가**를 Windows 로컬 CUDA GPU에서 실행한다. `-EarlyStopping`을 추가하면 검증 loss에 따라 학습 횟수를 결정하는 별도 실험을 실행한다. `-PublicFinetuning`은 전체 Swin과 LR warm-up/cosine을 쓰는 로컬 후속 recipe이며 기존 U04/U05와 구분한다. 실제 학습 결과가 나오기 전에는 성능 개선을 주장하지 않는다.

학습 계산은 `experiments/spatial_train.py`를 재사용하고, 로컬 실행기가 Kaggle 경로를 찾는 두 함수를 해당 프로세스 안에서 연결한다. 로컬 실험 ID는 기본 모드 `r06_local_warmup`, early stopping 모드 `local_earlystop_p3`이며 Kaggle 실험과 별도로 기록한다. 2026-10-23 공통 학습 루프에 선택적 early stopping과 매 epoch 기록 저장을 추가했다. 각 로컬 실행은 해당 소스 hash를 저장하며, 기존 Kaggle 캠페인의 저장된 notebook·상태 파일은 수정하지 않는다.

## 2026-10-23: 검증 loss 기반 파인튜닝

사용자 지시에 따라 고정 1/3 epoch 실험과 별도로 **weak validation BCE가 3 epoch 연속 최저값을 갱신하지 않으면 종료**한다. 값이 같아도 미개선으로 세며, 조금이라도 낮아지면 대기 횟수를 0으로 초기화한다(`min_delta=0`). 과적합이 전혀 없음을 보장하는 기준은 아니다.

- 초기값: 원래 동결 S11 encoder + 새 head warm-up 1 epoch. 이전 R06의 마지막 상태에서 이어 학습하지 않는다.
- 학습 대상: 기존 Swin 마지막 stage/norm과 head. 학습률은 backbone `1e-5`, head `1e-4`를 유지한다.
- 데이터: weak train 3,835건 / weak validation 457건. Gold58은 backbone 학습과 early stopping 모두에서 제외한다.
- 최대 30 epoch. 기존 Kaggle용 FT 7시간 제한 대신 이 로컬 모드에만 72시간 제한을 둔다. 시간 제한은 실패로 기록하며 수렴으로 간주하지 않는다.
- 매 epoch 검증 후 최저 loss의 `backbone_best.pt`를 저장한다. 종료 시 **최적 backbone을 복원**하고 전체 특징 재추출 → 기존 Gold58 OOF 평가 → 최종 head 학습·bundle 저장을 수행한다. 마지막 epoch의 head를 최종 제출 head로 쓰지 않는다.
- `backbone_history.csv`와 `backbone_audit.json`을 매 epoch 갱신한다. audit에 best epoch, 완료 epoch, 미개선 횟수, 종료 사유(`early_stopping` / `max_epochs`)를 남긴다. 학습 도중 사유는 `running`이다.
- 실행 중 파일 잠금으로 동일 로컬 작업 중복을 막고 Windows 자동 절전을 억제한다. 수동 절전·재부팅은 막지 않으며 epoch 중간 재개는 지원하지 않는다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_training.ps1 -Mode check -EarlyStopping
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_training.ps1 -Mode smoke -EarlyStopping
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_local_training.ps1 -Mode train -EarlyStopping
```

출력은 `outputs/local/earlystop-p3-rtx5090-smoke/`와 `outputs/local/earlystop-p3-rtx5090-train/`으로 분리한다. 이미 존재하는 출력에는 쓰지 않으며 재실행은 새로운 `-Output`이 필요하다. 다운로드는 반복하지 않고 준비된 MRI를 사용한다. agent의 지속 모니터링 없이 일반 프로세스로 실행한다.

비교 기준인 R06은 **2026-10-22 17:59:42–21:46:27 KST**, 약 **3시간 47분**에 정상 완료됐다. OOF AUC **0.727763**, weak validation BCE **0.572002**이다. R06 소요 시간은 백본·warm-up 약 1시간 52분, 특징 재추출 약 1시간 47분이었다. 새 실행의 총 시간은 실제 중단 epoch에 따라 달라지며 고정 3 epoch 예상 시간과 구분한다.

검증: 전체 자동 테스트 119개 중 CPU 환경의 GPU 전용 3개를 제외한 116개가 통과했다. GPU 환경에서는 관련 11개(해당 GPU 전용 3개 포함)가 모두 통과했다. 실제 MRI 3건 smoke도 성공했고, best epoch 2 이후 3번 미개선으로 **epoch 5에서 종료**했다. smoke 학습·검증 시간 13.531초, 최대 CUDA 할당량 810.481 MiB이며 전체 학습 성능·메모리 추정값으로 사용하지 않는다. CPU 테스트의 기존 TorchScript/Python 3.14 경고와 실제 GPU 학습의 Python 3.11 환경을 구분한다.

**2026-10-23 09:38 KST** 숨김 백그라운드 실행을 시작했다(launcher PID 4388). 4,350건·선택 시리즈 21,045개 입력 검사를 통과하고 **09:38:43 KST 실제 worker PID 53160, `running`**을 확인했다. 실행 명령과 시작 시간은 [launch.json](../outputs/local/earlystop-p3-unattended-20261023/launch.json), 표준 출력은 [runner.log](../outputs/local/earlystop-p3-unattended-20261023/runner.log), 오류 출력은 [runner.stderr.log](../outputs/local/earlystop-p3-unattended-20261023/runner.stderr.log)에 기록한다. [local_run.json](../outputs/local/earlystop-p3-rtx5090-train/local_run.json)에 실제 학습 PID·설정·시작 시간·완료/실패 상태가 저장된다. 현재 결과를 이전 R06이나 smoke 결과와 혼동하지 않는다.

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

Python 진입점은 `python -m scripts.train_local --help`로 확인한다. 기본 모드는 `check`다. 임의 학습률과 입력 구조를 바꾸는 옵션은 제공하지 않으며 기본 R06 비교 조건을 유지한다. 기본 R06의 FT/특징 추출 단계별 7시간 제한도 유지한다. early stopping과 공개 방법 모드는 FT 제한을72시간으로 늘리며 특징 추출 제한은 그대로다.

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

## 2026-10-23: 공개 방법 기반 후속 큐

[공개 원문 7개 계열과 점수 근거](kaggle_finetuning_evidence_20261023.md)를 수집하고, S6의 전체 encoder·LR warm-up/cosine과 ConvNeXt/CoAtNet의 checkpoint 가중치 평균을 현재 Swin에 적용한다. 공개 모델의 정확한 재현이 아니며 full scope와 scheduler가 함께 바뀌는 recipe 비교다. 공개 all-data 가중치는 사용하지 않는다.

현재 `earlystop-p3-rtx5090-train`이 **정상 완료한 다음** 다음 순서로 실행한다.

| 단계 | 실행 내용 | 출력 폴더 |
|---|---|---|
| smoke | 실제 MRI 3건으로 전체 encoder backward·checkpointing·종료 조건 확인 | `outputs/local/public-finetune-20261023/smoke/` |
| full | 전체 Swin + head warm-up 1 epoch + LR 5% warm-up/cosine + valid BCE patience3 | 같은 경로의 `full/` |
| average3 | full의 weak-validation BCE 상위3 encoder 가중치 평균 → 특징 재추출·OOF·최종 head 학습 | 같은 경로의 `average3/` |
| 비교 | 기존 부분 FT / full best / full average3의 OOF AUC·seed SD·BCE 저장 | 같은 경로의 `comparison.json` |

full 설정은 encoder/head 최대 LR **1e-5/1e-4**, AdamW WD.01, accumulation 8 studies, FP16, gradient checkpointing(chunk4), 최대30epoch·FT72시간 제한이다. LR는 최대30epoch의 예정 optimizer step 수를 기준으로 앞5%에서 선형 증가하고 이후 cosine으로 최대 LR의1%까지 감소한다. 이 수치들은 현재 코드에 맞춘 보수적 로컬 설정으로 S6의 backbone·LR·epoch·bf16을 그대로 복제한 것이 아니다. 입력·crop224·약라벨·weak3835/457 분할은 유지하고 Gold58은 backbone 학습·checkpoint 선택에서 제외한다. 최종 OOF head와 feature 추출은 기존 평가 경로를 재사용한다.

가중치 평균은 **encoder만** 평균한다. 같은 학습 궤적의 가장 낮은 weak-validation BCE 3개를 선택하고 각 파일 SHA256을 확인한다. 정수 buffer는 서로 동일해야 하며 NaN/Inf·다른 recipe·미완료 원본을 거부한다. 평균 encoder에 대해 head는 새로 학습한다. 평균 결과가 단일 best보다 우수하다고 자동 선언하거나 Public 결과로 해석하지 않는다.

실행기 [run_local_public_methods.py](../scripts/run_local_public_methods.py)는 source snapshot과 입력 hash를 고정한다. 현재 학습의 이전 소스도 해당 실행 폴더의 `source_snapshot/`에 원래 hash와 일치하는 상태로 보존했다. 후속 worker는 큐의 `source/` 복사본에서 실행하므로 이후 작업공간 코드 변경과 구분된다. 같은 GPU 잠금을 공유하며 선행 실패·사라진 worker의 stale 상태·입력 변경·smoke 실패는 후속 학습을 중단한다. 기존 출력을 덮어쓰거나 자동 재학습하지 않는다. LLM 모니터링·공식 제출은 없다.

```powershell
# 새 큐를 준비할 때 1회 실행. 기존 큐 폴더가 있으면 중단한다.
.\.venv\Scripts\python.exe -m scripts.run_local_public_methods --prepare
# 실제 실행은 큐 source/를 작업 디렉터리로 사용하며, --queue-dir에 위 큐의 절대 경로를 전달한다.
```

상태는 `outputs/local/public-finetune-20261023/status.json`, 로그는 같은 폴더의 `runner.log` / `runner.stderr.log`, 고정 내역은 `pins.json`에 남긴다. 실제 MRI smoke는 현재 작업의 GPU 잠금이 해제된 뒤 자동 실행한다. 새 full 실험의 소요 시간·VRAM·OOF는 아직 측정 전이다.

**2026-10-23 09:50 KST** 백그라운드 큐를 시작했고 `waiting_for_predecessor`를 확인했다. launcher PID **48208**, 실제 worker PID **49900**. 현재 부분 FT의 PID53160은 계속 실행 중이며 후속 full 학습이 이미 시작됐다는 의미가 아니다. 독립 소스40개를 고정했고 해당 복사본에서 CUDA·4,350건·21,045시리즈 입력 검사를 통과했다. 전체 자동 테스트125개 중 CPU121개 통과, GPU 전용4개는 별도 GPU 실행에서 통과했다(관련 GPU 환경 테스트10개 모두 성공). 기존 독립 Kaggle notebook 생성 회귀도 통과했다.

정규 대회 마감은 **2026-10-23 08:59 KST**였다. Late submission 활성화는 확인되지 않았으므로 이 큐는 로컬 개발 평가까지 진행하며 Public 점수 갱신을 약속하지 않는다. [공식 Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)

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
