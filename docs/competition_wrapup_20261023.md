# 대회 종료 정리 — 2026-10-23

정리 시점: **2026-10-23 09:53 KST**. 정규 대회 마감은 같은 날 **08:59 KST**였다. 이 문서는 완료된 실험과 실행 중인 후속 연구를 구분해 보존한다. 최종 Private 점수·순위와 late submission 가능 여부는 확인하지 않았다. [공식 Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline)

## 대회 제출 결과

2026-10-22 사용자가 제공한 제출 화면의 **Public Score** 기록이다. 선택 상태도 해당 화면 시점의 상태이며 OOF와 다른 평가다.

| 제출 | 버전 | Public | 화면의 선택 상태 |
|---|---:|---:|---|
| V02 slice blend | 1 | **0.805** | 선택됨 |
| R01 E09/S13 blend | 2 | 0.797 | 미선택; 같은 점수로 2회 제출 |
| E02b Swin | 1 | 0.688 | 미선택 |
| 초기 baseline | 1 | 0.642 | 미선택 |

기록된 최고 Public은 baseline보다 **+0.163**, R01보다 **+0.008** 높다. [기존 제출 원장](notebook_run_results.md)

## 수행한 실험과 확인된 결과

저장된 `metrics.json`에서 **Kaggle 41개 구성 + 로컬 R06 1개 = 42개 구성**의 OOF 집계를 수집했다. [전체 CSV](results/2026-10-23/experiment_metrics.csv)에 실험 ID, AUC, seed SD, BCE, 원본 경로와 SHA256을 보존했다. 검사별 정답·예측은 포함하지 않는다. 이번 정리는 기존 집계 파일을 수집한 것으로, 모든 원시 OOF를 다시 계산한 작업은 아니다.

| 단계 | 수행 내용 | 관찰 |
|---|---|---|
| E00–E05 | ResNet/DINOv2/Swin/ConvNeXt/ViT, attention·위치·방향 | 초기 동결 backbone 비교에서 Swin 선택 |
| E06–E12 | gold 대조군, 보고서 약라벨, 2.5D, 촬영 조건, MIL·부분 FT | E06→E07 OOF **0.585972→0.741941**; 약라벨 확대가 큰 효과 |
| S00–S13 | 인접 장수, 위치 인코딩, gate/cross-attention, 부분 FT, blend | S02 5장이 S03 10장보다 높음. S11→S12는 **0.736386→0.718888**로 하락 |
| R01/R04/R05 | 고정 ensemble, confidence 필터, 소견별 loss | R01 **0.771432**; R04/R05는 개선 확인 안 됨 |
| T00–T02 | 방향 × 깊이 위치 표현 | 대조군 **0.722923**, 합 **0.733206**, 결합 **0.716154** |
| V00–V04 | R01과 다른 head blend, slice dropout, seed head 평균 | 최고 개발 OOF V00 **0.780610**; 최고 기록 Public은 V02 |
| 로컬 R06 | weak-only head warm-up 1 epoch + Swin 마지막 stage/norm 1 epoch | OOF **0.727763**; S12보다 높고 frozen S11보다 낮음 |

E00–E12는 E02의 4개 backbone 변형을 포함해 16개 구성이다. 각 행은 서로 다른 조건을 포함하므로 하나의 누적 개선 곡선으로 해석하지 않는다. [초기 33개 상세 분석](portfolio_results.md), [소견별 오류 분석](experiment_error_analysis.md), [연구 캠페인 설계](research_campaign_20261021.md)

| 주요 비교 | OOF macro AUC | seed SD | OOF BCE |
|---|---:|---:|---:|
| V00: R01 80% + S10 20% | **0.780610** | 0.007659 | 0.545497 |
| V01: R01 80% + T01 20% | 0.776232 | 0.008682 | 0.547965 |
| V02: R01 80% + S02 20% | 0.776593 | 0.004799 | **0.544974** |
| V03: S10 + slice dropout | 0.744693 | 0.005729 | 0.612449 |
| V04: S10 seed head 평균 | 0.769419 | 0.004740 | 0.552988 |
| S11: frozen cross-attention | 0.736386 | 0.017466 | 0.618533 |
| S12: warm-up 없는 부분 FT 1 epoch | 0.718888 | 0.010687 | 0.648535 |
| 로컬 R06: warm-up 후 부분 FT 1 epoch | 0.727763 | 0.006425 | 0.625453 |

공통 개발 평가는 **Gold58, 3-fold × seeds 42/2026/3407**의 seed별 전체 OOF macro AUC 평균이다. V04는 각 평가 seed 안에서 head 3개의 예측을 평균한다. 같은 Gold58을 반복해서 개발·선택에 사용했고 약라벨 teacher의 Gold 독립성도 입증되지 않았으므로 독립 일반화 성능을 뜻하지 않는다. seed SD는 신뢰구간이 아니다.

## 로컬 GPU 파인튜닝

기존 **Windows Python 3.11.9 / PyTorch 2.7.1+cu128 / RTX 5090 32GB**를 재사용했다. `.venv-5090`은 기존 GPU PyTorch를 공유하며 CPU용 `.venv`와 분리했다. S11 train v1 입력·약라벨·특징 캐시 hash, TorchScript/eager 복원과 실제 CUDA forward/backward를 확인했다.

선택 시리즈 21,045개, DICOM 706,256개를 준비했다. 총 4,350 study 중 backbone은 **weak train 3,835 / weak validation 457**을 사용하며 Gold58은 backbone·warm-up·checkpoint 선택에서 제외한다. 적응 후 Gold58 OOF head 평가와 전체 head 학습을 별도로 수행한다.

완료된 R06은 **10월22일 17:59:42–21:46:27 KST**, 약 **3시간47분** 걸렸다. warm-up·backbone 약1시간52분, 특징 재추출 약1시간47분이며 다운로드·압축 해제 시간은 제외한다. 최적 epoch는1, weak validation BCE는 **0.572002**다. 이 값과 위 표의 Gold58 OOF BCE는 서로 다르다. 로컬 R06의 공식 Public 점수는 없다.

## 정리 시점에 남은 작업

| 작업 | 확인 상태 | 자동 진행 조건 |
|---|---|---|
| `local_earlystop_p3` | **실행 중**; worker PID53160, epoch1 `400/3835` 로그 확인 | weak valid BCE 최저값을 3epoch 연속 갱신하지 않으면 종료; 상한30epoch/FT72시간 |
| 공개 방법 full FT 큐 | **선행 학습 대기**; worker PID49900, `waiting_for_predecessor` | 위 학습 정상 완료 → 실제 MRI3건 smoke → full Swin → best3 평균 평가 |
| V05/V06, R02/R03, 조건부 R07, U00–U08 | 이번 집계에서 완료 지표 미확보 | 기존 계획·코드는 보존; 완료나 성능 개선으로 계산하지 않음 |

부분 FT는 원래 S11에서 새 head warm-up 후 시작했고, 종료 시 best encoder를 복원한다. 후속 full FT는 전체 encoder·gradient checkpointing·5% LR warm-up/cosine·patience3을 적용한다. encoder/head 최대 LR는1e-5/1e-4다. 이후 **weak-validation BCE가 가장 낮은 3개 encoder 가중치**를 평균하고 head를 새로 학습해 단일 best와 비교한다.

현재 두 프로세스는 유지한다. 후속 큐는 소스40개와 입력 hash를 고정한 독립 복사본을 사용한다. Git 문서 정리·push·merge는 해당 실행 스냅샷을 수정하지 않는다. LLM 모니터링이나 자동 공식 제출 없이 일반 Python 프로세스로 진행하며, 선행 실패·입력 변경·smoke 실패에는 중단한다. 실제 full FT의 시간·VRAM·OOF는 아직 측정하지 않았다.

상태 확인은 아래 로컬 파일에서 한다. 이 파일들은 Git에서 제외돼 있어 다른 PC에서는 열리지 않는다.

- 부분 FT: `outputs/local/earlystop-p3-rtx5090-train/local_run.json`, `run/backbone_history.csv`, `run/backbone_audit.json`.
- 부분 FT 로그: `outputs/local/earlystop-p3-unattended-20261023/runner.log`.
- 공개 방법 큐: `outputs/local/public-finetune-20261023/status.json`, `runner.log`; 완료 후 `comparison.json`.
- 완료 R06: `outputs/local/r06-rtx5090-train/run/metrics.json`, `timings.json`, `backbone_audit.json`.

## 공개 방법 조사와 코드에 반영한 내용

[7개 계열 원문 조사](kaggle_finetuning_evidence_20261023.md)에 notebook 버전, 실제 학습 설정, 공식 historical best와 저자 보고를 구분해 기록했다. 핵심 근거는 S6의 전체 encoder·warm-up/cosine, ConvNeXt/CoAtNet의 epoch 가중치 평균이다. 고득점 제출 중에는 새 학습 없이 기존 모델의 TTA·blend만 수행하는 코드도 있었다.

현재 Swin에 필요한 방법만 적용했으며 공개 모델 전체의 정확 재현은 아니다. full encoder와 scheduler가 함께 바뀌므로 각각의 효과를 분리할 수도 없다. 공개 all-data 가중치는 Gold58을 이미 학습했을 가능성이 있어 로컬 OOF 초기값으로 사용하지 않았다. 문헌상의 점수 상승이 우리 실험의 상승을 뜻하지 않는다.

이번 보존 범위는 `experiments/spatial_train.py`의 선택적 early stopping·full FT·LR 일정, `experiments/public_finetune.py`의 checkpoint 평균, 로컬 CLI/PowerShell 실행기, 소스 고정 후속 큐와 회귀 테스트다. 기존 고정1epoch R06 설정과 독립 Kaggle notebook 생성 동작도 유지했다. [실행 명령·입력·smoke 기록](local_gpu_training.md)

## 병합 전 검증과 재현 한계

2026-10-23 새로 실행한 검증:

- CPU 전체 suite: **125개 중121개 통과, CUDA 전용4개 건너뜀**.
- GPU 관련 suite: **10개 통과**, 위 CUDA 전용4개 포함. patience 초기화·동률 처리·최적 가중치 복원·전체 encoder 갱신·checkpoint 평균·큐 입력 고정을 확인했다.
- 기존 독립 notebook 생성 회귀 포함. CPU Python3.14의 기존 TorchScript 경고는 남아 있으며 실제 GPU 학습은 검증된 Python3.11에서 실행한다.

원본 영상·보고서·검사별 라벨과 예측·가중치·인증 정보·실행 캐시는 Git에 넣지 않는다. 재현에는 사용자 권한으로 확보한 대회 데이터와 원래 S11 산출물이 필요하다. 이번 집계와 테스트는 후속 학습 완료나 최종 Private 성능을 대신하지 않는다.
