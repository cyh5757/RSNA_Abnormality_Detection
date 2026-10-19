# MRI slice/RoPE/fusion 자동 실험 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 기존 E06~E12가 모두 성공한 뒤 인접 3/5/10장, 순서, RoPE, 방향 embedding, fusion, fine-tuning, 앙상블을 독립 Kaggle 학습/제출 Notebook으로 자동 실행한다.

**Architecture:** 기존 소스와 E06~E12 설정은 보존한다. 새 spatial 모듈에 시리즈별 token·좌표·마스크와 후단 모델을 둔다. 기존 upload/version pin/resume/검증 runner는 선택적인 Notebook builder만 받아 재사용하며, 새 shell은 기존 큐의 성공 상태와 전역 lock을 확인한 뒤 실행한다.

**Tech Stack:** 현재 .venv, Python/PyTorch/torchvision, PowerShell 5.1, Kaggle CLI. 새 패키지 없음.

**Spec:** [설계 검토](../../mri_rope_design_audit.md). 사용자가 기획 및 자동 shell 작성을 요청했으므로 앞서 설명한 설계의 구현까지 진행한다. 현재 큐가 끝난 뒤 사용자가 shell을 실행할 수도 있고 지금 실행해 성공 완료를 기다릴 수도 있다. 이 작업에서는 새 GPU 큐를 미리 시작하지 않는다.

## Global Constraints

- Swin-T/224px, Pilkwang v1 hash, gold 58 + weak 4292, 기존 3fold/3seed 유지.
- 장수와 encoder 입력 표현을 분리: 3채널 triplet 대조군 → 한 장당 token 3/5/10장.
- 각 슬롯의 중앙 25~75%에서 중심 5개. 주변 slice의 중복 제거, 짧은 시리즈 padding mask. 10장은 offsets -4..5로 고정.
- 같은 시리즈의 물리 좌표·원본 정렬 index 보존. mm 없으면 index fallback과 좌표 종류 표식 기록. 방향별 위치를 1D로 연결하지 않는다.
- 후단 공통 차원128/4heads/1block, 20epochs, batch16, AdamW lr.001/wd.01. encoder는 FT 단계 전까지 동결.
- 학습은 private GPU/Internet ON, 제출용 Notebook은 Internet OFF + 해당 학습 버전만. 공식 leaderboard 제출은 수동.
- E06~E12 소스 hash 불변, 기존 출력 보존, 새 출력 `outputs/automation/spatial/`.
- 현재 PC에서 전역 runner lock 공유. 선행 큐 실패·미완료면 새 업로드 없음. 재실행은 완료 작업 재업로드 없음.

## 실험과 비교 기준

| ID | 내용 | 비교 대상 / 캐시 의존 |
| --- | --- | --- |
| s00_triplet | 인접3채널, projection128 + class MIL + concat linear 대조군 | 새 후단/학습 예산 기준. E10과 동일 모델이라는 주장은 하지 않음 |
| s01_slice3 | 인접3장을 각각 독립 token | s00 |
| s02_slice5 | 인접5장 | s01 |
| s03_slice10 | 인접10장 | s02 |
| s04_sequence | s02 + 위치 없는 slice Transformer | s02 |
| s05_position | s04 + index sin/cos 위치 덧셈 | s04 |
| s06_rope_index | s04 + 원본 index RoPE | s04 |
| s07_rope_mm | s04 + 상대 mm / 고정 3mm 단위 RoPE | s04 |
| s08_plane | s07 + plane/protocol embedding | s07 |
| s09_mlp | s08 + 소견별 concat MLP | s08 |
| s10_gate | s08 + 소견별 슬롯 gate | s08 |
| s11_cross | s08 + 12 소견 query cross-attention | s08 |
| s12_finetune | s11 마지막 Swin stage+norm 1epoch FT 후 새 특징 추출 및 동일 head CV | s11 |
| s13_ensemble | s11/s12 확률 평균 50:50 | s11+s12 |

실험군이 길기 때문에 완료된 결과를 캐시/재개하고, 실패 시 중단한다. 첫 비교는 모두 고정 설정이며 gold/LB 최고점에 따라 다음 설정을 자동 선택하지 않는다. 추가 학습량 조정은 결과를 본 후 별도 실행한다.

## Review Focus

- 기존 큐 미완료/실패 또는 lock 점유 시 새 업로드 금지.
- positions를 train/FT/submit 모두 같은 방식으로 전달하고 padding/전체 결측 슬롯에서 NaN 방지.
- RoPE token+좌표 동시 재배열 불변, 좌표 변경 효과, pairwise 상대거리 일관성.
- 캐시 입력/encoder hash 불일치와 구버전 모델을 거부, 최종 학습과 오프라인 제출 일치.
- FT의 gold 제외·report group 분리·gradient 범위, 알려지지 않은 label의 zero loss.

## Tasks

### Task 1 — 데이터·모델

Create `experiments/spatial_core.py`, `spatial_configs.py`; test `tests/test_spatial_experiments.py`.
Interface: `spatial_views(...) -> images,mask,coords`; coords 마지막 축은 index/mm/physical_valid. `SpatialHead.forward(tokens,mask,coords)->[B,12]`.

- [x] 선택·중복·경계·좌표·RoPE·결측 테스트를 먼저 추가하고 실패 확인.
- [x] token extractor, 좌표, slice Transformer, fusion을 구현.
- [x] 순서/위치/mask/shape 및 gradient 테스트 통과.

### Task 2 — 학습·제출 Notebook

Create `spatial_train.py`, `spatial_submit.py`, `spatial_notebooks.py`.
Interface: `train_spatial_experiment(config,factory,data,out)` / `submit_spatial_experiment(config,data,bundle,out)` / `build_spatial(configs,output_root,training_version=None)`.

- [x] bundle roundtrip, standalone Notebook, FT, cache 검증 테스트 추가.
- [x] 기존 라벨·fold·metrics·manifest·encoder export를 재사용하고 위치를 전달.
- [x] CPU synthetic 학습→bundle→offline 예측, 전체 Notebook 구문/정의 실행 검증.

### Task 3 — 선행 큐 대기 및 자동 실행

Create `experiments/run_spatial_queue.py`, `scripts/run_spatial_experiments.ps1`; extend `run_queue.run(..., builder=None)` only.

- [x] 미완료 gate, 실제 OS lock, 버전 pin, 재개, 업로드 없는 dry-run 테스트 추가.
- [x] 기존 E06~E12 state의 전체 verified 확인 후 lock 획득·재확인, 기존 runner 재사용.
- [x] 기획 및 실행 가이드 작성, 14개 train/submit 쌍 생성, 전체 회귀 테스트.

## Progress / rulings

- 계획 작성. Git 저장소가 없으므로 worktree/commit 단계는 적용하지 않는다.
- 기존 `run_queue.py`는 현재 fingerprint 입력에 포함되지 않는다. 선택적 builder 인자 추가는 기존 기본 호출과 이미 실행 중인 프로세스에 영향을 주지 않는다. fingerprint 입력인 기존 core/weak/build 소스는 수정하지 않는다.
- 설계의 기존 비교 항목은 이번 실행 가능한 14개로 고정한다. 전역/국소 sampling sweep와 patch/3D 확장은 추가하지 않는다.
- 완료 검증: 전체 **48 tests OK**, 새 28개 Notebook 구문/config/오프라인 metadata 확인, 실제 shell `-DryRun` 통과. 별도 검토자도 주요 결함을 발견하지 않았으며 CPU FP16-autocast에서 네 fusion의 forward/backward를 확인했다.
- 기존 실행 fingerprint `199344fc4c8ee7748b5651642d19bd943aa8bf769ce0910859fcf09b1b93898f`의 불변을 확인했다. 새 shell은 작성·검증만 했고 대기 프로세스나 GPU 작업은 시작하지 않았다.
- 실제 Kaggle GPU/전체 MRI에서의 소요 시간·VRAM·점수는 후속 실행에서 측정한다. [실행 가이드](../../spatial_experiments.md).
