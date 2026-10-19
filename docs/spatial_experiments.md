# 인접 슬라이스·RoPE·cross-attention 자동 실험

2026-10-16. [기획 및 비교표](superpowers/plans/2026-10-16-spatial-experiments.md), [현재 모델과 HTML 설계 대조](mri_rope_design_audit.md).

**기존 E06~E12를 전부 성공 완료한 뒤 S00~S13을 순서대로 실행하는 shell을 준비했다. 새 학습은 아직 시작하지 않았다.** 지금 실행하면 기존 큐가 끝날 때까지 대기하고, 종료 후 실행하면 성공 기록을 확인하여 바로 시작한다. 별도 Codex 호출은 필요 없다.

프로젝트 폴더의 PowerShell에서:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_spatial_experiments.ps1
```

업로드·대기 없는 사전 확인:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_spatial_experiments.ps1 -DryRun
```

## 자동으로 하는 일

1. `outputs/automation/weak-labels/state.json`에서 E06~E12 모두 train/submit 완료와 검증을 확인한다. 완료 산출물의 config·manifest·CSV도 재검사한다.
2. 현재 큐가 사용하는 Windows 파일 lock이 풀릴 때까지 기다린다. 두 번째 대기 shell도 중복 실행하지 않는다.
3. 각 새 실험의 private Kaggle GPU 학습 Notebook을 업로드하고 완료를 기다린다.
4. 반환된 실제 학습 버전을 고정해서 Internet OFF 제출용 Notebook을 업로드한다.
5. `submission.csv`, OOF AUC/BCE, 소견별 AUC, history, 모델·입력 정보를 회수·검증하고 다음 실험으로 넘어간다.

**대회 리더보드 제출 버튼은 자동으로 누르지 않는다.** 기존 방식처럼 선택한 실험의 완료된 `…-submit` Notebook 버전을 대회 제출 화면에서 고른다. 자동 업로드와 제출 파일 생성까지 준비되어 있다.

## 실험 순서

S00은 E10과 동일 모델이 아니라, 새 공통 후단과 학습 조건으로 만든 3채널 입력 대조군이다. 모든 실험은 Swin-T, 224px, 같은 약라벨·gold 분할을 사용한다. 장수 비교 외에는 인접 5장을 기준으로 한다.

| 순서 | 실험 | 이전 무엇과 비교하는가 |
| --- | --- | --- |
| S00 | `s00_triplet`: 인접3장을 3채널, class MIL + concat | 새 실험군의 기준 |
| S01 | `s01_slice3`: 인접3장을 각각 token | S00: 입력 표현의 차이 |
| S02 | `s02_slice5`: 인접5장 | S01: 장수 |
| S03 | `s03_slice10`: 인접10장 | S02: 장수 |
| S04 | `s04_sequence`: 위치 없는 slice Transformer | S02: 슬라이스 관계 학습 |
| S05 | `s05_position`: 일반 sin/cos 위치 덧셈 | S04 |
| S06 | `s06_rope_index`: 원본 index RoPE | S04 |
| S07 | `s07_rope_mm`: 상대 mm RoPE | S04 |
| S08 | `s08_plane`: 방향·지방 억제 embedding | S07 |
| S09 | `s09_mlp`: concat + MLP | S08 |
| S10 | `s10_gate`: 소견별 슬롯 gate | S08 |
| S11 | `s11_cross`: 소견 query cross-attention | S08 |
| S12 | `s12_finetune`: 마지막 Swin stage + norm 1 epoch | S11 |
| S13 | `s13_ensemble`: S11/S12 확률 50:50 | S11, S12 |

현재 점수가 가장 좋은 분기로 자동 변경하지 않는다. 비교 기준을 사전에 고정하여 작은 gold58 또는 public LB에 맞춘 선택이 자동 누적되지 않게 했다.

## 데이터와 모델의 의미

- 방향 3종 × 지방 억제 여부의 최대 6슬롯을 사용한다. 슬롯마다 한 촬영 시리즈를 선택한다.
- 시리즈 중앙 25~75%의 중심 5개 주변에서 인접 3/5/10장을 고른다. 인접10장은 offset -4..5로 고정한다. 개별 token 입력은 중복 slice를 제거하고 짧은 시리즈의 빈 자리를 mask한다.
- 영상 feature·원본 정렬 index·상대 mm 좌표·물리 좌표 유효 여부를 별도로 보존한다. mm가 없는 시리즈는 index로 대체하고 좌표 종류를 embedding과 실행 기록에 표시한다. mm RoPE는 공통 3mm 단위를 사용한다.
- RoPE는 같은 시리즈의 self-attention Q/K에 적용한다. 서로 다른 방향의 깊이를 하나의 물리적 순서로 붙이지 않는다.
- 후단 차원128, 4heads, slice block 1개, head 20epochs/batch16/lr.001/wd.01로 시작한다. frozen encoder 단계는 같은 입력과 encoder hash의 특징 캐시를 재사용한다.
- S12는 gold와 중복 보고서를 제외한 weak만 사용하고 report group별 약10%를 검증에 둔다. raw 이미지부터 마지막 stage로 gradient를 전달한 뒤, encoder를 다시 동결하여 전체 특징을 재추출하고 같은 gold head CV를 수행한다.
- S13은 두 구성의 원래 전처리와 학습 bundle을 보존한다. 앙상블 비율을 gold58에서 최적화하지 않는다.

## 결과 위치와 재개

```text
outputs/automation/spatial/
  runner.log                         # 대기·업로드·완료·오류
  state.json                         # 실제 원격 버전 및 재개 기록
  summary.csv                        # 완료된 실험의 OOF AUC/BCE와 submit URL
  notebooks/<실험 ID>/train|submit/  # 그 실행에 사용한 독립 Notebook
  <실험 ID>/train/run/                # metrics, folds, history, audit 등
  <실험 ID>/submit/submission.csv
```

배포용 템플릿도 `notebooks/experiments/s00_triplet/`부터 `s13_ensemble/`까지 생성해 두었다. 각 폴더에 완전한 `train/train.ipynb`와 `submit/submit.ipynb`가 있다. 직접 UI로 실행할 때 템플릿의 기본 `/1` 입력 버전은 실제 완료 버전과 맞춰야 한다. shell은 이 버전을 자동으로 연결한다.

중단 뒤 **같은 명령**을 다시 실행하면 기존 Kaggle 버전과 다운로드 기록을 확인하고 재개한다. 모델 코드·설정이 바뀐 경우 기존 실행 이름은 거부하며 새 이름을 사용한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_spatial_experiments.ps1 -RunName spatial-trial2
```

`-RunName`은 로컬 결과/상태 폴더를 분리한다. 원격 Notebook slug는 실험별로 고정되어 새 실행은 새 버전을 올리므로, 다른 PC나 Kaggle UI에서 같은 Notebook을 동시에 실행·수정하지 않는다. 이전 실행은 원격 버전이 달라지면 안전하게 중단한다.

선행 E06~E12가 미완료인데 실행 중인 runner가 없으면 바로 오류로 끝난다. 이 경우 기존 큐를 먼저 재개한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_weak_experiments.ps1
```

현재 PC를 켜 두고 shell을 유지해야 다음 업로드와 결과 회수가 이어진다. PC가 꺼져도 이미 업로드한 Kaggle 작업은 계속될 수 있지만 다음 단계는 로컬 shell 재개가 필요하다. 인증은 기존 PC의 Kaggle 설정을 사용하며 토큰을 shell·Notebook에 넣지 않는다.

## 검증과 실행 한계

로컬 테스트는 위치/순서/RoPE, 인접 선택·중복 제거, 결측 슬롯, 학습 통계의 검증 데이터 독립성, raw encoder→새 head gradient, 오프라인 bundle 복원·CSV 생성, standalone Notebook, dependency 버전 pin, 중복 lock, 선행 큐 대기, 재개를 확인한다. CPU 합성 데이터 및 mocked Kaggle 테스트는 실제 GPU 학습이나 리더보드 결과가 아니다.

작성 시 전체 **48개 테스트 통과**, 새 **28개 Notebook** 구문·config·Internet OFF 제출 설정 검증, 실제 PowerShell `-DryRun` 통과. 기존 실행 fingerprint도 변경되지 않았음을 확인했다.

14개 실험은 단일 학습보다 GPU 시간을 많이 쓴다. 실제 소요 시간은 아직 측정하지 않았으며 Kaggle 한도·할당량·OOM·실행 오류가 나면 이후 실험은 중단한다. 자동으로 유료 GPU를 구매하거나 실험 예산을 늘리지 않는다. S12의 1 epoch도 첫 처리량 비교 예산이며 충분한 수렴을 보장하지 않는다.
