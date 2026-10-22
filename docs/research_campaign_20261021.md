# 공개 방법 조사와 자동 실험 캠페인 — 2026-10-21

사용자 요청: [대회 Code 목록](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code)의 방법을 수집하고, 추가 제안을 합쳐 실험한다. 준비가 끝나면 Python 실행기가 진행하고 사용자가 다시 물을 때까지 agent가 사후 확인하지 않는다.

## 수집 범위와 근거

Kaggle 연결 도구로 Most Votes 20개와 Hotness 20개 목록을 조회했다. 서로 다른 방법을 대표하는 12개 공개 원문을 저장하고 주요 설명·활성 설정·추론 경로를 검토했다. 목록 40개 전체의 코드를 정밀 감사했다는 뜻은 아니다. 원문을 실행하지 않았으며, Notebook 제목의 LB 숫자는 이번 조사에서 재현하거나 공식 점수로 재검증하지 않았다.

- 목록: `outputs/research/kaggle_code_20261021/selection.json`
- 원문 버전·원본 파일 SHA256: `outputs/research/kaggle_code_20261021/snapshots.json`
- 각 원문의 `metadata.json`, `source.ipynb` 또는 `source.py`, `code.txt`, `markdown.md` 보관.
- 기존 조사와 함께 읽기: [2026-10-19 공개 코드 조사](kaggle_hotness_votes_095_refresh.md).

| 공개 원문 / 저장 버전 | 확인한 방법 또는 역할 | 이번 실험과 연결 |
|---|---|---|
| [Pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), v15 | 판독 보고서에서 감독 신호 구성, 촬영 조건별 입력, 물리 단위 전처리, 영상 모델 학습 | 기존 약라벨 경로 유지; U01 crop |
| [goodpjw CoAtNet + R6 + MIL](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), v27 | 2.5D ConvNeXt reader, 강한 다른 reader에 작은 비율로 결합, mirror TTA·rank 결합; 큰 reader 비중의 실패 기록도 있음 | V00–V02, V06 고정 80:20; U06/U08 |
| [nartaa S6 all-data](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo), v2 | 활성 `W_A=0, W_B=1`; ConvNeXt-small 주력, anatomy 기반 시야·spatial/window attention. 실패 검사 fallback은 별도 | V05 backbone 다양성; U01–U05. 상단 과거 70:30 설명과 활성 설정 구분 |
| [Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs), v38 | DINO 계열과 CoAtNet reader; 소견별 공간 특징과 global residual, study attention | U03 공간 token, 장기 후보로 다른 backbone 계열 |
| [Heliosli latest dataset](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), v1 | 별도 학습된 CoAtNet/ConvNeXt 계열 가중치, 여러 folds, 물리 crop·rank 결합·일부 state stacker | V05/V06, U01/U07/U08; 외부 가중치를 그대로 결합하지 않음 |
| [D4 blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend), v3 | 여러 reader의 결합과 DINO 공유 prefix, 여러 추론 변형; 전처리/모델 연산 재사용 | V04 같은 encoder 특징을 공유하는 seed-head 평균; U06은 단일 flip 비교 |
| [One Free Flip](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up), v1 | 기존 checkpoint의 mirror/flip 경로와 다중 모델 결합 | U06. 우리의 영상 내 horizontal flip은 공개 코드의 완전한 해부학 mirror 재현이 아님 |
| [mavit 288](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu), v4 | 6 views × 8 slices, 288 크기, 외부 `model`/`my_fastread_knee` 모듈과 checkpoint를 쓰는 추론 예제 | 추론 시간도 함께 기록. 제목만으로 별도 모델 구조를 확정하지 않음 |
| [Four public models, one rank blend](https://www.kaggle.com/code/rudrpratapsingh/rsna-knee-four-public-models-one-rank-blend), v2 | 복수 공개 reader의 결합, checkpoint manifest와 flip/state 옵션, rank 처리 | U08 고정 기준분포 CDF 결합으로 순위 결합 효과 비교 |
| [Read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee), v62 | 보고서 파싱과 여러 모델/약라벨 실험 경로, report 중복 분리, rank 결합 | 기존 보고서 중복 제외·UNK 마스크 유지. 새로운 LLM 라벨 생성은 하지 않음 |
| [Twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), v7 | CoAtNet, 140 mm crop, 인접 3장 window, 촬영 조건별 장수와 넓은 window coverage | R02/R03 입력 범위·밀도, U01 물리 crop |
| [Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb), v82 | Efficiency Prize 결과를 표시하는 Notebook | 새 모델 학습법이 아님. 시간·메모리 결과도 보존 |

공통 방향은 **감독 데이터 확대, 영상 encoder 적응, 해부학/물리 시야, pooling 전 공간 정보, 상보적인 모델 결합**이다. 공개 제출 Notebook 상당수는 학습을 재현하는 코드가 아니라 이미 학습된 checkpoint의 추론·결합 코드다. 이를 새로운 독립 학습 방법 개수로 세지 않는다.

## 확정한 실행 계획

먼저 V00–V04의 저비용 실험을 실행하고, 이후 무료 GPU quota 갱신을 기다리며 V05/V06 → 기존 R → U 순서로 진행한다. 총 **19개 구성 + 조건부 R07 1개**이며 U04/U05를 위해 별도 adaptation 작업 12개가 포함된다. 완료된 기존 36개 구성은 그대로 재사용한다.

| ID | 실험 | 변경 근거 / 비교 대상 |
|---|---|---|
| V00 | R01 80% + S10 20% | 서로 다른 pooling 결합. 직전 대화의 탐색 OOF 0.780610은 독립 검증 성능이 아님 |
| V01 | R01 80% + T01 20% | 위치 표현 모델과 기존 최고 모델의 보완성 |
| V02 | R01 80% + S02 20% | 다른 후단 구조의 보완성 |
| V03 | S10 + slice dropout 15% | 추가 제안. 학습 시 일부 slice를 가려 과도한 의존을 줄이는지 확인; 활성 slot당 최소 1장 유지 |
| V04 | S10 3-seed head 확률 평균 | 추가 제안. 동일 fold의 학습 자료만으로 세 head 학습; encoder 특징 공유 |
| V05 | ConvNeXt-tiny + S10과 동일 입력·약라벨·head | 공개 ConvNeXt 활용에서 도출. frozen encoder만 교체하는 통제 실험 |
| V06 | R01 80% + V05 20% | backbone 다양성이 결합에 도움이 되는지 확인 |
| R02/R03 | 입력 범위 확대 / 중심점 밀도 증가 | 기존 계획 재개; 물리 crop과는 다른 입력 coverage 실험 |
| R06 / 조건부 R07 | weak head warmup 후 부분 FT / 3epoch | 기존 코드의 사전 조건 충족 시에만 R07 실행 |
| U00/U01/U02 | 대조군 / 140 mm crop / IOP 방향 처리 | 공개 방법에서 가져온 전처리 비교 |
| U03/U04/U05 | 공간 token / 전체 backbone 1epoch / 3epoch | 이미지 내부 위치 정보 및 encoder 적응 |
| U06/U07/U08 | flip TTA / 확률 평균 / 고정 reference CDF 평균 | 같은 구성에서 추론·결합 방법 비교 |

V04는 CV seed 42/2026/3407 각각에 대해 `[s,s+1,s+2]`를 학습해 예측 확률을 평균한다. 제출용은 `[42,43,44]`이다. 3개의 점수를 평균하는 기존 평가와 구별되며, 동일 gold fold를 재사용한다.

V05는 공개 ConvNeXt-small·CoAtNet의 고득점 학습 recipe 재현이 아니다. 현재 코드에 설치된 torchvision ConvNeXt-tiny를 사용하며 crop·resize·interpolation을 S10과 동일하게 고정한다. U01도 segmentation 중심 ROI가 아닌 영상 중심 physical crop, U03도 전체 patch가 아닌 2×2 영역 token이다.

이번 유한 캠페인에서 제외한 항목: Gold58에 맞춘 12개 소견별 가중치 탐색/stacking, 외부 공개 checkpoint의 무검증 결합, 새 LLM 라벨 생성, 고해상도 CoAtNet/DINO 전체 학습, 별도의 평균+최댓값 pooling. 각자의 비용·독립 검증·외부 의존성 문제가 있어 동일 캠페인에 모두 넣지 않았다. 효과가 확인된 방향을 다음 실험의 근거로 남긴다.

## 자동 실행과 결과 위치

2026-10-21 10:34:54 KST 숨김 실행기를 시작했다(PID 35436). 첫 [V00 train](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-v00-gate-blend-train) 버전 1의 업로드와 `RUNNING` 상태만 확인하고 agent 작업을 종료한다. 준비 검증은 전체 105개 테스트, dry-run, 독립 코드 리뷰를 통과했다. 이는 전체 실험 완료를 의미하지 않는다.

실행/재개:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_research_campaign.ps1
```

동일 명령은 저장된 버전에서 이어가며, 이미 runner가 있으면 공유 잠금으로 중복 실행을 막는다. 학습 → 완료 대기 → Internet OFF 추론 → 파일 다운로드/검증 → 분석까지 Python/Kaggle API로 수행한다. 자동 agent·LLM 호출이나 예약 Codex 작업은 없다. 공식 leaderboard 제출은 하지 않고 제출 가능한 CSV까지 생성한다.

- 전체 상태: `outputs/automation/research-campaign-20261021/queue_status.json`
- GPU 부족 상태: 같은 폴더의 `quota_wait.json` (10분 간격 갱신)
- 실행 로그: 같은 폴더의 `runner.log`, `launcher.stdout.log`, `launcher.stderr.log`
- 고정 입력·코드 해시: 같은 폴더의 `inputs.json`
- V/U 개별 결과: `outputs/automation/research-campaign-20261021-<실험ID>/`
- R 재개 결과: `outputs/automation/followup-20261019/`
- U adaptation 상태: `outputs/automation/research-campaign-20261021-adapt/`
- 전체 비교표/소견별 지표/혼동행렬: `outputs/analysis/research-campaign-20261021/`
- 구현·검증 계획: [실행 계획](superpowers/plans/2026-10-21-research-campaign.md).

무료 quota가 부족하면 구매 없이 기다린다. 한 번에 하나의 Kaggle 작업을 실행하므로 완료까지 여러 quota 주기가 필요할 수 있다. 업로드 결과 불명확, 원격 실패, source 변경, 검증 실패는 상태를 보존하고 중단한다. 무한 재학습하거나 agent가 자동으로 문제를 고치지는 않는다.

로컬 후속 진행에는 PC와 네트워크가 켜져 있어야 한다. 종료/절전 중에도 이미 실행된 Kaggle 작업은 원격에서 계속되지만 다음 업로드는 PC에서 재개해야 한다. 실행 중 `experiments/` 및 관련 실행/분석 script를 변경하면 고정 소스 검사가 중단시킨다.

## 해석 범위

동일 gold58의 3-fold 개발 OOF이며, 반복 후보 선택과 공개 약라벨 teacher의 독립성 문제가 남는다. 후보별 차이·seed 변동·12개 소견의 FN/FP를 기록하되 공식 LB 상승 또는 임상 성능으로 표현하지 않는다. 원문에 쓰인 LB 숫자, 이번 개발 OOF, 향후 공식 제출 점수는 서로 구분한다.
