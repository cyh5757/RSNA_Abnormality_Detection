# Kaggle knee MRI 파인튜닝 근거 확인 — 2026-10-22

## 판단

현재 공개 코드에는 **ImageNet/DINOv2 사전학습 encoder 자체를 MRI에 맞게 학습하는 사례가 분명히 있다.** 따라서 frozen feature 중심의 현재 파이프라인에서 encoder 파인튜닝을 확대할 근거는 있다. 다만 공개 0.924·0.949·0.950은 저자가 보고한 전체 제출 점수이며, **encoder를 풀었다는 한 가지 변화만으로 얻은 상승폭은 아니다.** 라벨, 해상도, slice 수, MIL, crop, 외부 데이터, checkpoint 선택까지 함께 다르다. 사용자의 0.805와 직접 빼서 기대 상승폭으로 해석하면 안 된다. [Dread 추론](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), [CoAtNet 저자 설명](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792), [S6 저자 설명](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)

## 확인한 원문과 버전

2026-10-22에 Kaggle 연결 도구로 notebook 원문·metadata, discussion 본문·댓글, dataset 설명·파일 목록을 조회했다. S6 설정 JSON·실행 인자·trainer는 공개 dataset v5에서 텍스트로 읽었다. 외부 코드를 실행하지 않았다.

| 원문 | 현재 버전 | 확인 내용 |
|---|---:|---|
| [nartaa S6 추론](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) | 2 | 저자 보고 public 0.950, 공개 가중치 추론 |
| [nartaa S6 weights/training code](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) | 5 | 2026-10-17 갱신, 정확한 trainer·argv·targets·pool 공개 |
| [Dread 학습 코드](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) | 8 | backbone과 head의 별도 optimizer group, 전체 backbone 기본 학습 |
| [Dread 추론 코드](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) | 7 | 저자 보고 public 0.924, 단일 CoAtNet |
| [Pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 15 | DINOv2 후반 6개 block + 마지막 norm 학습 경로 |

캐시 교차확인: Dread 학습 원문은 `outputs/research/kaggle_code_20261019/`의 해당 source.py, nartaa S6 추론 원문은 `outputs/research/kaggle_code_20261021/`의 해당 source.py와 줄바꿈·끝 공백 정규화 후 동일했다. 아래 결론은 10월 22일 실시간 원문을 기준으로 한다.

## S6: 가장 구체적으로 공개된 전체 encoder 학습 사례

S6는 `convnext_small.fb_in22k_ft_in1k` 초기 가중치에서 시작해 소견별 공간 attention과 window MIL을 함께 학습한다. 320 px cache, 288 px crop, full field of view와 100 mm 국소 crop, 3-slice window, 3.5 mm spacing을 쓴다. 학습당 study별 64 windows, batch 2, 16 epochs, seed 42, AdamW encoder LR **1e-4**, head LR **1e-3**, weight decay **0.05**, 전체 step의 **5% warm-up 뒤 cosine decay**다. gradient checkpointing과 bf16을 사용하며 T4 추론에서는 fp16으로 바꾼다. [설명](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359), [v5 설정·trainer](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights)

`E_prior_s42_fall.run_config.json`에는 `freeze_blocks=16`이 있지만, trainer에서 이 인자는 **OFL transformer 분기에만** 적용된다. ConvNeXt 분기는 encoder를 동결하지 않고 학습 가능 encoder 파라미터를 optimizer에 넣는다. 따라서 이 S6를 “ConvNeXt의 마지막 일부만 학습”으로 읽으면 잘못이다. 근거 파일은 `training_code/s6/train_s6f0_all.py`의 `Net.__init__`, `_timm_encoder`, optimizer 구성과 `run_all_alldata.sh`다. [공개 원본 파일 묶음](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights)

**최초 게시물보다 후속 정정이 우선한다.** Discussion 749359의 “unaddressed prior fill / per-cell weight” 표현을 그대로 가져오면 안 된다. 저자가 750180에서 공개한 실제 target 구성은 다음과 같다. [정정·재현 설명](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/discussion/750180)

- non-gold의 12개 `prior` 칸은 `labels_v1_blendjev`로 모두 채우고, `strict` 칸은 비운다. 해당 실행의 BCE 가중치는 모든 칸에서 1이다.
- `blendjev`는 기존 `labels_v1_blend`와 JEV soft label의 칸별 평균이고, JEV가 NaN이면 기존 blend를 유지한다. 공개 parquet와 재생성 스크립트가 있다.
- pool은 4,321건 = 학습 4,263건 + 제외된 gold 58건이다. 나머지 86건은 **저자가 디스크 공간 확보를 위해 로컬 DICOM을 삭제했기 때문**이며, 주최 측 제거 또는 품질 필터가 아니다.
- S6 최종 가중치는 **고정 epoch 16**, EMA/SWA 없음, Gold-58 기반 epoch 선택 없음이다. 학습에 OAI는 쓰지 않았다.
- 배포 추론은 S6가 읽지 못하는 study에 0.949 CoAtNet fallback을 사용한다. S6 자체의 OAI 미사용과 fallback 모델의 학습 이력은 구분해야 한다.

저자 보고 점수는 folds 1–4 학습 S6 **0.948**, all-data S6 **0.950**이다. 이것은 학습 데이터량 비교이며 frozen-vs-finetuned ablation은 아니다. 저자 역시 0.001–0.002 차이를 과대해석하지 말라고 명시한다. [점수 원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)

## Dread / CoAtNet: 학습 코드는 있지만 라벨과 선택 절차가 중요

Dread v8은 초기값을 timm ImageNet 가중치로 설정하고, 기본 `freeze_blocks=0`으로 backbone 전체를 학습한다. encoder/head LR 기본값은 **3e-5 / 1e-3**, AdamW weight decay **0.02**, **OneCycleLR(pct_start=0.15)**다. 이는 앞 15%에서 LR를 올리는 schedule이지 별도의 head-only warm-up이 아니다. CLI 기본은 12 epochs지만 소개된 CoAtNet 384 예시는 **16 epochs**, batch 8, train/eval windows 12/24, gradient checkpointing이다. 이 기본 설정을 실제 0.924 제출의 정확한 모든 hyperparameter라고 단정할 수 없다. [학습 원문](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)

보고서에서 만든 4,349건의 soft label로 학습하고 gold 58건은 학습에서 뺀다. **이 notebook은 원래 학습 라벨 parquet를 제공하지 않는다고 명시한다.** 또한 best epoch와 상위 3개 epoch 평균은 Gold-58 AUC로 고른다. 따라서 저자 보고 gold 0.9167을 독립 검증으로 간주할 수 없다. 현재 학습용 공개 corpus 설명은 44 slices, 추론 v7 설명은 64 slices로 달라, 단순 복사만으로 0.924를 정확 재현한다고 약속할 수도 없다. [학습](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model), [추론](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)

nartaa의 0.949 CoAtNet은 같은 계열을 재학습한 별도 결과다. 저자는 16 epochs, seed 42, encoder/head LR **4.01e-5 / 0.00195**, weight decay **0.0588**, batch 1·gradient accumulation 16, windows 12/24를 명시한다. 4,349 RSNA + 2,399 OAI knees, 세 소견만 외부 supervision, Gold-58로 고른 epoch 14/11/12의 가중치 평균, native 320 crop와 해부학적 mirror TTA가 함께 들어간다. 0.924→0.949를 파인튜닝 단독 이득으로 볼 수 없다. OAI exact linkage manifest는 제한 접근이며 완전 재현에는 별도 접근 자격이 필요하다. [저자 recipe·한계](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792), [학습 공개 설명](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/discussion/750180)

Pilkwang v15도 encoder 학습을 명시적으로 구현한다. DINOv2 마지막 **6 blocks + layernorm**, encoder/head LR **8e-6 / 1e-3**, 10 epochs, batch 8, weight decay 0.02, OneCycleLR `pct_start=0.15`다. 단, pretrained member package가 붙으면 학습 대신 즉시 추론하는 분기가 있으므로 공개 submission notebook을 실행했다는 사실만으로 그 실행에서 파인튜닝했다고 판단하면 안 된다. 현재 확인한 원문에는 고정 encoder 대비 통제된 LB 상승폭이 없다. [원문](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)

## 현재 프로젝트에 적용할 판단

우리 프로젝트도 파인튜닝이 전혀 없었던 것은 아니다. 저장된 개발 OOF 결과에서 **E10 0.713055 → E11 0.713889(차이 +0.000833)**, **S11 0.736386 → S12 0.718888(차이 −0.017499)**다. 이는 last-stage + norm만 1 epoch 학습한 제한적 실험이다. 차이는 원래 정밀도에서 계산한 뒤 반올림했으며, 통계적 유의성 또는 공식 LB 개선을 뜻하지 않는다. [저장 결과](results/2026-10-19/experiments.csv), [spatial trainer](../experiments/spatial_train.py)

S12에는 head warm-up 기본값이 0이고 새 head에서 시작하는 경로가 있어, 충분히 학습된 head를 붙여 encoder를 안정적으로 적응시키는 실험과 동일하지 않다. 이것이 S12 하락의 원인이라고 검증된 것은 아니다. **R06의 weak-only head warm-up 1 epoch 후 부분 파인튜닝**은 이 차이를 점검할 기존 실험이고, **U03 frozen control → U04 전체 Swin 1 epoch → U05 연속 3 epochs**도 이미 구현되어 있다. R06 이후 U 실험을 진행하는 것이 현재 코드 재사용 측면에서 가장 작은 다음 단계다. 단순히 V05 ConvNeXt를 기다리는 것은 encoder 파인튜닝을 시작하는 것과 다르다. [spatial trainer](../experiments/spatial_train.py), [전체 backbone adaptation](../experiments/anatomy_adapt.py), [U 실험 계획](public_method_experiments.md)

Kaggle API에서 확인한 현재 GPU 잔여는 약 **2.83시간**, 갱신은 **2026-10-24 09:00 KST**다. 과거 notebook 전체 실행은 E11 약 **6.16시간**, S12 약 **8.85시간**이었고 S12 feature 재추출만 약 **4.52시간**이었다. 이는 과거 전체 작업 관측치이지 새 학습 시간의 예측값은 아니다. 현재 캠페인의 R06 시작 기준은 11.5시간, U adaptation은 분할 작업과 U00~U03 준비가 필요하다. 따라서 순서를 조정해 재개하려면 GPU 갱신 후 R06을 먼저 실행하는 방식이 현실적이며, 현재 대기 runner를 이 조사 중 변경하지 않았다.

## 검증 한계

이번 작업은 공개 1차 소스와 로컬 결과의 조사다. 새 학습·제출·외부 코드 실행·runner 변경은 하지 않았다. 남의 public score는 저자 보고이며, 별도의 재현 제출이나 frozen-vs-finetuned 통제 비교로 검증하지 않았다. 우리 gold 58건도 반복 사용한 개발 지표이므로 비교에는 같은 split·라벨·전처리·head 조건과 untouched holdout이 필요하다. 공개 all-data competition 가중치를 로컬 OOF의 초기값으로 쓰면 held-out study를 이미 학습했을 수 있으므로, encoder 적응 효과를 평가할 때는 provenance가 명확한 사전학습 초기값에서 각 split에 맞게 학습해야 한다.
