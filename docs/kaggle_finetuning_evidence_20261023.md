# Kaggle 파인튜닝·점수 개선 근거 재수집 — 2026-10-23

## 이번 조사에서 확인한 결론

공개 고득점 코드는 encoder를 MRI에 맞게 실제 학습하고, 충분한 epoch·다중 slice/series·물리적 시야 정규화·약라벨·checkpoint 선택·추론 결합을 함께 사용한다. 현재 프로젝트에 바로 적용할 수 있는 근거는 **전체 encoder 학습, 작은 encoder 학습률, LR warm-up/cosine, validation으로 고른 checkpoint의 가중치 평균 비교**다. 이것이 우리 Public 0.805를 공개 0.95로 올린다고 입증된 것은 아니다. 서로 다른 모델과 라벨·입력을 사용한 공개 결과다. [S6 recipe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359), [새 ConvNeXt 학습 원문](https://www.kaggle.com/code/kutaydogann/rsna-knee-convnext-tiny-single-model-public-0), [CoAtNet 개선 기록](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)

이전 [10월 22일 조사](kaggle_finetuning_evidence_20261022.md), [0.95 근거 감사](kaggle_095_evidence_audit.md), [U 실험 계획](public_method_experiments.md)을 읽은 뒤, 이번에는 점수순 상위 30개와 최신 `train` 검색 30개를 조회하고 주요 원문을 다시 받았다. 60개가 모두 독립 모델이거나 파인튜닝 성공 사례라는 뜻은 아니다. **새 자료는 kutaydogann의 ConvNeXt-Tiny 학습 v1**이다. source·metadata·discussion·공식 점수 응답을 [조사 캐시](../outputs/research/finetuning-20261023/)에 저장했다. 외부 notebook 코드는 실행하지 않았다.

## 점수의 근거를 구분한 원문 목록

`bestPublicScore`는 Kaggle 공식 `KernelsService.ListKernels`가 반환한 **그 notebook의 과거 최고 Public 점수**다. 현재 내려받은 버전이 그 점수를 냈거나 우리가 재현했다는 뜻은 아니다. 작성자 본문의 개선 표는 별도 제출 원장을 받은 것이 아니므로 **저자 보고**로 표기한다. `get_notebook_info` 자체에는 점수 필드가 없다. [공식 Code 목록](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code), [직접 조회 응답](../outputs/research/finetuning-20261023/official_scores.json)

| 원문·확인 버전 | 확인한 점수 근거 | 실제 역할 |
|---|---|---|
| [kutaydogann ConvNeXt-Tiny 학습](https://www.kaggle.com/code/kutaydogann/rsna-knee-convnext-tiny-single-model-public-0), v1, notebook ID 140290111 | 저자 표: 단일 epoch 0.944 → 3 epoch 가중치 평균 0.950. 직접 Public metadata 연결은 미확인 | 전체 ConvNeXt 학습·검증·early stopping·EMA/SWA·offline 평균 소스 |
| [nartaa S6 추론](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo), v2 + [훈련 bundle](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights), dataset v5 | 공식 historical best **0.950**; 저자 보고 folds 1–4 0.948 → all-data 0.950 | ConvNeXt-small 전체 encoder, 공간 attention + window MIL |
| [Dread 학습](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model), v8 + [추론](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), v7 | 추론 notebook 공식 historical best **0.924** | 전체 CoAtNet/MIL 학습 원형 |
| [nartaa CoAtNet](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror), v2 + [변경별 저자 기록](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) | 저자 보고 0.939 → 0.949; 같은 가중치 anatomical mirror 0.948 → 0.949 | 재학습·라벨·외부 데이터·해상도·checkpoint 평균·TTA가 결합된 recipe |
| [Pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), v15 | 공식 historical best **0.891**; frozen 대비 상승폭 없음 | DINOv2 후반 6 blocks + norm 학습 경로와 이미 학습된 package 추론 경로 |
| [goodpjw CoAtNet + R6 + reader](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), v27 | 공식 historical best **0.958**; 단계별 변화는 저자 보고 | 이미 학습된 여러 모델의 해상도·TTA·rank blend |
| [goodpjw 후속 실패 실험](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957), v8 | 저자 보고 0.958 → S6 추가 0.957, gold-58 최적 가중치 0.955 | 모델 추가·검증셋 최적화가 실제 Public 개선을 보장하지 않는 반례 |

조회 중 kutaydogann notebook은 `convnext-rsna-knee-training`에서 현재 링크의 slug/title로 바뀌었다. ID 140290111·v1·본문은 동일하다. 검색 결과의 실행 시각은 **2026-10-23 00:38:44.687 UTC**였다. 일부 `get_notebook_info.last_run_time` 값은 검색 결과/공식 버전 생성일과 일치하지 않아 실제 학습 시각 증거로 사용하지 않았다. [초기 검색 snapshot](../outputs/research/finetuning-20261023/search-new-train.json), [현재 metadata](../outputs/research/finetuning-20261023/kutay-current.metadata.json)

## 새 자료: ConvNeXt-Tiny 전체 학습과 checkpoint 평균

kutaydogann v1은 `convnext_tiny.fb_in22k_ft_in1k_384`의 encoder 전체와 head를 함께 학습한다. `SliceEncoder`는 **한 slice를 1채널로 입력**하고 global feature를 만든다. series 내부와 series 사이에 소견별 attention을 쓰며 plane/fluid/fat metadata가 attention에 들어간다. 제목의 2.5D를 3개 인접 slice를 RGB 채널에 묶는 구현으로 오해하면 안 된다. [원문, model 부분](https://www.kaggle.com/code/kutaydogann/rsna-knee-convnext-tiny-single-model-public-0)

| 항목 | 공개 v1 설정 |
|---|---|
| 이미지 | **160 mm FOV, 320×320**, series별 최대 32 slice cache; train 최대 16 slice 랜덤 부분집합, eval 전체 |
| 좌우·명암 | 왼무릎을 오른무릎 방향으로 정규화; sagittal은 stack 순서 반전, 다른 plane은 영상 flip; series 1–99 percentile |
| 증강 | 동일 series의 모든 slice에 같은 affine/gamma/coarse dropout/noise; series dropout |
| 학습 | batch 1, gradient accumulation **16**, AdamW **encoder/head 모두 2e-4**, weight decay **0.01**, clip norm 1, **constant LR / scheduler 없음** |
| validation | non-gold를 seed42로 **85/15** study split. 보고서 soft label로 BCE 학습, val AUC에는 soft label을 0.5에서 이진화 |
| 종료 | split 실행 상한 **60 epoch**, **val AUC patience10 / tolerance1e-4**. 우리 요청인 valid BCE patience3과 다름 |
| checkpoint | best AUC/loss/gold/last 별도 저장; 본문은 gold를 epoch 선택에 쓰지 않는다고 명시. 최종 all-data 실행은 20 epoch 기본, gold도 포함 |
| 평균 | EMA decay .998 또는 SWA 시작 epoch index10; 공개 기본은 non-EMA. 저장된 epoch12/13/14를 offline 가중치 평균하는 코드 제공 |
| 추론 | non-EMA는 series별 affine/gamma/brightness-contrast **4회 TTA**, EMA는 깨끗한 1회 |

저자 표는 **epoch14 단독 .944, EMA epoch23 .948, epoch12+14 예측 평균 .947, epoch12/13/14 가중치 평균 .950**다. 따라서 후속 실험의 “최저 weak-validation loss checkpoint 대 상위 3 checkpoint 가중치 평균”에는 구체적 근거가 있다. 단, 그 표는 저자 보고이며 우리 데이터에서 같은 효과가 확인된 것은 아니다. 같은 본문의 첫 소개는 해당 모델을 .948이라 쓰고 이후 .950이라 써 표현도 완전히 일치하지 않는다. 본문의 private .934도 독립 확인하지 않았다. [원문, 9. Averaging saved epochs offline](https://www.kaggle.com/code/kutaydogann/rsna-knee-convnext-tiny-single-model-public-0)

전체 데이터를 포함해 재학습한 공개 최종 가중치는 로컬 OOF의 초기값으로 쓰지 않는다. 해당 모델은 gold와 validation study를 이미 학습했을 수 있다. 이 자료에서는 **방법을 참고하고 ImageNet 초기값부터 우리 split으로 학습**하는 것이 비교에 맞다. 공개 입력은 `kutaydogann/labels-calib`, `kutaydogann/rsna-train`이며 우리 weak label과 동일하다는 증거는 없다. [원문, Labels/Full training](https://www.kaggle.com/code/kutaydogann/rsna-knee-convnext-tiny-single-model-public-0)

## S6: 전체 encoder + 5% warm-up/cosine의 직접 recipe

S6 dataset **v5(2026-10-17 갱신)**와 추론 v2는 이전 조사와 같은 버전이다. 오늘 dataset 설명·파일 목록·저자 discussion을 다시 대조했다. 정확한 trainer는 `training_code/s6/train_s6f0_all.py`, 실제 argv는 `run_all_alldata.sh`, 설정은 `E_prior_s42_fall.run_config.json`에 있다. [dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights), [recipe](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)

- **전체 ConvNeXt-small** ImageNet-22k 초기값, 소견별 spatial attention과 window MIL. 설정의 `freeze_blocks=16`은 OFL transformer 분기에 쓰는 값이며 ConvNeXt 동결을 뜻하지 않는다. 이전 동일 버전 trainer 감사 내용을 유지한다.
- 320px cache → 288px crop, 전체 FOV + anatomy anchor 기반 **100mm local crop**, 3-slice window, 3.5mm spacing, study당 64 windows.
- **16 epoch, encoder LR1e-4/head LR1e-3, AdamW WD.05, batch2, 5% LR warm-up 후 cosine**, bf16, seed42, gradient checkpointing.
- 최종 **epoch16 고정**, EMA/SWA 없음. gold58을 제외한 **4,263 study** 학습. gold로 epoch를 고르지 않음.
- 0.948→0.950은 folds1–4 대 all-data 학습량 차이다. full encoder나 scheduler의 단독 개선 폭이 아니다.
- 배포는 읽기 실패 study에 0.949 CoAtNet fallback을 쓴다. S6 자체 OAI 미사용과 fallback 모델의 OAI 사용은 별개다.

라벨·pool은 최초 소개보다 **후속 저자 정정**을 따른다. 실제 target은 `labels_v1_blendjev`의 12개 soft label, 칸별 loss 가중치 1이다. “미언급 prior fill”을 추가 재현하지 않는다. pool4321에서 gold58을 뺀4263이고, 누락86건은 저자의 로컬 디스크 정리 때문이다. 외부 품질 제외 규칙이 아니다. [정정 및 exact training 공개](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/discussion/750180)

## CoAtNet·DINOv2와 0.958 ensemble에서 가져올 것

**Dread v8:** 기본 `freeze_blocks=0`으로 전체 CoAtNet을 학습한다. AdamW backbone/head LR **3e-5/1e-3**, WD.02, **OneCycleLR pct_start=.15**. CLI 기본12epoch와 예제16epoch를 구분한다. 예제384px·batch8·train/eval12/24windows, checkpointing이다. 보고서 soft label parquet는 원문에 미제공이라고 명시되어 있고 gold58로 best/top3 epoch를 선택한다. 이 gold 성능은 독립 검증이 아니다. [training source](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)

**nartaa CoAtNet:** encoder/head LR **4.01e-5/.00195**, WD.0588, batch1×accum16, 16epoch seed42, 12/24windows. 4,349 RSNA weak +2,399 OAI knees, 외부 label은 3소견만 mask supervision한다. accuracy 모델은 Gold58로 고른 **epoch14/11/12 평균**을 쓰고 384 cache의 native320crop, plane에 맞는 anatomical mirror를 평균한다. 단일 epoch14가 동일 조건에서 .947, 평균이 .948이었다는 저자 기록도 있다. 우리 평균 선택은 **gold 대신 weak-validation BCE**를 써야 한다. [변경·실패까지 기록한 원문](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792)

**Pilkwang v15:** DINOv2-small 마지막6blocks+norm, encoder/head **8e-6/1e-3**, WD.02, batch8,10epoch, OneCycle15%. 224/336 해상도를 비교하고 130mm crop·20–80% slice band 등을 쓴다. val AUC로 best checkpoint를 고르며 gold 점수는 진단용으로 출력한다. weights package가 있으면 학습을 건너뛰는 inference 분기가 있으므로 제출 notebook 실행 자체를 새 파인튜닝으로 세지 않는다. [source](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)

**goodpjw v27:** 0.958은 CoAtNet384/224, compact R6 mirror TTA, 2.5D reader와 소견별 rank blend의 조합이다. 원문의 자체 reader 단독 점수는 .929다. score순 첫 `evgendvorkin/rsna-versia-7` v23도 본문에서 goodpjw recipe 재현임을 명시한다. 두 notebook을 두 개의 독립적인 새 파인튜닝 성공으로 세지 않는다. [원저자](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [재현 설명](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7)

S6 추가나 gold58로 12개 blend 가중치를 최적화한 후 Public이 떨어진 공개 반례가 있다. 따라서 학습 횟수·모델 수·gold 점수 증가만으로 채택하지 않고 동일 OOF와 weak validation에서 비교한다. [후속 v8](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

## 현재 로컬 후속 실험에 적용할 범위

현재 부분 파인튜닝을 보존하면서, 다음 **한 개의 후속 학습**에서 기존 입력·weak split·라벨을 유지하고 전체 Swin을 해제한다. encoder를 head보다 낮은 학습률로 학습하며 5% LR warm-up/cosine, 요청한 **valid BCE patience3**, 최저 loss checkpoint 복원을 적용한다. 이 변경은 full encoder와 scheduler가 함께 바뀌는 **recipe 비교**이며 scheduler만의 효과를 분리한 실험이 아니다. S6/CoAtNet/ConvNeXt-Tiny 공개 모델의 정확 재현도 아니다. [full+schedule 근거](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)

학습이 끝나면 **최저 weak-validation BCE checkpoint**와 **weak-validation BCE 상위3 checkpoint 가중치 평균**을 각각 평가한다. epoch 평균은 best 단독보다 좋아질 수도 나빠질 수도 있으므로 비교 결과를 저장하고 자동으로 우월하다고 선언하지 않는다. 이번에 checkpoint 선택에 gold58나 Public을 쓰지 않는다. 이 평균 선택은 공개 저자의 Gold58 선택을 그대로 복사하지 않는 로컬 변형이다. [평균 소스·점수표](https://www.kaggle.com/code/kutaydogann/rsna-knee-convnext-tiny-single-model-public-0), [CoAtNet 평균과 selection 한계](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/discussion/750180)

160/140mm crop, 넓은 slice coverage, anatomy ROI, ConvNeXt 교체, 약라벨 blend, TTA·ensemble은 근거를 수집했지만 이 후속 실행에 한꺼번에 넣지 않는다. 전처리·구조는 [기존 U 실험](public_method_experiments.md)에서 별도로 비교할 수 있다. `patience=3`은 미래 개선 가능성을 증명하거나 실제 과적합 직전 시점을 아는 기준이 아니라, 미리 정한 검증 지표 정체 규칙이다.

## 제출 마감과 조사 한계

공식 API와 Timeline의 정규 제출 마감은 **2026-10-22 23:59 UTC = 10월23일08:59 KST**다. 이번 조회는 마감 후이므로 새 실험을 대회 기간 안의 공식 점수 갱신으로 약속할 수 없다. API/overview에 late 제출 활성화 확인 값은 없었다. [공식 Timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/timeline), [조회 snapshot](../outputs/research/finetuning-20261023/competition.json)

[Future submissions discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/751698)에는 참가자가 마감 후100회/일이라고 답했지만 호스트 확인은 아니다. **실제 late submission 가능 여부는 미확인**이며 조사 중 시험 제출도 하지 않았다.

S6 `training_code/`는 저자가 Apache-2.0, weights는 CC BY-NC4.0이라고 명시했다. 다른 source의 정확한 license를 모두 검증하지 않았고 이번 적용은 공개 코드를 실행·복사하지 않고 개념을 기존 코드에 맞춰 구현하는 범위다. 공개 all-data checkpoint의 OOF 누수, Gold58의 반복 개발 사용, weak label의 독립성 한계는 남는다. [S6 license·provenance](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights)

새 학습의 실제 시작/완료·성능은 이 조사 문서가 검증하지 않는다. 실행 상태는 로컬 runner와 별도 결과 파일로 확인해야 한다.
