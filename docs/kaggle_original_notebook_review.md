# Kaggle 고득표 원본 3종 코드 검토

조사일: 2026-10-16 KST. Kaggle MCP `get_notebook_info`가 돌려준 실제 notebook source와 `get_dataset_info`/`list_dataset_files`/`get_model_variation`의 공개 metadata를 읽었다. notebook 실행, GPU 학습, 다운로드한 코드 실행, 제출은 하지 않았다. 아래 `cell index`는 **0부터 시작**하며, ID가 있는 셀은 ID를 우선 기재한다. 세 source 모두 저장된 실행 결과와 execution count가 없어, 코드에 존재하는 동작과 실제 성공한 실행을 구분해야 한다.

현재 비교 기준은 **frozen Swin-T, 평면별 5장, 완전 정답 58검사, 3-fold × 3 seeds × 5 epochs, weighted BCE**이며, E03 attention → E04 상대 위치 → E05 방향 임베딩이 준비돼 있다. 따라서 아래 원본의 public 점수는 현재 OOF와 직접 비교할 수 없다. [현재 실험 설정](experiment_notebooks.md), [실행 결과](notebook_run_results.md)

| 원본 | 확인 버전·표 | 실제 제공하는 것 | 우리 실험에 유용한 부분 |
| --- | --- | --- | --- |
| [Pilkwang: RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | v15 · 736 | 공개 DINOv2 체크포인트 추론 + 가중치 미첨부 시 자체 학습 | 약라벨·gold 결합, 6개 촬영 슬롯, 소견별 attention, 전처리 계약 |
| [Mattia Angeli: Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) | v38 · 328 | 여러 공개 모델의 **추론 전용 앙상블** | 다양한 인코더/시야의 예측을 결합하는 후반 실험 참고 |
| [prvsiyan: read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | metadata v62 · 267 | 보고서 분석 + Pilkwang 추론 기반 + 여러 누적 학습/추론 분기 | 약라벨 합의, frozen 의료 인코더+작은 query head, 선택 편향 경고 |

prvsiyan source 내부에는 V63·V64 이름도 있어 metadata 버전과 내부 실험 이름이 일치하지 않는다. 제목이나 최신 셀의 이름만으로 실행된 모델을 단정할 수 없다.

## 1. Pilkwang: 가장 먼저 읽을 학습 원본

**학습/추론 분기.** `code-33/main()`은 `find_weights()`가 패키지를 찾으면 `infer_from_package()` 실행 후 반환한다. 실제 첨부 목록에 `pilkwang/rsna-knee-weights`가 있으므로 공개 구성은 이 경로를 우선한다. 패키지에는 `.pt` 20개, `manifest.json`, `oof.npz`, `merge_gain.npz`가 존재한다. 가중치를 붙이지 않으면 아래 자체 학습 경로가 살아 있다. **학습 코드 기본값과 20개 패키지 멤버 각각의 학습 조건이 같다고 가정하면 안 된다.** [원본, `code-30`, `code-33`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), [weights 목록](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights)

- **감독 신호:** 전체 train 보고서에서 12개 soft target과 confidence를 얻는다. `read_labels()`는 공개 `report_labels_v2.csv`를 우선하며 표에서 빠진 검사는 9개 언어 규칙 추출기로 보충한다. 완전 gold는 실제 정답으로 대체하고 가중치 3.0, 나머지는 `0.25 + 0.75 × confidence`를 BCE에 곱한다. 이것은 현재 클래스 불균형용 `pos_weight`와 다른 가중치다. 규칙 추출기의 미언급은 score 0.28/confidence 0.05여서 loss weight 0.2875로 남는다. **미언급을 제외하는 label mask가 아니다.** [`code-11/extract`, `code-14/read_labels`, `code-33`의 Y/W 구성·loss](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
- **영상 입력:** 기본값은 `GROUP=3`, `N_GROUP_MAX=1`, 6 slots, stack 20–80%에서 균등 추출, 130 mm 중앙 crop, cache 336 px이며 224/336 px 두 학습 설정을 비교한다. 6개 슬롯은 sagittal/coronal/axial fluid+fat-suppressed, sagittal fluid non-fat-suppressed, coronal/sagittal T1이다. 슬롯에 맞는 시리즈가 없으면 비워 presence mask를 유지한다. 같은 슬롯 후보 중 slice 수가 큰 시리즈를 고른다. 기본 3장은 stack에 퍼진 3채널 입력이므로 **인접 3장이라고 부르면 틀린다.** [`code-13`, `code-16/pick_slots`, `code-19/read_slot`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
- **전처리:** 위치와 방향으로 계산한 slice-normal 투영 순서, 실패 시 InstanceNumber; rescale slope/intercept; sampled series의 1/99 percentile; 물리 crop 후 resize; ImageNet 정규화. 우측은 coronal/axial 수평 반전, sagittal stack 역순으로 정규화한다. 학습된 패키지에는 별도 legacy ordering/laterality/slot fallback/decode-fill 규칙도 있어 manifest 설정을 그대로 재현한다. 단순히 최신 전처리로 고쳐서 기존 가중치와 결합하지 않는다. [`code-15`, `code-19`, `code-21`, `code-26`, `code-30/adopt_config_globals`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
- **모델:** DINOv2-small CLS+평균 patch 특징을 연결한다. `SlotHead`는 256차원 투영, learned slot embedding, **12개 소견별 query**로 슬롯 attention을 계산하며 missing slot을 softmax 전에 가린다. 기본 학습은 마지막 6 transformer blocks와 최종 LayerNorm을 열고 backbone LR 8e-6/head LR 1e-3, AdamW, 10 epochs다. 현재 frozen encoder 실험과 다른 비교다. [`code-25/SlotHead`, `code-26/Model`, `code-28/build_model`, `code-33`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
- **검증:** 보고서 원문 MD5 `% 5 == 0`을 고정 holdout으로 삼아 같은 보고서를 묶는다. weak target을 0.5로 이진화한 holdout macro AUC로 epoch와 해상도를 선택한다. gold 검증은 holdout에 들어온 소수 gold만 별도 보고한다. 전체 gold 58검사의 반복 OOF와 같은 평가가 아니다. native 학습 기본값은 5-fold CV가 아니라 **하나의 80/20 holdout**이다. 패키지 멤버 학습 provenance는 manifest를 더 읽어야 확정된다. [`md-32`, `code-33`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)

**재사용 결론:** 큰 앙상블보다 이 원본의 label-source 검사, protocol slot mask, 소견별 query 구조를 먼저 참고한다. 현재 E03–E05의 방향 내 Transformer attention과 여기의 소견별 슬롯 attention은 서로 다른 구조다. 이를 동일한 attention 기법의 재현이라고 기록하지 않는다.

## 2. Mattia Angeli: 후반 앙상블 참고용

원본 첫 markdown에 inference-only라고 명시하며, 확인한 실행 경로도 학습된 가중치 로딩·예측·혼합이다. label dataset을 첨부했더라도 이 notebook이 원본 가중치의 weak-label 학습을 재현하는 것은 아니다. 훈련 검사 수·teacher 구성·loss·fold별 누수 여부는 **이 추론 source만으로 확인 불가**다. [원본, cell index 0–20](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs)

- **DINO 기반:** Pilkwang 계열 `Model`/`SlotHead`를 사용하고 20/20 checkpoint fingerprint 일치를 다음 arm의 전제 조건으로 둔다. 해상도·slice·crop은 멤버 config를 따른다. [`cell index 8–14`](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs)
- **A5/fold arm:** 6 plane×fat-suppression slots, 슬롯당 16장, 336 px, 130 mm crop, slice band 12–88%, slice별 1/99 percentile이다. checkpoint config로 backbone·readout을 구성한다. padding/slice mask를 전달하고 DINO 결과와 rank 45%로 혼합한다. `mask[s_i]=len(picks)`라서 모든 선택 slice의 decode 성공까지 보장하는 mask는 아니다. [`cell index 14–18`, `build_study`, `_micro`, `A5_W=0.45`](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs)
- **의료 encoder arm:** RadImageNet ResNet-50와 5-fold heads, E13 heads를 hash 검증 후 불러온다. 8 slices/slot, 224 px 경로가 보인다. 해당 셀은 head 학습 코드가 아닌 inference 코드다. [`cell index 19`, `_rad_load_public_heads`, `_rad_load_e13_heads`, `_rad_main`](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs)
- **CoAtNet/Raptor arm:** `cell index 20`의 embedded `_KE_SRC`는 `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, 384 px backbone과 336/384 px cache, 서로 다른 slice span의 공개 Raptor checkpoints를 로딩한다. `_coat_substitute()`는 별도 공개 CoAt residual-gated inference module을 읽어 Raptor rank 60%+CoAt rank 40%를 만든다. 마지막에는 이 결과를 기존 arm과 소견별로 섞는다: 기본 60%, ACL/Lateral OA/Fracture 75%, Medial Meniscus 80%, Lateral Meniscus 100%. 이 숫자는 코드상 고정 혼합 비율이며, 우리 58건에서 최적이라는 근거가 아니다. [`cell index 20`, `_KE_SRC`, `_coat_substitute`, `_coatnet_weight`](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs)

첫 markdown은 CoAt arm을 384 px 인접 3장, 소견별 공간 gated head와 study-level attention, checkpoint 3개의 rank 평균으로 설명한다. 해당 모델의 상세 학습 코드/성능을 이번 검토에서 독립 재현한 것은 아니다. 공개 CoAt dataset v3는 존재하고 버전 메모에 geometry 및 Gold58 portability audit 수정이 적혀 있다. [원본 설명](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs), [공개 CoAt 패키지](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3)

**재사용 결론:** 단일 모델 개선의 출발점으로는 복잡하다. 서로 다른 인코더를 같은 고정 fold에서 독립 학습하고 OOF가 쌓인 후, 전체 소견 공통 평균 또는 rank 평균 정도부터 비교하는 참고자료로 둔다. source에 보이는 소견별 수동 비율을 현재 58검사에 그대로 가져올 이유는 없다.

## 3. prvsiyan: 약라벨과 frozen 의료 encoder 참고, 누적 분기는 분리해서 읽기

초반은 4,407개 보고서/58 gold의 다국어 rule extractor 분석이다. 그러나 `v10-proven-ensemble`부터 실제 기반은 Pilkwang 20-checkpoint inference로 바뀌고 Synovitis specialist, PCA, EfficientNet-B3, RadImageNet, header, 외부 encoder, Curia-2 분기가 계속 추가된다. 최신 source 전체를 하나의 간단한 학습 레시피로 취급하면 안 된다. [`v10-proven-ensemble`, `v26-synovitis-runtime`, `v34-train-only-pca-specialists-runtime`, `v50-orthofoundation`, `curia2-tpu-guarded`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)

- **3-source teacher:** `v50-orthofoundation/make_targets()`는 Pilkwang `report_labels_v2.csv`, Steven `llm_labels_v2.csv`, Lixin `labels_llm_gpt56sol.csv`를 study ID로 정렬한다. 소견마다 최소 2개 teacher가 있어야 하고 평균 soft label을 쓴다. 가중치는 `0.15 + 0.85 × (0.65 × agreement + 0.35 × certainty)`이고 gold는 실제 정답/3.0으로 대체한다. confidence가 낮아도 floor가 있어 **unknown mask가 아니다**. 예컨대 세 teacher가 모두 0.5면 agreement=1/certainty=0이므로 가중치가 0.7025다. B3 `build_public_consensus()`의 별도 기본 floor는 0.20이다. [`v50-orthofoundation/make_targets`, cell index 46 `build_public_consensus`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)
- **우리가 비교하기 쉬운 arm:** frozen RadImageNet ResNet-50 → 2048차원 slice feature → 512차원 query head. plane embedding, slice-position embedding, 12개 소견 query, 8-head cross-attention, masked global mean과의 결합을 사용한다. 기본 8 slices/slot·224 px이며 `CROP_MM=10000`으로 사실상 물리 중앙 crop을 끈 경로다. encoder state hash와 parameter count를 검사한다. [`v50-orthofoundation`, `load_radimagenet`, `FoundationQueryHead`, `encode_radimagenet`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)
- **head 학습/검증:** 정규화한 보고서 hash로 `GroupKFold(5)`를 나누고 fold 밖 head로 gold 58건을 예측한다. BCE를 가중치 합으로 정규화하며 최대 24 epochs, weak-val AUC 기준 early stopping을 쓴다. global blend weight도 outer fold에서 선택하는 코드가 있다. 현재 58 gold·5 epochs 실험과 학습량/감독/모델선택이 다르다. [`v50-orthofoundation/report_groups`, `train_fold`, `main_v52`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)
- **다른 분기의 상태:** 공개 B3 학습 source는 있지만 누적 cell마다 frozen/final-block fine-tune/추론 전용 경로가 다르다. cell index 64는 명시적으로 five-fold inference-only이며 특정 fold dataset을 요구한다. `curia2-tpu-guarded`는 Curia-2 frozen encoder+query heads를 TPU에서 시도하고 실패 시 과거 artifact로 돌아가는 코드다. 해당 input 일부는 metadata slug가 빈 문자열이므로 이번 정보만으로 모든 파일의 공개 재사용 가능성을 보증할 수 없다. [`cell index 44–64`, `curia2-tpu-guarded`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)

**점수는 작성자 보고와 분리한다.** 첫 scorecard는 Pilkwang 0.891, 이 notebook V52 0.906을 주장한다. `curia2-tpu-frontier`는 이후 label-wise multi-encoder overlay가 0.887로 하락했다고 명시한다. 이번 작업에서 해당 공식 제출 row를 독립 조회하거나 점수를 재현하지 않았으므로 **원본에 기록된 결과**로만 인용한다. 현재 0.619577 OOF 대비 개선량으로 뺄 수 없다. 같은 58건을 보고 반복 선택하면 과적합할 수 있다는 점을 원본 자체가 드러낸다. [scorecard cell index 1 및 `curia2-tpu-frontier`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)

## 4. 약라벨·가중치 공개 상태와 중요한 한계

| 자원 | 확인한 공개 상태/표시 라이선스 | 해석 |
| --- | --- | --- |
| [Pilkwang report labels](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels) | CC0, `report_labels_v2.csv`와 `api_labeler.py` 존재 | YES/NO/UNK와 confidence 기반 표; 생성 모델/전체 프롬프트는 `api_labeler.py`를 더 읽어야 확인 |
| [Steven labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) | CC0, v2/v4 표 제공 설명 | v2 Synovitis 미언급을 Effusion으로 채우는 변경을 같은 58 gold 점수로 결정했다고 설명 |
| [Lixin labels](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56) | CC0, dataset metadata 확인 | 제목 외 생성 과정의 상세 설명은 없음 |
| [Pilkwang weights](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights) | CC0, 20 checkpoints와 manifest/OOF 존재 | 전체 패키지 약 1.79 GB; fold provenance는 파일명만으로 보장 안 됨 |
| [DINOv2-small](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small) | Apache 2.0, Meta model instance | 공식 model metadata가 facebookresearch/dinov2를 provenance로 연결 |
| [Mattia fold weights](https://www.kaggle.com/datasets/mattiaangeli/knee-mri-fold-weights), [CoAt weights](https://www.kaggle.com/datasets/mattiaangeli/rsna-knee-coat-resgated-ep10-top3) | 각각 CC0 | 게시된 패키지의 표시 라이선스이며 모든 상위 pretrained 자산의 조건을 대신하지 않음 |
| [RadImageNet ResNet-50 mirror](https://www.kaggle.com/datasets/marwanmath/resnet-50-radimagenet-marwan) | **CC BY-NC-SA 4.0** | DINOv2와 같은 이용 조건이 아님; notebook의 “official” 표기는 저자 설명이며 미러 provenance까지 검증한 것은 아님 |

**약라벨 선택 누수:** Steven v2 설명은 같은 58 gold를 보고 Synovitis proxy를 수정했다고 밝힌다. 따라서 image model의 fold를 분리해도 teacher 생성/보정 과정까지 해당 gold와 독립이라고 할 수 없다. 공개 라벨 품질 AUC와 image model OOF, Kaggle public 점수는 서로 다른 측정이다. frozen encoder+weak label 비교는 유용하지만, 해당 58건 성능을 최종 일반화 성능으로 제시하지 않는다. [Steven 원문 설명](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)

노트북 코드 자체의 라이선스는 이번 `get_notebook_info` metadata에 없었다. dataset CC0를 notebook 전체 또는 모든 외부 모델의 라이선스로 확대 해석하지 않는다. 체크포인트/embedded payload를 실행하지 않았으며, 모든 공개 파일을 실제 내려받아 hash 일치를 검증하지 않았다.

## 5. 현재 파이프라인에 가져올 순서

1. **E03–E05 비교를 먼저 끝낸다.** 동일 Swin-T·5장·58 gold·고정 fold에서 attention/위치/방향의 효과를 분리한다. 원본에는 이 구성요소가 모두 있지만, 우리 조건에서 개선된다는 실험 증거는 아니다.
2. **다음 별도 실험은 frozen Swin-T 그대로 약라벨 학습을 추가한다.** backbone까지 동시에 바꾸지 않는다. 검증 gold와 같은 보고서 그룹은 그 fold의 weak training에서도 제외하고, gold/weak/unknown을 별도 표시한다. 공개 source ID·버전과 보정 이력을 남기며 미언급을 자동 음성으로 만들지 않는다. 현재 loss의 클래스 `pos_weight`와 샘플/소견 confidence weight는 구분한다.
3. **그다음 촬영 조건과 pooling을 비교한다.** 현재 3개 plane에서 6 protocol slots로 확장하거나 소견별 query attention을 추가하는 실험은 각각 분리한다. missing slot을 억지로 복제하지 않고 mask를 유지한다. 물리 crop·laterality·slice band 변경도 고정 전처리 대비 별도 실험이다.
4. **감독 신호가 늘어난 후 partial fine-tune 또는 frozen 의료 encoder를 비교한다.** Pilkwang의 마지막 6블록 개방을 58 gold 조건에 바로 복사하지 않는다. RadImageNet query head는 frozen 구조의 다음 후보지만 표시 라이선스와 input provenance를 함께 기록한다.
5. **앙상블은 OOF가 확보된 후 최소 구성부터.** 동일 study 순서의 예측을 정렬하고 전체 소견 공통 평균/rank 평균을 비교한다. 58건에서 소견별 모델·비율을 반복 최적화하는 Mattia/prvsiyan식 큰 누적 앙상블은 현재 단계의 우선순위가 아니다.

이번 검토는 코드와 공개 metadata 분석이다. 대회 점수 재현, GPU 비용/시간 측정, 원본 전체 학습, 모든 가중치 provenance·라이선스 전수 확인은 생략했다. 가장 큰 잔여 위험은 **동일 58 gold를 통한 teacher 보정과 반복 모델 선택으로 OOF가 낙관화되는 것**이다.
