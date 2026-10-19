# Most Votes 공개 Notebook에서 가져올 방법

조사일: 2026-10-16. [대회 Code](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code)의 Most Votes에 대응하는 Kaggle MCP `search_notebooks`의 `sortBy=VoteCount`로 상위 20개를 조회하고, 아래 10개의 원본 코드·설명을 확인했다. 투표 수와 버전은 조회 시점의 값이다. 공개 코드는 실행하지 않았다.

**추천은 `Pilkwang baseline`의 학습·약라벨 처리와 `goodpjw2008`의 2.5D reader 구조를 참고하는 것이다.** 모델 이름만 바꾸는 것보다, 지금 사용하지 않는 보고서 학습 데이터와 MRI의 인접 슬라이스·촬영 프로토콜을 어떻게 사용할지 먼저 실험할 가치가 있다. 이는 코드 비교에 근거한 실험 우선순위이며, 우리 데이터에서의 성능 향상이 검증되었다는 뜻은 아니다.

## 원본을 확인한 목록

순위는 상위 20개 조회 결과 안에서의 위치다. LB 숫자는 작성자 설명에 나온 값이며 직접 제출하여 확인한 점수가 아니다.

| 순위 / 투표 | Notebook | 실제 내용과 활용 판단 |
| --- | --- | --- |
| 1 / 736 | [Pilkwang — RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), v15 | **학습 출발점으로 추천.** 보고서 약라벨 + gold, DINOv2 일부 fine-tuning, 여러 촬영 프로토콜의 특징 통합. 기본 설정에서는 첨부 가중치로 바로 추론하므로 학습 분기를 확인해야 한다. |
| 2 / 591 | [Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb), v77 | 미리 계산한 효율성 CSV를 보여주는 Notebook. **모델 학습 코드가 아니다.** 실행 시간 비교용이다. |
| 3 / 428 | [RSNA_Versia_7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7), v22 | R7 ConvNeXt-small + Raptor CoAtNet 등 공개 결과를 재구성한 추론 앙상블. 독립적인 새 backbone 실험으로 세면 안 된다. |
| 4 / 328 | [Bend the knee to the dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs), v38 | DINO·의료 encoder·CoAtNet 등을 조합한 **추론 전용 앙상블**. 후반 앙상블 참고용이며 backbone 하나의 학습 코드로 취급하면 안 된다. [상세 조사](kaggle_original_notebook_review.md) 참고. |
| 5 / 310 | [Blend Gold — old dataset](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-old-dataset), v2 | Raptor + S6 ConvNeXt-small 4-fold의 순위 기반 앙상블. 이전 Dataset 버전에 고정된 코드로, 새 학습 구조보다 공개 가중치 재현 참고용이다. |
| 6 / 267 | [Read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee), metadata v62 | 보고서 라벨의 불확실성·합의·평가 문제를 읽을 가치가 있다. 현재 버전은 여러 모델·분기가 누적되어 전체를 하나의 학습 레시피로 복사하기 어렵다. |
| 8 / 249 | [Fast 2×T4 inference](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference), v11 | **속도 개선 참고.** 같은 DINO 앞쪽 블록 공유, CPU/GPU 작업 중첩, DICOM 메타데이터·픽셀 캐시, 메모리 제한. 복잡한 앙상블 추론 코드이며 첫 학습 실험에는 과하다. |
| 11 / 218 | [R7 + Raptor latest dataset](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), v1 | R7 ConvNeXt-small 4-fold + Raptor, label별 percentile rank를 50:50 혼합. 높은 점수의 제출 참고용이나 학습 전 과정이 이 Notebook에 담긴 것은 아니다. |
| 14 / 197 | [Twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), v7 | **단순한 모델 구조 설명으로 추천.** 2.5D CoAtNet + 소견별 attention pooling. 단독 public LB 0.924라고 작성자가 보고. 공개 Notebook은 추론 코드이고 첨부 Dataset은 가중치 2개다. |
| 19 / 166 | [CoAtNet + R6 + 2.5D MIL](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957), v26 | **현재 질문에 가장 잘 맞는 구조 참고.** Part B의 자체 ConvNeXt reader: 2.5D + 촬영 슬롯/위치 임베딩 + Transformer + 소견별 pooling. 전체 앙상블과 reader 단독 성능을 구분해야 한다. |

## 1. 가장 직접적인 구조 참고: 2.5D ConvNeXt reader

[원본 Notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957)의 Part B에 `preprocess.py`, `knee.py`, `infer.py`를 생성하는 코드가 있다. 조회한 v26 기준 셀 인덱스 18~20(0부터 계산)이다. `knee.py`의 `StudyDataset`, `window_centres`, `KneeNet`이 핵심이다.

```text
MRI 검사
  → 방향 3종 × 지방 억제 여부 2종 = 최대 6개 촬영 슬롯
  → 각 슬롯에서 여러 중심 위치 선택
  → 같은 series의 [앞 슬라이스, 현재 슬라이스, 뒤 슬라이스]를 3채널로 입력
  → ConvNeXt-Tiny로 각 3장 묶음의 특징 추출
  → 특징 투영 + 촬영 슬롯 임베딩 + 정규화된 슬라이스 위치 임베딩
  → 모든 슬롯/위치의 token에 작은 Transformer 적용
  → 12개 소견마다 서로 다른 attention으로 특징 통합
  → 소견별 분류 가중치 → 12 logits → sigmoid → 12개 예측 점수
```

기본 `KneeNet`은 특징을 384차원으로 투영하고, 2층·8-head Transformer를 사용한다. 위치는 0~1로 정규화한 슬라이스 인덱스를 MLP에 넣는다. 위치 임베딩이 절대적인 해부학적 위치의 일치를 보장하는 것은 아니므로, 방향 정렬·series 선택도 같이 이해해야 한다. 결측 촬영 슬롯에는 attention mask를 적용한다. [원본 코드](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957)

이 구조는 모든 소견이 같은 평균 특징을 쓰게 하지 않는다. 예를 들어 ACL 점수와 Effusion 점수가 서로 다른 슬라이스에 큰 가중치를 줄 수 있다. 이는 사용자가 우려한 “특징을 평균하면서 중요한 부분을 잃는 문제”를 직접 실험할 수 있는 방법이다. 단, attention만으로 정확한 병변 위치가 보장되는 것은 아니다.

전처리에는 DICOM 방향·위치를 이용한 정렬, PixelSpacing 기반 크기 보정, series 단위 명암 정규화가 있다. `MM_PER_PX=0.4`, `SIZE=384`이므로 준비된 영상의 시야는 약 153.6mm다. 위치 샘플링 범위는 4~96%, 학습에서는 구간 안에서 흔들고 추론에서는 일정하게 뽑는다. 이 값들을 한꺼번에 가져오면 어느 변화가 효과를 냈는지 알 수 없으므로 별도 비교가 필요하다. [원본 코드](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957)

재현 자료도 비교적 잘 공개되어 있다. 연결된 [2.5D ConvNeXt reader Dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)의 v1 파일 목록에서 `train.py`, `make_labels.py`, `oof.py`, 전처리·모델·추론 코드와 3개 fold 가중치를 확인했다. Dataset 표기 라이선스는 Apache 2.0이다. Notebook 안의 추론·모델 소스는 검토했지만, 별도 Dataset의 학습 파일 본문과 전체 재학습 가능성은 아직 검증하지 않았다.

**제목의 0.957을 이 reader의 점수로 보면 안 된다.** 작성자는 reader 단독 3-fold public LB를 **0.929**로 설명한다. 0.957은 Raptor/R6/R7 등을 조합한 더 큰 앙상블의 과거 실행 설명이다. v26의 결과까지 독립 확인한 것은 아니다. 작성자 기록에는 자체 reader를 크게 혼합했을 때 점수가 하락한 사례도 있어, 새 모델을 더하면 반드시 좋아진다고 볼 수 없다. [Notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-957), [reader 설명](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)

## 2. 학습 데이터 활용 참고: Pilkwang baseline

현재 프로젝트는 확정 정답 58건만으로 분류기를 학습한다. 반면 [Pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)에는 보고서를 이용해 만든 약라벨과 gold 라벨을 섞는 학습 코드가 있다. 라벨이 없는 나머지 검사를 학습에 활용하는 방법이므로, backbone 교체와 별개의 큰 변화다.

조회한 v15에는 DINOv2-small의 마지막 6개 블록과 LayerNorm을 학습하는 코드, 낮은 backbone 학습률과 더 높은 head 학습률, confidence에 따른 BCE 가중치가 있다. 즉, 현재의 **동결된 DINO 특징 + 58건 head 학습**과 같은 조건이 아니다. 현재 실험에서 DINO가 Swin보다 낮았다는 이유로 이 접근까지 부적합하다고 결론 내릴 수 없다. [원본](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), [현재 실행 결과](notebook_run_results.md)

약라벨을 사용할 때는 “보고서에 언급 없음”을 정상 0으로 볼지, 불확실한 target으로 볼지, loss에서 제외할지 먼저 고정해야 한다. 공개 구현의 confidence weight는 항상 완전한 mask인 것도 아니다. Pilkwang은 약라벨에 `0.25 + 0.75 × confidence`를 쓰므로 confidence가 0이어도 학습에 남는다. gold 가중치는 3이다. 세부 분기·평가 분할은 [원본 3종 상세 조사](kaggle_original_notebook_review.md)에 기록한다.

## 3. 가볍게 이해할 수 있는 소견별 pooling: 단일 CoAtNet

[Dread Development의 단일 모델](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)은 `RaptorClassifier.head`가 짧아 소견별 pooling을 이해하기 좋다. 각 입력 묶음의 특징에서 12개 attention 값을 만들고, 입력 묶음 축으로 softmax를 적용하여 소견별 특징 12개를 만든다. 마지막에 소견별 가중치로 각각 점수를 계산한다.

전처리는 5개 촬영 슬롯에 64장, 6~94% 범위, 140mm crop이다. v7의 추론 설정은 42개 3채널 window, CoAtNet `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`다. 단독 public LB 0.924, gold 58건 AUC 0.9167은 **작성자가 보고한 값**이다. [코드](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model), [가중치 설명](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

그대로 복사할 때 주의할 코드도 있다. 원본은 여러 슬롯을 하나의 volume에 이어 붙인 뒤 `_eval_centers`/`eval_windows`에서 triplet을 만든다. 해당 함수는 series 경계를 명시적으로 검사하지 않아 경계의 묶음이 서로 다른 series를 포함할 가능성이 있다. 우리에게는 **같은 series 안에서 인접한 슬라이스를 묶는 방식**이 더 명확하다. 원본 가중치를 재사용할 때 입력 규약을 임의로 바꾸면 학습 시 입력과 달라지므로 별도 검증해야 한다.

## 4. 현재 코드와 차이

현재 구현 근거는 [experiments/core.py](../experiments/core.py)의 `image_tensor`, `study_tokens`, `StudyHead`와 [experiments/train.py](../experiments/train.py)의 gold 선택 및 encoder 동결이다.

| 항목 | 현재 실행한 backbone 비교 E00~E02d | 참고할 변화 |
| --- | --- | --- |
| 학습 대상 | gold 58건 | 보고서 기반 약라벨 4,349건도 활용; gold 처리/평가를 별도로 고정 |
| Backbone | 사전학습 가중치 동결 | 충분한 학습 데이터로 마지막 블록 일부 fine-tuning |
| 촬영 선택 | 방향별 series 1개, 최대 3개 | 방향과 지방 억제/영상 대조 특성에 따른 여러 슬롯 |
| 위치 범위 | series 중앙 25~75% | 범위를 넓혀 주변 구조도 포함하는 별도 실험 |
| 입력 3채널 | 같은 회색조 한 장을 3번 복제 | 같은 series의 인접한 세 장 |
| 특징 통합 | 방향별 평균 → concat | 소견별 attention pooling; 필요하면 슬롯 간 Transformer |
| 위치 정보 | E02까지 없음 | E04의 상대 위치, 더 나아가 protocol slot 정보 |

준비된 E03~E05는 방향별 작은 Transformer → 평균 → concat 구조다. 위 reader의 **모든 촬영 슬롯 간 Transformer + 소견별 12개 pooling**과는 다르다. 새 실험에 가져올 때 이 차이를 하나씩 분리해야 한다.

현재 최고인 Swin-T의 **0.619577은 58건, 3-fold·3-seed OOF macro AUC 평균**이다. 다른 작성자의 public LB 0.929/0.957이나 gold holdout 점수와 직접 비교할 수 없다. [현재 측정 조건과 결과](notebook_run_results.md)

## 5. 다음 실험에 반영할 우선순위

기존 E03→E04→E05는 같은 조건에서 끝내고, 아래는 후속 실험 후보로 둔다. 이번 조사에서는 자동 실행 큐나 학습 코드를 바꾸지 않았다.

1. **약라벨과 평가 기준부터 준비한다.** Study ID 중복·결측, label 순서, 언급 없음 처리, public teacher가 gold 정답을 본 적이 있는지를 확인한다. 이미지 학습은 분리해도 teacher가 검증 정답을 보고 조정되었다면 완전한 독립 검증은 아니다.
2. **Backbone을 유지하고 약라벨 추가 효과를 측정한다.** 가장 먼저 학습 데이터 변화의 효과를 분리한다. 정해진 split·평가 방식은 이후에도 유지한다.
3. **MRI 입력을 하나씩 바꾼다.** 물리적 crop, 샘플링 범위, 촬영 슬롯 확장, 2.5D를 각각 별도 실험으로 비교한다. 2.5D와 촬영 슬롯을 동시에 바꾸지 않는다.
4. **소견별 attention pooling을 비교한다.** 기존 평균 pooling과 동일한 입력에서 비교한 다음, 필요한 경우 전체 슬롯 간 Transformer를 추가한다. 위치·슬롯 임베딩도 분리해서 평가한다.
5. **부분 fine-tuning을 비교한다.** 같은 데이터·입력·head로 frozen 대비 효과를 측정한다. 그다음 CoAtNet/ConvNeXt/DINO를 다시 비교하면 backbone 차이를 해석하기 쉽다.
6. **좋은 모델이 남으면 앙상블과 추론 최적화를 적용한다.** [Fast 2×T4](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference)의 캐시·작업 중첩을 참고한다. 같은 58건이나 public LB에 반복해서 맞춘 소견별 혼합 비율을 그대로 가져오지 않는다.

이 순서는 실험 제안이다. 각 단계의 AUC·소견별 AUC·학습/추론 시간·메모리를 보고 유지 여부를 결정하며, 성능 향상을 가정하지 않는다.

## 평가·재사용의 확인 범위

- 공개 약라벨의 주의점: [Steven Lee의 라벨 Dataset](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)은 같은 58 gold를 이용해 Synovitis의 미언급 처리 정책을 개선했다고 설명한다. 이를 사용한 우리의 58건 평가도 teacher 선택의 영향을 받을 수 있다. 자세한 출처와 가중치 해석은 [상세 조사](kaggle_original_notebook_review.md) 참고.
- R7 계열은 fastMRI 보조 학습이나 OAI 출처도 언급한다. 공개 Notebook에 첨부되어 있다는 사실만으로 모든 외부 데이터의 접근·이용 조건을 확인한 것은 아니다. 원본의 Dataset 버전과 학습 데이터 출처를 함께 확인해야 한다. [R7 설명](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click)
- Dataset metadata에서 ConvNeXt reader의 Apache 2.0, 단일 Raptor 가중치의 CC0 표기를 확인했다. 이는 확인한 파일 묶음의 표기이며 모든 상위 학습 데이터의 조건까지 보장하지 않는다.
- 공개 코드의 동작·실행 시간·hidden test 완주·LB 점수는 재현하지 않았다. 조회한 버전의 소스와 Dataset metadata/file list에 근거한 조사다. 추후 재사용하면 Notebook과 Dataset 버전을 고정해야 한다.
