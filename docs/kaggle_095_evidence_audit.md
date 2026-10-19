# 공개 0.95 계열: 실제 개선 원리와 근거 재점검

확인일: 2026-10-19. 기존 25개 원문 검토와 저장된 실행 코드·manifest를 다시 대조했다. 이 중 **22개는 고득점 후보, 3개는 학습·설계 대조 자료**다. 25개의 독립적인 0.95 이상 성공을 검증했다는 뜻은 아니다. 전체 목록은 [원문별 검토](kaggle_095_twenty_notebook_review.md), 비교 계획은 [우리 코드와의 차이](kaggle_095_comparison_and_plan.md)에 있다.

Kaggle MCP로 원저자 두 페이지를 다시 조회했다. goodpjw의 0.958 문서는 v27·370 votes, heliosli의 0.9586 문서는 v7·162 votes였다. `get_notebook_info` metadata에는 점수가 없었지만, 이번 재조사에서는 **공식 웹 API `KernelsService.ListKernels`의 `bestPublicScore`도 별도로 확보**했다. goodpjw와 heliosli 두 Notebook의 공식 최고 공개 점수는 각각 **.958, .958**이다. heliosli 제목의 .9586을 API의 표시 정밀도와 같은 값으로 다루지 않는다. 표 수는 조회 시점에 따라 달라진다. [goodpjw 원본](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [heliosli 원본](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586), [공식 점수 응답 snapshot](../outputs/research/kaggle_code_20261019_refresh/official_score_summary.json)

`bestPublicScore`는 **해당 Notebook의 과거 최고 public 점수**다. 현재 내려받은 버전의 특정 가중치·branch가 그 점수를 냈음을 입증하는 필드는 아니다. 아래 단계별 과거 점수는 작성자 기록으로 구분하며, 우리 재현 실행 결과로 표현하지 않는다. 기존 문서의 “공식 숫자 점수 미확인”은 첫 수집 당시의 한계이고 이 재조사에서 일부 보완했다.

## 어떤 순서로 좋아졌다고 보고했나

goodpjw가 남긴 버전 기록은 단순한 높은 점수 제목보다 도움이 된다. 아래는 저자가 기록한 경로를 요약한 것으로, 여러 요소가 함께 변한 단계도 있어 각 요소의 단독 효과로 해석하지 않는다. [실험 기록 원문](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

| 구성 | 저자 보고 public LB | 읽어야 할 의미 |
|---|---:|---|
| 공개 nartaa CoAtNet | .949 | 충분히 학습된 이미지 encoder가 출발점 |
| 자체 ConvNeXt 2.5D reader 단독, 3fold | .929 | 방향·위치·Transformer가 있는 모델도 단독으로 .95는 아님 |
| CoAtNet + reader, 소견별 결합 | .950 | 오류가 다른 모델의 보완 |
| CoAtNet + compact S6 계열 | .954 | 다른 학습 계열 추가 |
| CoAtNet + compact R7 | .956 | 변경된 compact 모델과 조합 |
| Raptor 384/224 + R6 mirror TTA 등 | .957 | 입력 시야·학습 가중치·추론 조합 |
| 소견별 Raptor 비율을 사용한 v25/v26 | .958 | 현재 v27은 v25 recipe와 설명 갱신 |

반대로 같은 계열의 후속 실험에서 **S6를 추가하자 .958→.957**, gold58에서 12개 결합 비율을 최적화하자 **.958→.955**라고 보고했다. 작은 검증셋의 최적값이나 모델 수 증가를 개선의 보장으로 볼 수 없다. [실패까지 기록한 후속 실험](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

## 우리에게 적용할 가치가 큰 6가지

### 1. 픽셀 크기와 해부 방향을 먼저 일관되게 만든다

Raptor의 활성 추론은 PixelSpacing 기반 **140 mm crop**, 384 렌더 후 **320 입력**을 사용한다. compact 계열은 전체 시야와 **100 mm 국소 시야**를 함께 구성하고 무릎·슬개골 anchor 및 좌우 처리 경로를 둔다. 이는 위치 숫자를 token에 붙이는 것보다 앞 단계인 **이미지 자체의 크기·시야·방향 정규화**다. [Raptor 경로](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [compact 경로](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

- 실제 근거: Raptor `cell_3_code.txt` 119·122행의 `INPUT_RES`/`CROP_MM`, 복원 reader의 `mm_crop_resize`; compact `px_cache.py` 83행 `series_views`, 185행 부근 thin-stack 처리.
- 우리 차이: 물리 좌표 정렬과 mm RoPE는 있지만 이를 곧바로 mm 시야 정규화와 동일하게 볼 수 없다.
- 대조 실험: 같은 backbone·token 수로 mm crop만 변경하고, 이후 좌우 정규화를 따로 추가한다. 해상도와 전처리를 동시에 바꾸면 원인을 분리하기 어렵다.

### 2. 인접 장수만 늘리기보다 검사 전체의 coverage를 확보한다

현재 Raptor는 넓은 범위에서 구성한 96-slice stack의 **94개 triplet window**를 평가한다. goodpjw reader는 series의 **4–96% 범위에서 16개 중심**을 선택한다. compact R7 manifest는 `train_windows=176`, `phys_mm=3.5`, coverage sampling을 기록한다. 여기서 “3장 입력”은 한 window의 채널 수이지 검사 전체에서 3장만 본다는 뜻이 아니다. [Raptor·reader](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [R7 manifest 원본](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss)

- 실제 근거: Raptor `K_EVAL=94`; reader `window_centres(n,k,train,lo=.04,hi=.96)`; R7 manifest의 window/spacing/sampling 필드.
- 우리 차이: S02→S03은 주변 이웃 폭 5→10 비교다. 같은 장수로 범위를 넓히는 실험과 다르다.
- 대조 실험: 최종 unique token 수를 고정한 중앙 밀집 대 넓은 범위 균등 선택. 기존 R02/R03 결과도 이 관점에서 해석한다.

### 3. 이미지 encoder를 MRI 과제에 실제로 적응시킨다

R7 manifest는 **ConvNeXt-small, 16epoch, 4fold**, backbone LR `1e-4`, head LR `1e-3`를 기록한다. 소스의 `freeze_blocks`는 OFL 분기에 사용되므로 그 설정 숫자만 보고 ConvNeXt도 frozen이라고 읽으면 안 된다. Raptor 학습 원형 역시 backbone과 MIL head를 함께 최적화한다. [R7 데이터·설정](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss), [실행 가능한 학습 원형](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)

- 실제 근거: `compact_train.py` 382–384행의 encoder 분기; manifest `epochs/folds/lr/head_lr`; Raptor 학습의 backbone/head optimizer group과 `OneCycleLR`.
- 우리 차이: 주력은 Swin-T의 고정된 이미지 vector 위 head 학습이다. S12의 마지막 stage 1epoch 결과가 낮았다는 사실만으로 backbone 적응이 무효라고 결론낼 수 없다.
- 대조 실험: 이미 계획한 head warmup→제한된 backbone 해제의 결과를 먼저 읽고, 학습률·epoch·선택 기준을 고정해 확장한다. R7의 학습률을 우리 작은 batch/다른 backbone에 그대로 복사하지 않는다.

### 4. 이미지 내부 patch를 합치기 전에 소견별로 읽는다

R7 manifest는 `spatial_attn=true`다. 실제 `spatial_pool`은 encoder의 공간 격자를 유지하고 **각 patch의 x/y에 16차원 sin/cos 위치 특징**을 더해 소견별 query로 가중합한다. 그 뒤 window 간 소견별 pooling을 수행한다. 이는 이미지별 global vector를 만든 다음 study cross-attention을 적용하는 우리 S11과 다르다. [활성 소스가 들어 있는 원본](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

- 실제 근거: `compact_train.py` 492행 부근의 global pooling 해제, 541행 `spatial_pool`, 603행 `head`; [복원 소스](../outputs/research/kaggle_code_20261019/heliosli__this-is-best-public-dataset-new-version-0-9586/inspection/cell_6_compact_train.py.txt).
- 우리 차이: S11은 방향과 slice 사이에서 어떤 이미지 vector를 읽을지 학습한다. 이미 global pooling에서 없어진 이미지 내부 위치를 되살리지는 못한다.
- 대조 실험: 같은 frozen backbone에서 global average 대 class별 spatial pooling만 비교한다. 이때의 **x/y 2D 위치**와 새 T계열의 **촬영 방향×slice 깊이**는 별도의 좌표 체계다.

### 5. 약라벨의 의미와 geometry를 함께 다룬다

goodpjw의 `KneeNet`은 **slot embedding + 연속 위치 MLP → 2층 study Transformer → 12개 attention pool**을 사용한다. compact는 slot/view embedding과 `rel_pos, log_spacing, extent, fluid`의 선형 geometry embedding을 사용한다. 따라서 방향·위치 정보를 쓰려는 우리의 아이디어는 공개 구현에도 나타나지만, 모든 고득점 모델이 RoPE나 study Transformer를 쓰는 것은 아니다. [reader 원본](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [compact 원본](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

R7은 report-derived targets 외에도 상태 분류·extent·pair ranking·fastMRI 보조 학습 설정을 기록한다. 우리 confidence-weighted soft BCE와 같은 감독이라고 묶으면 안 된다. **소스에 학습 함수가 있다는 것과 원저자의 모든 target table을 확보해 재학습할 수 있다는 것은 다르다.** [R7 manifest](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss)

- 실제 근거: reader `cell_16_code.txt` 139–151행; compact `geom = Linear(4,hidden)`와 metadata 6열; R7 `state_weight=.5`, `extent_weight=.2`, `pair_rank=.5`, `fm_aux_weight=.3`.
- 대조 실험: 새 T계열로 방향·깊이 결합 효과를 먼저 확인한다. label audit에서는 UNK/부정/불확실·소견별 coverage를 조사한다. 보조 loss는 그에 맞는 target을 준비한 뒤 하나씩 추가한다.

### 6. 다른 오류를 내는 모델·시야를 검증해 결합한다

224 branch는 384 모델에 작은 입력을 넣는 것뿐 아니라 **별도 학습 checkpoint**다. 공개 조합은 Raptor 384/224와 compact R6/R7, 소량의 별도 reader를 섞는다. plain/mirror와 fold 평균, 확률 평균, rank 평균도 서로 다른 연산이다. 같은 공개 가중치를 쓰는 파생 Notebook 여러 개는 독립적인 학습 성공 여러 건으로 세지 않는다. [224 branch 원문](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956), [mirror 원문](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up)

- 실제 근거: 224의 별도 `c08_r224_swa.pt`, Raptor 내부 384:224=.65:.35; `_mirror_tri`의 sagittal 채널 역순과 coronal/axial 가로 반전. compact mirror는 좌표와 medial/lateral 출력 처리까지 연동한다.
- 우리 차이: R01은 E09/S11/S12의 확률 평균이며, 3seed 검증을 했다는 것이 배포 모델도 3seed/fold ensemble이라는 뜻은 아니다.
- 대조 실험: 기존 OOF로 고정 비율의 probability/rank 결합부터 비교한다. 이후 canonical 방향에서만 mirror TTA를 검증한다. 58건에 12개 가중치를 직접 최적화하는 방식은 별도 내부 검증 없이 채택하지 않는다.

## 과장되기 쉬운 해석을 바로잡기

1. **“0.95 모델을 찾았으니 구조만 복사하면 된다”**: 대부분 학습된 공개 가중치와 전처리·추론 recipe의 결합이다. 학습 데이터, target table, checkpoint 선택까지 같은 조건은 아니다.
2. **“Transformer여서 .95가 나왔다”**: compact의 주력 backbone은 ConvNeXt-small이고, class별 spatial/window pooling 경로다. reader Transformer 단독 보고는 .929였다.
3. **“.9586이 .958보다 개선된 원인이 증명됐다”**: 현재 원문에 원인별 통제 ablation이 없다. 공식 `bestPublicScore`에서는 두 Notebook 모두 .958로 확인되며 더 정밀한 차이나 현재 버전별 효과는 이 필드로 증명할 수 없다.
4. **“제목에 있는 모든 branch가 실행된다”**: 일부 Notebook은 가중치 누락·시간 초과 시 추가 branch를 생략한다. 실제 receipt를 봐야 배포 구성을 알 수 있다.
5. **“우리 .7714와 공개 .95의 차이가 모델 차이다”**: 우리는 gold58 OOF, 공개 문서는 대체로 public hidden-test LB다. 표본과 모델 선택 절차가 다르다. 우리 gold도 반복 개발에 사용했다.

이 재점검은 코드·설정의 근거 확인이다. 타인의 가중치를 실행하거나 새 학습·Kaggle 제출을 수행하지 않았다. 현재 자동 큐도 변경하지 않았다. **우리에게 가장 큰 추가 연구 축은 영상의 물리적 정규화, patch 정보 보존, backbone 적응량**이며, 위치 encoding은 그중 별도 대조 실험으로 해석한다. 구체적인 후속 순서는 기존 [P00–P08 계획](kaggle_095_comparison_and_plan.md)을 따른다.
