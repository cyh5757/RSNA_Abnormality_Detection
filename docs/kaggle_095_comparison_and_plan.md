# 공개 0.95 계열과 우리 실험의 차이, 다음 검증 계획

**추가 조사:** [Hotness·Most Votes와 공식 최고 공개 점수 대조](kaggle_hotness_votes_095_refresh.md). 기존 22개 고득점 후보의 역사상 최고 AUC ≥.950을 확인했다. 새로 검토한 nartaa S6는 .950이며, encoder/head를 함께 16epochs 학습한 조건을 저자가 구체적으로 공개했다. 아래 원문·OOF 비교는 기존 snapshot이며 최신 버전 점수를 뜻하지 않는다.

기준일: 2026-10-19. 우리 결과는 `outputs/analysis/2026-10-19`의 **31개 완료 실험 snapshot**을 기준으로 한다. 진행 중인 후속 실험을 완료 성적으로 합치지 않는다. 공개 Notebook 원문은 22개 고득점 후보와 3개 학습·설계 원형, 총 25개를 확인했다. 원문별 검토는 [1–10번](kaggle_095_review_part_a.md), [11–25번](kaggle_095_review_part_b.md)에 있다.

## 먼저 내릴 수 있는 결론

공개 0.95 계열의 공통점은 거대한 Transformer 하나가 아니다. **보고서 기반 학습 표본 확보, 해부학적으로 일관된 입력, 더 넓고 촘촘한 시야, 영상 backbone 적응, 서로 다른 가중치의 결합**이 묶여 있다. 같은 Raptor/compact 가중치의 파생 Notebook이 많으므로 “20명이 독립적으로 검증한 20개 방법”으로 해석해서는 안 된다. 최신 버전의 점수가 없거나 추가 가중치가 비공개인 사례도 포함했다. [R7 원본 계열](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click), [R6+reader 실험](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [224 추가](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956)

**0.95는 95% 정확도가 아니다.** 12개 소견별 ROC-AUC의 평균이다. 우리 R01의 0.7714는 58 gold의 3-fold OOF를 seed 3개에서 평가한 평균이고, 공개 글의 .95x는 대부분 작성자가 보고한 public hidden-test LB다. 평가 표본·학습량·모델 선택 절차가 달라 둘을 빼서 순수 모델 격차라고 표현할 수 없다. 우리 gold58도 지금까지 반복적으로 모델 선택에 사용했기 때문에 완전히 손대지 않은 최종 test는 아니다. 세 seed가 174명의 독립 환자를 만드는 것도 아니다. [공식 평가](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), [로컬 결과](followup_experiment_results.md)

## 계보를 먼저 묶어서 이해하기

| 계보 | 실제 중심 구성 | 이번 조사에서의 위치 | 우리가 얻을 것 |
|---|---|---|---|
| Raptor / CoAtNet | adjacent triplet → 이미지 encoder → 소견별 MIL, SWA; 최신 nartaa 384·224 공개 가중치 | goodpjw·heliosli 및 다수 파생의 한 축 | 학습된 이미지 feature, 입력 크기/시야 다양성 |
| compact R6/R7 | ConvNeXt-small + knee-centred views + metadata/geometry + class별 pooling + 추가 head/stacker | heliosli 원본과 goodpjw, jenillangaliya 등 | 물리적 전처리, patch 보존, 모델·설정 manifest |
| 2.5D reader | ConvNeXt + slot/position + study Transformer + class별 attention | goodpjw, hitarthjain0, rabari 등 | 우리의 token attention과 가까운 비교 대상 |
| 다중 해상도/TTA 조합 | 기존 가중치의 384/224, mirror, rank 평균 | varshithakolupuri, laimer, yuvraj 등 | 새 학습 없이 검증할 수 있는 추론 대조군 |
| 대형 추론 graph | DINOv2 여러 member, A5/Rad/Raptor, 추가 CoAt family | mattiaangeli, 일부 prvsiyan 역사 경로 | patch/depth 설계와 실행 재현성; 전체 복사는 고비용 |

계보 근거: [Raptor 학습](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model), [R7 manifest dataset](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss), [reader 소스](https://www.kaggle.com/code/hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950), [Speedy Raptors](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original). 이름이 같은 R6/R7이라도 bundle ID가 다르면 같은 모델로 보지 않았다. **공개 R7 모델 세대와 우리의 R07 실험 ID는 무관**하다.

## 현재 우리 구현과의 구체적 비교

| 항목 | 우리 E/S/R 실험 | 공개 사례에서 확인한 차이 | 해석 |
|---|---|---|---|
| 정답/약라벨 | 58 gold + 4,292 report weak, UNK zero mask, confidence weighting | 여러 report reader/soft label 조합, R7 state·extent 보조 target | 표본수만의 차이보다 label 의미·coverage·가중치 차이가 중요 |
| 이미지 backbone | 현재 주력은 **Swin-T**. 대부분 frozen; S12만 마지막 stage+norm 1epoch 적응 | R7 ConvNeXt-small 16epoch 설정·4fold, Raptor end-to-end 학습 원형 | ResNet→Transformer 교체는 이미 했지만 영상 적응량이 다름 |
| 입력 크기/시야 | resize232 → center crop224, percentile 정규화 | R7 render384/crop352, Raptor render384/crop320;224 별도 모델 | 큰 숫자보다 병변의 실제 크기·field of view 보존 여부 |
| 물리적 처리 | DICOM normal로 slice 정렬, 상대 mm를 token 좌표로 사용 | PixelSpacing 기반 mm crop, 방향/좌우 정렬, anatomy crop | **mm RoPE와 mm 이미지 정규화는 다른 기능** |
| 슬라이스 선택 | 기본 중심 25–75%, 5 centres와 이웃 수3/5/10; dedup | Raptor96 stack/94windows, R7 dense coverage·phys_mm3.5 | “3장”은 입력 채널수일 수 있음. study 전체3장이라는 뜻이 아님 |
| 이미지 안 공간정보 | Swin의 global pooling 후768차원 vector를128차원으로 투영 | R7 활성 spatial-attention, 일부CoAt spatial/depth head | 우리의 cross-attention은 이미 잃은 patch 정보를 복구하지 못함 |
| study fusion | plane×FS 6slot, slice self-attention/RoPE, S10gate/S11class-query cross | reader Transformer도 있지만 R7은 class별 spatial/window pooling 중심 | 모든 상위 모델이 study Transformer인 것은 아님 |
| 학습 목표 | confidence/mask weighted BCE; 평가gold unweighted BCE + macroAUC | R7 manifest에 state .5, extent .2, pair-rank .5, fastMRI aux .3 | loss항 추가는 label·sampling까지 갖춰 별도 실험해야 함 |
| 배포 결합 | final full-fit seed42; R01=E09 .5+S11 .25+S12 .25 **확률 평균** | fold ensemble/SWA/EMA/해상도별 rank평균/TTA | 우리3seed평가가 배포3seed/fold앙상블이라는 뜻은 아님 |
| 검증 | 같은58gold OOF,3seeds; 반복개발에 사용 | gold로checkpoint 선택한 모델도 존재, LB주장·원예측 누락 | 점수 숫자만 정렬하면 잘못된 결론 |

우리 구현 근거: `experiments/spatial_core.py`의 `spatial_order`, `selected_windows`, `spatial_views`, `SliceBlock`, `SpatialHead`; `experiments/core.py`와 spatial/weak-label 학습 경로. 공개 R7 근거는 다운로드한 manifest와 SHA가 일치한 `compact_train.py`/`compact_infer.py`다. R7의 `E_lrprior` 경로는 likelihood-ratio 기반 감독이므로 우리의 soft BCE와 구분한다. `freeze_blocks=16`은 generic 설정이지만 ConvNeXt 분기에 그대로 적용되는 freeze가 아니므로 “R7도 encoder frozen”이라고 읽으면 안 된다. [R7 설정 원본](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss)

## 우리의 수치로 판단하면

| 비교 | 동일 gold OOF 3seed 평균 macroAUC | 현재 가능한 결론 |
|---|---|---|
| E06 gold control → E07 weak | .585972 → .741941 | 이 비교에서 가장 큰 개선은 보고서 약라벨 확대 |
| E07 → E08 adjacent triplet → E09 protocol | .741941 → .742926 → .744113 | 방향/촬영조건·인접 입력의 개선은 작으며 표본 불확실성이 큼 |
| S01 이웃3 → S02 이웃5 → S03 이웃10 | .716563 → .732541 → .724005 | 이 설정에선5가 가장 높음; 일반적인최적장수 결론은 아님 |
| S02 → S04sequence → S05additive | .732541 → .726607 → .713061 | 복잡한 sequence/position을 추가해도 단독 성능은 개선되지 않음 |
| S06indexRoPE → S07mmRoPE | .721530 → .711794 | 좌표 활용이 자동으로 도움이 되는 것은 아님 |
| S09MLP → S10gate / S11cross | .724314 → .739986 / .736386 | class별융합은 가능성이 있지만 cross가gate보다 우세하지 않음 |
| S11cross → S12FT | .736386 → .718888 | 현재의1epoch적응은 하락; 모든fine-tuning이 무효라는 뜻은 아님 |
| S13ensemble → R01E09혼합 | .748420 → .771432 | 서로 다른 경로의 결합이 가장 높은 개발OOF |

직접 근거: [전체 실험 노트](followup_experiment_results.md), `outputs/analysis/2026-10-19/experiments.csv`. R01 seed SD=.008916, BCE=.554393. S13→R01 paired bootstrap의 개선구간은 약+.0042~+.0423이지만 **반복된모델선택에 따른 낙관편향을 보정한 구간이 아니다**. 단순 평균이 유리한 경우도 있으므로 “항상 class별 수동비율이 좋다”는 일부공개설명은 채택하지 않는다.

S13→R01 seed42, threshold=.5 기준 FN은 75→67, FP는 144→132로 줄었다. 하지만 MCL FN은 6→7, Fracture FN은 11→13으로 악화했다. **macro AUC 최고 모델이 모든 소견에서 최선인 것은 아니다.** 각 클래스의 2×2 혼동행렬·recall·precision을 함께 본다. 이 threshold는 분석용이며 제출할 때 12개 확률/score를 0/1로 잘라서는 안 된다. [혼동행렬 비교](../outputs/analysis/2026-10-19/confusion_comparison.html)

## 5장과10장, 그리고 attention에 대한 실험적 해석

우리 S02/S03의 5/10은 **각 중심 주변의 이웃 폭**이다. 서로 겹친 slice를 중복 제거하므로 encoder token이 정확히 2배가 되지 않는다. 실제 snapshot은 409,239→526,504개(+28.7%), feature 추출 시간은 4.60→5.52시간이었다. 따라서 “정보 2배인데 성능이 떨어졌다”는 설명은 맞지 않는다. [실험 노트의 sampling audit](followup_experiment_results.md)

가능한 가설은 ① 같은 부위의 중복 정보 증가, ② 주변 배경·다른 해부구조 추가, ③ 고정 feature에서 분류 head의 표본 대비 복잡도 증가, ④ 학습률/epoch 최적점 차이, ⑤ 58 gold 표본 변동이다. 현재 결과로 원인을 확정할 수 없다. 새 실험은 **중심 coverage**, **이웃 간 mm 간격**, **최종 unique token budget**을 각각 고정/변경해 구분한다. 부족한 영상 정보는 downstream attention으로 되살릴 수 없으므로 spatial patch 보존을 별도 축으로 검증한다.

## 다음 실험: 기존 자동 queue와 분리한 계획

현재 후속 queue는 R04 confidence → R05 label 균형 → R02 중심 범위 확대 → R03 중심 9개 → R06 head warmup 후 FT → 조건부 R07 장기 FT 순서다. 이번 조사는 이 순서나 실행 중인 Notebook을 변경하지 않는다. 아래 P계열은 **새 후보 계획이며 아직 실행/성공을 주장하지 않는다**.

| 순서 | 가설/실험 | 바꾸는 항목 | 반드시 고정할 항목 | 판단과 비용 |
|---|---|---|---|---|
| P00 | 기존 R01 대 probability/rank ensemble | OOF 결합 연산만; 비율은 고정 | 같은 OOF·seed·클래스 순서 | GPU 0; AUC·AP·상관·bootstrap. rank는 보정 확률이 아님 |
| P01a | image mm crop 유무 | PixelSpacing 기반 일정 시야 | Swin/head/표본/seed/token budget | feature 재추출 필요; 유효 header·시야 누락 감사 |
| P01b | 좌우 정규화 추가 | DICOM 좌표/방향 정규화 | P01a의 나머지 전부 | medial/lateral 오류와 전체 AUC; 단순 가로 flip과 구분 |
| P02 | breadth 대 redundancy | 넓은 범위 균등 선택 vs 중앙의 인접 선택 | **unique token 수를 같게**, 동일 encoder/head | 기존 R02/R03 결과와 함께 coverage 가설 판단 |
| P03 | 이미지 patch 정보가 병목인가 | global avg 대 class별 spatial pool | 입력·backbone freeze·학습 표본 고정 | 유효 patch mask/실제 사용량 기록; 단독 효과 검증 |
| P04 | 영상 적응량이 부족한가 | warmup→last stage 1/3/6 epoch 단계적 | head·전처리·약라벨·LR 원칙 | R06/R07 결과 선행; 학습곡선·seed·소견 악화 함께 평가 |
| P05 | 안정화가 도움 되는가 | best checkpoint 대 EMA/사전 지정 epoch 평균 | 같은 학습 run/선택 규칙 | 새 추론 모델 추가 없는 대안; BN 재계산 조건 확인 |
| P06 | 유효한 해부 mirror TTA | plain 대 canonical mirror 평균 | 동일 가중치·OOF·sampling | 추론 시간 증가; plane/token 좌표 변환 동기화 |
| P07 | 개선 feature에서 backbone 재비교 | Swin-T 대 ConvNeXt-small/CoAtNet | 동일 label/split/input/training budget | frozen 58 gold 순위가 재학습 순위와 같은지 검증 |
| P08 | 소견별 결합이 필요한가 | global blend 대 regularized class blend | member 고정; inner 검증 선택 | nested/holdout 설계 없이 58 gold에 12비율 최적화 금지 |

P00의 rank 검증은 fold별 held-out 표본에서만 정해진 변환을 적용하거나 train-only reference CDF를 고정해 평가한다. 전체 OOF를 한꺼번에 rank해 선택하고 숨은 test에는 다르게 처리하는 방식은 피한다. confidence/BCE·혼동행렬 평가는 확률 출력을 따로 유지한다. raw probability가 더 좋을 수도 있다.

채택 기준은 사전 등록한 대조군보다 평균 AUC 개선, 3 seed 중 적어도 2개의 일관성, 주요 취약 소견의 악화 유무, paired uncertainty, 실제 GPU 시간을 같이 기록하는 것이다. 작고 불확실한 차이는 보류한다. 새 가중치의 학습 조건을 입력 모양만 맞춰 추정하지 말고 preprocessing/label/version을 함께 고정한다.

공개 R7의 fastMRI 보조 학습/OAI 관련 최신 Raptor 가중치는 추가 데이터·학습 설계까지 포함한 결과다. 데이터를 실제로 확보하고 그 학습을 재현하지 않은 상태에서 우리 방법과 동일 조건 비교라고 부르지 않는다. 보고서 약라벨의 품질 개선은 LLM 교체만이 아니라 UNK/부정/불확실 표현, 소견별 confidence와 coverage 점검을 포함한다. [R7 manifest](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss), [Raptor 원형](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)

## 공개할 우리 Code의 범위

실험 목적, 31개 완료 결과, 실패한 변경, 소견별 혼동행렬, 실제 `SpatialHead` 코드와 합성 tensor 예제, 위 후속 가설을 실험일지형 Notebook으로 공개한다. 공개한 숫자는 gold 개발 OOF이며 .95 LB 성공을 주장하지 않는다. 환자 식별자·report 원문·영상·환자별 예측은 포함하지 않는다. 원저자가 제공한 학습 데이터/방법은 링크로 출처를 남긴다.

공개 Notebook은 CPU로 표·그림·tensor 예제를 다시 생성하는 **방법론/오류 분석 Code**다. 학습·hidden-test 제출 Notebook과 역할을 구분한다. 업데이트할 때 검증 완료 결과만 새 snapshot에 반영하고 각 버전을 보존한다. 대회 마감 이후에도 계속 연구할 수 있지만 마감 후 실험은 대회 제출 개선과 별도로 기록한다.
