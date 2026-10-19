# Kaggle RSNA Knee 공개 Code 25개 검토와 우리 개선 계획

**공식 점수 확인 추가(2026-10-19):** [Hotness·Most Votes 재조사](kaggle_hotness_votes_095_refresh.md)에서 Kaggle `bestPublicScore`를 조회했다. 아래의 고득점 후보 22개 모두 역사상 최고 공개 AUC ≥.950을 확인했으며, 일부 제목의 .956+는 실제 조회값 .950이었다. 아래 표의 ‘미확인’은 최초 원문 검토 당시의 상태다. 새 조회값은 notebook의 과거 최고점이므로 최신 버전의 개별 점수·독립 재현 여부와는 구별한다.

조사일: 2026-10-19. **0.95 이상을 보고하거나 그 계보에 속하는 후보 22개 + 학습·설계 원형 3개**, 총 25개 원문을 읽었다. 제목 검색에 그치지 않고 실제 실행 셀, 내장 모듈, 모델 manifest, 현재/과거 버전의 차이를 확인했다.

우리 방법도 [Kaggle 실험일지 v1](https://www.kaggle.com/code/choiyoungwhan/rsna-knee-experiments-and-error-analysis)로 저장해 실행 완료·출력 일치를 확인했다. 현재는 **비공개**다. 공개 요청은 Kaggle의 대회 source 제한으로 거부됐으며, [게시 상태와 검증 기록](kaggle_experiment_journal_publication.md)에 정확한 응답을 남겼다.

**22개 모두의 0.95를 독립 재현한 결과가 아니다.** 아래에는 원저자 점수의 인용, 파생 코드, 최신 버전 미측정, 비공개 추가 가중치, 외부 실행기까지 구분했다. 상세 증거가 부족한 경우 그대로 표시했다. 공개 source는 실행하지 않았고 hidden-test 제출도 하지 않았다.

읽는 순서: 먼저 비교·실험 계획을 읽고, 이어지는 A1–A10 및 11–25번에서 개별 원문과 코드 근거를 확인하면 된다. [검토 목록 CSV](../outputs/research/kaggle_code_20261019/review_matrix.csv), [원문 버전·SHA 목록](../outputs/research/kaggle_code_20261019/snapshots.json)도 보존했다.

## 전체 검토 목록

| 번호 | 원본 | 버전 | 점수 표기 / 증거 범위 |
|---|---|---|---|
| 1 | [goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | v27 | 0.958 — cell1: versions25/26; current27 equals25 recipe |
| 2 | [goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957) | v8 | 0.957 — cell1: v6 .957/v7 .955; current8=v6 source |
| 3 | [heliosli/this-is-best-public-dataset-new-version-0-9586](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | v7 | 0.9586 — title only; no numeric official metadata |
| 4 | [heliosli/this-one-is-old-version-0-9573-newone-you-can-use](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use) | v2 | 0.9573 — title only |
| 5 | [heliosli/blend-mine-in-private-lb-luck-to-all-upvoters](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) | v1 | 0.958 — A1/A2 cite this exact recipe as .958; not own title |
| 6 | [heliosli/0-957-latest-dataset-score-in-one-click](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | v1 | 0.957 — title approx .957; A1/A7 report deployed recipe .956 |
| 7 | [evgendvorkin/rsna-versia-7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | v22 | 0.956 — cell1 explicitly attributes .956 to original public solution, own score unconfirmed |
| 8 | [prvsiyan/the-bee-s-knees-final-rsna-push](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push) | v47 | unverified — cell145 upstream358332966 .957; explicitly no new official score; cell150 unvalidated contrast |
| 9 | [kaiwalyaatulraut/rsna-knee-competition-solution](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-competition-solution) | v3 | 0.958 — cell0 inherited figure references v25/v26 despite currentv3 |
| 10 | [kiyoshiohno/rsna-knee-template](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template) | v27 | unverified — no >=.95 numerical evidence in source |
| 11 | [bliverrigpdoge/r7-raptor-knee-centred-convnext](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext) | v1 | .956→.957 reported — private extra branch; public base fallback |
| 12 | [laimerbotrolenk/rsna-four-model-ensemble](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble) | v1 | .957 reference only — current V33 candidate explicitly unmeasured |
| 13 | [leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl) | v12 | .957 reported — analytical article; no executable inference in v12 |
| 14 | [jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble](https://www.kaggle.com/code/jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble) | v1 | .956 reported — Raptor + R7, reused inference recipe |
| 15 | [jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble](https://www.kaggle.com/code/jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble) | v4 | .954 reported — Raptor .7 + older compact .3 |
| 16 | [varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956) | v1 | .956 displayed — 224 branch addition; rank movement author report |
| 17 | [varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up) | v1 | .956 displayed — 224 anatomical mirror; rank movement author report |
| 18 | [hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950](https://www.kaggle.com/code/hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950) | v5 | .950 historical — new V5 routing not independently scored; negative blending results |
| 19 | [rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader](https://www.kaggle.com/code/rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader) | v11 | .9564+ claimed — probit/cross-prior claims not independently verified |
| 20 | [yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956) | v2 | >.956 claimed — same 3-branch family; gold table provenance incomplete |
| 21 | [rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951](https://www.kaggle.com/code/rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951) | v4 | .951+ claimed — two CoAt scales + 3-fold reader, class routing |
| 22 | [yuvrajlamba01/rsna-knee-v15-tri-hybrid-ensemble-0-954](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-v15-tri-hybrid-ensemble-0-954) | v8 | .954 title — old bundle; only two active branches |
| 23 | [pilkwang/rsna-knee-baseline-v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | v15 | training reference — DINOv2/protocol/geometry; not counted as independent >=.95 |
| 24 | [dreaddevelopment/knee-mri-training-the-twelve-finding-model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) | v8 | .924 origin reference — requires own soft labels; gold checkpoint selection |
| 25 | [mattiaangeli/bend-the-knee-to-speedy-raptors-the-original](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original) | v39 | architecture reference — large inference graph; current v39 score not verified |


---

# 공개 0.95 계열과 우리 실험의 차이, 다음 검증 계획

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


---

# Kaggle 0.95 이상 후보 소스 검토 A — 처음 10개

검토일: 2026-10-19. `selection.json`의 처음 10개를 대상으로 **실제로 실행되는 코드, 내장된 Python 소스, 고정 모델 manifest**를 읽었다. 공개 노트북·내장 모듈·체크포인트는 실행하거나 로드하지 않았다. base64/zlib, base91/LZMA 자료는 바이트와 텍스트로만 복원했고 가능하면 원문 SHA256과 대조했다. 셀 번호는 JSON의 **0부터 시작하는 인덱스**다.

**핵심 결론:** 이 10개는 독립적인 10개 학습 성공 사례가 아니다. 대부분 nartaa의 CoAtNet/Raptor 두 체크포인트, heliosli의 R6 또는 R7, goodpjw2008의 2.5D reader를 다시 조합한 추론 노트북이다. 사용자의 방식과 가장 큰 차이는 물리 단위 영상 처리, 넓고 조밀한 슬라이스·국소 시야, 여러 epoch의 영상 encoder 학습, 보고서의 상태/정도 감독, 서로 다른 학습 계열의 혼합이다. 단순히 후단 attention을 추가해서 0.95가 된 사례로 읽으면 잘못이다.

## 점수와 버전의 증거 범위

수집된 Kaggle metadata에는 공식 제출의 숫자 점수가 없다. 아래 숫자는 **제목·본문·그림에 저자가 기재한 값**이다. 타인의 점수 인용, 과거 버전 점수, 현재 코드 점수를 구분했다. 특히 `.95`라는 숫자가 EMA·분위수 등 하이퍼파라미터에 등장하는 것은 성능 증거로 쓰지 않았다. 공식 제출 API를 통한 점수 검증이나 이 환경의 재현 실행을 했다는 뜻이 아니다.

| # | 노트북 / 고정 버전 | 원문 SHA256 | 점수 증거와 한계 |
|---|---|---|---|
| A1 | [goodpjw: CoAtNet + R6 + 2.5D, v27](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | `a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c` | 셀1: v25/v26 **.958**, 현재 v27은 v25 recipe의 문서 갱신이라고 명시 |
| A2 | [goodpjw: Does S6 help, v8](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957) | `64ac2a3070ef3b48dc14a433119b41d2812c14280922acb3ee1c50bb6db61c3f` | 셀1: v4/v5 .957, v6 .957, v7 .955; v8은 v6 코드와 결과 기록, 새 점수 주장 없음 |
| A3 | [heliosli: best public dataset, v7](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | `cb0e1eac33a00f96b7d1bcc0e9914f7eb045af2bc9cd01bc290b2213e7f17ddd` | 제목 **.9586**; 현재 v7의 공식 점수·각 변경의 개별 상승분은 원문으로 검증 불가 |
| A4 | [heliosli: old version .9573, v2](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use) | `85c694c903ec6841488b49d5ecbe2838b8fad7db283f1f952c59f48d936e0b24` | 제목 **.9573**; 현재 bundle은 R7, 이름에 R6가 있는 dataset이어도 R6라고 읽으면 안 됨 |
| A5 | [heliosli: blend mine, v1](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) | `ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d` | 자체 제목에는 숫자 없음. A1/A2가 이 recipe를 **.958**이라고 명시적으로 인용 |
| A6 | [heliosli: .957 one-click, v1](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | `cb2a0f57962ca9b6668661e34461aad772ed9ed10d576d32772e7c2898f2e315` | 제목 약 **.957**, A1/A7 본문은 같은 R7+Raptor를 **.956**으로 설명. 둘을 합쳐 .957 확정으로 쓰지 않음 |
| A7 | [evgendvorkin: RSNA Versia 7, v22](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | `1b2282b6e5bfe83f77eb07e09b830d5ba1033733bb483487f6843abcff685bc2` | 셀1: 약 **.956은 원저자의 점수**, 자신의 재현 점수는 별도 확인해야 한다고 명시 |
| A8 | [prvsiyan: bee's knees, v47](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push) | `07a50c3e6e5af6616cd6cd41a647adb40d30141e33f80cf9822e1cf8f8c41d3e` | 셀145: upstream **.957**, **자신의 새 공식 점수 없음**. 셀150: .55/.45 변형은 검증되지 않은 대조 실험 |
| A9 | [kaiwalya: Competition Solution, v3](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-competition-solution) | `62a7587c2963f61c0dc5783c62a3509e3a25ec75d291208f6087cee7d818b09d` | 셀0 그림 **.958** 및 “v25/v26, current=v25”를 계승. 실제 수집 버전은 v3이므로 그 문구를 이 노트북의 버전 실험으로 해석 불가 |
| A10 | [kiyoshiohno: Template, v27](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template) | `66fbf17addfacbaca201153700220102ce8dda636d6e2b4851a8ac5150a7c66e` | 본문/코드에 0.95 이상 수치 근거 없음. 외부 pipeline 실행기라 이 소스만으로 모델·점수 확인 불가 |

원문·metadata·셀 인덱스는 [수집 디렉터리](../outputs/research/kaggle_code_20261019/selection.json)에 연결되어 있다. 위 후보 10개를 모두 읽었다는 사실과 “0.95 이상을 독립적으로 검증한 10개”라는 주장은 다르다.

## 공통 구성요소를 실제 경로로 분해

### R384: 이름은 384, 현재 추론 입력은 320

A1 셀3과 A3~A6 셀2의 실행 상수는 `IMG=384`, `INPUT_RES=320`, `CROP_MM=140`, `K_EVAL=94`다. 실제 `ARMS`에는 `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` **한 개**만 있다. 체크포인트 SHA256은 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`. 과거 주석의 “5-fold”, 본문의 64장/42창/336px를 현재 실행 설정으로 인용하면 안 된다. 모델 계열은 `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`이고 최종 임베딩을 `LayerNorm → class-specific tanh attention → class-specific linear readout`으로 모은다. `RaptorClassifier.head`에는 명시적인 slice-position embedding이나 study Transformer가 없다. [A1 셀3](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

5개 슬롯은 Sag fluid/nonfluid 26/22장, Cor fluid/nonfluid 18/12장, Ax 18장으로 총96장이다. `Fluid_Sensitive`를 우선하여 중복 series를 피한다. 안전하게 복원한 `_FASTREAD_B64`의 `make_reader_v2`, `build_study_v2`, `pick_indices`에서 확인한 활성 범위는 **2~98%**다. DICOM `ImagePositionPatient · cross(ImageOrientationPatient)`로 정렬하며 실패 시 InstanceNumber를 쓴다. PixelSpacing에 근거한 중심140mm crop을384로 만들고, 추론 시 중앙320을 취한다. 강도는 선택한 series 픽셀의 2/98분위수 창과 ImageNet 정규화다. 고정2.5D triplet으로94개 window를 점수화한다. 이 Raptor 구현에서 방향·좌우 표준화를 R6와 같은 수준으로 한다고 주장할 근거는 없고, 물리정렬/물리crop과 아래 mirror 처리는 확인된다. [A1 셀3의 내장 reader](../outputs/research/kaggle_code_20261019/goodpjw2008__rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/inspection/cell_3_embedded__FASTREAD_B64.txt)

`_mirror_tri`는 Sag 창의 세 채널을 역순으로, Cor/Ax는 가로방향으로 반전한다. **uint8 단계에서 반전한 뒤 강도 LUT를 적용**한다. `infer_pair_probs`는 plain/mirror 확률을 .5/.5 평균한다. R6의 클래스 permutation 및 logit 평균과 동일한 방식이 아니다. [A1 셀3 `_mirror_tri`, `infer_pair_probs`, `gpu_windows`](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

### R224: 별도 해상도로 학습된 두 번째 체크포인트

같은 nartaa dataset의 `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`를 사용한다. 384 크기로 만든140mm field의 **0.8 중심crop(약112mm)을224로 resize**한다. 단순히 R384 모델의 입력 해상도만 낮추는 방법이 아니다. A3/A4/A5/A8은 현재 실행상 **single view**, A1/A2/A9는224도 mirror 확률 평균을 한다. 내부 Raptor 혼합은 `.65 rank(R384) + .35 rank(R224)`다. A6/A7에는 R224 자체가 없다. [A1 셀6](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958), [A3 셀4](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

### C6/C7: 영상 전체를 학습한 compact ConvNeXt 계열

R6 `fe055406fad6a12f`의 고정 manifest는 A8 셀148에 든 package를 안전하게 복원하여 확보했다. SHA256 `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648`. R7 `8c7b666e27dc0e8f` manifest는 공개 dataset에서 읽었다. 둘 다 **ConvNeXt-small,384 render/352crop,seed0의4fold,16epoch,train_windows176,weak pool4344,Gold58 학습 제외,backbone LR1e-4/head LR1e-3**를 기록한다. `freeze_blocks=16` 필드는 있지만 `Net` 구현에서 이는 **OFL encoder 전용**이다. 실제 ConvNeXt 경로 `_timm_encoder`를 frozen으로 해석하면 틀린다. [R6 manifest 복원본](../outputs/research/kaggle_code_20261019/prvsiyan__the-bee-s-knees-final-rsna-push/inspection/bee_R6_FROZEN_V1_MANIFEST.json.txt), [R7 manifest](../outputs/research/kaggle_code_20261019/manifests/r7/manifest.json), [R7 dataset](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss)

공통 입력은 `views=v4`,3-slice,`phys_mm=3.5`,eval_stride1이다. `px_cache.series_views`의 v4는 full field와100mm local을 함께 보며, Sag/Ax는 joint·patella·posterior, Cor는 medial/lateral local을 둔다. SKM-TEA anatomy checkpoint에서 얻은 knee/patella anchor와 DICOM geometry를 쓴다. 1.75mm보다 얇은3D stack은 약3.5mm slab로 평균하여 얇은 series가 대부분 버려지거나 과도하게 반복되지 않도록 한다. Sag right knee의 물리좌표 부호를 바꾸고, laterality가 불명확하면 side classifier로 재렌더링하는 경로도 활성 manifest에 포함된다. R6/R7 모두 단순 중앙5장보다 훨씬 촘촘하고 여러 시야를 사용한다. [A3 셀6 `px_cache.py:series_views/StudySlices`, `compact_train.py:window_rows`, `compact_infer.py:CStudies`](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

`Net._build_head/head`는 projection384차원, slot embedding10개, view embedding6개, **상대위치·log spacing·시야 extent·fluid metadata의 linear embedding**, class별 window attention을 쓴다. R7 manifest는 추가로 `spatial_attn=true`: backbone spatial map 위의 class별 query attention(`spatial_pool`,16차원2D sin/cos 좌표)을 활성화한다. 따라서 사용자의 Swin 전체벡터 후단 attention과 달리 **영상 내 위치에서도 소견별 pooling을 하는 경로**다. R7은 `extent=true`의 OA 면적 proxy, report-state head, `syn_decomp=true`도 활성이다. 이는 study self-attention/RoPE와는 다른 구조다. R6의 축약 manifest는 spatial/extent flag를 모두 보이지 않으므로 같은 generic 소스가 있다고 그 flag까지 동일하다고 추정하지 않았다. [R7 manifest](../outputs/research/kaggle_code_20261019/manifests/r7/manifest.json), [A3 셀6 `compact_train.py:Net`](../outputs/research/kaggle_code_20261019/heliosli__this-is-best-public-dataset-new-version-0-9586/inspection/cell_6_compact_train.py.txt)

R7 학습 설정은 `E_lrprior=lr:lr_prior:w`,report-state loss weight.5,matched-pair rank loss.5,rare boosting,mirror augmentation.5,EMA.999,coverage sampling stride3,fastMRI auxiliary weight.3/96windows를 명시한다. `lr_loss`는 likelihood-ratio 감독이며 사용자의 confidence-weighted soft BCE와 같지 않다. R7의 “fastMRI 활용”은 manifest와 generic 학습 코드 양쪽에서 확인되지만, **R6 대비 각 기법의 개별 효과를 보여주는 통제된 ablation은 여기 없다**. R6 축약 manifest의 run tag만으로 동일한 fastMRI 세부값을 확정하지 않는다. [R7 manifest](../outputs/research/kaggle_code_20261019/manifests/r7/manifest.json), [A6 셀0·4](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click)

배포 시4fold logit을 평균하고 Effusion은 이미지에서 산출한 fluid-volume stacker, Baker's는 report-state 분포 stacker로 보정한다. R6 flip-TTA는 Sag 채널순서+상대좌표 부호,Cor/Ax 가로반전,Cor local view2↔3을 함께 바꾸고 medial/lateral meniscus·OA 출력의 순서를 복구한 다음 **logit 평균**한다. MCL은 plain만 쓴다. R7 검토 노트북은 `flip_tta=false`다. “모든 영상에 단순 가로flip 후 평균”으로 복제하면 원문과 다르다. [A3 셀6 `compact_infer.py:mirror_windows/score_v4c/apply_stacker/apply_state_stacker`](../outputs/research/kaggle_code_20261019/heliosli__this-is-best-public-dataset-new-version-0-9586/inspection/cell_6_compact_infer.py.txt)

**학습 공개 범위:** `compact_train.py`230,308자와 관련 모듈은 소스에 포함되어 있어 학습 구조·loss·augmentation 구현을 읽을 수 있다. 그러나 현재 노트북이 training loop를 실행하는 것은 아니며, 실제 target table/state target 생성물·전체 run artifacts가 모두 담긴 것도 아니다. 원문도 report/label table을 제공하지 않는다고 명시한다. 따라서 “원학습 코드는 전혀 없다”도 부정확하고 “노트북만 Run All 하면 동일모델을 처음부터 재학습한다”도 부정확하다. [A3 셀1·6](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

### G: goodpjw 독립 2.5D ConvNeXt-Tiny reader

A1 셀15~17의 별도 모델이다.0.4mm/pixel·384×384=153.6mm field,IOP 기반 in-plane orientation과 IPP 물리순서,series당 최대64장,1/99.5분위수 intensity를 쓴다. six slots는3planes×fat-suppression이며 missing fs/nonfs를 counterpart로 채운다. 중심은 **4~96%에16개**,3-slice windows다. backbone 출력에384차원 projection과slot/연속위치 MLP embedding을 더하고 **2-layer study Transformer → class별 attention pooling**을 한다. 모델 입력 resize는 checkpoint의 `args.res`를 읽으므로 generic `forward(res=256)` 기본값만 보고 정확한 배포 res를 확정하지 않았다. [A1 셀15~17 `load_series`, `window_centres`, `KneeNet`, `load_models`](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

원저자는 report text로 묶은5fold 학습,4개 공개 soft-label table,Gold58 미학습을 기재한다. 배포는 `cnxt_v0_fold[012].pt`의 **3fold**이며 확률 평균이다. 공개 소스에는 preprocessing/dataset/model/inference가 있고 이 노트북에서는 학습하지 않는다. 학습 hyperparameter 전부가 노트북에 없으므로 저자 진술과 실행 소스 확인을 구분한다. 혼합 가중치는 Baker .035,Contusion .03,Medial OA/ACL .025,Medial Meniscus/MCL .02,Lateral Meniscus/Fracture/Lateral OA/PF OA .008,Effusion .004,Synovitis .003이다. **독립 모델 자체의 보고점수는 .929**이며 .958은 다른 대형 계열과의 혼합 결과다. [A1 셀1·18](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

## 개별 노트북에서 실제로 무엇이 달라졌나

### A1 — goodpjw CoAtNet + R6 + 2.5D, v27

활성 경로는 R384 mirror(셀3/4)+R224 mirror(6/7)+C6 flip(9/10)→소견별 rank blend(12)→G의 작은 보정(15~18)이다. Raptor share는 Synovitis.8,Lateral OA.2,PF OA.6,Baker.35,나머지.5. Raptor 내부는 .65/.35다. G stage가 시간/추론 오류로 실패하면 Part A 제출을 유지하는 구조이므로 결과 receipt 없이 모든 stage가 실제 실행됐다고 단정하면 안 된다. [A1 셀12·18](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

비교 증거는 비교적 구체적이다. 저자는 단일CoAtNet .949→G의 소견별 혼합 .950,이후 S6계열 .954→R7계열 .956→R6 flip .957→소견별 가중치 .958을 기록했다. 그러나 이 전체 경로는 학습·데이터·추론·가중치가 같이 바뀌므로 각각의 순수 인과효과가 아니다. **G full weight를 .956 base에 더하면 .954로 하락**,S6-distilled reader는 gold58 .931이어도 blend .953이었다고 기록한다. v25/v26은 R7을 추가해도 둘 다 .958이며 별도 이득이 입증되지 않았다. 채택할 교훈은 독립성·실제 holdout 검증이고, “모델을 많이 추가할수록 오른다”가 아니다. [A1 셀1](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

### A2 — Does S6 help next to R6, v8

제목에 S6가 있지만 **현재 v8 활성 코드는 R6만**이다. A1과 동일한 R384/R224/C6/G를 쓰고,셀12의 Raptor weight만 ACL.3,Effusion.7,Fracture.8로 바꾼다. 나머지는 A1의 가중치다. 이는 현재 소스에서 S6를 찾는 것과 과거 실험 결과를 읽는 것을 분리해야 하는 사례다. [A2 셀9·12](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

실험 기록: S6를 compact slot의30%로 추가한 v4/v5는 **.958→.957**. Gold58에서 큰 차이가 난3개 소견의 weight를 바꾼 v6도 **.957**. Gold58에서12개 weight를 최적화하고.1~.9로 제한한 v7은 **.955**. 본문은 Gold58의 소견별 양성9~35건에서 나온 차이로 weight를 맞추지 말라고 결론짓는다. 58건에서 내부 AUC가 높은 구성과 더 나은 LB 구성이 같지 않다는, 사용자의 후속실험 설계에 직접 중요한 **반증 자료**다. [A2 셀1](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

### A3 — heliosli best public dataset .9586, v7

R384 mirror+R224 single+C6 flip을 소견별 rank blend하고 마지막에 G를 A1과 동일한 작은 weight로 넣는다. 현재 셀8은 Raptor share **Lateral OA.15,PF OA.65,Synovitis.85,Baker.30**,나머지.5다. A5의 .2/.6/.8/.35와 다르다. 셀0의 일반적인 “.5/.5” 요약보다 실제 셀8을 우선했다. 셀9의 `_PB_FILES`에는 G의3개 파일이 들어 있고 SHA를 점검한 뒤 추론한다. [A3 셀2·4·6·8·9](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586)

새 학습을 하는 노트북이 아니라 이미 공개된 세 계열을 재조합한다. 제목의 .9586을 읽을 수 있지만 A5→A3에서 가중치4개 변화와 G 추가의 각각 독립된 before/after는 없다. “G 때문에 .0006 상승” 또는 “이 소견 가중치가 일반화 최적”으로 해석하면 안 된다.

### A4 — heliosli old version .9573, v2

R384 mirror+R224 single+C7 no-flip. 최종 `.5 C7 + .325 R384 + .175 R224`의 percentile ranks다. dataset slug는 `rsna-knee-r6-frozen-fe055406`인데 `BUNDLE_ID=8c7b666e27dc0e8f`,주석은 그 dataset v2가 R7이라고 명시한다. **이름·최신 dataset 선택만으로 R6/R7 재현을 보장할 수 없다.** manifest를 고정해야 한다. [A4 셀0·6·8](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use)

A6보다224 체크포인트를 더 사용하는 구조는 확인되지만 제목 .9573을 A6의 공식 점수와 같은 조건에서 잰 상승분으로 직접 빼지 않는다. 학습 스크립트는 C6/C7 공통 소스이고 실제 배포 bundle만 다르다.

### A5 — heliosli Blend mine, v1

R384 mirror+R224 single+C6 flip. 최종 Raptor share는 Synovitis.8,Lateral OA.2,PF OA.6,Baker.35,나머지.5이며 G는 없다. C6 bundle을 별도의 `rsna-knee-r6-frozen-fe055406-w1`로 지정하여 원래 dataset 최신버전이 R7로 바뀐 문제를 피한다. A1/A2는 이 recipe를 .958로 인용하지만 이 수집본의 metadata에서 공식 숫자를 검증한 것은 아니다. [A5 셀6·8](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters)

사용자가 참고할 수 있는 것은 서로 다른 모델의 소견별 강점을 이용하는 **고정 rank-blend 구조**다. 가중치가 어떤 독립 validation 과정으로 결정됐는지는 여기만으로 재현되지 않는다. 셀8은 Raptor가 Gold58에서 선택되었으므로 weight fitting에 쓸 깨끗한 labelled set이 없다고 주석을 남긴다. Gold58이 gradient 학습에서 빠졌다는 주장과 개발상 완전히 미접촉이었다는 주장을 혼동하면 안 된다.

### A6 — heliosli .957 latest one-click, v1

가장 단순한 C7 조합이다. **R384 mirror 하나+C7 4fold**,각 소견별 percentile rank를 .5/.5로 평균한다. R224/G는 없고,같은 데이터 decode를 공유하여 C7의4fold를 실행한다. 현재 recipe의 .956 설명과 제목 약 .957은 구분해야 한다. 활성R7 모델은 위 manifest의 spatial pooling·report-state·extent·fastMRI auxiliary 등을 갖지만 이 노트북이 그것들을 새로 학습하지 않는다. [A6 셀0·2·4·5·6](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click)

### A7 — evgendvorkin RSNA Versia 7, v22

긴 설명은9개 과거 공개 해법의 진화를 다루지만 현재 추론은 **A6와 같은 R384+C7 .5/.5**다. 셀3/7/9/11과 A6 셀2/4/5/6을 Python AST로 비교했을 때 주석/docstring을 제외한 실행 구조가 동일했다. 자체 학습 향상 사례로 한 표를 추가하지 않고 **문서화·재현 파생물**로 분류한다. 본문도 .956은 원저자의 점수이며 자신의 재현 점수는 별도 확인 대상이라고 명시한다. [A7 셀1·3·7·9·11](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7)

과거DINO·RadImageNet·S5/S6 등의 설명이 현재 활성 ensemble에 모두 포함된다고 읽으면 안 된다. 채택 가치는 계보 이해와 inference 재현성이다.

### A8 — prvsiyan bee's knees, v47

151개 셀 중 현재 `code`는 **셀148 하나**다. 과거DINO/A5/RadImageNet/Raptor·독립reader 실험은 raw 또는 markdown 셀로 보존되며 실행되지 않는다. 셀148의 base91/LZMA package를 바이트로 복원하고7개 payload SHA256을 모두 확인했다. 복원된 `runner.source.py`는 고정 upstream notebook의 셀을 사용하고 `blend55_patch.py`로 최종 가중치만 바꾼다. 이 현재 recipe는 **R384320px paired + R224224px0.8crop single + C6 flip**,final weights **C6.55,R384.2925,R224.1575**다. [A8 셀145·148·150](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push), [복원된 blend patch](../outputs/research/kaggle_code_20261019/prvsiyan__the-bee-s-knees-final-rsna-push/inspection/bee_blend55_patch.py.txt)

셀149에는224 mirror 설명이 남아 있지만 마지막 셀150과 활성 package는 **추가mirror inference 없이** 가중치만 바꾸는 실험을 명시한다. 셀145는 upstream `heliosli/...this-is-new-version-0-9576-in-one-click?scriptVersionId=358332966`의.957을 인용하면서 **자신의 새 공식 점수는 없다고 명시**한다. 따라서 고점수 후보 목록에서 상세검토했으나 새0.95 달성 사례로 세면 안 된다. 반복가능한 source pin과 baseline 대비 동일예측 비교를 중요하게 다룬다는 점은 참고할 만하다.

### A9 — kaiwalya Competition Solution, v3

R384 mirror+R224 mirror+C6 flip+G의 작은 가중치라는 A1 recipe다. A9 셀1/3/5/7/9/10/11/12가 A1 셀3/6/9/12/15/16/17/18과 **주석/docstring을 제거한 AST에서 동일**했다. figure는 “this notebook's reader”,“versions25/26 .958”라는 A1 계보의 설명까지 계승한다. 이 사용자의 독립적인 v25/v26 학습 실험이라고 볼 수 없다. [A9 셀0~12](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-competition-solution)

단순 제목 요약에서는 놓치기 쉬운 파생 관계다. 동일모델·동일혼합을 재실행한 노트북을 독립ensemble 다양성으로 추가하면 실질적인 새 정보가 없다.

### A10 — kiyoshiohno Template, v27

셀2는42줄 launcher다. `rsna_knee_manifest.json` 중 `bundle_id=rsna-knee-local-logic`,`runner=rsna_knee_pipeline.py`인 단일 dataset을 찾고 그 runner를 실행한다. 노트북 자체는 해상도·crop·pooling·학습·혼합을 정의하지 않으며 SHA 검증도 하지 않는다고 문서에 명시한다. metadata에는G/C6/nartaa/code dataset과 **빈 dataset ref**가 있다. 공개 launcher만으로 그 빈 항목의 실제 pipeline/버전/성능을 복원할 수 없다. [A10 셀0~2](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template), [metadata](../outputs/research/kaggle_code_20261019/kiyoshiohno__rsna-knee-template/metadata.json)

참고할 아이디어는 코드와 추론환경을 dataset으로 버전화하는 배포 패턴이다. 어떤 모델이 점수를 올렸는지 판단할 증거는 부족하다. 이 한계를 채우기 위해 무관한 private 입력에 접근하거나 임의 실행하지 않았다.

## 사용자의 현재 방식과 비교

사용자 R01 **.771432는 Gold58 개발 OOF**이고 위 공개값은 대부분 저자보고 public LB다. 둘을 빼서 “.187 차이”처럼 효과를 계산할 수 없다. 사용자 실험은 동일Gold58의3fold와3seed이고,외부 공개model의 training/selection split은 다르다. 공개모델을Gold58에 그대로 대입해 나온값도 기존OOF와 동일한 검증조건이라고 볼 수 없다. [현재 실험 결과](followup_experiment_results.md)

| 축 | 사용자 현재 | 반복하여 확인된 공개 고점수 recipe | 의미 / 다음 판단 |
|---|---|---|---|
| 영상학습 | Swin-T 대부분 고정특징,일부weak-only1epoch FT | C6/C7 ConvNeXt-small 영상encoder와 head16epoch,4fold | 현재의128차원 후단만 조정하는 탐색과 학습량·표현 적응의 규모가 다름. 같은 자료/OOF에서 단계적으로 비교할 근거 |
| 정보 범위 | 중심25~75%,5centers,각5single neighbors,224px | Raptor2~98%,96slice/94triplets; G4~96%/16triplets; C6/C7 physical dense coverage+full/local | MCL·lateral meniscus 등 주변 구조가 입력에서 잘리는지 먼저 확인. Raptor 상수를 복제하기보다 wider-span·physical coverage를 분리 실험 |
| 물리 크기 | IPP/IOP 순서는 사용,PixelSpacing 기반mm crop 없음 |140mm /153.6mm 표준FOV,100mm local,3.5mm through-plane | mm기반 위치RoPE를 썼다는 것과 입력의물리scale 정규화는 다른 개선축 |
| 좌우/시야 | 명시적 laterality normalization 부족 | C6/C7 side-canonical geometry와knee/patella anchor,불명확side분류; 해부학별mirror | 정렬방향·좌우·crop을 함께 일관되게 처리해야 함. 임의hflip을TTA로 추가하지 않음 |
| pooling/position |768→128,self-attention/RoPE/gate/cross;Swin global feature 이후 | Raptor class attention;G384dim2layerTransformer;R7 spatial+window class attention+geometry | 이미 attention은 있으므로 “attention의 유무”가 핵심 격차라는 근거 부족. 입력의 coverage와 spatial pooling 위치를 구분 |
| 약라벨 |Gold58+weak4292,UNK weight0,confidence-weighted softBCE |C6/C7 weak4344,Gold gradient 제외,report-state/정도/extent/ranking;Raptor는여러label source+OAI | 코드만으로teacher우열은 못정함. 같은 표본에서known/unknown·정도기준·label source차이를검증해야함 |
| 외부자료 |현재경로경쟁영상중심 |nartaa OAI,compact SKM-TEA anchors,R7 fastMRI auxiliary | 외부자료와모델훈련효과를후단구조효과로오인하지않기 |
| ensemble |R01확률50:50,실제E09.5/S11.25/S12.25 |독립학습계열의rankblend,224/320multi-res,TTA,class weights | 다른구조가아닌동일feature family만섞는것과차이. 다만A1/A2의추가모델악화사례로각arm검증필수 |

사용자 코드의 근거: [spatial_core.py](../experiments/spatial_core.py), [weak_core.py](../experiments/weak_core.py), [followup_experiment_results.md](followup_experiment_results.md). 기존R01의개선은같은OOF체계안의유효한관찰이며,공개점수와동일척도로직접비교하지않는것이그관찰을약화시키지않는다.

## 이 표본에서 채택할 것과 추정하면 안 되는 것

1. **우선할 증거 기반 변화:** 물리 FOV·좌우 정규화·넓은 slice coverage를 원영상 샘플로 확인하고, 변경별 OOF를 고정해 비교한다. 그다음 backbone 미세조정의 충분한 학습량, 가능하면 spatial feature pooling을 검토한다. 소스상 차이가 크다는 근거와 실제 점수 개선 예측은 별개다.
2. **의미 있는 구조 차이:** R7은 report-state/정도와 영상 내 class별 spatial pooling이 있다. 공개 입력 label/보조자료까지 같아야 그 학습을 정확히 재현할 수 있다. generic 코드에 있는 모든 옵션을 R6/R7에 적용됐다고 기재하지 않는다.
3. **앙상블은 독립성부터:** A1의 G 단독 .929가 좋은 초기 보완이었지만 강한 base에 큰 weight로 추가하면 오히려 하락했다. 복사 노트북의 개수를 독립 reader 수로 세지 않는다.
4. **Gold58에 대한 과적합 증거:** A2의 Gold 최적 weights가 LB .958→.955로 하락했다. 사용자의31개 개발 결과에서 소견별 최고모델을 선택해 혼합하는 것은 별도 검증 없이 성공으로 기록하기 어렵다.
5. **점수 provenance 유지:** 제목 .9586, 과거버전 .958, 타인 .956 인용, 새 점수 없음을 각각 유지한다. 버전·checkpoint hash·실제 사용 manifest·stage 성공 receipt가 없는 실행은 같은 recipe인지조차 불명확할 수 있다.

이는 새 훈련·공개모델 실행을 수행한 결과가 아니라, 고정된 공개소스와 manifest를 읽어 얻은 증거 검토다. 제안한 변경으로 사용자의 독립검증 성능이 얼마나 오를지는 이 자료만으로 산출할 수 없다.


---

# 공개 Code 원문 검토: 11–25번

조사일: 2026-10-19. 아래 점수는 **저자의 본문·제목에 기재된 값**이다. 이번 조사에서 해당 가중치로 hidden test를 재실행하거나 제출 이력의 숫자 점수를 독립적으로 검증하지 않았다. 셀 번호는 다운로드한 원본의 **0부터 시작하는 번호**다. 소스는 실행하지 않고 읽었다. 제목보다 최종 실행 경로와 설정을 우선했다.

## 11. Knee-centred ConvNeXt 추가 — bliverrigpdoge, v1

[원본](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext)

- 저자 보고: R7+Raptor 기본 조합 **0.956 → 0.957**. 기본 결과에 자체 knee-centred 모델의 rank를 20% 섞는다.
- 셀 2–6은 Raptor SWA와 compact R7의 추론·rank 결합이다. 셀 7은 별도 `leg_main.py`와 `members.json`을 찾고 자체 모델을 실행한다. 셀 9의 실제 가중치는 `base=0.8, dm=0.2`다.
- 자체 branch의 ConvNeXt-small/EfficientNetV2-S 및 medial/lateral crop 설명은 저자 본문에 있다. **그 추가 가중치가 비공개**라는 사실도 명시돼 있다. 셀 7·9에서 해당 입력이 없으면 추가 branch를 제외하고 기본 CSV를 유지한다. 따라서 공개 fork가 저자의 0.957 모델 전체를 재현한다고 볼 수 없다.
- 우리에게 필요한 것은 해당 숫자의 복제가 아니라 **서로 다른 시야의 모델이 다른 오류를 내는지 확인하는 방법**이다. 작은 장기·부위 crop을 별도 모델로 학습하고 OOF 오류 상관과 blend 효과를 확인하는 후보로 삼는다. 성능이 낮은 모델이면 다양성만으로 도움이 되지 않는다.

## 12. Four-model ensemble — laimerbotrolenk, v1 / 내부 실험명 V33

[원본](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble)

- 기준 조합의 저자 보고 점수는 **0.957**. 그러나 **현재 mirror-mix 후보의 점수는 미측정**이라고 셀 0과 20이 명확히 구분한다. 이 후보를 0.957 개선 성공 사례로 집계하지 않는다.
- 셀 3의 기본 선택은 `mirror_mix_half`. R7 0.30 + S6 0.20 + Raptor384 TTA 0.325 + Raptor224 plain-rank 0.0875 + Raptor224 TTA-rank 0.0875다. 기준 조합은 마지막 두 항 대신 224 TTA-rank 0.175를 쓴다.
- 셀 11에서 이미 계산한 plain/mirror 출력을 보관하고 셀 20에서 조합한다. **확률 75:25 평균과 rank 공간 50:50 혼합은 다른 연산**이다. 셀 20은 두 후보 CSV와 `measured_public_auc=None`인 receipt를 남긴다. `REQUIRE_S6=True`면 S6 누락 시 네 모델이라는 이름으로 다른 조합을 내보내지 않는다.
- 우리가 배울 점은 후보/대조군을 같은 실행에서 만들고, 실제 사용된 branch·시간·실패 여부까지 기록하는 방식이다. 이미 동작하는 자동화의 검증 기록에 TTA 유효 여부를 추가하는 데 참고할 수 있다.

## 13. 0.957 Explained — leoprovorov, v12

[원본](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl)

- 저자는 10월 15일 public LB **0.957, 206위**와 기준 0.956을 보고한다. 설명된 조합은 Raptor384 0.325 + Raptor224 mirror 0.175 + R7 0.30 + S6 0.20이다.
- **이 v12 소스의 실행 셀은 HTML handbook 표시, 통계 계산, 그림 표시다.** 본문에 제출 코드가 포함됐다는 설명이 있지만 셀 22는 markdown 설명이며, 여기서 추론을 수행하거나 `submission.csv`를 생성하는 실행 셀은 확인되지 않았다. 실행 가능한 네 모델 추론 예시는 12번에서 따로 확인했다.
- Gold58에서 Raptor 계열을 선택했다는 평가 한계를 설명하는 부분은 유용하다. 반면 hidden public 표본수 약 640, 모델 오차의 동일상관 가정으로 구한 성능 상한 등은 **추정 모형**이다. 실제 test 크기·달성 가능한 상한으로 인용하지 않는다.
- 우리 노트에도 `실측 OOF / 저자 보고 LB / 아직 실행하지 않은 가설`을 분리한다. 외부 모델의 gold 점수가 높아도 그 gold가 checkpoint 선택에 쓰였다면 우리 OOF와 동일한 검증 조건이 아니다.

## 14. Raptor + ConvNeXt R7 — jenillangaliya, v1

[원본](https://www.kaggle.com/code/jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble)

- 제목/본문 보고 **0.956**. 셀 2의 CoAtNet Raptor SWA, 셀 4의 bundle `8c7b666e27dc0e8f`, 셀 6의 **Raptor 0.5 + R7 0.5 rank blend**가 실제 경로다.
- 독립적인 새 backbone 학습보다는 heliosli/nartaa의 공개 추론 조합과 설명을 제공한다. R7의 class별 spatial attention, fastMRI 보조 학습 등은 R7 manifest 및 원 training code를 함께 확인해야 한다. 이름에 OrthoFoundation이 있다고 최종 분류 backbone이 OFL Transformer인 것은 아니다. 이 경로는 ConvNeXt-small이다.
- Raptor 이름의 384는 원 렌더 크기이며 이 소스의 실제 입력은 320 center crop이다. 96개 stack에서 94개 triplet window를 사용한다. `5-fold`라는 잔존 주석과 달리 현재 `ARMS`는 SWA checkpoint 한 개다.
- 우리에게는 **Dense input + 학습된 backbone + 보완적인 두 모델**이라는 패키지 비교 대상이다. 이 결과만으로 CNN 대 Transformer 우열을 판정할 수 없다.

## 15. Dual SOTA 0.954 — jenillangaliya, v4

[원본](https://www.kaggle.com/code/jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble)

- 본문 보고 **0.954**. 셀 5의 구세대 compact bundle `209a32c8b50a6b1c`와 셀 7의 **Raptor 0.70 + compact S6 0.30** rank 결합을 확인했다. 14번 R7의 50:50과 다르다.
- 모델별 subprocess로 메모리를 반환하고 파일·체크포인트를 확인한다. 그러나 본문의 “OOM 위험 0”, “실패 없음”은 보장이 아니다. 실행 오류와 예측 실패는 여전히 발생할 수 있다.
- 세대와 결합 비율이 동시에 달라지므로 14번과의 0.002 차이를 R7 한 요소의 효과로 분리할 수 없다. 우리 실험은 같은 split·head·비율을 유지한 상태에서 한 요소씩 비교해야 한다.

## 16. Unused 224 checkpoint — varshithakolupuri, v1

[원본](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956)

- 저자 보고: 공개 표시 점수 **0.956 유지**, 같은 날짜 순위 233→156. 앞선 S6 조합에서는 0.954 표시를 유지하며 437→320이라고 설명한다. **순위 변화만으로 정확한 AUC 증가량이나 통계적 유의성을 확정하지 않는다.** 표시 동점 정렬 원인도 이 조사에서 별도 검증하지 않았다.
- 셀 9–12: 공개 가중치 묶음 안의 `c08_r224_swa.pt`를 추가로 실행한다. Raptor 부분만 384:224=0.65:0.35로 바꾼다. 전체 비율은 R7 0.5, Raptor384 0.325, Raptor224 0.175다.
- 224 branch는 시간 여유가 없거나 실패하면 생략된다. 이때 결과는 새 조합이 아니라 원래 두 모델 조합이다. 원본의 `R224_OK` 및 실행 로그 확인이 필수다.
- 우리 R01의 서로 다른 입력/head 결합과 가장 가까운 아이디어다. 우선 기존 OOF에서 확률 평균 대 rank 평균을 검증하고, 그다음 전처리·시야가 다른 학습 모델을 추가하는 순서가 경제적이다.

## 17. One free flip — varshithakolupuri, v1

[원본](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up)

- 저자 보고: **0.956 표시 유지**, 10월 14일 순위 357→289. 16번과 날짜가 다른 순위를 직접 효과량으로 비교하면 안 된다.
- 224 branch에 plain + mirror 확률 평균을 추가한다. `_mirror_tri`는 sagittal window에서 세 채널의 순서를 뒤집고, coronal/axial에서는 가로축을 뒤집는다. 그 후 rank를 계산한다. 따라서 모든 방향의 이미지를 똑같이 가로 반전하는 것과 다르다.
- 새 학습 없이 추론 pass를 늘린 사례다. 제목의 “free”는 추가 학습이 없다는 의미에 가깝고 **추론 비용은 증가**한다.
- 우리 적용 시 DICOM 방향 정규화와 각 token 좌표·plane metadata를 함께 변환해야 한다. 공개 recipe의 mirror 규칙을 현재 raw orientation에 그대로 붙이면 해부학적 좌우 반전이 아닐 수 있다. OOF에 같은 TTA를 적용해 먼저 평가한다.

## 18. CoAtNet + ConvNeXt MIL — hitarthjain0, v5

[원본](https://www.kaggle.com/code/hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950)

- 셀 12의 저자 결과표: 기본 **0.950**, 자체 EfficientNet-B3 10% 결합 **0.948**, 더 큰 자체 모델 비중 **0.944**, 자체 모델 단독 **0.902**. 새 모델의 다양성이 품질 부족을 반드시 보상하지 않는 실제 반례다.
- 셀 8 `KneeNet`: ConvNeXt의 이미지 vector → 384차원 → slot embedding + 위치 MLP → depth=2 TransformerEncoder → 12개 소견별 attention pooling. **사용자의 위치+방향+attention 구상과 유사한 구현이 이미 있다.** 이 구조가 0.950 전체 점수를 혼자 만든 것은 아니며 공개 CoAtNet과 섞는다.
- v5의 셀 9 실제 자체 reader TTA는 **256/320 두 크기, slice offset 0/+1, 가로 flip**이다. 본문의 384/512 표현과 구분해야 한다. 셀 10은 class별 V5 가중치와 작은 tie-break 항을 적용한다.
- 셀 11의 파일 모양 검사는 점수의 보장이 아니다. 특히 최신 V5의 목표 점수와 셀 12의 과거 0.950 실측 보고를 같은 것으로 보지 않는다. 모든 plane에 대한 가로 flip이 17번의 해부학적 mirror와 같은 것도 아니다.

## 19. Micro-boost tri-reader — rabari9999, v11

[원본](https://www.kaggle.com/code/rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader)

- 본문은 **0.9564+** 및 여러 단계의 CV/LB 상승표를 보고한다. 다만 해당 단계별 OOF 예측, split·checkpoint 선택 과정과 공식 제출 점수를 이 소스만으로 재검증하지 못했다. **저자 주장으로 보존하고 개선 효과의 증거로 확정하지 않는다.**
- 셀 15의 실제 코드는 Raptor384/224 + 3개 fold의 ConvNeXt reader를 class별로 섞는다. percentile→probit 변환, 온도 1.02, clipping, medial→lateral 12% 혼합, 다른 소견 score 기반 보정, 작은 tie-break 후 최종 rank 순서다.
- 수학적 검토: **양의 상수로 나누는 온도 조정만으로는 순위가 변하지 않아 AUC가 개선되지 않는다.** clipping은 동점을 만들 수 있다. 단조 변환 이전의 모델 결합이나 소견 간 혼합은 순위를 바꿀 수 있지만 개선은 보장되지 않는다. 따라서 “tail을 날카롭게 하면 AUC가 반드시 오른다”, “probit가 항상 낫다”는 설명을 채택하지 않는다.
- 우리에게는 우선순위가 낮다. 58 gold에 수십 개 class별 상수를 맞추면 개발 표본 과적합 위험이 크다. 임상적 인과 설명으로 가중치를 정당화하기보다 독립 검증 또는 nested 평가가 필요하다.

## 20. Multi-scale OrthoFoundation — yuvrajlamba01, v2

[원본](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956)

- 제목/본문 **>0.956**, 순위 156과 gold AUC 0.9512를 보고한다. 단계별 gold 표의 원 예측과 검증 설계는 이 소스에서 확인되지 않아 독립 CV로 취급하지 않는다.
- 셀 6은 R7 `8c7b666e…`, 셀 9는 224 crop, 셀 12는 `A224=.35`, Raptor:R7=.5:.5다. 실제 핵심은 **16번의 공개 3-branch 조합**이다. 새로 학습된 OrthoFoundation backbone의 별도 성공 사례로 집계하지 않는다.
- 최신 branch 수, fallback 여부, 실제 blend 계수는 다이어그램보다 코드로 확인해야 한다. 16번과 함께 한 계보로 묶고 증거를 두 배로 세지 않는다.

## 21. Dual-engine / triple-reader — rabari9999, v4

[원본](https://www.kaggle.com/code/rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951)

- 본문 보고 **0.951+**. 셀 3–5에서 Raptor224와384, 셀 7–9에서 2.5D reader를 정의하고 셀 10에서 결합한다.
- 실제 `W_C224=.3`, `W_OWN=0`이다. 후자는 ConvNeXt reader가 없다는 뜻이 아니라 **추가 own2 모델만 비활성**이라는 뜻이다. `cnxt_v0_fold*.pt` 세 개는 활성이다.
- CoAtNet 두 모델의 rank를 70:30으로 섞고 다시 rank, 이후 class별 ConvNeXt 비중(Baker's .35, Contusion .30 등)을 적용한다. 19번의 probit·cross-prior 방식과는 구분한다.
- 우리 코드와 닮은 study token Transformer 경로가 있지만, 공개 학습 가중치와 후처리를 함께 쓰는 구성이다. 우리 S11의 단일 head 변경 결과와 직접 대조하는 ablation은 아니다.

## 22. V15 tri-hybrid — yuvrajlamba01, v8

[원본](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-v15-tri-hybrid-ensemble-0-954)

- 제목은 **0.954**, 본문에는 다른 높은 점수 표현도 있어 버전별 대응을 확정할 수 없다.
- 셀 5는 `BUNDLE_ID=209a32c8b50a6b1c`, 후보 dataset은 `compact-r6cnxfsss` / `compact-v1`이며 manifest ID를 검증한다. 따라서 dataset 이름에 r6가 있다고 최신 R6/R7와 동일하게 취급하면 안 된다.
- 셀 7의 실제 결합은 Raptor .5 + compact .5다. 224 추가 branch가 실행되지 않는다. **tri-hybrid라는 제목을 독립적인 세 backbone 앙상블의 근거로 쓰지 않는다.** 셀 8의 분포 그래프도 hidden 정답을 쓰는 성능 검증이 아니다.
- 얻을 교훈은 bundle hash와 활성 경로를 기록하는 것이다. 우리 자동화의 source/config/checkpoint pinning은 이미 이 문제를 다루고 있다.

## 23. 학습 원형: Pilkwang baseline — v15

[원본](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)

이 항목은 ≥0.95 독립 성공 건수에 넣지 않은 **학습·전처리 대조군**이다. 셀 13·19·21·25–33에서 확인한 핵심은 DINOv2, 130 mm crop, 좌우 정규화, protocol slot, class별 slot attention, backbone 마지막 6 block 적응이다. 기본 head LR=1e-3, backbone LR=8e-6, 10 epochs이며 feature를 영구 고정한 head 학습과 다르다. 3장은 spread된 선택 위치를 채널에 넣는 방식이므로 우리 adjacent triplet과도 구분한다.

공개 weight package가 붙으면 셀 30에서 **여러 사전학습 member 추론** 경로가 실행된다. 기본 학습 분기는 그 모든 배포 가중치의 재학습을 보증하지 않는다. label의 confidence floor와 unknown 처리도 우리의 UNK zero mask와 다르다. 검증 split·teacher 선택 과정까지 동일하다고 가정하지 않는다.

## 24. 학습 원형: Training the twelve-finding model — dreaddevelopment, v8

[원본](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model)

이 역시 ≥0.95 건수에 넣지 않았다. 소스 머리말은 **0.924 추론 모델의 학습 원형**이라고 설명한다. `StudyWindows`, `RaptorClassifier`, `main`을 확인했다.

- 44×336×336 uint8 corpus에서 adjacent triplet을 뽑아 학습 때 random K, 검증 때 균등 K를 쓴다. 기본 CLI는 ViT-small/224/12epochs이고, CoAtNet384/16epochs는 제시된 실행 예시다. 예시를 기본값으로 잘못 적지 않는다.
- 이미지 backbone과 class별 attention-MIL을 함께 학습한다. AdamW backbone LR3e-5/head1e-3, OneCycleLR, BCEWithLogitsLoss(pos_weight), gradient clipping. ImageNet 초기화가 현재 기본이며 이름에 Raptor가 있다고 비공개 SSL 가중치가 반드시 필요한 것은 아니다.
- 58 gold는 gradient 학습에서 제외하지만 **best epoch·top3 checkpoint 선택에 사용**한다. top3 weight average와 gold 성능도 출력한다. 그래서 이 gold 점수는 최종 독립 평가가 아니다.
- report soft label 파일은 제공하지 않으며 사용자가 준비해야 한다. 없으면 정상 exit 0으로 끝난다. **프로세스 성공 = 학습 완료**가 아니라는 사례다.
- 우리에게 가장 직접적인 후속 방향은 고정 feature 한계 확인 후 제한된 backbone 해제, 적절한 학습률·warmup, 검증 기준을 고정한 weight averaging이다. 이 원형을 복사한다고 nartaa의 OAI 추가학습 최신 .949 가중치가 자동 재현되는 것은 아니다.

## 25. Speedy Raptors — mattiaangeli, v39

[원본](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original)

주요 역할은 **학습/추론 설계 대조군**이며 이 v39의 ≥0.95 새 점수를 확인한 것으로 집계하지 않는다. 이전 버전 분석 문서와 달라져 현재 활성 경로를 다시 확인했다.

- 셀 13·15의 DINOv2 20 member, 이후 A5/Rad 및 Raptor 경로, 셀 22의 추가 CoAt family가 누적되는 큰 추론 graph다. 셀 23은 DINO fingerprint 20개와 calibrator 적용을 확인한 뒤 마지막 CSV를 쓴다.
- 셀 22에는 public Raptor의 여러 resolution/span branch 외에 residual gated spatial head, Global96, D4 depth-zone, Repair-v1 child가 들어 있다. 성공한 member의 **확률 평균을 rank**하는 경로와, 확률 파일이 없을 때 기존 rank 혼합 경로가 구분돼 있다. child 실패 시 family가 달라질 수 있다.
- D4의 depth 구획, class별 spatial pooling과 보호된 global residual은 사용자가 제기한 **“이미지 feature가 global pooling에서 너무 사라지는가?”**에 직접 대응하는 아이디어다. 현재 ours의 cross-attention은 이미지별 pooled vector 다음에 있으므로 patch 내부 정보를 되살릴 수 없다.
- 다만 복잡한 전체 graph의 효과를 특정 head 하나의 개선으로 해석하면 안 된다. 우리는 patch pooling만 바꾼 단일 대조 실험을 먼저 하고, 성공한 경우에 depth-zone과 앙상블을 붙이는 편이 해석 가능하다. 속도 개선도 새 정확도 증거와 별개다.

## 검토 증거 보존

25개 소스의 버전·표 수·SHA256·셀 개수는 `outputs/research/kaggle_code_20261019/snapshots.json`, 각 원문과 cell index는 같은 디렉터리의 저자별 폴더에 있다. R7 설정은 `manifests/r7/manifest.json`을 별도 확보했다. 내려받은 타인 코드는 실행하지 않았다. 이 문서는 소스 수준 분석이며, 타인의 hidden-test 제출 점수나 공개 가중치의 전체 학습 과정을 재현한 결과는 아니다.
