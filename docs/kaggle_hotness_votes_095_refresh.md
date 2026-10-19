# Hotness·Most Votes로 다시 확인한 RSNA Knee 0.95 이상 코드

조사일: 2026-10-19 KST. **이전 25개 원문 검토에 더해, 이번에는 Kaggle 공개 API의 실제 `bestPublicScore`를 대조했다.** 제목·작성자 주장만으로 점수를 확정하지 않는다.

## 조사 범위와 새로 확인한 사실

- Hotness·Most Votes 각각 상위 20개를 우선 읽을 목록으로 확인했다. 아래에 40개 목록을 보존했다.
- 공식 웹 조회 API는 Hotness·VoteCount·ScoreDescending 각각 100개씩 조회했다. 중복 제거 **224개 metadata**이며, 이 중 최고 공개 점수 0.950 이상은 **115개**였다. 224개 모두의 전체 코드를 읽었다는 뜻은 아니다.
- 이전에 읽은 **25개 중 고득점 후보 22개 모두에서 공식 최고 공개 점수 ≥0.950을 확인**했다. 나머지 3개는 학습·설계 참고 자료다.
- 이번에는 notebook 원문 **8개**를 추가로 확인했다. 기존 goodpjw v27 재확인 1개와 새 자료 7개이며, 실행 셀·활성 설정·점수 조회 코드·저자 학습 설명을 대조했다. 원문·버전·SHA를 저장했다.
- Hotness 상위 20개 중 14개, Most Votes 상위 20개 중 8개가 ≥0.950이다. 추천 수와 정확도 순위는 다르다.

**`bestPublicScore`는 notebook의 역사상 최고 공개 점수다. 지금 보이는 최신 버전이 그 점수를 냈다는 의미는 아니다.** 또한 이 환경에서 모델을 다시 학습·제출해 재현한 결과도 아니다. 점수는 12개 소견의 macro ROC-AUC이며 95% 정답률이 아니다.

API 근거: Kaggle 공개 `kernels.KernelsService/ListKernels`의 `scriptUrl`, `bestPublicScore`, `totalVotes`, `scriptVersionId`. 코드 목록 [원페이지](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code). 조회 방법은 공개된 [실제 점수 대조 코드](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook)의 셀 1·6에서 확인하고, 자체 작성한 읽기 전용 script로 조회했다. 공개 원문 자체를 실행하지는 않았다.

## 먼저 읽을 만한 코드와 진행 방식

| 우선순위 | 원문 | 공식 최고 공개 AUC | 실제 확인한 내용 / 용도 |
|---:|---|---:|---|
| 1 | [goodpjw CoAtNet + R6 + 2.5D MIL](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | **.958** | Most Votes 4위, 370표. CoAtNet 384·224 + compact R6 + 독립 reader; mirror TTA와 소견별 rank 결합. 버전별 성공·실패 기록이 상세함 |
| 2 | [nartaa S6 ConvNeXt](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) | **.950** | Hotness 1위. 단일 주력 모델을 별도 약라벨로 재학습. 전체 시야 + 무릎 중심 crop + spatial/window attention. 소수 실패 검사는 CoAtNet fallback |
| 3 | [heliosli R7 + Raptor](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | **.956** | Most Votes 12위. compact ConvNeXt와 CoAtNet의 보완적 결합. 제목의 약 .957과 조회값을 구별 |
| 4 | [heliosli 소견별 결합](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) | **.958** | Most Votes 17위. 강한 서로 다른 모델의 소견별 결합. 모델 하나를 처음부터 학습하는 notebook이 아님 |
| 5 | [224 checkpoint의 mirror 추가](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up) | **.956** | Hotness 11위. 이미 학습된 224 모델에도 해부학적 mirror 적용. 표시 점수와 순위 변화, 실제 학습 개선을 구별 |
| 6 | [RSNA Versia 7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | **.956** | Most Votes 3위. 주요 공개 모델을 결합한 추론 경로; 다른 notebook과 가중치 계보가 겹침 |

## 실제로 어떻게 학습했나: nartaa의 구체적인 0.950 사례

이 사례는 단순 복사·blend 설명보다 **우리의 재학습 실험 설계에 직접 참고할 내용**이 많다. 원저자는 Heliosli의 S6 구조와 전처리를 사용하되, 자신의 보고서 약라벨로 가중치를 새로 학습했다고 설명했다.

- ConvNeXt-small backbone과 소견별 **이미지 내부 spatial attention → window MIL**을 학습한다.
- 전체 영상과 anatomy network로 찾은 무릎 중심의 **100 mm 국소 crop**을 함께 사용한다. 320 px cache, 288 crop, window당 3장, 물리 간격 3.5 mm 설정이다.
- 약라벨 4,263검사, gold58 제외. 학습 시 검사당 64 windows, **16 epochs**.
- AdamW: encoder LR 1e-4, head LR 1e-3, weight decay .05, batch 2, cosine schedule와 5% warmup. 알려지지 않은 소견은 prior fill, loss는 weight-normalized BCE로 설명한다.
- folds 1–4만 학습한 모델 .948, 전체 약라벨 학습 모델 .950을 보고했다. 국소 OOF 순위와 LB 순위가 달랐던 사례도 공개했다.

학습 조건은 [저자 원문 discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359)에 근거한다. 현재 v2 source의 `B_MODEL`은 16epochs/4263건, `B_GEOMETRY`는 ConvNeXt-small/320/288/3장/3.5 mm를 검사한다. `W_A=0, W_B=1`이므로 정상 검사의 주력 예측은 S6이고, 파일 상단에 남아 있는 과거 `.7/.3 blend` docstring을 현재 활성 설정으로 읽으면 안 된다. 소수 전처리 실패 검사에는 CoAtNet을 쓰므로 배포 경로 전체에 모델이 하나뿐이라고도 표현하지 않는다.

## goodpjw의 0.958까지의 변화에서 배울 점

저자는 CoAtNet 단독 .949 → 자체 reader 결합 .950 → S6 계열 결합 .954 → R7 계열 .956 → R6 mirror와 소견별 결합 .958의 경로를 기록했다. 이는 여러 변화가 함께 들어간 개발 이력이며 각 변경의 독립적인 상승분을 측정한 통제 실험은 아니다. **최신 v27은 .958을 기록한 v25 recipe의 설명 갱신**이라고 명시한다.

자체 reader는 ConvNeXt 특징에 slot embedding과 연속 slice 위치를 더하고 study Transformer와 소견별 attention으로 읽는다. 즉 사용자가 제안한 방향·깊이 활용과 가까운 구성도 실제 쓰인다. 다만 **reader 단독은 저자 보고 .929**, .958은 다른 강한 모델들과의 결합 결과다. 위치 encoding만 추가해서 .958이 된 것으로 해석하면 안 된다. [원문](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958)

반례도 중요하다. 강한 base에 reader를 큰 비중으로 넣으면 .956→.954로 내려갔고, 별도 S6를 추가해도 개선되지 않았다. gold58에서 12개 소견별 blend 비율을 최적화한 다른 실험은 .958 기준에서 .955까지 내려갔다고 기록한다. [S6 추가 실험](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957)

## 우리와의 차이: attention이 없는 것이 아니다

| 항목 | 현재 우리 주력 | 확인한 0.95 계열의 차이 |
|---|---|---|
| 영상 학습 | 대부분 ImageNet Swin-T 고정 feature, head 20epochs. 일부 last-stage 1epoch 실험 | S6는 encoder에도 LR을 주고 16epochs 학습. head epoch와 backbone epoch는 다름 |
| 입력 시야 | 224 pixel crop, 중앙 중심점 위주 | 물리 mm crop, 전체·국소 영상, 더 넓은 범위 및 여러 window |
| 이미지 내부 위치 | slice 전체를 global pooling한 벡터 | compact는 pooling 전에 patch 위치·소견별 공간 attention을 활용 |
| slice/방향 관계 | plane×FS, mm RoPE, class-query cross-attention 구현됨 | reader도 slot+depth+Transformer 사용. 이것만으로 격차가 설명되지는 않음 |
| 약라벨 | 보고서 약라벨 이미 사용, UNK mask와 confidence | 다른 label 조합·prior fill·state/extent 등의 감독 차이. 단순 표본 수만 비교할 수 없음 |
| ensemble | E09·S11·S12의 확률 평균 | 서로 다른 학습, 해상도, fold/SWA, mirror, rank 결합 |

우리 결과가 ‘ResNet feature concat에 머물러 있다’는 설명은 현재 코드에는 맞지 않는다. **핵심 가설은 이미지에서 추출되는 feature 자체, 공간정보를 줄이는 시점, 해부학적 입력 정렬과 backbone 적응량**이다. 이 중 무엇이 실제 병목인지는 우리 검증 분할에서 따로 비교해야 한다. 상세 소스 대조는 [근거 audit](kaggle_095_evidence_audit.md), [기존 비교·계획](kaggle_095_comparison_and_plan.md)에 있다.

## 제목만 보고 0.956 이상으로 취급하면 안 되는 사례

| Notebook | 제목의 표현 | 공식 최고 공개 AUC |
|---|---|---:|
| [Clinical Tri-Reader](https://www.kaggle.com/code/rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader) | 0.956+ | **.950** |
| [Dual-Engine Stack](https://www.kaggle.com/code/rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951) | 0.951+ | **.950** |
| [Heliosli latest](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | 약 0.957 | **.956** |
| [Heliosli new dataset](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | 0.9586 | **.958** |

‘목표 점수’·원저자 점수·표시 정밀도·과거 버전이 섞일 수 있다. 예를 들어 [haideptry V5](https://www.kaggle.com/code/haideptry/rsna-knee-v2-vs-apex-comparative-study)는 과거 V3/V4 .950과 현재 목표 .952~.955를 구별해서 읽어야 한다. 예외 때 이전 예측으로 돌아가는 fallback이 있다고 새 모델의 최소 AUC를 보장하지는 않는다.

Hotness 2위 [MaViT baseline](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu)은 실제 최고 .935로, 고득점 재현보다 추론 속도 비교용이다. 원문은 6 views, 8 slices, 288 px, 130 mm crop, 외부 `model.py`를 사용한다. 추천 수 1위 baseline은 .891이고, 2위 efficiency leaderboard는 모델 학습 코드가 아닌 표 출력 notebook이다.

## 다음에 반영할 우선순위

1. **현재 R/T 실험 순서는 유지**한다. 이번 조사는 notebook 업로드나 GPU 실행, 대회 제출을 하지 않았다.
2. 추가 연구는 물리적 시야/좌우 방향 정규화 → token 수를 맞춘 coverage → pooling 전 공간 attention → backbone warmup·점진적 fine-tuning 순으로 비교한다. 기존 P01~P04 계획과 연결한다.
3. 방향×깊이 T00/T01/T02는 유용한 가설이지만, global pooling 이전 patch 정보 유지와는 별도 축이다. 상위 코드의 실제 patch `(x,y)` 위치 encoding과 두 요소 위치 encoding을 혼동하지 않는다.
4. 빠른 공개 기준선 확인과 직접 학습 경험을 구별한다. 공개 .958 notebook을 실행하면 보통 **이미 학습된 남의 가중치로 추론**한다. 동일 모델을 직접 재학습하려면 학습 code 외에 label table·data provenance·전처리·checkpoint 선택 조건까지 확보해야 한다.
5. 첫 재학습 참고 대상으로는 조건이 구체적인 nartaa S6, 결합·실패 기록 참고 대상으로는 goodpjw .958을 우선한다. 공개 가중치를 별도 대조군으로 실행할지 여부는 이번 조사에서 결정하거나 실행하지 않았다.

우리 R01 .7714는 gold58 개발 OOF이고 위 .95x는 공개 hidden-test LB이므로 직접 차감해서 모델 격차로 해석하지 않는다. 현재 변경들에 대한 새 Kaggle 결과가 나오면 별도 실험 결과 문서에 반영한다.

## Hotness 상위 20개

| 순위 | Notebook | 추천 수 | 공식 최고 공개 AUC |
|---:|---|---:|---:|
| 1 | [rsna-knee-s6fall-solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) | 35 | 0.950 |
| 2 | [mavit-288-baseline-01 : 1.5sec/study @ 1gpu](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu) | 51 | 0.935 |
| 3 | [RSNA · four model ensemble ](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble) | 31 | 0.957 |
| 4 | [RSNA Knee · LB 0.95+](https://www.kaggle.com/code/rabari9999/rsna-knee-lb-0-95) | 38 | 0.950 |
| 5 | [RSNA Knee SOTA Dual-Engine Stack · LB 0.951+](https://www.kaggle.com/code/rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951) | 33 | 0.950 |
| 6 | [RSNA Knee Abnormality Pipeline](https://www.kaggle.com/code/mayankraaz/rsna-knee-abnormality-pipeline) | 31 | 0.956 |
| 7 | [RSNA Knee 0.956+:Micro-Boost & Clinical Tri-Reader](https://www.kaggle.com/code/rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader) | 56 | 0.950 |
| 8 | [rsna-knee-blend3 please upvote if useful](https://www.kaggle.com/code/samanyu1808/rsna-knee-blend3-please-upvote-if-useful) | 27 | 0.951 |
| 9 | [RSNA Knee: real scores of every public notebook](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-real-scores-of-every-public-notebook) | 17 | 미표시 |
| 10 | [RSNA Knee: Raptor student blend (LB 0.944)](https://www.kaggle.com/code/bliverrigpdoge/rsna-knee-raptor-student-blend-lb-0-944) | 27 | 0.944 |
| 11 | [RSNA 0.956: One Free Flip, 68 Ranks Up](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up) | 14 | 0.956 |
| 12 | [RSNA Knee 0945 Efficient 224crop](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop) | 24 | 0.945 |
| 13 | [RSNA Knee · Gold-Gated Triple Reader ](https://www.kaggle.com/code/kozykappa/rsna-knee-gold-gated-triple-reader) | 24 | 0.950 |
| 14 | [RSNA Knee  V15 Tri-Hybrid Ensemble (0.954+) ](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-v15-tri-hybrid-ensemble-0-954) | 22 | 0.954 |
| 15 | [RSNA Knee D4](https://www.kaggle.com/code/matterhorn3838/rsna-knee-d4) | 20 | 0.950 |
| 16 | [RSNA Knee · Rank Blend: Apex 70 + nartaa 30](https://www.kaggle.com/code/karttikjangid05/rsna-knee-rank-blend-apex-70-nartaa-30) | 8 | 0.950 |
| 17 | [Inside the Machine: a 60-second knee film (ep. 2)](https://www.kaggle.com/code/nilsfl/inside-the-machine-a-60-second-knee-film-ep-2) | 12 | 미표시 |
| 18 | [RSNA Knee 🦴 CoAtNet + ConvNeXt MIL · LB 0.950](https://www.kaggle.com/code/hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950) | 60 | 0.950 |
| 19 | [RSNA Knee: What the Top Notebooks Are Made Of](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-what-the-top-notebooks-are-made-of) | 5 | 미표시 |
| 20 | [RSNA Knee V2 vs Apex Comparative Study](https://www.kaggle.com/code/haideptry/rsna-knee-v2-vs-apex-comparative-study) | 61 | 0.950 |

## Most Votes 상위 20개

| 순위 | Notebook | 추천 수 | 공식 최고 공개 AUC |
|---:|---|---:|---:|
| 1 | [RSNA Knee baseline v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 743 | 0.891 |
| 2 | [RSNA Knee Abnormalities - Efficiency LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb) | 599 | 미표시 |
| 3 | [RSNA_Versia_7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | 435 | 0.956 |
| 4 | [RSNA Knee · CoAtNet + R6 + 2.5D MIL · LB 0.958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | 370 | 0.958 |
| 5 | [Bend the Knee to the Dinosaurs](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) | 330 | 0.941 |
| 6 | [rsna-knee-blend-gold⭐⭐⭐（old dataset）](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-old-dataset) | 310 | 0.956 |
| 7 | [RSNA Knee: read the report, then the knee](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee) | 269 | 0.906 |
| 8 | [rsna-knee-d4-blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend) | 265 | 0.946 |
| 9 | [RSNA Knee Fast 2xT4 Inference](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) | 249 | 0.943 |
| 10 | [Bend the Knee to DinoV3 (ensembled)](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled) | 232 | 0.922 |
| 11 | [Bend the Knee to Speedy Raptors-The Original ⭐⭐⭐⭐⭐](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original) | 227 | 0.943 |
| 12 | [≈0.957 · Latest dataset· score in one click!](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | 226 | 0.956 |
| 13 | [rsna-knee-blend-gold⭐⭐(Dataset restored)](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold-dataset-restored) | 206 | 0.954 |
| 14 | [Head and shoulders, knees and toes](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes) | 205 | 0.937 |
| 15 | [Knee MRI: twelve findings from a single model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) | 202 | 0.924 |
| 16 | [rsna-knee-0942-restructured](https://www.kaggle.com/code/maverickss26/rsna-knee-0942-restructured) | 185 | 0.942 |
| 17 | [Blend mine in—private LB luck to all upvoters 🍀](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) | 180 | 0.958 |
| 18 | [This one is old version 0.9573 Newone you can use↑](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use) | 179 | 0.957 |
| 19 | [RSNA Knee · DINOsaur V5? 🦖 ](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v5) | 172 | 0.950 |
| 20 | [RSNA Knee · DINOsaur V4 🦖 ](https://www.kaggle.com/code/romantamrazov/rsna-knee-dinosaur-v4) | 170 | 0.937 |

## 이전 25개 원문 검토와 공식 점수의 연결

기존 원문 해석은 [25개 검토 문서](kaggle_095_twenty_notebook_review.md)에 보존하고, 당시 미확인이던 공식 최고 공개 점수는 아래 감사표로 보완했다. 최신 버전별 제출 점수는 여전히 별도 확인이 필요하다.

| 기존 번호 | 원문 | 공식 역사상 최고 AUC |
|---:|---|---:|
| 1 | [goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) | 0.958 |
| 2 | [goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957](https://www.kaggle.com/code/goodpjw2008/rsna-knee-does-s6-help-next-to-r6-lb-0-957) | 0.957 |
| 3 | [heliosli/this-is-best-public-dataset-new-version-0-9586](https://www.kaggle.com/code/heliosli/this-is-best-public-dataset-new-version-0-9586) | 0.958 |
| 4 | [heliosli/this-one-is-old-version-0-9573-newone-you-can-use](https://www.kaggle.com/code/heliosli/this-one-is-old-version-0-9573-newone-you-can-use) | 0.957 |
| 5 | [heliosli/blend-mine-in-private-lb-luck-to-all-upvoters](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) | 0.958 |
| 6 | [heliosli/0-957-latest-dataset-score-in-one-click](https://www.kaggle.com/code/heliosli/0-957-latest-dataset-score-in-one-click) | 0.956 |
| 7 | [evgendvorkin/rsna-versia-7](https://www.kaggle.com/code/evgendvorkin/rsna-versia-7) | 0.956 |
| 8 | [prvsiyan/the-bee-s-knees-final-rsna-push](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push) | 0.957 |
| 9 | [kaiwalyaatulraut/rsna-knee-competition-solution](https://www.kaggle.com/code/kaiwalyaatulraut/rsna-knee-competition-solution) | 0.958 |
| 10 | [kiyoshiohno/rsna-knee-template](https://www.kaggle.com/code/kiyoshiohno/rsna-knee-template) | 0.958 |
| 11 | [bliverrigpdoge/r7-raptor-knee-centred-convnext](https://www.kaggle.com/code/bliverrigpdoge/r7-raptor-knee-centred-convnext) | 0.957 |
| 12 | [laimerbotrolenk/rsna-four-model-ensemble](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble) | 0.957 |
| 13 | [leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl](https://www.kaggle.com/code/leoprovorov/rsna-knee-0-957-explained-why-the-open-ensembl) | 0.957 |
| 14 | [jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble](https://www.kaggle.com/code/jenillangaliya/rsna-knee-0-956-raptor-convnext-r7-ensemble) | 0.956 |
| 15 | [jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble](https://www.kaggle.com/code/jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble) | 0.954 |
| 16 | [varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956) | 0.956 |
| 17 | [varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up) | 0.956 |
| 18 | [hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950](https://www.kaggle.com/code/hitarthjain0/rsna-knee-coatnet-convnext-mil-lb-0-950) | 0.950 |
| 19 | [rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader](https://www.kaggle.com/code/rabari9999/rsna-knee-0-956-micro-boost-clinical-tri-reader) | 0.950 |
| 20 | [yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-multiscale-orthofoundation-sota-0-956) | 0.956 |
| 21 | [rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951](https://www.kaggle.com/code/rabari9999/rsna-knee-sota-dual-engine-stack-lb-0-951) | 0.950 |
| 22 | [yuvrajlamba01/rsna-knee-v15-tri-hybrid-ensemble-0-954](https://www.kaggle.com/code/yuvrajlamba01/rsna-knee-v15-tri-hybrid-ensemble-0-954) | 0.954 |
| 23 | [pilkwang/rsna-knee-baseline-v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 0.891 |
| 24 | [dreaddevelopment/knee-mri-training-the-twelve-finding-model](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) | 미표시 |
| 25 | [mattiaangeli/bend-the-knee-to-speedy-raptors-the-original](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original) | 0.943 |

근거 파일:

- [공식 점수·순위 snapshot](../outputs/research/kaggle_code_20261019_refresh/official_score_summary.json)
- [이전 25개 공식 점수 대조 CSV](../outputs/research/kaggle_code_20261019_refresh/prior_25_official_score_audit.csv)
- [이번 원문 8개 버전·SHA](../outputs/research/kaggle_code_20261019_refresh/source_manifest.json)
- [저자 S6 학습 설명 원문](../outputs/research/kaggle_code_20261019_refresh/nartaa_discussion_749359_body.json)
- [재조회 script](../outputs/research/kaggle_code_20261019_refresh/snapshot_public_scores.py)
